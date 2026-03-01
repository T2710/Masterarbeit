import os
import math
import time
import json
import glob
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken

from my_gpt2.source.model import GPT, GPTConfig


class GSM8KSFTDataset(Dataset):
    """
    Loads all .pt shards in a split directory. Each record is expected to be:
        {"ids": List[int], "mask": List[int]}

    ids: full tokenized sequence including prompt + target (+ optional eos)
    mask: 1 only where loss should be applied
    """
    def __init__(self, data_root: str, split: str):
        self.data_root = data_root
        self.split = split
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"split dir not found: {split_dir}")

        shard_paths = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        if len(shard_paths) == 0:
            raise FileNotFoundError(f"no shard files found in: {split_dir}")

        records: List[Dict[str, List[int]]] = []
        for shard_path in shard_paths:
            shard = torch.load(shard_path)
            if not isinstance(shard, list):
                raise ValueError(f"unexpected shard format in {shard_path}")
            records.extend(shard)

        if len(records) == 0:
            raise ValueError(f"no records found in split: {split_dir}")

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.records[idx]


class SFTCollator:
    """
    Converts variable-length full sequences into padded LM inputs.

    Given ids=[t0,t1,...,tn] and mask aligned to ids,
    we train on:
        idx     = ids[:-1]
        targets = ids[1:]
        loss_mask = mask[1:]

    Padding:
        idx pad with eos_id
        targets pad with -1 (ignored by CE)
        loss_mask pad with 0
    """
    def __init__(self, eos_id: int, max_seq_len: int):
        self.eos_id = eos_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_x, batch_y, batch_m = [], [], []
        max_len = 0

        for rec in batch:
            ids = rec["ids"]
            mask = rec["mask"]
            if len(ids) != len(mask):
                raise ValueError("ids/mask length mismatch in batch record")
            if len(ids) < 2:
                raise ValueError("each example must contain at least 2 tokens")
            if len(ids) > self.max_seq_len:
                raise ValueError(
                    f"example length {len(ids)} exceeds max_seq_len={self.max_seq_len}. "
                    "Filter long samples during dataset build."
                )

            x = ids[:-1]
            y = ids[1:]
            m = mask[1:]
            max_len = max(max_len, len(x))
            batch_x.append(x)
            batch_y.append(y)
            batch_m.append(m)

        x_out = torch.full((len(batch), max_len), self.eos_id, dtype=torch.long)
        y_out = torch.full((len(batch), max_len), -1, dtype=torch.long)
        m_out = torch.zeros((len(batch), max_len), dtype=torch.float32)

        for i, (x, y, m) in enumerate(zip(batch_x, batch_y, batch_m)):
            L = len(x)
            x_out[i, :L] = torch.tensor(x, dtype=torch.long)
            y_out[i, :L] = torch.tensor(y, dtype=torch.long)
            m_out[i, :L] = torch.tensor(m, dtype=torch.float32)

        return x_out, y_out, m_out


MODEL_CFG_MAP = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
}


def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP mode requires CUDA"
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(ddp_local_rank)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def cleanup_ddp(ddp: bool):
    if ddp:
        dist.destroy_process_group()



def build_model(args) -> Tuple[GPT, str]:
    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if "config" in ckpt:
            config = ckpt["config"]
        else:
            mcfg = MODEL_CFG_MAP[args.model]
            config = GPTConfig(vocab_size=args.vocab_size, block_size=args.max_seq_len, **mcfg)
        model = GPT(config)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            raise RuntimeError(f"unexpected keys when loading checkpoint: {unexpected}")
        run_model_name = ckpt.get("model_tag", args.model)
        if len(missing) > 0:
            print(f"warning: missing keys when loading checkpoint: {missing}")
        return model, run_model_name

    mcfg = MODEL_CFG_MAP[args.model]
    model = GPT(GPTConfig(vocab_size=args.vocab_size, block_size=args.max_seq_len, **mcfg))
    return model, args.model


@torch.no_grad()
def evaluate(model, val_loader, device, device_type, ddp: bool) -> float:
    model.eval()
    val_loss_sum = 0.0
    val_steps = 0

    for x, y, loss_mask in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            _, loss = model(x, y, loss_mask=loss_mask)
        val_loss_sum += loss.detach()
        val_steps += 1

    if val_steps == 0:
        raise RuntimeError("validation loader is empty")

    val_loss = val_loss_sum / val_steps
    if ddp:
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    return val_loss.item()


def _clean_decoded(text: str) -> str:
    return text.replace("<|endoftext|>", "").strip()


def _extract_question_from_prompt(prompt_text: str) -> str:
    text = _clean_decoded(prompt_text)
    if "Question:" in text:
        text = text.split("Question:", 1)[1]
    if "Reasoning:" in text:
        text = text.split("Reasoning:", 1)[0]
    elif "Final Answer:" in text:
        text = text.split("Final Answer:", 1)[0]
    return text.strip()


@torch.no_grad()
def preview_generations(model, dataset, device, device_type, eos_id: int, max_new_tokens: int = 120, num_samples: int = 5):
    """
    Prints a few qualitative validation examples during training.
    For each example, shows:
      - Question
      - Model Answer
      - True Solution
    """
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    shown = min(num_samples, len(dataset))

    for i in range(shown):
        rec = dataset.records[i]
        ids = rec["ids"]
        mask = rec["mask"]
        prompt_ids = [tid for tid, m in zip(ids, mask) if m == 0]
        target_ids = [tid for tid, m in zip(ids, mask) if m == 1]

        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        block_size = model.module.config.block_size if hasattr(model, "module") else model.config.block_size

        for _ in range(max_new_tokens):
            if x.size(1) >= block_size:
                break
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
                logits, _ = model(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        prompt_text = enc.decode(prompt_ids)
        target_text = _clean_decoded(enc.decode(target_ids))
        model_text = _clean_decoded(enc.decode(x[0].tolist()[len(prompt_ids):]))
        question_text = _extract_question_from_prompt(prompt_text)

        print("-" * 100)
        print(f"INDEX: {i}")
        print("\nQUESTION:")
        print(question_text)
        print("\nMODEL ANSWER:")
        print(model_text)
        print("\nTRUE SOLUTION:")
        print(target_text)



def main():
    ap = argparse.ArgumentParser(description="Instruction/Math SFT training for GPT-2-style models")
    ap.add_argument("--model", type=str, default="gpt2-medium", choices=list(MODEL_CFG_MAP.keys()))
    ap.add_argument("--init_checkpoint", type=str, default=None,
                    help="Path to a pretrained checkpoint from pretrain.py. Strongly recommended for SFT.")
    ap.add_argument("--data_root", type=str, default="my_gpt2/source/datasets/gsm8k_ab")
    ap.add_argument("--train_split", type=str, default="train_direct")
    ap.add_argument("--val_split", type=str, default="val_direct")
    ap.add_argument("--output_root", type=str, default="my_gpt2/results/instruction_sft")
    ap.add_argument("--run_name", type=str, default=None,
                    help="If omitted, uses '<train_split>_<model>'.")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Micro-batch size per process in number of examples.")
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--max_lr", type=float, default=2e-5)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--vocab_size", type=int, default=50304)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--preview_every", type=int, default=0,
                    help="If > 0, print a few generations every N optimizer steps on rank 0.")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Optional hard cap on optimizer steps.")
    ap.add_argument("--use_compile", action="store_true")
    args = ap.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]

    train_dataset = GSM8KSFTDataset(args.data_root, args.train_split)
    val_dataset = GSM8KSFTDataset(args.data_root, args.val_split)
    collator = SFTCollator(eos_id=eos_id, max_seq_len=args.max_seq_len)

    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )

    model, run_model_name = build_model(args)
    model.to(device)
    if args.use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        device_type=device_type,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    min_lr = args.max_lr * args.min_lr_ratio

    def get_lr(it: int) -> float:
        if it < args.warmup_steps:
            return args.max_lr * (it + 1) / max(1, args.warmup_steps)
        if total_steps <= args.warmup_steps:
            return min_lr
        if it >= total_steps:
            return min_lr
        decay_ratio = (it - args.warmup_steps) / (total_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.max_lr - min_lr)

    run_name = args.run_name or f"{args.train_split}_{run_model_name}"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "log.txt")
    args_file = os.path.join(out_dir, "args.json")
    if master_process:
        with open(log_file, "w", encoding="utf-8") as f:
            pass
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

        print(f"train examples: {len(train_dataset)}")
        print(f"val examples: {len(val_dataset)}")
        print(f"steps_per_epoch: {steps_per_epoch}")
        print(f"total optimizer steps: {total_steps}")
        print(f"output dir: {out_dir}")
        print(f"device: {device}")
        print(f"ddp_world_size: {ddp_world_size}")

    optimizer_step = 0
    best_val_loss = float("inf")
    stop_training = False

    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        micro_step = 0

        while True:
            if optimizer_step >= total_steps:
                stop_training = True
                break

            pending_batches = []
            for _ in range(args.grad_accum_steps):
                try:
                    pending_batches.append(next(train_iter))
                except StopIteration:
                    break

            if len(pending_batches) == 0:
                break

            current_accum_steps = len(pending_batches)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            tokens_supervised_accum = 0.0
            t0 = time.time()

            for accum_idx, (x, y, loss_mask) in enumerate(pending_batches):
                if ddp:
                    model.require_backward_grad_sync = (accum_idx == current_accum_steps - 1)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss_mask = loss_mask.to(device, non_blocking=True)
                tokens_supervised_accum += loss_mask.sum().item()

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
                    _, loss = model(x, y, loss_mask=loss_mask)
                loss = loss / current_accum_steps
                loss_accum += loss.detach()
                loss.backward()
                micro_step += 1

            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(optimizer_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()
            if device_type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            if master_process:
                supervised_tok_per_sec = tokens_supervised_accum / max(dt, 1e-9)
                msg = (
                    f"step {optimizer_step:5d} | epoch {epoch:2d} | "
                    f"loss {loss_accum.item():.6f} | lr {lr:.4e} | "
                    f"norm {grad_norm:.4f} | dt {dt*1000:.2f}ms | "
                    f"sup_tok/s {supervised_tok_per_sec:.2f}"
                )
                print(msg)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{optimizer_step} train {loss_accum.item():.6f}\n")

            do_eval = (optimizer_step % args.eval_every == 0) or (optimizer_step == total_steps - 1)
            if do_eval:
                if ddp and val_sampler is not None:
                    val_sampler.set_epoch(epoch)
                val_loss = evaluate(model, val_loader, device, device_type, ddp)
                if master_process:
                    print(f"validation loss: {val_loss:.6f}")
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"{optimizer_step} val {val_loss:.6f}\n")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(out_dir, "best.pt")
                        torch.save({
                            "model": raw_model.state_dict(),
                            "config": raw_model.config,
                            "step": optimizer_step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "best_val_loss": best_val_loss,
                            "train_split": args.train_split,
                            "val_split": args.val_split,
                            "run_name": run_name,
                        }, best_path)

            do_ckpt = (optimizer_step > 0 and optimizer_step % args.ckpt_every == 0) or (optimizer_step == total_steps - 1)
            if master_process and do_ckpt:
                ckpt_path = os.path.join(out_dir, f"ckpt_{optimizer_step:05d}.pt")
                torch.save({
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": optimizer_step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                    "run_name": run_name,
                }, ckpt_path)

            if do_eval and master_process:
                preview_generations(
                    model,
                    val_dataset,
                    device,
                    device_type,
                    eos_id=eos_id,
                    max_new_tokens=120,
                    num_samples=5,
                )

            optimizer_step += 1

        if stop_training:
            break

    cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
