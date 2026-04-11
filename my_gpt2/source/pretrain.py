import argparse
import math
import os
import random
import signal
import socket
import time

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from my_gpt2.source.hellaswag import render_example, iterate_examples
    from my_gpt2.source.model import GPT, GPTConfig
except ImportError:
    from hellaswag import render_example, iterate_examples
    from model import GPT, GPTConfig


"""
To optimize training time:
1. Max out Batch Size to still fit in GPU memory
2. Choose Float network precision (e.g. float32 , 16..) if supported by hardware
3. Use torch.compile
4. Numbers devisible by 2, 8, etc. are faster due to gpu architecture
"""
"""
Todo:
Fix torch compile
1. Pretraining
2. SFT
3. SFT auf Reasoningdaten
4. RL mit Reasoningdaten
"""

"""
Todo:
Evtl. Beim Pretraining shard bzw. damit Dokumentenreihenfolge zufällig mischen, damit keine dependezenzen zwischen shards (Reihenfolge) entstehen
wenn man mehrfach über die Daten geht
"""


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process, data_root):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.master_process = master_process
        assert split in {"train", "val"}

        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"data_root='{data_root}' not found. "
                f"Run the shard builder and point data_root to the right folder!"
            )
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def _load_shard(self, shard_index):
        self.current_shard = shard_index
        self.tokens = load_tokens(self.shards[self.current_shard])

    def reset(self):
        self._load_shard(0)
        self.current_position = self.B * self.T * self.process_rank

    def state_dict(self):
        shard_position_base = self.current_position - (self.B * self.T * self.process_rank)
        return {
            "split": self.split,
            "current_shard": self.current_shard,
            "shard_position_base": shard_position_base,
        }

    def load_state_dict(self, state):
        if state.get("split", self.split) != self.split:
            raise ValueError(
                f"Loader split mismatch: checkpoint={state.get('split')} current={self.split}"
            )

        shard_index = int(state["current_shard"])
        if not 0 <= shard_index < len(self.shards):
            raise ValueError(
                f"Checkpoint shard index {shard_index} out of range for split {self.split}"
            )

        self._load_shard(shard_index)
        shard_position_base = int(state["shard_position_base"])
        required_tokens = self.B * self.T * self.num_processes + 1
        max_position = len(self.tokens) - required_tokens
        if shard_position_base < 0 or shard_position_base > max_position:
            raise ValueError(
                f"Checkpoint shard position {shard_position_base} invalid for shard "
                f"{self.shards[shard_index]} with length {len(self.tokens)}"
            )
        self.current_position = shard_position_base + (self.B * self.T * self.process_rank)

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self._load_shard((self.current_shard + 1) % len(self.shards))
            self.current_position = B * T * self.process_rank
        return x, y


def numpy_rng_state_to_dict(state):
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "bit_generator": name,
        "state": keys.tolist(),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def numpy_rng_state_from_dict(state):
    return (
        state["bit_generator"],
        np.array(state["state"], dtype=np.uint32),
        state["pos"],
        state["has_gauss"],
        state["cached_gaussian"],
    )


def capture_local_rng_state():
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": numpy_rng_state_to_dict(np.random.get_state()),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def gather_rng_state(ddp, ddp_world_size):
    local_state = capture_local_rng_state()
    if ddp:
        gathered = [None for _ in range(ddp_world_size)]
        dist.all_gather_object(gathered, local_state)
    else:
        gathered = [local_state]
    return {"per_rank": gathered}


def normalize_rng_tensor(state):
    if state is None:
        return None
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.uint8)
    if state.dtype != torch.uint8:
        state = state.to(dtype=torch.uint8)
    if state.device.type != "cpu":
        state = state.cpu()
    return state.contiguous()


def restore_local_rng_state(rng_state, ddp_rank):
    if not rng_state:
        return

    per_rank = rng_state.get("per_rank", [])
    if not per_rank:
        return

    if ddp_rank < len(per_rank):
        local_state = per_rank[ddp_rank]
    elif len(per_rank) == 1:
        local_state = per_rank[0]
    else:
        raise ValueError(
            f"Checkpoint RNG state has {len(per_rank)} ranks, cannot restore rank {ddp_rank}"
        )

    random.setstate(local_state["python"])
    np.random.set_state(numpy_rng_state_from_dict(local_state["numpy"]))
    torch.set_rng_state(normalize_rng_tensor(local_state["torch"]))
    if torch.cuda.is_available() and local_state.get("cuda") is not None:
        cuda_states = [normalize_rng_tensor(state) for state in local_state["cuda"]]
        current_device = torch.cuda.current_device()
        if current_device < len(cuda_states):
            torch.cuda.set_rng_state(cuda_states[current_device])
        elif cuda_states:
            torch.cuda.set_rng_state(cuda_states[0])


# for hellaswag dataset evaluation
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def build_model_from_args(model_name: str):
    model_cfg_map = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
    }
    mcfg = model_cfg_map[model_name]
    return GPT(GPTConfig(vocab_size=50304, block_size=1024, **mcfg))


def atomic_torch_save(checkpoint, path):
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def build_args_snapshot(args, max_steps, ddp_world_size, grad_accum_steps):
    snapshot = dict(vars(args))
    snapshot["resolved_max_steps"] = max_steps
    snapshot["world_size"] = ddp_world_size
    snapshot["grad_accum_steps"] = grad_accum_steps
    return snapshot


def validate_resume_args(args, resume_ckpt, max_steps, ddp_world_size, grad_accum_steps):
    snapshot = resume_ckpt.get("args_snapshot")
    if snapshot is None:
        raise ValueError("Resume checkpoint does not contain an args_snapshot entry.")

    mismatches = []
    exact_fields = {
        "model": args.model,
        "B": args.B,
        "T": args.T,
        "total_batch_size": args.total_batch_size,
        "max_lr": args.max_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "tokens_target": args.tokens_target,
        "resolved_max_steps": max_steps,
        "world_size": ddp_world_size,
        "grad_accum_steps": grad_accum_steps,
    }

    for key, current_value in exact_fields.items():
        expected_value = snapshot.get(key)
        if expected_value != current_value:
            mismatches.append(
                f"{key}: checkpoint={expected_value} current={current_value}"
            )

    checkpoint_data_root = snapshot.get("data_root")
    if checkpoint_data_root is None:
        mismatches.append("data_root: missing from checkpoint args_snapshot")
    else:
        checkpoint_root = os.path.abspath(checkpoint_data_root)
        current_root = os.path.abspath(args.data_root)
        if checkpoint_root != current_root:
            mismatches.append(
                f"data_root: checkpoint={checkpoint_root} current={current_root}"
            )

    if mismatches:
        raise ValueError(
            "Resume checkpoint is incompatible with the current run:\n- "
            + "\n- ".join(mismatches)
        )


def default_log_dir(args, model_tag):
    if args.run_name is not None and args.run_name.strip() != "":
        return os.path.join(args.log_dir, args.run_name)
    return f"{args.log_dir}_{model_tag}"


def write_run_config(run_config_path, args, init_meta, extra_meta, append=False):
    mode = "a" if append else "w"
    with open(run_config_path, mode, encoding="utf-8") as f:
        if append:
            f.write("\n# resume_event\n")
            prefix = "resume_"
        else:
            prefix = ""

        for k, v in sorted(vars(args).items()):
            f.write(f"{prefix}arg_{k}={v}\n")
        for k, v in sorted(init_meta.items()):
            f.write(f"{prefix}meta_{k}={v}\n")
        for k, v in sorted(extra_meta.items()):
            f.write(f"{prefix}{k}={v}\n")


def load_model_for_run(args, device, master_process):
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("--init_checkpoint and --resume_checkpoint are mutually exclusive.")

    if args.resume_checkpoint is not None:
        if master_process:
            print(f"loading resume checkpoint: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", None)
        if config is None:
            raise ValueError("Resume checkpoint does not contain a 'config' entry.")
        state = ckpt.get("model", None)
        if state is None:
            raise ValueError("Resume checkpoint does not contain a 'model' state_dict entry.")

        model = GPT(config)
        model.load_state_dict(state)
        init_meta = {
            "resume_from_checkpoint": True,
            "warm_start_from_checkpoint": False,
            "start_step": int(ckpt.get("resume_step", ckpt.get("step", 0))),
            "source_checkpoint": args.resume_checkpoint,
            "source_step": ckpt.get("step", None),
            "source_resume_step": ckpt.get("resume_step", ckpt.get("step", None)),
            "source_val_loss": ckpt.get("last_val_loss", ckpt.get("val_loss", None)),
            "source_model_tag": ckpt.get("model_tag", None),
        }
        return model, init_meta, ckpt

    if args.init_checkpoint is None:
        model = build_model_from_args(args.model)
        init_meta = {
            "resume_from_checkpoint": False,
            "warm_start_from_checkpoint": False,
            "start_step": 0,
            "source_checkpoint": None,
        }
        return model, init_meta, None

    if master_process:
        print(f"loading init checkpoint: {args.init_checkpoint}")
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", None)
    if config is None:
        raise ValueError("Checkpoint does not contain a 'config' entry.")

    model = GPT(config)
    state = ckpt.get("model", None)
    if state is None:
        raise ValueError("Checkpoint does not contain a 'model' state_dict entry.")
    model.load_state_dict(state)

    init_meta = {
        "resume_from_checkpoint": False,
        "warm_start_from_checkpoint": True,
        "start_step": 0,
        "source_checkpoint": args.init_checkpoint,
        "source_step": ckpt.get("step", None),
        "source_resume_step": ckpt.get("resume_step", ckpt.get("step", None)),
        "source_val_loss": ckpt.get("last_val_loss", ckpt.get("val_loss", None)),
        "source_model_tag": ckpt.get("model_tag", None),
    }
    return model, init_meta, None


def build_checkpoint_base(
    raw_model,
    model_tag,
    args,
    total_batch_size,
    log_dir,
    last_val_loss,
    resume_step,
    save_reason,
):
    return {
        "model": raw_model.state_dict(),
        "config": raw_model.config,
        "step": resume_step,
        "resume_step": resume_step,
        "completed_step": resume_step - 1,
        "val_loss": last_val_loss,
        "last_val_loss": last_val_loss,
        "model_tag": model_tag,
        "tokens_target": args.tokens_target,
        "total_batch_size": total_batch_size,
        "init_checkpoint": args.init_checkpoint,
        "resume_checkpoint": args.resume_checkpoint,
        "data_root": args.data_root,
        "run_name": args.run_name,
        "log_dir": log_dir,
        "save_reason": save_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--data_root", type=str, default="my_gpt2/source/datasets/edu_fineweb10B")
    ap.add_argument("--tokens_target", type=int, default=10_000_000_000)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_lr", type=float, default=6e-4)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--total_batch_size", type=int, default=524288)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--val_loss_steps", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_dir", type=str, default="results/log_pretraining")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--init_checkpoint", type=str, default=None)
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    args = ap.parse_args()

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP mode requires CUDA."
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    signal_exit_requested = False

    def handle_preemption(signum, _frame):
        nonlocal signal_exit_requested
        signal_exit_requested = True
        print(
            f"rank {ddp_rank} received signal {signum}; "
            "will save checkpoint and exit after the current optimizer step.",
            flush=True,
        )

    signal.signal(signal.SIGUSR1, handle_preemption)

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    max_lr = args.max_lr
    min_lr = max_lr * 0.1
    warmup_steps = args.warmup_steps

    total_batch_size = args.total_batch_size
    B = args.B
    T = args.T

    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if args.max_steps is None:
        max_steps = math.ceil(args.tokens_target / total_batch_size)
    else:
        max_steps = args.max_steps

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    enc = tiktoken.get_encoding("gpt2")
    if master_process:
        print(f"model: {args.model}")
        print(f"data_root: {args.data_root}")
        print(f"total desired batch size: {total_batch_size}")
        print(f"calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"max_steps: {max_steps}")
        print(f"init_checkpoint: {args.init_checkpoint}")
        print(f"resume_checkpoint: {args.resume_checkpoint}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process, data_root=args.data_root)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process, data_root=args.data_root)
    torch.set_float32_matmul_precision('high')

    model, init_meta, resume_state = load_model_for_run(args, device, master_process)
    model.to(device)

    use_compile = False
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    if master_process:
        c = raw_model.config
        print(f"arch: n_layer={c.n_layer} n_head={c.n_head} n_embd={c.n_embd} block_size={c.block_size} vocab={c.vocab_size}")

    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=max_lr, device_type=device_type)

    if init_meta["resume_from_checkpoint"]:
        if resume_state.get("optimizer") is None:
            raise ValueError("Resume checkpoint does not contain an optimizer state.")
        if resume_state.get("train_loader_state") is None:
            raise ValueError("Resume checkpoint does not contain a train_loader_state entry.")
        validate_resume_args(args, resume_state, max_steps, ddp_world_size, grad_accum_steps)
        optimizer.load_state_dict(resume_state["optimizer"])
        train_loader.load_state_dict(resume_state["train_loader_state"])
        restore_local_rng_state(resume_state.get("rng_state"), ddp_rank)

    model_tag = args.model
    log_dir = default_log_dir(args, model_tag)
    if init_meta["resume_from_checkpoint"] and resume_state.get("log_dir"):
        stored_log_dir = resume_state["log_dir"]
        if master_process and os.path.normpath(stored_log_dir) != os.path.normpath(log_dir):
            print(f"resume checkpoint overrides requested log_dir with: {stored_log_dir}")
        log_dir = stored_log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    run_config_path = os.path.join(log_dir, "run_config.txt")
    resume_latest_path = os.path.join(log_dir, "resume_latest.pt")

    config_meta = {
        "resolved_max_steps": max_steps,
        "world_size": ddp_world_size,
        "grad_accum_steps": grad_accum_steps,
        "log_dir": log_dir,
    }
    if master_process:
        if init_meta["resume_from_checkpoint"]:
            write_run_config(run_config_path, args, init_meta, config_meta, append=True)
        else:
            write_run_config(run_config_path, args, init_meta, config_meta, append=False)

    if init_meta["resume_from_checkpoint"]:
        start_step = init_meta["start_step"]
        last_val_loss = resume_state.get("last_val_loss", resume_state.get("val_loss", float("nan")))
        if start_step >= max_steps:
            raise ValueError(
                f"Resume checkpoint start_step={start_step} is not smaller than max_steps={max_steps}."
            )
        if master_process:
            resume_banner = (
                f"=== RESUME from {args.resume_checkpoint} at step {start_step} ==="
            )
            print(resume_banner, flush=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(resume_banner + "\n")
    else:
        start_step = 0
        last_val_loss = float("nan")
        if master_process:
            with open(log_file, "w", encoding="utf-8"):
                pass

    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % args.eval_every == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_step = args.val_loss_steps
                for _ in range(val_loss_step):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    loss = loss / val_loss_step
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            last_val_loss = val_loss_accum.item()
            if master_process:
                print(f"validation loss: {last_val_loss:.4f}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} val {last_val_loss:.4f}\n")

        if (step % args.eval_every == 0 or last_step) and (not use_compile):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        if ((step > 0 and step % args.eval_every == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(xgen)
                    logits = logits[:, -1, :]
                    if logits.size(-1) > enc.n_vocab:
                        logits[:, enc.n_vocab:] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    topk = min(50, probs.size(-1))
                    topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        exit_requested_tensor = torch.tensor(
            int(signal_exit_requested),
            device=device if device_type == "cuda" else "cpu",
            dtype=torch.int32,
        )
        if ddp:
            dist.all_reduce(exit_requested_tensor, op=dist.ReduceOp.MAX)
        exit_after_step = bool(exit_requested_tensor.item())

        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        resume_step = step + 1
        should_save_checkpoint = (
            resume_step % args.ckpt_every == 0
            or last_step
            or exit_after_step
        )
        if should_save_checkpoint:
            rng_state = gather_rng_state(ddp, ddp_world_size)
            if master_process:
                if last_step:
                    save_reason = "final"
                elif exit_after_step:
                    save_reason = "signal"
                else:
                    save_reason = "interval"

                checkpoint_base = build_checkpoint_base(
                    raw_model=raw_model,
                    model_tag=model_tag,
                    args=args,
                    total_batch_size=total_batch_size,
                    log_dir=log_dir,
                    last_val_loss=last_val_loss,
                    resume_step=resume_step,
                    save_reason=save_reason,
                )

                archive_checkpoint = dict(checkpoint_base)
                archive_path = os.path.join(log_dir, f"{model_tag}_model_{resume_step:05d}.pt")
                torch.save(archive_checkpoint, archive_path)

                resume_checkpoint = dict(checkpoint_base)
                resume_checkpoint["optimizer"] = optimizer.state_dict()
                resume_checkpoint["train_loader_state"] = train_loader.state_dict()
                resume_checkpoint["rng_state"] = rng_state
                resume_checkpoint["args_snapshot"] = build_args_snapshot(
                    args=args,
                    max_steps=max_steps,
                    ddp_world_size=ddp_world_size,
                    grad_accum_steps=grad_accum_steps,
                )
                resume_checkpoint["job_meta"] = {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "saved_at_unix": time.time(),
                    "rank": ddp_rank,
                    "world_size": ddp_world_size,
                }
                atomic_torch_save(resume_checkpoint, resume_latest_path)
                print(
                    f"saved checkpoints for resume_step {resume_step} ({save_reason})",
                    flush=True,
                )
            if ddp:
                dist.barrier()

        if exit_after_step:
            if master_process:
                exit_msg = f"graceful stop requested; exiting after checkpoint at resume_step {resume_step}"
                print(exit_msg, flush=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(exit_msg + "\n")
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
