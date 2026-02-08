import os
import glob
import time
import random
import argparse
from contextlib import nullcontext
from typing import List, Tuple

import torch
from my_gpt2.source.model import GPT, GPTConfig


# ----------------------------
# Utils
# ----------------------------
def list_shards(data_dir: str, prefix: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No shards found in {data_dir} with pattern {prefix}_*.pt")
    return paths


def load_shard(path: str) -> List[Tuple[List[int], List[int]]]:
    # shard contains list[(ids, mask)]
    return torch.load(path)


def collate_batch(batch, pad_id: int, block_size: int):
    """
    collate:
      - inputs padded with pad_id
      - targets padded with -1 (ignore_index)
      - cap seq length so inputs/targets never exceed block_size
      - left-truncate (keep the end of the conversation)
    """
    B = len(batch)
    # ids length must be <= block_size+1 (because x=ids[:-1] => length block_size)
    max_ids_len = min(max(len(ids) for ids, _ in batch), block_size + 1)
    T = max_ids_len - 1

    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), -1, dtype=torch.long)

    for i, (ids, mask) in enumerate(batch):
        if len(ids) != len(mask):
            raise ValueError("ids and mask length mismatch in batch example")

        # left-truncate if too long
        if len(ids) > max_ids_len:
            ids = ids[-max_ids_len:]
            mask = mask[-max_ids_len:]

        n = len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long)

        # inputs
        x[i, : n - 1] = ids_t[:-1]

        # targets: only supervised tokens (mask==1), others set to -1
        tgt = ids_t[1:].clone()
        m = torch.tensor(mask[1:], dtype=torch.long)
        tgt[m == 0] = -1
        y[i, : n - 1] = tgt

    return x, y


def infinite_example_stream(
    shard_paths: List[str],
    shuffle_shards: bool,
    shuffle_within_shard: bool,
    seed: int,
    max_examples: int = -1,
):
    """
    Yields individual examples (ids, mask).
    - If max_examples == -1 -> loops forever over shards
    - If max_examples != -1 -> yields exactly max_examples then stops
    Memory-safe: loads 1 shard at a time.
    """
    rng = random.Random(seed)
    total_yielded = 0

    while (max_examples == -1) or (total_yielded < max_examples):
        paths = shard_paths[:]
        if shuffle_shards:
            rng.shuffle(paths)

        for sp in paths:
            shard = load_shard(sp)
            if shuffle_within_shard:
                rng.shuffle(shard)

            for ex in shard:
                yield ex
                total_yielded += 1
                if (max_examples != -1) and (total_yielded >= max_examples):
                    return


def batch_generator(example_stream, batch_size: int):
    """
    Turns a stream of examples into batches of size batch_size.
    """
    batch = []
    for ex in example_stream:
        batch.append(ex)
        if len(batch) == batch_size:
            yield batch
            batch = []


@torch.no_grad()
def eval_loss(model, val_batch_gen, device, pad_id, steps: int, autocast_ctx):
    """
    Evaluate average loss for up to `steps` batches.
    If val_batch_gen runs out (StopIteration), we stop early and average over what we got.
    """
    model.eval()
    losses = []
    for _ in range(steps):
        try:
            batch = next(val_batch_gen)
        except StopIteration:
            break
        x, y = collate_batch(batch, pad_id=pad_id, block_size=model.config.block_size)
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)  # requires ignore_index=-1 in your forward loss
            if loss is None:
                raise RuntimeError("Model returned loss=None during eval")
        losses.append(loss.item())
    model.train()

    if len(losses) == 0:
        raise RuntimeError("Validation produced 0 batches (check val_dir/prefix or max_val_examples/eval_steps).")
    return sum(losses) / len(losses)


def load_config_from_ckpt(ckpt) -> GPTConfig:
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("Checkpoint has no 'config'")

    # your code sometimes saves config as dict (__dict__)
    if isinstance(cfg, dict):
        return GPTConfig(**cfg)

    # or it may be a GPTConfig already
    if isinstance(cfg, GPTConfig):
        return cfg

    # last resort: try to use as-is
    return cfg


# ----------------------------
# Main train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # dataset
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)

    # optional explicit val dataset (if not provided -> auto split from train shards)
    ap.add_argument("--val_dir", type=str, default="", help="Optional. If set, use this dir as validation set.")
    ap.add_argument("--val_prefix", type=str, default="", help="Optional. If empty, uses --prefix for val_dir.")

    # auto-split parameters (only used when val_dir is empty)
    ap.add_argument("--val_fraction_shards", type=float, default=0.02, help="fraction of shards for val split (auto)")

    # checkpoints
    ap.add_argument("--pretrain_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # how many examples to use
    ap.add_argument("--max_train_examples", type=int, default=-1, help="-1 = infinite looping over shards")
    ap.add_argument("--max_val_examples", type=int, default=-1, help="-1 = infinite looping over val shards")

    # batching
    ap.add_argument("--device_batch_size", type=int, default=4)
    ap.add_argument("--target_examples_per_step", type=int, default=32)

    # training horizon
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)

    # optimization
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=42)

    # shuffling
    ap.add_argument("--no_shuffle_shards", action="store_true")
    ap.add_argument("--no_shuffle_within_shard", action="store_true")

    args = ap.parse_args()

    print("sft_train.py loaded from:", __file__, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    print("Device:", device, flush=True)

    # ----------------------------
    # Load shard lists
    # ----------------------------
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    train_shards_all = list_shards(args.data_dir, args.prefix)

    if args.val_dir.strip():
        # Explicit val set
        if not os.path.isdir(args.val_dir):
            raise FileNotFoundError(f"val_dir not found: {args.val_dir}")

        val_prefix = args.val_prefix.strip() if args.val_prefix.strip() else args.prefix
        val_shards = list_shards(args.val_dir, val_prefix)
        train_shards = train_shards_all

        print("Using explicit validation directory.", flush=True)
        print(f"Train: dir={args.data_dir} prefix={args.prefix} shards={len(train_shards)}", flush=True)
        print(f"Val:   dir={args.val_dir} prefix={val_prefix} shards={len(val_shards)}", flush=True)
    else:
        # Auto-split by shards
        rng = random.Random(args.seed)
        all_shards = train_shards_all[:]
        rng.shuffle(all_shards)

        n_shards = len(all_shards)
        n_val_shards = max(1, int(args.val_fraction_shards * n_shards))
        val_shards = all_shards[:n_val_shards]
        train_shards = all_shards[n_val_shards:]

        print("Using automatic shard split for validation.", flush=True)
        print(f"Found shards: {n_shards} total | {len(train_shards)} train | {len(val_shards)} val", flush=True)

    # ----------------------------
    # Load pretrained model
    # ----------------------------
    if not os.path.exists(args.pretrain_ckpt):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {args.pretrain_ckpt}")

    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    config = load_config_from_ckpt(ckpt)

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("Loaded pretrain ckpt:", args.pretrain_ckpt, flush=True)

    # pad token = EOS (works because padded targets are ignored via -1)
    pad_id = 50256

    # mixed precision
    if args.dtype == "bfloat16" and device == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device_type=device_type,
    )

    # ----------------------------
    # grad accumulation
    # ----------------------------
    if args.target_examples_per_step % args.device_batch_size != 0:
        raise ValueError("target_examples_per_step must be divisible by device_batch_size")
    grad_accum_steps = args.target_examples_per_step // args.device_batch_size
    print("device_batch_size:", args.device_batch_size, flush=True)
    print("target_examples_per_step:", args.target_examples_per_step, flush=True)
    print("=> grad_accum_steps:", grad_accum_steps, flush=True)

    # If using finite train stream, ensure max_steps feasible
    if args.max_train_examples != -1:
        max_possible_steps = args.max_train_examples // args.target_examples_per_step
        if max_possible_steps <= 0:
            raise ValueError("max_train_examples too small for one step. Increase max_train_examples.")
        if args.max_steps > max_possible_steps:
            print(
                f"⚠️ reducing max_steps from {args.max_steps} to {max_possible_steps} "
                f"(because max_train_examples={args.max_train_examples})",
                flush=True,
            )
            args.max_steps = max_possible_steps

    # If using finite val stream, ensure eval_steps feasible (or eval will stop early)
    if args.max_val_examples != -1:
        max_possible_eval_steps = args.max_val_examples // args.device_batch_size
        if max_possible_eval_steps <= 0:
            raise ValueError("max_val_examples too small to produce even 1 val batch.")
        if args.eval_steps > max_possible_eval_steps:
            print(
                f"⚠️ reducing eval_steps from {args.eval_steps} to {max_possible_eval_steps} "
                f"(because max_val_examples={args.max_val_examples})",
                flush=True,
            )
            args.eval_steps = max_possible_eval_steps

    # ----------------------------
    # streams
    # ----------------------------
    shuffle_shards = not args.no_shuffle_shards
    shuffle_within = not args.no_shuffle_within_shard

    train_stream = infinite_example_stream(
        train_shards,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within,
        seed=args.seed,
        max_examples=args.max_train_examples,
    )
    train_batches = batch_generator(train_stream, args.device_batch_size)

    # LR schedule: linear decay (clamped)
    def lr_for_step(step: int) -> float:
        frac = 1.0 - (step / max(1, args.max_steps))
        return args.lr * max(0.0, frac)

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    t_print = time.time()

    model.train()
    for step in range(args.max_steps):
        # eval
        if step % args.eval_every == 0:
            # IMPORTANT: for stable eval, do not shuffle
            val_stream = infinite_example_stream(
                val_shards,
                shuffle_shards=False,
                shuffle_within_shard=False,
                seed=args.seed + 123,
                max_examples=args.max_val_examples,
            )
            val_batches = batch_generator(val_stream, args.device_batch_size)

            vloss = eval_loss(
                model,
                val_batches,
                device,
                pad_id,
                steps=args.eval_steps,
                autocast_ctx=autocast_ctx,
            )
            print(f"[eval] step {step:05d} | val_loss {vloss:.4f}", flush=True)

            if vloss < best_val:
                best_val = vloss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "config": config.__dict__ if hasattr(config, "__dict__") else config,
                        "val_loss": vloss,
                    },
                    best_path,
                )
                print(f"✅ saved best.pt -> {best_path}", flush=True)

        # save periodic
        if step > 0 and step % args.save_every == 0:
            path = os.path.join(args.out_dir, f"ckpt_{step:06d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "config": config.__dict__ if hasattr(config, "__dict__") else config,
                },
                path,
            )
            print(f"saved checkpoint -> {path}", flush=True)

        # set LR
        lr = lr_for_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_supervised_tokens = 0

        for micro in range(grad_accum_steps):
            try:
                batch = next(train_batches)
            except StopIteration:
                raise RuntimeError(
                    "Train data exhausted (StopIteration). "
                    "Increase --max_train_examples or reduce --max_steps / --target_examples_per_step."
                )

            x, y = collate_batch(batch, pad_id=pad_id, block_size=model.config.block_size)
            x, y = x.to(device), y.to(device)
            total_supervised_tokens += (y != -1).sum().item()

            with autocast_ctx:
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("Model returned loss=None during training")
                loss = loss / grad_accum_steps

            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # periodic print
        if step % 50 == 0:
            dt = time.time() - t_print
            print(
                f"step {step:05d}/{args.max_steps} | "
                f"train_loss {total_loss:.4f} | lr {lr:.2e} | "
                f"sup_toks {total_supervised_tokens:,} | dt {dt:.2f}s",
                flush=True,
            )
            t_print = time.time()

    # final save
    final_path = os.path.join(args.out_dir, "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "step": args.max_steps,
            "config": config.__dict__ if hasattr(config, "__dict__") else config,
        },
        final_path,
    )
    print(f"saved final.pt -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
