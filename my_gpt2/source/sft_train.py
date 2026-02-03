# my_gpt2/source/sft_train.py

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
        raise FileNotFoundError(f"No shards found in {data_dir} with prefix {prefix}_*.pt")
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
        assert len(ids) == len(mask)

        # left-truncate if too long
        if len(ids) > max_ids_len:
            ids = ids[-max_ids_len:]
            mask = mask[-max_ids_len:]

        n = len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long)

        # inputs
        x[i, :n - 1] = ids_t[:-1]

        # targets (mask out unsupervised tokens)
        tgt = ids_t[1:].clone()
        m = torch.tensor(mask[1:], dtype=torch.long)
        tgt[m == 0] = -1
        y[i, :n - 1] = tgt

    return x, y


def infinite_example_stream(
    shard_paths: List[str],
    shuffle_shards: bool,
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

    while True:
        paths = shard_paths[:]
        if shuffle_shards:
            rng.shuffle(paths)

        for sp in paths:
            shard = load_shard(sp)
            if shuffle_shards:
                rng.shuffle(shard)

            for ex in shard:
                yield ex
                total_yielded += 1
                if max_examples != -1 and total_yielded >= max_examples:
                    return

        # if max_examples == -1: loop forever
        if max_examples != -1:
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
def eval_loss(model, val_batch_gen, device, pad_id, steps: int, autocast_ctx=nullcontext()):
    model.eval()
    losses = []
    for _ in range(steps):
        batch = next(val_batch_gen)
        x, y = collate_batch(batch, pad_id=pad_id, block_size=model.config.block_size)
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)  # requires ignore_index=-1 in your forward loss
            if loss is None:
                raise RuntimeError("Model returned loss=None during eval")
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ----------------------------
# Main train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="my_gpt2/source/datasets/ultrachat_sft_examples")
    ap.add_argument("--prefix", type=str, default="ultrachat_all")
    ap.add_argument("--pretrain_ckpt", type=str, default="my_gpt2/results/pretraining/log_pretraining_gpt2/gpt2_model_70000.pt")
    ap.add_argument("--out_dir", type=str, default="my_gpt2/results/sft_ultrachat")

    # how many examples to use (this is what you wanted!)
    ap.add_argument("--max_train_examples", type=int, default=150_000, help="-1 = use all available")
    ap.add_argument("--max_val_examples", type=int, default=2_000, help="-1 = use all available")
    ap.add_argument("--val_fraction_shards", type=float, default=0.02, help="fraction of shards for val split")

    # batching
    ap.add_argument("--device_batch_size", type=int, default=4)
    ap.add_argument("--target_examples_per_step", type=int, default=32)

    # training horizon
    ap.add_argument("--max_steps", type=int, default=8000, help="SFT should be short (2k-8k often enough)")
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)

    # optimization
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    print("Device:", device)

    # ----------------------------
    # Load shards and split train/val by shard list
    # ----------------------------
    all_shards = list_shards(args.data_dir, args.prefix)

    # ✅ Fix 3: shuffle shard list once (so val is not always the first files)
    rng = random.Random(args.seed)
    rng.shuffle(all_shards)

    n_shards = len(all_shards)
    n_val_shards = max(1, int(args.val_fraction_shards * n_shards))
    val_shards = all_shards[:n_val_shards]
    train_shards = all_shards[n_val_shards:]
    print(f"Found shards: {n_shards} total | {len(train_shards)} train | {len(val_shards)} val")

    # ----------------------------
    # Load pretrained model
    # ----------------------------
    if not os.path.exists(args.pretrain_ckpt):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {args.pretrain_ckpt}")

    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")

    cfg = ckpt.get("config", None)
    if isinstance(cfg, dict):
        config = GPTConfig(**cfg)
    elif cfg is not None:
        config = cfg
    else:
        raise RuntimeError("Checkpoint has no config")

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("Loaded pretrain ckpt:", args.pretrain_ckpt)

    # pad token = EOS (works because padded targets are ignored)
    pad_id = 50256

    # mixed precision
    if args.dtype == "bfloat16" and device == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    # optimizer (your model sets betas internally)
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device_type=device_type,
    )

    # ----------------------------
    # grad accumulation
    # ----------------------------
    examples_per_step = args.device_batch_size  # single GPU
    assert args.target_examples_per_step % examples_per_step == 0, \
        "target_examples_per_step must be divisible by device_batch_size"
    grad_accum_steps = args.target_examples_per_step // examples_per_step
    print("device_batch_size:", args.device_batch_size)
    print("target_examples_per_step:", args.target_examples_per_step)
    print("=> grad_accum_steps:", grad_accum_steps)

    # ✅ Fix 2: ensure max_steps is feasible with max_train_examples
    if args.max_train_examples != -1:
        max_possible_steps = args.max_train_examples // args.target_examples_per_step
        if max_possible_steps <= 0:
            raise ValueError("max_train_examples too small for one step. Increase max_train_examples.")
        if args.max_steps > max_possible_steps:
            print(f"⚠️ reducing max_steps from {args.max_steps} to {max_possible_steps} "
                  f"(because max_train_examples={args.max_train_examples})")
            args.max_steps = max_possible_steps

    # ----------------------------
    # build streaming batch generator for training (train can stop after max_examples)
    # ----------------------------
    train_stream = infinite_example_stream(
        train_shards,
        shuffle_shards=True,
        seed=args.seed,
        max_examples=args.max_train_examples,
    )
    train_batches = batch_generator(train_stream, args.device_batch_size)

    # LR schedule: linear decay (clamped)
    def lr_for_step(step: int) -> float:
        frac = 1.0 - (step / max(1, args.max_steps))
        if frac < 0.0:
            frac = 0.0
        return args.lr * frac

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val = 1e9
    t0 = time.time()

    model.train()
    for step in range(args.max_steps):
        # build a fresh val loader for every evaluation 
        if step % args.eval_every == 0:
            val_stream = infinite_example_stream(
                val_shards,
                shuffle_shards=False,
                seed=args.seed + 1 + step,   # step-dependent seed is fine
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
            print(f"[eval] step {step:05d} | val_loss {vloss:.4f}")

            if vloss < best_val:
                best_val = vloss
                torch.save(
                    {"model": model.state_dict(), "step": step, "config": config.__dict__, "val_loss": vloss},
                    os.path.join(args.out_dir, "best.pt"),
                )
                print("✅ saved best.pt")

        # save periodic
        if step > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "config": config.__dict__},
                os.path.join(args.out_dir, f"ckpt_{step:06d}.pt"),
            )
            print("saved checkpoint")

        # set LR
        lr = lr_for_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_supervised_tokens = 0

        for micro in range(grad_accum_steps):
            batch = next(train_batches)
            x, y = collate_batch(batch, pad_id=pad_id, block_size=model.config.block_size)
            x, y = x.to(device), y.to(device)

            # count supervised tokens
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

        if step % 50 == 0:
            dt = time.time() - t0
            print(
                f"step {step:05d}/{args.max_steps} | "
                f"train_loss {total_loss:.4f} | lr {lr:.2e} | "
                f"sup_toks {total_supervised_tokens:,} | dt {dt:.2f}s"
            )
            t0 = time.time()

    # final save
    torch.save(
        {"model": model.state_dict(), "step": args.max_steps, "config": config.__dict__},
        os.path.join(args.out_dir, "final.pt"),
    )
    print("saved final.pt")


if __name__ == "__main__":
    main()
