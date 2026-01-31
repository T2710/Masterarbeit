import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

"""
Build UltraChat SFT examples as streaming .pt shards (nanochat-style).

Each saved example is a tuple: (ids, mask)
- ids: List[int] tokens
- mask: List[int], same length
  mask=1 for assistant CONTENT tokens (supervised)
  mask=0 otherwise

We do NOT do train/val split here.
We build ALL examples once and decide later in training how many to use.

Output files:
  out_dir/
    ultrachat_all_000000.pt
    ultrachat_all_000001.pt
    ...
    meta.json

Run:
  python -m my_gpt2.source.ultrachat --out_dir my_gpt2/source/datasets/ultrachat_sft_examples --num_examples -1
"""

SYSTEM_PROMPT = "You are a helpful assistant.\n"


def _encode(enc, text: str) -> List[int]:
    if not text:
        return []
    return enc.encode_ordinary(text)


def tokenize_ultrachat_example(example, enc, eos_id: int,
                               add_system_prompt: bool = True,
                               supervise_eos: bool = True) -> Tuple[List[int], List[int]]:
    """
    Convert one UltraChat example to (ids, mask).

    UltraChat schema:
      example["messages"] = [{"role": "user"/"assistant"/"system", "content": "..."}]
    """
    msgs = example.get("messages", None)
    if not msgs:
        prompt = (example.get("prompt") or "").strip()
        msgs = [{"role": "user", "content": prompt}]

    ids: List[int] = []
    mask: List[int] = []

    # BOS: use EOS token like GPT-2 (nanochat also uses special tokens)
    ids.append(eos_id)
    mask.append(0)

    # fixed system prompt (recommended)
    if add_system_prompt:
        sys_ids = _encode(enc, SYSTEM_PROMPT)
        ids.extend(sys_ids)
        mask.extend([0] * len(sys_ids))

    for msg in msgs:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()

        if role == "system":
            # ignore dataset system messages (we already inject our own)
            continue

        if role in {"user", "human"}:
            seg = f"User: {content}\n"
            seg_ids = _encode(enc, seg)
            ids.extend(seg_ids)
            mask.extend([0] * len(seg_ids))

        elif role in {"assistant", "gpt"}:
            # prefix NOT supervised
            prefix = "Assistant: "
            prefix_ids = _encode(enc, prefix)
            ids.extend(prefix_ids)
            mask.extend([0] * len(prefix_ids))

            # assistant content supervised
            content_ids = _encode(enc, content)
            ids.extend(content_ids)
            mask.extend([1] * len(content_ids))

            # newline after assistant supervised (helps formatting)
            nl_ids = _encode(enc, "\n")
            ids.extend(nl_ids)
            mask.extend([1] * len(nl_ids))

        else:
            # unknown role -> treat as user
            seg = f"User: {content}\n"
            seg_ids = _encode(enc, seg)
            ids.extend(seg_ids)
            mask.extend([0] * len(seg_ids))

    # exactly ONE EOS at end
    ids.append(eos_id)
    mask.append(1 if supervise_eos else 0)

    return ids, mask


def save_shard(buffer, out_dir: Path, shard_idx: int, prefix: str):
    out_path = out_dir / f"{prefix}_{shard_idx:06d}.pt"
    torch.save(buffer, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="my_gpt2/source/datasets/ultrachat_sft_examples")
    ap.add_argument("--repo", type=str, default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--split", type=str, default="train_sft")
    ap.add_argument("--num_examples", type=int, default=-1, help="-1 = all examples")
    ap.add_argument("--shard_size_examples", type=int, default=10_000, help="examples per shard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_system_prompt", action="store_true")
    ap.add_argument("--no_supervise_eos", action="store_true")
    ap.add_argument("--prefix", type=str, default="ultrachat_all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]  # 50256

    add_system_prompt = not args.no_system_prompt
    supervise_eos = not args.no_supervise_eos

    print("Loading dataset...")
    ds = load_dataset(args.repo, split=args.split)

    total = len(ds) if args.num_examples == -1 else min(args.num_examples, len(ds))
    print(f"Building examples: {total} (split={args.split})")
    print(f"Output dir: {out_dir}")
    print(f"Shard size: {args.shard_size_examples}")
    print(f"System prompt: {add_system_prompt}")
    print(f"Supervise EOS: {supervise_eos}")

    buffer = []
    shard_idx = 0
    n_written = 0

    pbar = tqdm(total=total)
    for i in range(total):
        ex = ds[i]
        ids, mask = tokenize_ultrachat_example(
            ex, enc, eos_id,
            add_system_prompt=add_system_prompt,
            supervise_eos=supervise_eos
        )
        # safety
        if len(ids) != len(mask):
            raise ValueError("ids and mask length mismatch")

        buffer.append((ids, mask))

        # write shard if full
        if len(buffer) >= args.shard_size_examples:
            path = save_shard(buffer, out_dir, shard_idx, args.prefix)
            n_written += len(buffer)
            buffer = []
            shard_idx += 1
            pbar.set_postfix_str(f"saved {path.name} | total={n_written}")

        pbar.update(1)

    # flush remaining
    if len(buffer) > 0:
        path = save_shard(buffer, out_dir, shard_idx, args.prefix)
        n_written += len(buffer)
        pbar.set_postfix_str(f"saved {path.name} | total={n_written}")

    pbar.close()

    meta = {
        "repo": args.repo,
        "split": args.split,
        "num_examples": total,
        "written_examples": n_written,
        "shard_size_examples": args.shard_size_examples,
        "num_shards": shard_idx + (1 if n_written % args.shard_size_examples != 0 else 0),
        "prefix": args.prefix,
        "system_prompt": SYSTEM_PROMPT if add_system_prompt else None,
        "supervise_eos": supervise_eos,
        "tokenizer": "tiktoken:gpt2",
        "eos_id": eos_id,
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n Done.")
    print(f"Wrote {n_written} examples into {out_dir}")
    print("Meta saved to meta.json")


if __name__ == "__main__":
    main()
