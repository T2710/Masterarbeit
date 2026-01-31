import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = "You are a helpful assistant.\n"


def _encode(enc, text: str) -> List[int]:
    if not text:
        return []
    return enc.encode_ordinary(text)


def parse_gsm8k_answer(ans: str) -> Tuple[str, str]:
    """
    Returns (reasoning, final_answer).
    GSM8K commonly formats the final answer as: '#### <number>'
    """
    if ans is None:
        return "", ""
    ans = ans.strip()
    if "####" in ans:
        parts = ans.split("####", 1)
        reasoning = parts[0].strip()
        final = parts[1].strip()
        return reasoning, final
    lines = [l.strip() for l in ans.splitlines() if l.strip()]
    if len(lines) == 0:
        return "", ""
    final = lines[-1]
    reasoning = "\n".join(lines[:-1]).strip()
    return reasoning, final


def tokenize_gsm8k_example(
    example,
    enc,
    eos_id: int,
    mode: str,
    add_system_prompt: bool = True,
    supervise_eos: bool = True,
) -> Tuple[List[int], List[int]]:
    question = (example.get("question") or "").strip()
    answer = (example.get("answer") or "").strip()

    reasoning, final = parse_gsm8k_answer(answer)

    if mode not in {"cot", "direct"}:
        raise ValueError("mode must be 'cot' or 'direct'")

    ids: List[int] = []
    mask: List[int] = []

    ids.append(eos_id)
    mask.append(0)

    if add_system_prompt:
        sys_ids = _encode(enc, SYSTEM_PROMPT)
        ids.extend(sys_ids)
        mask.extend([0] * len(sys_ids))

    user_seg = f"User: {question}\n"
    user_ids = _encode(enc, user_seg)
    ids.extend(user_ids)
    mask.extend([0] * len(user_ids))

    prefix = "Assistant: "
    prefix_ids = _encode(enc, prefix)
    ids.extend(prefix_ids)
    mask.extend([0] * len(prefix_ids))

    if mode == "cot":
        content = reasoning.strip()
        if content:
            content = content + "\n"
        content = content + f"Final Answer: {final}".strip()
    else:
        content = f"Final Answer: {final}".strip()

    content_ids = _encode(enc, content)
    ids.extend(content_ids)
    mask.extend([1] * len(content_ids))

    nl_ids = _encode(enc, "\n")
    ids.extend(nl_ids)
    mask.extend([1] * len(nl_ids))

    ids.append(eos_id)
    mask.append(1 if supervise_eos else 0)

    if len(ids) != len(mask):
        raise ValueError("ids and mask length mismatch")

    return ids, mask


def save_shard(buffer, out_dir: Path, shard_idx: int, prefix: str):
    out_path = out_dir / f"{prefix}_{shard_idx:06d}.pt"
    torch.save(buffer, out_path)
    return out_path


def decode_supervised(enc, ids: List[int], mask: List[int]) -> str:
    sup_ids = [tid for tid, m in zip(ids, mask) if m == 1]
    try:
        return enc.decode(sup_ids)
    except Exception:
        return "<decode failed>"


def decode_full(enc, ids: List[int]) -> str:
    try:
        return enc.decode(ids)
    except Exception:
        return "<decode failed>"


def build_split(
    out_dir: Path,
    repo: str,
    config_name: Optional[str],
    split: str,
    num_examples: int,
    shard_size_examples: int,
    seed: int,
    add_system_prompt: bool,
    supervise_eos: bool,
    prefix: str,
    mode: str,
    do_test_prints: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]  # 50256

    ds = load_dataset(repo, config_name, split=split) if config_name else load_dataset(repo, split=split)
    total = len(ds) if num_examples == -1 else min(num_examples, len(ds))

    print(f"\nBuilding GSM8K examples | mode={mode} | split={split}")
    print(f"repo={repo} config={config_name} out_dir={out_dir}")
    print(f"num_examples={total} shard_size_examples={shard_size_examples}")
    print(f"system_prompt={add_system_prompt} supervise_eos={supervise_eos}")

    buffer = []
    shard_idx = 0
    n_written = 0

    pbar = tqdm(total=total)
    for i in range(total):
        ex = ds[i]
        ids, mask = tokenize_gsm8k_example(
            ex, enc, eos_id,
            mode=mode,
            add_system_prompt=add_system_prompt,
            supervise_eos=supervise_eos,
        )
        buffer.append((ids, mask))

        if len(buffer) >= shard_size_examples:
            path = save_shard(buffer, out_dir, shard_idx, prefix)
            n_written += len(buffer)
            buffer = []
            shard_idx += 1
            pbar.set_postfix_str(f"saved {path.name} | total={n_written}")

        pbar.update(1)

    if len(buffer) > 0:
        path = save_shard(buffer, out_dir, shard_idx, prefix)
        n_written += len(buffer)
        pbar.set_postfix_str(f"saved {path.name} | total={n_written}")

    pbar.close()

    meta = {
        "repo": repo,
        "config": config_name,
        "split": split,
        "mode": mode,
        "num_examples": total,
        "written_examples": n_written,
        "shard_size_examples": shard_size_examples,
        "prefix": prefix,
        "system_prompt": SYSTEM_PROMPT if add_system_prompt else None,
        "supervise_eos": supervise_eos,
        "tokenizer": "tiktoken:gpt2",
        "eos_id": eos_id,
        "seed": seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote {n_written} examples into {out_dir}")

    if do_test_prints:
        shard_paths = sorted([p for p in out_dir.glob(f"{prefix}_*.pt")])
        if len(shard_paths) == 0:
            print("No shards found for test prints.")
            return
        shard0 = torch.load(str(shard_paths[0]))
        print(f"\nTest prints | mode={mode} | split={split}")
        print(f"Loaded shard: {shard_paths[0].name} | examples in shard: {len(shard0)}")

        for j in range(min(3, len(shard0))):
            ids, mask = shard0[j]
            full = decode_full(enc, ids)
            sup = decode_supervised(enc, ids, mask)
            sup_cnt = sum(mask)
            print("\n" + "-" * 80)
            print(f"example[{j}] | total_tokens={len(ids)} | supervised_tokens={sup_cnt}")
            print("FULL (tail 800 chars):")
            print(full[-800:])
            print("\nSUPERVISED (tail 800 chars):")
            print(sup[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="my_gpt2/source/datasets/gsm8k_sft_examples")
    ap.add_argument("--repo", type=str, default="openai/gsm8k")
    ap.add_argument("--config", type=str, default="main")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--num_examples", type=int, default=-1, help="-1 = all examples")
    ap.add_argument("--shard_size_examples", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_system_prompt", action="store_true")
    ap.add_argument("--no_supervise_eos", action="store_true")
    ap.add_argument("--mode", type=str, default="both", choices=["cot", "direct", "both"])
    ap.add_argument("--prefix", type=str, default="gsm8k")
    ap.add_argument("--no_test_prints", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    add_system_prompt = not args.no_system_prompt
    supervise_eos = not args.no_supervise_eos
    do_test_prints = not args.no_test_prints

    config_name = args.config if args.config not in {"", "none", "None"} else None

    if args.mode in {"cot", "both"}:
        build_split(
            out_dir=out_dir / f"{args.prefix}_cot",
            repo=args.repo,
            config_name=config_name,
            split=args.split,
            num_examples=args.num_examples,
            shard_size_examples=args.shard_size_examples,
            seed=args.seed,
            add_system_prompt=add_system_prompt,
            supervise_eos=supervise_eos,
            prefix=f"{args.prefix}_cot",
            mode="cot",
            do_test_prints=do_test_prints,
        )

    if args.mode in {"direct", "both"}:
        build_split(
            out_dir=out_dir / f"{args.prefix}_direct",
            repo=args.repo,
            config_name=config_name,
            split=args.split,
            num_examples=args.num_examples,
            shard_size_examples=args.shard_size_examples,
            seed=args.seed,
            add_system_prompt=add_system_prompt,
            supervise_eos=supervise_eos,
            prefix=f"{args.prefix}_direct",
            mode="direct",
            do_test_prints=do_test_prints,
        )


if __name__ == "__main__":
    main()
