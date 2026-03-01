import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def parse_gsm8k_answer(answer: Optional[str]) -> Tuple[str, str]:
    if answer is None:
        return "", ""
    text = answer.strip()
    if not text:
        return "", ""
    if "####" in text:
        reasoning, final_answer = text.split("####", 1)
        return reasoning.strip(), final_answer.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return "", lines[0]
    return "\n".join(lines[:-1]).strip(), lines[-1]


def normalize_final_answer(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_prompt_and_target(question: str, final_answer: str, reasoning: str, mode: str) -> Tuple[str, str]:
    question = question.strip()
    final_answer = normalize_final_answer(final_answer)
    reasoning = reasoning.strip()

    if mode == "direct":
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Final Answer:\n"
        )
        target = final_answer
        return prompt, target

    if mode == "cot":
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Reasoning:\n"
        )
        if reasoning:
            target = reasoning + "\n\nFinal Answer:\n" + final_answer
        else:
            target = "Final Answer:\n" + final_answer
        return prompt, target

    raise ValueError(f"Unsupported mode: {mode}")


def encode_plain(enc: tiktoken.Encoding, text: str) -> List[int]:
    return enc.encode_ordinary(text) if text else []


def tokenize_example(
    example: Dict,
    enc: tiktoken.Encoding,
    eos_id: int,
    mode: str,
    supervise_eos: bool = True,
) -> Optional[Dict[str, List[int]]]:
    question = (example.get("question") or "").strip()
    answer = (example.get("answer") or "").strip()
    reasoning, final_answer = parse_gsm8k_answer(answer)
    if not question or not final_answer:
        return None

    prompt_text, target_text = build_prompt_and_target(question, final_answer, reasoning, mode)
    prompt_ids = encode_plain(enc, prompt_text)
    target_ids = encode_plain(enc, target_text)
    newline_ids = encode_plain(enc, "\n")

    ids = [eos_id]
    mask = [0]
    ids.extend(prompt_ids)
    mask.extend([0] * len(prompt_ids))
    ids.extend(target_ids)
    mask.extend([1] * len(target_ids))
    ids.extend(newline_ids)
    mask.extend([1] * len(newline_ids))
    ids.append(eos_id)
    mask.append(1 if supervise_eos else 0)
    assert len(ids) == len(mask)

    full_text = prompt_text + target_text + "\n"

    return {
        "ids": ids,
        "mask": mask,
        "question": question,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "full_text": full_text,
    }


def make_train_val_indices(num_examples: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    indices = list(range(num_examples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(num_examples * val_fraction)))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return train_indices, val_indices


def save_shard(records: Sequence[Dict[str, List[int]]], out_dir: Path, prefix: str, shard_idx: int) -> Path:
    out_path = out_dir / f"{prefix}_{shard_idx:06d}.pt"
    torch.save(list(records), out_path)
    return out_path


def maybe_clear_existing_files(out_dir: Path, prefix: str, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for path in out_dir.glob(f"{prefix}_*.pt"):
        path.unlink(missing_ok=True)
    for extra_name in ["meta.json", "indices.json", "preview.txt"]:
        extra_path = out_dir / extra_name
        if extra_path.exists():
            extra_path.unlink()


def build_mode_split(
    ds,
    indices: Sequence[int],
    out_dir: Path,
    prefix: str,
    mode: str,
    shard_size_examples: int,
    max_seq_len: int,
    supervise_eos: bool,
    overwrite: bool,
    do_preview: bool,
    repo: str,
    config_name: Optional[str],
    hf_split: str,
    split_name: str,
    seed: int,
) -> None:
    maybe_clear_existing_files(out_dir, prefix, overwrite)
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]

    buffer = []
    shard_idx = 0
    kept = dropped_missing = dropped_too_long = 0
    total_supervised_tokens = total_tokens = 0
    preview_samples = []

    pbar = tqdm(indices, desc=f"Build {split_name}/{mode}")
    for idx in pbar:
        record = tokenize_example(ds[idx], enc, eos_id, mode, supervise_eos)
        if record is None:
            dropped_missing += 1
            continue
        if len(record["ids"]) > max_seq_len:
            dropped_too_long += 1
            continue
        buffer.append({"ids": record["ids"], "mask": record["mask"]})
        kept += 1
        total_tokens += len(record["ids"])
        total_supervised_tokens += sum(record["mask"])

        if do_preview and len(preview_samples) < 5:
            preview_samples.append(
                "=" * 100
                + f"\nMODE: {mode} | SPLIT: {split_name} | INDEX: {idx}\n"
                + "-" * 100
                + "\nSEQUENCE WRITTEN TO SHARD:\n"
                + record["full_text"]
                + "\n"
            )

        if len(buffer) >= shard_size_examples:
            shard_path = save_shard(buffer, out_dir, prefix, shard_idx)
            shard_idx += 1
            buffer = []
            pbar.set_postfix_str(f"kept={kept} last={shard_path.name}")
    if buffer:
        shard_path = save_shard(buffer, out_dir, prefix, shard_idx)
        pbar.set_postfix_str(f"kept={kept} last={shard_path.name}")

    with open(out_dir / "indices.json", "w", encoding="utf-8") as f:
        json.dump({"split_name": split_name, "mode": mode, "num_indices": len(indices), "indices": list(indices)}, f, indent=2)

    meta = {
        "repo": repo,
        "config": config_name,
        "hf_split": hf_split,
        "local_split": split_name,
        "mode": mode,
        "tokenizer": "tiktoken:gpt2",
        "eos_id": eos_id,
        "shard_size_examples": shard_size_examples,
        "max_seq_len": max_seq_len,
        "supervise_eos": supervise_eos,
        "seed": seed,
        "num_source_indices": len(indices),
        "num_written_examples": kept,
        "dropped_missing_examples": dropped_missing,
        "dropped_too_long_examples": dropped_too_long,
        "avg_total_tokens": (total_tokens / kept) if kept else 0.0,
        "avg_supervised_tokens": (total_supervised_tokens / kept) if kept else 0.0,
        "sequence_format": {
            "direct": "Question:\\n<task>\\n\\nFinal Answer:\\n<answer>\\n",
            "cot": "Question:\\n<task>\\n\\nReasoning:\\n<reasoning>\\n\\nFinal Answer:\\n<answer>\\n",
        },
        "note": "preview.txt shows the actual text sequence written into the shard; no separate TARGET label is part of the training text.",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if do_preview and preview_samples:
        with open(out_dir / "preview.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(preview_samples))

    print(
        f"Finished {split_name}/{mode}: written={kept}, dropped_missing={dropped_missing}, "
        f"dropped_too_long={dropped_too_long}, avg_tokens={(total_tokens / kept) if kept else 0:.1f}, "
        f"avg_supervised={(total_supervised_tokens / kept) if kept else 0:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GSM8K direct/CoT SFT datasets with exact-sequence preview.")
    parser.add_argument("--out_dir", type=str, default="my_gpt2/source/datasets/gsm8k_ab")
    parser.add_argument("--repo", type=str, default="openai/gsm8k")
    parser.add_argument("--config", type=str, default="main")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="both", choices=["direct", "cot", "both"])
    parser.add_argument("--shard_size_examples", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--supervise_eos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_name = None if args.config in {"", "none", "None"} else args.config

    ds_train = load_dataset(args.repo, config_name, split="train") if config_name else load_dataset(args.repo, split="train")
    ds_test = load_dataset(args.repo, config_name, split="test") if config_name else load_dataset(args.repo, split="test")

    train_indices, val_indices = make_train_val_indices(len(ds_train), args.val_fraction, args.seed)
    test_indices = list(range(len(ds_test)))
    requested_modes = ["direct", "cot"] if args.mode == "both" else [args.mode]

    print("Building datasets with exact-sequence preview:")
    print(f"repo={args.repo} config={config_name}")
    print(f"train_examples={len(train_indices)} val_examples={len(val_indices)} test_examples={len(test_indices)}")
    print(f"modes={requested_modes} max_seq_len={args.max_seq_len} shard_size_examples={args.shard_size_examples}")
    print(f"output_root={out_dir}")

    for mode in requested_modes:
        build_mode_split(
            ds_train, train_indices, out_dir / f"train_{mode}", f"gsm8k_{mode}", mode,
            args.shard_size_examples, args.max_seq_len, args.supervise_eos, args.overwrite,
            not args.no_preview, args.repo, config_name, "train", "train", args.seed
        )
        build_mode_split(
            ds_train, val_indices, out_dir / f"val_{mode}", f"gsm8k_{mode}", mode,
            args.shard_size_examples, args.max_seq_len, args.supervise_eos, args.overwrite,
            not args.no_preview, args.repo, config_name, "train", "val", args.seed
        )
        build_mode_split(
            ds_test, test_indices, out_dir / f"test_{mode}", f"gsm8k_{mode}", mode,
            args.shard_size_examples, args.max_seq_len, args.supervise_eos, args.overwrite,
            not args.no_preview, args.repo, config_name, "test", "test", args.seed
        )


if __name__ == "__main__":
    main()
