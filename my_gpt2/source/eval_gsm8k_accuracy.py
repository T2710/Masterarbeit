# my_gpt2/source/eval_gsm8k_accuracy.py

import os
import re
import glob
import argparse
from typing import List, Tuple, Optional

import torch
import tiktoken
from contextlib import nullcontext

from my_gpt2.source.model import GPT, GPTConfig


# ----------------------------
# Loading GSM8K SFT shards (list[(ids, mask)])
# ----------------------------
def list_shards(data_dir: str, prefix: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No shards found in {data_dir} with prefix {prefix}_*.pt")
    return paths


def load_all_examples(data_dir: str, prefix: str) -> List[Tuple[List[int], List[int]]]:
    shards = list_shards(data_dir, prefix)
    examples = []
    for p in shards:
        examples.extend(torch.load(p))
    return examples


# ----------------------------
# Parse ground-truth answer from supervised tokens
# ----------------------------
FINAL_RE = re.compile(r"Final Answer:\s*([^\n<]+)", re.IGNORECASE)


def decode_supervised(enc, ids: List[int], mask: List[int]) -> str:
    sup_ids = [tid for tid, m in zip(ids, mask) if m == 1]
    return enc.decode(sup_ids)


def extract_final_answer_text(text: str) -> Optional[str]:
    m = FINAL_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()
    ans = ans.replace("\u00a0", " ").strip()
    return ans


def normalize_numberish(s: str) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return s.lower().strip()

    num = m.group(0).replace(",", "")
    if num in ("", "-", ".", "-."):
        return None

    try:
        if "." in num:
            x = float(num)
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return format(x, "g")
        return str(int(num))
    except Exception:
        return num.strip()


def get_ground_truth_norm(enc, ids: List[int], mask: List[int]) -> Optional[str]:
    sup_text = decode_supervised(enc, ids, mask)
    gt_text = extract_final_answer_text(sup_text)
    return normalize_numberish(gt_text)


# ----------------------------
# Prompt extraction: slice until first supervised token
# ----------------------------
def first_supervised_index(mask: List[int]) -> int:
    for k, m in enumerate(mask):
        if m == 1:
            return k
    return len(mask)


def decode_prompt(enc, ids: List[int], mask: List[int]) -> str:
    k = first_supervised_index(mask)
    return enc.decode(ids[:k])


# ----------------------------
# Model loading
# ----------------------------
def load_model_from_ckpt(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", None)
    if isinstance(cfg, dict):
        config = GPTConfig(**cfg)
    else:
        config = cfg
    if config is None:
        raise RuntimeError("Checkpoint has no config")

    model = GPT(config)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


# ----------------------------
# Greedy generation (temp=0)
# ----------------------------
@torch.no_grad()
def greedy_generate(model: GPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
    x = input_ids
    for _ in range(max_new_tokens):
        if x.size(1) > model.config.block_size:
            x = x[:, -model.config.block_size:]

        logits, _ = model(x)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)

        if next_id.item() == eos_id:
            break
    return x


def extract_pred_norm_from_generation(full_text: str) -> Optional[str]:
    ans_text = extract_final_answer_text(full_text)
    return normalize_numberish(ans_text)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--print_misses", type=int, default=10)
    args = ap.parse_args()

    device = args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]

    examples = load_all_examples(args.data_dir, args.prefix)
    if args.limit != -1:
        examples = examples[:args.limit]

    model = load_model_from_ckpt(args.ckpt, device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (args.amp and device_type == "cuda")
        else nullcontext()
    )

    correct = 0
    total = 0
    misses_shown = 0

    for ids, mask in examples:
        gt = get_ground_truth_norm(enc, ids, mask)
        if gt is None:
            continue

        k = first_supervised_index(mask)
        prompt_ids = ids[:k]
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        with autocast_ctx:
            out = greedy_generate(model, x, max_new_tokens=args.max_new_tokens, eos_id=eos_id)

        gen_text = enc.decode(out[0].tolist())
        pred = extract_pred_norm_from_generation(gen_text)

        total += 1
        if pred == gt:
            correct += 1
        else:
            if misses_shown < args.print_misses:
                prompt_text = enc.decode(prompt_ids)
                print("--------------------------------------------------------------------------------")
                print(f"GT:   {gt}")
                print(f"PRED: {pred}")
                print("PROMPT (tail):")
                print(prompt_text[-800:])
                print("GEN (tail):")
                print(gen_text[-1200:])
                misses_shown += 1

    acc = correct / max(1, total)
    print("====================================")
    print(f"ckpt={args.ckpt}")
    print(f"dataset_dir={args.data_dir} prefix={args.prefix}")
    print(f"evaluated={total} correct={correct} acc={acc:.4f}")
    print("====================================")


if __name__ == "__main__":
    main()
