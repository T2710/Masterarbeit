import os
import re
import json
import glob
import argparse
from typing import List, Dict, Tuple

import torch
import tiktoken

try:
    from my_gpt2.source.model import GPT, GPTConfig
except Exception:
    from model import GPT, GPTConfig


def load_split_records(data_root: str, split: str) -> List[Dict]:
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    shard_paths = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
    if len(shard_paths) == 0:
        raise FileNotFoundError(f"no shard files found in: {split_dir}")

    records = []
    for p in shard_paths:
        shard = torch.load(str(p), map_location="cpu")
        if not isinstance(shard, list):
            raise ValueError(f"unexpected shard format in {p}")
        records.extend(shard)
    return records


def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def split_prompt_and_target(ids, mask, eos_id: int) -> Tuple[List[int], List[int]]:
    ids = to_list(ids)
    mask = to_list(mask)
    if len(ids) != len(mask):
        L = min(len(ids), len(mask))
        ids = ids[:L]
        mask = mask[:L]

    first_target = None
    for i, m in enumerate(mask):
        if int(m) == 1:
            first_target = i
            break
    if first_target is None:
        raise ValueError("no supervised target tokens in example")

    prompt_ids = ids[:first_target]
    target_ids = ids[first_target:]

    while len(prompt_ids) > 0 and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]
    while len(target_ids) > 0 and target_ids[-1] == eos_id:
        target_ids = target_ids[:-1]

    return prompt_ids, target_ids


def extract_question_from_prompt(prompt_text: str) -> str:
    if "Question:" in prompt_text:
        q = prompt_text.split("Question:", 1)[1]
        if "Reasoning:" in q:
            q = q.split("Reasoning:", 1)[0]
        elif "Final Answer:" in q:
            q = q.split("Final Answer:", 1)[0]
        return q.strip()
    return prompt_text.strip()


def maybe_crop_prediction(text: str) -> str:
    t = text.strip()
    if "Final Answer:" in t:
        pre, post = t.rsplit("Final Answer:", 1)
        post = post.strip()
        first_line = post.splitlines()[0].strip() if post else ""
        out = pre.strip()
        if out:
            out = out + "\n\nFinal Answer:\n" + first_line
        else:
            out = "Final Answer:\n" + first_line
        return out.strip()
    return t


def extract_final_answer(text: str) -> str:
    t = text.strip()
    if not t:
        return ""

    m = re.search(r"Final Answer:\s*([^\n\r]+)", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_answer(ans: str) -> str:
    ans = (ans or "").strip()
    ans = re.sub(r"\s+", " ", ans)
    ans = ans.rstrip(".")
    return ans


def detect_mode_from_split(test_split: str) -> str:
    low = test_split.lower()
    if "cot" in low:
        return "cot"
    return "direct"


def split_reasoning_and_final(text: str) -> Tuple[str, str]:
    t = text.strip()
    if "Final Answer:" in t:
        reasoning, final_part = t.rsplit("Final Answer:", 1)
        final_line = final_part.strip().splitlines()[0].strip() if final_part.strip() else ""
        return reasoning.strip(), final_line
    return t, extract_final_answer(t)


@torch.no_grad()
def generate(
    model,
    prompt_ids: List[int],
    device,
    device_type: str,
    eos_id: int,
    max_new_tokens: int,
    stop_on_final_answer: bool = True,
) -> str:
    enc = tiktoken.get_encoding("gpt2")
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    block_size = model.module.config.block_size if hasattr(model, "module") else model.config.block_size

    generated_ids = []
    for _ in range(max_new_tokens):
        if x.size(1) >= block_size:
            break

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            logits, _ = model(x)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = next_token.item()
        x = torch.cat([x, next_token], dim=1)

        if token_id == eos_id:
            break

        generated_ids.append(token_id)
        gen_text = enc.decode(generated_ids)

        if stop_on_final_answer:
            m = re.search(r"Final Answer:\s*([^\n\r]+)", gen_text, flags=re.IGNORECASE)
            if m and m.group(1).strip():
                break

    return enc.decode(generated_ids)


def load_model(checkpoint_path: str, model_name: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "config" in ckpt:
        config = ckpt["config"]
        model = GPT(config)
    else:
        cfg_map = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config = GPTConfig(vocab_size=50304, block_size=1024, **cfg_map[model_name])
        model = GPT(config)

    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(f"unexpected keys when loading checkpoint: {unexpected}")
    if len(missing) > 0:
        print(f"warning: missing keys when loading checkpoint: {missing}")

    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="my_gpt2/source/datasets/gsm8k_ab")
    ap.add_argument("--test_split", type=str, default="test_direct")
    ap.add_argument("--model", type=str, default="gpt2-medium")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_save_examples", type=int, default=20)
    ap.add_argument("--save_dir", type=str, default="my_gpt2/results/eval_gsm8k")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    torch.set_float32_matmul_precision("high")
    mode = detect_mode_from_split(args.test_split)

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]

    records = load_split_records(args.data_root, args.test_split)
    if args.limit is not None:
        records = records[:args.limit]

    model = load_model(args.checkpoint, args.model, device)

    exact_correct = 0
    total = 0
    saved = []

    for idx, rec in enumerate(records):
        ids = rec["ids"]
        mask = rec["mask"]

        prompt_ids, target_ids = split_prompt_and_target(ids, mask, eos_id)
        prompt_text = enc.decode(prompt_ids)
        target_text = enc.decode(target_ids)

        raw_pred = generate(
            model=model,
            prompt_ids=prompt_ids,
            device=device,
            device_type=device_type,
            eos_id=eos_id,
            max_new_tokens=args.max_new_tokens,
            stop_on_final_answer=True,
        )
        pred_text = maybe_crop_prediction(raw_pred)

        question_text = extract_question_from_prompt(prompt_text)

        pred_reasoning, pred_final_raw = split_reasoning_and_final(pred_text)
        sol_reasoning, sol_final_raw = split_reasoning_and_final(target_text)

        pred_final = normalize_answer(pred_final_raw)
        target_final = normalize_answer(sol_final_raw)

        is_correct = (pred_final == target_final)
        exact_correct += int(is_correct)
        total += 1

        if len(saved) < args.num_save_examples:
            saved.append({
                "index": idx,
                "prompt": prompt_text,
                "question": question_text,
                "pred_reasoning": pred_reasoning,
                "pred_final": pred_final,
                "solution_reasoning": sol_reasoning,
                "solution_final": target_final,
                "raw_pred": raw_pred,
                "pred_text": pred_text,
                "target_text": target_text,
                "is_correct": is_correct,
            })

    exact_accuracy = exact_correct / total if total > 0 else 0.0

    metrics = {
        "checkpoint": args.checkpoint,
        "test_split": args.test_split,
        "mode": mode,
        "num_examples": total,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_accuracy,
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.save_dir, "mistakes.json"), "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)

    preview_lines = []
    for item in saved:
        preview_lines.append("=" * 100)
        preview_lines.append(f"INDEX: {item['index']}")
        preview_lines.append("")
        preview_lines.append("PROMPT:")
        preview_lines.append(item["prompt"])
        preview_lines.append("")
        preview_lines.append("Question:")
        preview_lines.append(item["question"])
        preview_lines.append("")
        if mode == "cot":
            preview_lines.append("Reasoning:")
            preview_lines.append(item["pred_reasoning"])
            preview_lines.append("")
            preview_lines.append("Final Answer:")
            preview_lines.append(item["pred_final"])
            preview_lines.append("")
            preview_lines.append("Solution:")
            preview_lines.append("Reasoning:")
            preview_lines.append(item["solution_reasoning"])
            preview_lines.append("")
            preview_lines.append("Final Answer:")
            preview_lines.append(item["solution_final"])
            preview_lines.append("")
        else:
            preview_lines.append("Model Answer:")
            preview_lines.append(item["pred_final"])
            preview_lines.append("")
            preview_lines.append("True Solution:")
            preview_lines.append(item["solution_final"])
            preview_lines.append("")

    with open(os.path.join(args.save_dir, "mistakes_preview.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(preview_lines))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
