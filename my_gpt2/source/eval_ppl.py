import argparse
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_my_model(ckpt_path: str, device: str):
    try:
        from my_gpt2.source.model import GPT, GPTConfig
    except Exception:
        from model import GPT, GPTConfig

    # Explicit weights_only=False because the checkpoint contains config objects.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", ckpt.get("model_args", None))
    if cfg is None:
        raise ValueError("Checkpoint has no 'config' or 'model_args' field.")

    if isinstance(cfg, dict):
        model = GPT(GPTConfig(**cfg))
    else:
        model = GPT(cfg)

    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_text(dataset_name: str, split: str) -> str:
    if dataset_name == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        texts = ds["text"]
    elif dataset_name == "wikitext103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        texts = ds["text"]
    else:
        raise ValueError("dataset_name must be one of: wikitext2, wikitext103")
    return "\n\n".join(texts)


def encode_text(tokenizer: GPT2TokenizerFast, text: str, device: str, max_eval_tokens: Optional[int] = None) -> torch.Tensor:
    # Avoid tokenizer warnings about model_max_length during evaluation.
    old_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    finally:
        tokenizer.model_max_length = old_model_max_length

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    if max_eval_tokens is not None and input_ids.size(1) > max_eval_tokens:
        input_ids = input_ids[:, :max_eval_tokens]
    return input_ids


@torch.no_grad()
def compute_ppl_from_logits(
    input_ids: torch.Tensor,
    forward_logits_fn: Callable[[torch.Tensor, Optional[torch.dtype]], torch.Tensor],
    device: str,
    max_length: int = 1024,
    stride: int = 1024,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, int]:
    """
    Sliding-window perplexity with correct overlap handling.

    We evaluate each token exactly once (except the very first token, which has no
    left context for next-token prediction). This matches the intended HF-style
    sliding-window perplexity but uses explicit logits for both HF and custom models,
    so the comparison is exactly apples-to-apples.
    """
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("nan"), 0

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for end_loc in range(max_length, seq_len + stride, stride):
        end_loc = min(end_loc, seq_len)
        begin_loc = max(end_loc - max_length, 0)
        trg_len = end_loc - prev_end
        if trg_len <= 0:
            break

        x = input_ids[:, begin_loc:end_loc]  # (1, chunk_len)
        chunk_len = x.size(1)
        if chunk_len < 2:
            prev_end = end_loc
            if end_loc == seq_len:
                break
            continue

        logits = forward_logits_fn(x, amp_dtype)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = x[:, 1:].contiguous()

        # Score only the newly introduced tokens of this window.
        # shift_labels[k] corresponds to original token x[:, k+1].
        score_start = max(0, chunk_len - trg_len - 1)
        loss_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        loss_mask[:, score_start:] = 1.0

        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)

        nll_sum += (per_tok * loss_mask).sum().item()
        n_tokens += int(loss_mask.sum().item())

        prev_end = end_loc
        if end_loc == seq_len:
            break

    ppl = math.exp(nll_sum / max(1, n_tokens))
    return ppl, n_tokens


def build_hf_forward(model: GPT2LMHeadModel, device: str):
    def forward_logits(x: torch.Tensor, amp_dtype: Optional[torch.dtype]) -> torch.Tensor:
        if amp_dtype is not None and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(x).logits
        return model(x).logits

    return forward_logits


def build_my_forward(model, device: str):
    def forward_logits(x: torch.Tensor, amp_dtype: Optional[torch.dtype]) -> torch.Tensor:
        if amp_dtype is not None and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(x)
                return logits
        logits, _ = model(x)
        return logits

    return forward_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "wikitext103"])
    ap.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    ap.add_argument("--hf_model", type=str, default="gpt2-medium")
    ap.add_argument("--my_ckpt", type=str, default=None, help="Path to your .pt checkpoint (optional)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--max_eval_tokens", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None

    print(f"device={device}")
    print(
        f"dataset={args.dataset} split={args.split} "
        f"max_length={args.max_length} stride={args.stride}"
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text = load_text(args.dataset, args.split)
    input_ids = encode_text(tokenizer, text, device, max_eval_tokens=args.max_eval_tokens)
    print(f"encoded_tokens={input_ids.size(1)}")

    hf_model = GPT2LMHeadModel.from_pretrained(args.hf_model).to(device)
    hf_model.eval()
    ppl_hf, n_tok_hf = compute_ppl_from_logits(
        input_ids=input_ids,
        forward_logits_fn=build_hf_forward(hf_model, device),
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        amp_dtype=amp_dtype,
    )
    print(f"[HF {args.hf_model}] PPL={ppl_hf:.4f} (scored_tokens={n_tok_hf})")

    if args.my_ckpt:
        my_model = load_my_model(args.my_ckpt, device)
        ppl_my, n_tok_my = compute_ppl_from_logits(
            input_ids=input_ids,
            forward_logits_fn=build_my_forward(my_model, device),
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            amp_dtype=amp_dtype,
        )
        print(f"[MY ckpt={args.my_ckpt}] PPL={ppl_my:.4f} (scored_tokens={n_tok_my})")


if __name__ == "__main__":
    main()
