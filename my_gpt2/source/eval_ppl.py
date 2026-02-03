import argparse
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# optional: dein eigenes Modell
def load_my_model(ckpt_path: str, device: str):
    from my_gpt2.source.model import GPT, GPTConfig
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"] if "config" in ckpt else ckpt.get("model_args", None)
    if cfg is None:
        raise ValueError("Checkpoint hat kein 'config' Feld. Erwartet: checkpoint['config'].")
    # cfg kann schon GPTConfig sein; ansonsten dict
    if isinstance(cfg, dict):
        model = GPT(GPTConfig(**cfg))
    else:
        model = GPT(cfg)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def ppl_sliding_window_hf(model, tokenizer, text, device, max_length=1024, stride=1024, max_eval_tokens=None, amp_dtype=None):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if max_eval_tokens is not None and input_ids.size(1) > max_eval_tokens:
        input_ids = input_ids[:, :max_eval_tokens]

    nll_sum = 0.0
    n_tokens = 0

    seq_len = input_ids.size(1)
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - begin
        if trg_len <= 0:
            break

        input_chunk = input_ids[:, begin:end]
        labels = input_chunk.clone()
        # GPT2-style: berechne Loss nur auf den Tokens im Fenster
        # (bei stride=max_length ist das identisch zu "no overlap")
        # Kein zusätzlicher Kontext außerhalb des Fensters.
        if amp_dtype is not None and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(input_chunk, labels=labels)
                loss = out.loss
        else:
            out = model(input_chunk, labels=labels)
            loss = out.loss

        nll_sum += loss.item() * trg_len
        n_tokens += trg_len

        if end == seq_len:
            break

    ppl = math.exp(nll_sum / max(1, n_tokens))
    return ppl, n_tokens

@torch.no_grad()
def ppl_sliding_window_my(model, tokenizer, text, device, max_length=1024, stride=1024, max_eval_tokens=None, amp_dtype=None):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if max_eval_tokens is not None and input_ids.size(1) > max_eval_tokens:
        input_ids = input_ids[:, :max_eval_tokens]

    nll_sum = 0.0
    n_tokens = 0

    seq_len = input_ids.size(1)
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - begin
        if trg_len <= 0:
            break

        x = input_ids[:, begin:end]               # (1, T)
        # logits: (1, T, vocab)
        if amp_dtype is not None and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(x)
        else:
            logits, _ = model(x)

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = x[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        # loss ist pro Token (für die shift_labels Länge = trg_len-1)
        eff_len = max(0, trg_len - 1)
        nll_sum += loss.item() * eff_len
        n_tokens += eff_len

        if end == seq_len:
            break

    ppl = math.exp(nll_sum / max(1, n_tokens))
    return ppl, n_tokens

def load_text(dataset_name: str, split: str):
    if dataset_name == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        texts = ds["text"]
    elif dataset_name == "wikitext103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        texts = ds["text"]
    else:
        raise ValueError("dataset_name must be one of: wikitext2, wikitext103")

    # Standard: Texte zu einem String konkatenieren (wie in den HF PPL-Beispielen)
    return "\n\n".join(texts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "wikitext103"])
    ap.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    ap.add_argument("--hf_model", type=str, default="gpt2", help="z.B. gpt2 oder gpt2-medium")
    ap.add_argument("--my_ckpt", type=str, default="my_gpt2/results/pretraining/log_pretraining_gpt2/gpt2_model_70000.pt", help="Pfad zu deinem .pt Checkpoint (optional)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--max_eval_tokens", type=int, default=None, help="optional: eval nur auf ersten N Tokens")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast auf CUDA")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None

    print(f"device={device}")
    print(f"dataset={args.dataset} split={args.split} max_length={args.max_length} stride={args.stride}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    text = load_text(args.dataset, args.split)

    # Baseline: HF GPT-2
    hf_model = GPT2LMHeadModel.from_pretrained(args.hf_model).to(device)
    hf_model.eval()
    ppl_hf, n_tok_hf = ppl_sliding_window_hf(
        hf_model, tokenizer, text, device,
        max_length=args.max_length, stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
        amp_dtype=amp_dtype,
    )
    print(f"[HF {args.hf_model}] PPL={ppl_hf:.4f} (tokens={n_tok_hf})")

    # Dein Modell (optional)
    if args.my_ckpt is not None:
        my_model = load_my_model(args.my_ckpt, device)
        ppl_my, n_tok_my = ppl_sliding_window_my(
            my_model, tokenizer, text, device,
            max_length=args.max_length, stride=args.stride,
            max_eval_tokens=args.max_eval_tokens,
            amp_dtype=amp_dtype,
        )
        print(f"[MY ckpt={args.my_ckpt}] PPL={ppl_my:.4f} (tokens={n_tok_my})")

if __name__ == "__main__":
    main()

# python -m my_gpt2.source.eval_ppl --dataset wikitext2 --split test --hf_model gpt2 --stride 1024 --max_length 1024 --amp
