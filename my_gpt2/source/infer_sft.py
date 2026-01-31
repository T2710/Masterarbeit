import os
import argparse
from my_gpt2.source import model
import torch
import tiktoken

from my_gpt2.source.model import GPT, GPTConfig


@torch.no_grad()
def generate(model, idx, enc, max_new_tokens=200, temperature=0.8, top_k=50, eos_token_id=50256):
    model.eval()
    prompt_len = idx.size(1)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

        decoded_new = enc.decode(idx[0, prompt_len:].tolist())
        if "\nUser:" in decoded_new:
            break

        if next_id.item() == eos_token_id:
            break

    return idx


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = ckpt.get("config", None)
    if isinstance(cfg, dict):
        config = GPTConfig(**cfg)
    elif cfg is not None:
        config = cfg
    else:
        # fallback (sollte selten passieren)
        config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)

    model = GPT(config).to(device)

    state = ckpt.get("model", ckpt.get("model_state_dict", None))
    if state is None:
        raise KeyError(f"Checkpoint keys: {list(ckpt.keys())} (expected 'model' or 'model_state_dict')")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False, default="/home/ul/ul_student/ul_raf24/project/Masterarbeit/my_gpt2/results/sft_ultrachat/best.pt")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_checkpoint(args.ckpt, device)

    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]  # 50256

    prompts = [
        "Hello whats up?",
        "Write a short email to a professor asking for an appointment next week.",
        "Why is the sky blue?",
        "Write a poem about why the sky blue is"
    ]

    for i, p in enumerate(prompts, 1):
        text = f"<|endoftext|>You are a helpful assistant.\nUser: {p}\nAssistant:"
        ids = enc.encode_ordinary(text)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        y = generate(model, x, enc, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, eos_token_id=eos_id)


        out = enc.decode(y[0].tolist())

        print("\n" + "=" * 80)
        print(f"[Prompt {i}]")
        print(out)
        print("=" * 80)


if __name__ == "__main__":
    main()

# python -m my_gpt2.source.infer_sft --temperature 0.7 --top_k 50 --max_new_tokens 160
