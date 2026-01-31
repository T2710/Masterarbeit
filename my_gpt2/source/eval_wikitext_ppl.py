import math
import argparse
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
import re
# dein Model
from my_gpt2.source.model import GPT, GPTConfig

EOT = 50256  # <|endoftext|> für GPT-2 BPE
TITLE_RE = re.compile(r"^=+ .+ =+$")  # Wikitext headings like "= Title ="

def load_my_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", None)
    if isinstance(cfg, dict):
        config = GPTConfig(**cfg)
    else:
        config = cfg
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def load_wikitext_articles(name="wikitext-103-v1", split="test"):
    ds = load_dataset("wikitext", name, split=split)
    texts = ds["text"]

    articles = []
    cur = []

    for line in texts:
        if line is None:
            continue

        line = line.rstrip()

        # Artikel-Header: starte neuen Artikel
        if TITLE_RE.match(line):
            if cur:
                articles.append("\n".join(cur).strip())
                cur = []
            # Header selbst NICHT als Content (optional)
            continue

        # Leere Zeilen innerhalb eines Artikels behalten wir als Absatztrenner
        # aber wir skippen lange Runs aus komplett leeren Lines nicht aggressiv.
        cur.append(line)

    if cur:
        articles.append("\n".join(cur).strip())

    # entferne komplett leere Artikel
    articles = [a for a in articles if len(a) > 0]
    return articles

def tokenize_docs(docs):
    enc = tiktoken.get_encoding("gpt2")
    tokenized = []
    for t in docs:
        ids = enc.encode_ordinary(t)
        if len(ids) > 0:
            tokenized.append(torch.tensor(ids, dtype=torch.long))
    return tokenized


def _autocast_ctx(device: str, use_bf16: bool):
    if device.startswith("cuda") and use_bf16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def nll_sliding_window_my_gpt(model, tokens_1d: torch.Tensor, block_size: int, stride: int,
                             device: str, use_bf16: bool = True):
    """
    Gibt (sum_nll, count_tokens) zurück (nicht gemittelt),
    damit wir korrekt über viele Doks aggregieren können.
    """
    model.eval()
    total_nll = 0.0
    total_count = 0

    autocast_ctx = _autocast_ctx(device, use_bf16)
    n = tokens_1d.numel()
    if n < 2:
        return 0.0, 0

    for i in range(0, n - 1, stride):
        begin = max(i + stride - block_size, 0)
        end = min(i + stride, n)

        input_ids = tokens_1d[begin:end].unsqueeze(0).to(device)
        L = input_ids.size(1)
        if L < 2:
            continue

        x = input_ids[:, :-1]
        y = input_ids[:, 1:].clone()

        trg_len = end - i
        trg_len = min(trg_len, y.size(1))

        if y.size(1) > trg_len:
            y[:, :-trg_len] = -1  # ignore_index=-1

        with autocast_ctx:
            logits, _ = model(x, None)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-1,
                reduction="sum"
            )

        count = (y != -1).sum().item()
        total_nll += float(loss.item())
        total_count += int(count)

    return total_nll, total_count


@torch.no_grad()
def nll_sliding_window_hf(model, tokens_1d: torch.Tensor, block_size: int, stride: int,
                          device: str, use_bf16: bool = True):
    model.eval()
    total_nll = 0.0
    total_count = 0

    autocast_ctx = _autocast_ctx(device, use_bf16)
    n = tokens_1d.numel()
    if n < 2:
        return 0.0, 0

    for i in range(0, n - 1, stride):
        begin = max(i + stride - block_size, 0)
        end = min(i + stride, n)

        input_ids = tokens_1d[begin:end].unsqueeze(0).to(device)
        L = input_ids.size(1)
        if L < 2:
            continue

        x = input_ids[:, :-1]
        y = input_ids[:, 1:].clone()

        trg_len = end - i
        trg_len = min(trg_len, y.size(1))

        if y.size(1) > trg_len:
            y[:, :-trg_len] = -100

        with autocast_ctx:
            out = model(input_ids=x)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum"
            )

        count = (y != -100).sum().item()
        total_nll += float(loss.item())
        total_count += int(count)

    return total_nll, total_count


def add_eot_between_docs(tokenized_docs, add_eot=True):
    """
    Option A: Dokumente getrennt evaluieren (empfohlen).
    Option B: Falls du *doch* concatenaten willst, dann mit EOT dazwischen.
    (Für paper-näher: getrennt evaluieren ist am saubersten.)
    """
    if not add_eot:
        return tokenized_docs
    out = []
    for d in tokenized_docs:
        out.append(d)
        out.append(torch.tensor([EOT], dtype=torch.long))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--my_ckpt", type=str, required=True)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=256)

    ap.add_argument("--dataset", type=str, default="wikitext-103-v1",
                    choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1", "wikitext-2-v1", "wikitext-103-v1"])
    ap.add_argument("--split", type=str, default="test", choices=["validation", "test"])

    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--also_hf_gpt2", action="store_true")
    ap.add_argument("--hf_model", type=str, default="gpt2")

    # wichtig: diese Flags steuern die “paper-nähere” Evaluation
    ap.add_argument("--doc_reset", action="store_true",
                    help="Evaluiert jedes Dokument separat (kein Cross-Doc-Kontext).")
    ap.add_argument("--insert_eot", action="store_true",
                    help="Fügt <|endoftext|> zwischen Dokumente ein (zusätzlich/alternativ).")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (not args.no_bf16)

    print("device:", device)
    print(f"loading wikitext docs: {args.dataset} / {args.split} ...")
    docs = load_wikitext_articles(args.dataset, args.split)
    tok_docs = tokenize_docs(docs)

    if args.insert_eot:
        tok_docs = add_eot_between_docs(tok_docs, add_eot=True)

    print("num docs/segments:", len(tok_docs))
    print(f"block_size={args.block_size} | stride={args.stride} | doc_reset={args.doc_reset} | insert_eot={args.insert_eot}")

    # --- dein Modell ---
    print("\n[MY MODEL]")
    my_model = load_my_model(args.my_ckpt, device)

    total_nll, total_cnt = 0.0, 0
    if args.doc_reset:
        for d in tok_docs:
            nll, cnt = nll_sliding_window_my_gpt(my_model, d, args.block_size, args.stride, device, use_bf16)
            total_nll += nll
            total_cnt += cnt
    else:
        # fallback: concatenation (nicht empfohlen fürs Paper)
        stream = torch.cat(tok_docs, dim=0)
        total_nll, total_cnt = nll_sliding_window_my_gpt(my_model, stream, args.block_size, args.stride, device, use_bf16)

    avg_nll = total_nll / max(1, total_cnt)
    ppl = math.exp(avg_nll)
    print(f"tokens_counted: {total_cnt} | avg_nll: {avg_nll:.4f} | ppl: {ppl:.2f}")

    # --- HF baseline ---
    if args.also_hf_gpt2:
        print(f"\n[HF MODEL: {args.hf_model}]")
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(args.hf_model).to(device)
        hf.eval()

        total_nll_hf, total_cnt_hf = 0.0, 0
        if args.doc_reset:
            for d in tok_docs:
                nll, cnt = nll_sliding_window_hf(hf, d, args.block_size, args.stride, device, use_bf16)
                total_nll_hf += nll
                total_cnt_hf += cnt
        else:
            stream = torch.cat(tok_docs, dim=0)
            total_nll_hf, total_cnt_hf = nll_sliding_window_hf(hf, stream, args.block_size, args.stride, device, use_bf16)

        avg_nll_hf = total_nll_hf / max(1, total_cnt_hf)
        ppl_hf = math.exp(avg_nll_hf)
        print(f"tokens_counted: {total_cnt_hf} | avg_nll: {avg_nll_hf:.4f} | ppl: {ppl_hf:.2f}")


if __name__ == "__main__":
    main()

# python -m my_gpt2.source.eval_wikitext_ppl --my_ckpt my_gpt2/results/pretraining/log_pretraining_gpt2/gpt2_model_70000.pt --insert_eot --also_hf_gpt2 --hf_model gpt2