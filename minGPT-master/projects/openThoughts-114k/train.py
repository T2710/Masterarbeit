import os
import time
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN, set_seed
from mingpt.bpe import BPETokenizer


class OpenThoughtsDataset(Dataset):
    """
    OpenThoughts-114k -> tokenisierte Blöcke, mit Cache auf Disk.

    Cache-Datei enthält:
      - data_int32: Tensor [N, block_size] (int32 kompatibel + sicher zu serialisieren)
      - meta: dict mit block_size, split, num_samples
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 256,
        log_every: int = 5000,
        cache_dir: str = "./cache",
        use_cache: bool = True,
    ):
        super().__init__()
        self.split = split
        self.block_size = int(block_size)
        self.tokenizer = BPETokenizer()

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"openthoughts_{split}_bs{self.block_size}.pt")

        # ---------- FAST PATH: load cache ----------
        if use_cache and os.path.exists(cache_path):
            print(f"✅ Lade Dataset-Cache: {cache_path}")
            obj = torch.load(cache_path, map_location="cpu")

            # neuer Key (int32)
            if "data_int32" in obj:
                self.data = obj["data_int32"]
            # fallback: falls du noch einen alten Cache hast (optional)
            elif "data_uint16" in obj:
                print("⚠️ Alter Cache-Key data_uint16 gefunden, konvertiere zu int32 (und speichere neu).")
                self.data = obj["data_uint16"].to(torch.int32)
                torch.save({"data_int32": self.data, "meta": obj.get("meta", {})}, cache_path)
            else:
                raise RuntimeError(f"Cache-Datei hat kein data_int32/data_uint16: {cache_path}")

            meta = obj.get("meta", {})
            print(f"✅ Cache geladen: {self.data.shape[0]} samples, block_size={self.data.shape[1]}, dtype={self.data.dtype}")
            if meta:
                print("Cache meta:", meta)
            return

        # ---------- SLOW PATH: build dataset ----------
        print("Lade Dataset von HuggingFace…")
        raw = load_dataset("open-thoughts/OpenThoughts-114k", split=split)
        n_raw = len(raw)
        print("Datensatzgröße (HF):", n_raw)
        print("Spalten:", raw.column_names)

        samples = []  # list[Tensor(block_size)] int32
        skipped = 0
        t0 = time.time()

        for i, ex in enumerate(raw, start=1):
            convs = ex.get("conversations", None)
            if not convs or len(convs) < 2:
                skipped += 1
                continue

            user_msg = (convs[0] or {}).get("value", "")
            assistant_msg = (convs[1] or {}).get("value", "")
            if not user_msg or not assistant_msg:
                skipped += 1
                continue

            text = f"User: {user_msg}\n\nAssistant: {assistant_msg}\n"
            ids = self.tokenizer(text)

            # normalize ids -> list[int]
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().view(-1).tolist()
            elif isinstance(ids, int):
                ids = [int(ids)]
            elif hasattr(ids, "tolist"):
                ids = ids.tolist()
                if isinstance(ids, int):
                    ids = [int(ids)]
            elif isinstance(ids, tuple):
                ids = list(ids)

            if not isinstance(ids, list):
                skipped += 1
                continue
            if len(ids) > 0 and isinstance(ids[0], list):
                ids = [t for sub in ids for t in sub]

            try:
                ids = [int(t) for t in ids]
            except Exception:
                skipped += 1
                continue
            if len(ids) < 2:
                skipped += 1
                continue

            # build blocks (pad or chunk)
            if len(ids) < self.block_size:
                pad_len = self.block_size - len(ids)
                ids = ids + [0] * pad_len
                chunk = torch.tensor(ids, dtype=torch.int32)
                samples.append(chunk)  # int32
            else:
                for j in range(0, len(ids) - self.block_size + 1, self.block_size):
                    chunk_ids = ids[j:j + self.block_size]
                    chunk = torch.tensor(chunk_ids, dtype=torch.int32)
                    samples.append(chunk)  # int32

            # progress
            if log_every > 0 and (i % log_every == 0 or i == n_raw):
                elapsed = time.time() - t0
                speed = i / max(elapsed, 1e-9)
                remaining = n_raw - i
                eta_sec = remaining / max(speed, 1e-9)
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)
                eta_s = int(eta_sec % 60)

                print(
                    f"[dataset:init] {i:>7d}/{n_raw} raw | "
                    f"samples {len(samples):>8d} | "
                    f"skipped {skipped:>6d} | "
                    f"{speed:6.1f} ex/s | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )

        if len(samples) == 0:
            raise RuntimeError("Dataset ist leer nach Preprocessing. Parsing/Tokenizer prüfen.")

        # stack into one tensor [N, block_size] int32
        self.data = torch.stack(samples, dim=0).contiguous()
        print(f"✅ Built dataset: {self.data.shape[0]} samples, block_size={self.data.shape[1]}, dtype={self.data.dtype}")

        # save cache
        if use_cache:
            meta = {
                "split": split,
                "block_size": self.block_size,
                "num_samples": int(self.data.shape[0]),
                "dtype": str(self.data.dtype),
            }
            torch.save({"data_int32": self.data, "meta": meta}, cache_path)
            print(f"💾 Cache gespeichert: {cache_path}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # int32 -> long (Embedding erwartet LongTensor)
        tokens = self.data[idx].to(torch.long)  # [block_size]
        x = tokens[:-1]                         # [block_size-1]
        y = tokens[1:]                          # [block_size-1]
        return x, y

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

def get_config():
    C = CN()

    # System
    C.system = CN()
    C.system.work_dir = "./out/gpt2_medium_reasoning"

    # Modell
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt2-medium"
    C.model.vocab_size = 50257
    C.model.block_size = 255  # weil x,y jeweils Länge 255 haben (block_size-1)

    # Trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.device = "auto"
    C.trainer.num_workers = 2
    C.trainer.batch_size = 4      # H100 sollte 4 locker schaffen
    C.trainer.max_iters = 300_000  
    C.trainer.learning_rate = 2e-4
    C.trainer.betas = (0.9, 0.95)
    C.trainer.weight_decay = 0.1
    C.trainer.grad_norm_clip = 1.0

    return C


# -----------------------------------------------------------
# TRAINING
# -----------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    config = get_config()
    print(config)

    os.makedirs(config.system.work_dir, exist_ok=True)

    # Dataset
    block_tokens = config.model.block_size + 1
    train_dataset = OpenThoughtsDataset(
        split="train",
        block_size=block_tokens,
        cache_dir="./cache",
        use_cache=True
    )

    print("Anzahl Trainingssamples:", len(train_dataset))
    if len(train_dataset) == 0:
        raise RuntimeError("Dataset ist leer – irgendwas ist mit der Sample-Erzeugung schiefgelaufen.")

    # Modell
    model = GPT(config.model)

    # Trainer (Karpathy-Signatur: Trainer(config, model, train_dataset))
    trainer = Trainer(config.trainer, model, train_dataset)

    total_iters = config.trainer.max_iters
    start_time = time.time()

    # Logging-Callback mit ETA
    def log_progress(tr):
        if tr.iter_num % 50 == 0:
            elapsed = time.time() - start_time
            avg_dt = elapsed / max(1, tr.iter_num)
            remaining = max(0, total_iters - tr.iter_num)
            eta_seconds = remaining * avg_dt
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            eta_s = int(eta_seconds % 60)

            print(
                f"iter {tr.iter_num:6d} | "
                f"loss {tr.loss.item():.4f} | "
                f"dt {tr.iter_dt * 1000:.1f}ms | "
                f"ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
            )

    trainer.set_callback("on_batch_end", log_progress)

    # Training starten
    trainer.run()

    # Modell speichern
    ckpt_path = os.path.join(config.system.work_dir, "model_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ Training completed — final model saved at:\n{ckpt_path}\n")
