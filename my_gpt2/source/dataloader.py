"""
Download and prepare datasets for training.

Supports the existing FineWeb builders unchanged and adds OpenWebMath.
The shard format remains identical:
- .npy shards with uint16 token ids
- first shard is val, remaining shards are train
- each document is prefixed with <|endoftext|>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


DATASETS = {
    "edu_fineweb10B": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "local_dir": "edu_fineweb10B",
        "shard_prefix": "edufineweb",
        "shard_size": int(1e8),
        "text_field": "text",
        "streaming": True,
    },
    "edu_fineweb100B": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "name": "sample-100BT",
        "split": "train",
        "local_dir": "edu_fineweb100B",
        "shard_prefix": "edufineweb",
        "shard_size": int(1e8),
        "text_field": "text",
        "streaming": True,
    },
    "openwebmath": {
        "repo": "open-web-math/open-web-math",
        "name": None,
        "split": "train",
        "local_dir": "openwebmath",
        "shard_prefix": "openwebmath",
        "shard_size": int(1e8),
        "text_field": "text",
        "streaming": True,
    },
}


class DatasetBuilder:
    def __init__(self, dataset, base_dir="my_gpt2/source/datasets"):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset = dataset
        self.config = DATASETS[dataset]
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / self.config["local_dir"]
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]

    def load(self):
        cfg = self.config
        kwargs = {
            "path": cfg["repo"],
            "split": cfg.get("split", "train"),
            "streaming": cfg.get("streaming", True),
        }
        if cfg.get("name"):
            kwargs["name"] = cfg["name"]
        return load_dataset(**kwargs)

    def tokenize(self, doc):
        text_field = self.config.get("text_field", "text")
        text = doc.get(text_field, "") or ""
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(text))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
            "token dictionary too large for uint16"
        )
        return tokens_np.astype(np.uint16)

    @staticmethod
    def write_datafile(filename, tokens_np):
        np.save(filename, tokens_np)

    def _new_progress_bar(self, shard_index, shard_size, max_tokens, total_written):
        pb_total = shard_size if max_tokens is None else min(shard_size, max_tokens - total_written)
        return tqdm(total=pb_total, unit="tokens", desc=f"Shard {shard_index}")

    def build_shards(self, max_tokens: int = None, write_meta: bool = True):
        cfg = self.config
        shard_size = cfg["shard_size"]
        shard_prefix = cfg["shard_prefix"]
        dataset = self.load()

        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        total_written = 0
        progress_bar = None
        stop_now = False

        for doc in dataset:
            tokens = self.tokenize(doc)

            if max_tokens is not None:
                remaining = max_tokens - total_written
                if remaining <= 0:
                    break
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]
                    stop_now = True

            if len(tokens) == 0:
                if stop_now:
                    break
                continue

            i = 0
            while i < len(tokens):
                if progress_bar is None:
                    progress_bar = self._new_progress_bar(
                        shard_index=shard_index,
                        shard_size=shard_size,
                        max_tokens=max_tokens,
                        total_written=total_written,
                    )

                space = shard_size - token_count
                take = min(space, len(tokens) - i)
                all_tokens_np[token_count: token_count + take] = tokens[i: i + take]
                token_count += take
                total_written += take
                progress_bar.update(take)
                i += take

                if token_count == shard_size:
                    split = "val" if shard_index == 0 else "train"
                    filename = self.dataset_dir / f"{shard_prefix}_{split}_{shard_index:06d}"
                    self.write_datafile(str(filename), all_tokens_np)

                    progress_bar.close()
                    progress_bar = None

                    shard_index += 1
                    token_count = 0

                    if max_tokens is not None and total_written >= max_tokens:
                        stop_now = True
                        break

            if stop_now:
                break

        if progress_bar is not None:
            progress_bar.close()

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = self.dataset_dir / f"{shard_prefix}_{split}_{shard_index:06d}"
            self.write_datafile(str(filename), all_tokens_np[:token_count])

        if write_meta:
            meta_path = self.dataset_dir / f"meta_{self.dataset}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": self.dataset,
                        "repo": cfg["repo"],
                        "name": cfg.get("name"),
                        "split": cfg.get("split", "train"),
                        "streaming": cfg.get("streaming", True),
                        "text_field": cfg.get("text_field", "text"),
                        "shard_prefix": shard_prefix,
                        "shard_size": shard_size,
                        "max_tokens": max_tokens,
                        "total_written": total_written,
                        "local_dir": cfg["local_dir"],
                    },
                    f,
                    indent=2,
                )

        print(f"Done. Wrote {total_written:,} tokens to {self.dataset_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="edu_fineweb10B", choices=list(DATASETS.keys()))
    ap.add_argument("--base_dir", type=str, default="my_gpt2/source/datasets")
    ap.add_argument("--max_tokens", type=int, default=None, help="If set, stop after writing this many tokens.")
    ap.add_argument("--no_meta", action="store_true", help="Do not write a metadata json file.")
    args = ap.parse_args()

    dataloader = DatasetBuilder(args.dataset, base_dir=args.base_dir)
    dataloader.build_shards(max_tokens=args.max_tokens, write_meta=not args.no_meta)


if __name__ == "__main__":
    main()
