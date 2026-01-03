"""
Download and prepare datasets for training.
"""

import os
import multiprocessing as mp
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
    },
}


class DatasetBuilder:
    def __init__(self, dataset, base_dir="source/datasets"):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset = dataset
        self.config = DATASETS[dataset]
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / self.config["local_dir"]
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        cfg = self.config
        return load_dataset(
            cfg["repo"],
            name=cfg.get("name"),
            split=cfg.get("split", "train"),
            cache_dir=str(self.dataset_dir),
        )

    def tokenize(self, doc):
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens["<|endoftext|>"]
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
            "token dictionary too large for uint16"
        )
        return tokens_np.astype(np.uint16)

    def write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def build_shards(self):
        cfg = self.config
        shard_size = cfg["shard_size"]
        shard_prefix = cfg["shard_prefix"]

        dataset = self.load()
        nprocs = max(1, os.cpu_count() // 2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for tokens in pool.imap(self.tokenize, dataset, chunksize=16):
                if token_count + len(tokens) < shard_size:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=shard_size,
                            unit="tokens",
                            desc=f"Shard {shard_index}",
                        )
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = self.dataset_dir / f"{shard_prefix}_{split}_{shard_index:06d}"
                    remainder = shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                    self.write_datafile(str(filename), all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder

            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = self.dataset_dir / f"{shard_prefix}_{split}_{shard_index:06d}"
                self.write_datafile(str(filename), all_tokens_np[:token_count])
