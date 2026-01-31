import os
import glob
import random
import torch
import tiktoken

DATA_DIR = "my_gpt2/source/datasets/ultrachat_sft_examples"
PREFIX = "ultrachat_all"

def main():
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc._special_tokens["<|endoftext|>"]

    shards = sorted(glob.glob(os.path.join(DATA_DIR, f"{PREFIX}_*.pt")))
    print("num shards:", len(shards))
    assert len(shards) > 0

    # load a random shard
    sp = random.choice(shards)
    shard = torch.load(sp)
    print("loaded shard:", sp)
    print("num examples in shard:", len(shard))

    # pick 3 random examples
    for k in range(3):
        ids, mask = random.choice(shard)
        assert len(ids) == len(mask)

        text = enc.decode(ids)
        print("\n" + "="*80)
        print(f"Example {k+1}")
        print("len:", len(ids), "| mask ones:", sum(mask), "| mask zeros:", len(mask) - sum(mask))
        print("--- TEXT ---")
        print(text[:2000])  # show first 2000 chars

        # show where mask starts (first 200 bits)
        print("\n--- first 200 mask bits ---")
        print("".join(str(x) for x in mask[:200]))

        # check: should contain User: and Assistant:
        ok_user = "User:" in text
        ok_ass = "Assistant:" in text
        print("\ncontains 'User:' ?", ok_user, "| contains 'Assistant:' ?", ok_ass)

        # check: last token should be eos
        print("last token id:", ids[-1], "| eos?", ids[-1] == eos_id)

if __name__ == "__main__":
    main()
