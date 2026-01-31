import math
import numpy as np
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from my_gpt2.source.dataloader import DatasetBuilder
from my_gpt2.source.hellaswag import render_example, iterate_examples
from my_gpt2.source.model import GPT, GPTConfig


"""
To optimize training time: 
1. Max out Batch Size to still fit in GPU memory
2. Choose Float network precision (e.g., float32 , 16..) if supported by hardware
3. Use torch.compile
4. Numbers devisible by 2, 8, etc. are faster due to gpu architecture
"""
"""
Todo:
Fix torch compile
1. Pretraining
2. SFT
3. SFT auf Reasoningdaten
4. RL mit Reasoningdaten
"""

"""
Todo:
Evtl. Beim Pretraining shard bzw. damit Dokumentenreihenfolge zufällig mischen, damit keine dependezenzen zwischen shards (Reihenfolge) entstehen
wenn man mehrfach über die Daten geht
"""

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process, data_root):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.master_process = master_process
        assert split in {"train", "val"}

        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"data_root='{data_root}' not found. "
                f"Run the shard builder and point data_root to the right folder!"
            ) 
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# for hellaswag datset evaluation
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ------------------- Training Setup -------------------
import tiktoken
import time
import argparse
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def main():
        
    """
    Set up DDP (distributed Data Parallel) if multiple GPUs are available.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--data_root", type=str, default="my_gpt2/source/datasets/edu_fineweb10B")
    ap.add_argument("--tokens_target", type=int, default=10_000_000_000)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_lr", type=float, default=6e-4)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--total_batch_size", type=int, default=524288)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--val_loss_steps", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_dir", type=str, default="results/log_pretraining")
    args = ap.parse_args()

    ddp = int(os.environ.get('RANK', -1)) != -1 # check if ddp is to be used
    if ddp:
        assert torch.cuda.is_available(), "DDP mode requires CUDA."
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK']) # global rank of this process
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # GPU index for this process
        ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of gpus/processes
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0 # this process (number 0) will do logging tasks
    else: 
        # non ddp mode
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Learning Rate adjustment over time
    max_lr = args.max_lr
    min_lr = max_lr *0.1
    warmup_steps = args.warmup_steps

    total_batch_size = args.total_batch_size
    B = args.B
    T = args.T

    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if args.max_steps is None:
        max_steps = math.ceil(args.tokens_target / total_batch_size)
    else:
        max_steps = args.max_steps

    def get_lr(it):
        # 1. linear warmup for warmup_steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2. if it > max_steps: return min_lr
        if it > max_steps:
            return min_lr
        # 3. cosine decay down to min_lr 
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    """
    Use Batch Size = 0.5 M because it matches the other Hyperparameters from the GPT paper
    Symiulate this by accumulating gradients over multiple smaller batches if necessary because of GPU memory limits
    16 * 1024 = 16384 tokens per step -> 524288 / 16384 = 32 gradient accumulation steps to reach total batch size
    Split the total batch size into smaller micro-batches bevor Backpropagation by accumulating gradients
    """
    enc = tiktoken.get_encoding("gpt2")
    if master_process:
        print(f"model: {args.model}")
        print(f"data_root: {args.data_root}")
        print(f"total desired batch size: {total_batch_size}")
        print(f"calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"max_steps: {max_steps}")

    # B = batch size, T = sequence length e.g. -> B = 6, T = 1024 = 6 Sequences of 1024 tokens each
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process, data_root=args.data_root)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process, data_root=args.data_root)
    torch.set_float32_matmul_precision('high')

    # Numbers devisible by 2 are faster due to gpu architecture, Add fake vocab size to make it devisible by 2, 8 etc.
    model_cfg_map = {
        "gpt2":        dict(n_layer=12, n_head=12, n_embd=768), # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }
    mcfg = model_cfg_map[args.model]
    model = GPT(GPTConfig(vocab_size=50304, block_size=1024, **mcfg))
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. 
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    if master_process:
        c = raw_model.config
        print(f"arch: n_layer={c.n_layer} n_head={c.n_head} n_embd={c.n_embd} block_size={c.block_size} vocab={c.vocab_size}")


    # optimize the model according to GPT-3 Paper, because they didnt publish in the GPT-2 paper
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=max_lr, device_type=device_type)

    # create log
    model_tag = args.model
    log_dir = f"{args.log_dir}_{model_tag}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    last_val_loss = float("nan")

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # evaluation validation loss
        if step % args.eval_every == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_step = args.val_loss_steps
                for _ in range(val_loss_step):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_step
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            last_val_loss = val_loss_accum.item()
            if master_process:
                print(f"validation loss: {last_val_loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {last_val_loss:.4f}\n")

        if master_process and step > 0 and (step % args.ckpt_every == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"{model_tag}_model_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': last_val_loss,
                'model_tag': model_tag,
                'tokens_target': args.tokens_target,
                'total_batch_size': total_batch_size,
            }
            torch.save(checkpoint, checkpoint_path)

        # evaluate hellaswag on the dataset from huggingface
        # once in a while evaluate hellaswag
        if (step % args.eval_every == 0 or last_step) and (not use_compile):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % args.eval_every == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # train loop
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        # in ddp, average the loss across processes
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # Clip gradients to prevent exploding gradients for example in bad Batches
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine the learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() 
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
