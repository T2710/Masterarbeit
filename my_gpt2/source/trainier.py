import math
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

"""
To optimize training time: 
1. Max out Batch Size to still fit in GPU memory
2. Choose Float network precision (e.g., float32 , 16..) if supported by hardware
3. Use torch.compile
4. Numbers devisible by 2, 8, etc. are faster due to gpu architecture
"""

# 1. SSFT with The Pile Dataset 2. SFT with OpenAssistent Dataset
# Implement Multi-Head Causal Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value c_projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output c_projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).tril().view(1, 1, config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        # B = batch size, T = sequence length, C = embedding dimension
        B, T, C = x.size()
        qkv = self.c_attn(x)
        # split into query, key, and value tensors
        q, k ,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        # compute attention scores
        """
        # original implementation from attention is all you need paper
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v 
        """
        # more efficient implementation from Flash Attention paper
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y) # output projection
        return y


# Implement Feedforward Neural Network 
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Fully connected layer expanding the embedding dimension to 4 times its size
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Activation function (GELU with tanh approximation)
        self.gelu = nn.GELU(approximate='tanh')
        # Fully connected layer projecting back to the original embedding dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Define Transformer components: Causal Self-Attention and MLP (Feed-Forward Network)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward (self, x):
        # Apply ln1, followed by causal self-attention, then add residual connection
        x = x + self.attn(self.ln_1(x))
        # Apply ln2, followed by MLP, then add residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # Sequence length (context window)
    vocab_size: int = 50257 # vocabulary size
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Transformer components 
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            # Position Embedding
            wpe = nn.Embedding(config.block_size,config.n_embd),
            # List of Transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer norm (matches GPT-2)
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # Linear layer to c_project the final hidden states to vocabulary size for language modeling
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # Token and Position embeddings
        tok_emb = self.transformer.wte(idx) # each index maps to a (n_embd,) vector
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T,)
        pos_emb = self.transformer.wpe(pos) # each position maps to a (n_embd,) vector
        x = tok_emb + pos_emb # shape (B,T,n_embd)
        # Forward pass through Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # Final layer norm
        x = self.transformer.ln_f(x)
        # Language modeling head
        logits = self.lm_head(x) # shape (B,T,vocab_size)
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    """
    This method configures the optimizer for the model.
    It collects all trainable parameters and splits them into two groups:
    1) Parameters with 2 or more dimensions (e.g. weight matrices, embeddings),
       which receive weight decay for regularization.
    2) Parameters with fewer than 2 dimensions (e.g. biases, LayerNorm weights),
       which do NOT receive weight decay to avoid harming sensitive parameters. 
    Weight decay is a regularization technique. It slightly pulls the weights toward zero at every optimization step.
    These two groups are passed to AdamW with different weight decay settings.
    The method also automatically enables the faster "fused" AdamW version
    when running on CUDA, if it is available.
    
    """
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    """
    Initialize weights using huggingface's method for GPT-2. Allows us to load pretrained models as well.
    """
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Load data batch
        with open('source/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Data has {self.tokens.size(0)} tokens")
        print(f"1 epoch = {self.tokens.size(0) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        # Reset if loading the next batch exceeds data length
        if self.current_position + B * T * self.num_processes + 1 >= self.tokens.size(0):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# ------------------- Training Setup -------------------
import os
import tiktoken
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

"""
Set up DDP (distributed Data Parallel) if multiple GPUs are available.
"""
ddp = int(os.environ.get('RANK', -1)) != -1 # check if ddp is to be used
if ddp:
    assert torch.cuda.is_available(), "DDP mode requires CUDA."
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # global rank of this process
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # GPU index for this process
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of gpus/processes
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0 # this process (number 0) will do logging tasks
else: 
    # non ddp mode
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Learning Rate adjustment over time
max_lr = 6e-4
min_lr = max_lr *0.1
warmup_steps = 10
max_steps = 50
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

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

"""
Use Batch Size = 0.5 M because it matches the other Hyperparameters from the GPT paper
Symiulate this by accumulating gradients over multiple smaller batches if necessary because of GPU memory limits
16 * 1024 = 16384 tokens per step -> 524288 / 16384 = 32 gradient accumulation steps to reach total batch size
Split the total batch size into smaller micro-batches bevor Backpropagation by accumulating gradients
"""
total_batch_size = 524288 # ~ 0.5M tokens per step
B = 4
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"calculated gradient accumulation steps: {grad_accum_steps}")

# B = batch size, T = sequence length e.g. -> B = 6, T = 1024 = 6 Sequences of 1024 tokens each
train_loader = DataLoaderLite(B = B, T = T, process_rank=ddp_rank, num_processes=ddp_world_size)
torch.set_float32_matmul_precision('high')

# Numbers devisible by 2 are faster due to gpu architecture, Add fake vocab size to make it devisible by 2, 8 etc.
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) # Use only on Uni Cluster
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


# optimize the model according to GPT-3 Paper, because they didnt publish in the GPT-2 paper
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate= 6e-4, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.requires_backward_grad_sync = (micro_step == grad_accum_steps - 1)
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
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_step = tokens_processed / dt
    if master_process:
        print(f"Step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_step:.2f}")

if ddp:
    destroy_process_group()

