import math
import numpy as np
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
"""
Todo:
Fix torch compile
1. Pretraining
2. SFT
3. SFT auf Reasoningdaten
4. RL mit Reasoningdaten
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
        # stabilisiert die Residual-Branch-Skalen über viele Layer
        self.c_proj.NANOGPT_SCALE_INIT = 1 
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
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

    def forward(self, idx, targets=None, loss_mask = None):
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
            # token level loss
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
                ignore_index=-1
            ).view_as(targets)
            if loss_mask is None:
                denom = (targets != -1).sum().clamp(min=1)
                loss = per_tok.sum() / denom
            else:
                denom = loss_mask.sum().clamp(min=1)
                loss = (per_tok * loss_mask).sum() / denom
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
