# Transformer architecture inspired by nanoGPT (Karpathy, 2022)
# https://github.com/karpathy/nanoGPT
# Adapted for symbolic music generation


import math
import inspect
from dataclasses import dataclass

import torch.nn as nn
import torch
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, features, bias, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class MusicConfig:
    block_size: int = 512

    pitch_size: int = 128
    velocity_size: int = 32
    duration_size: int = 64
    position_size: int = 32


    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

#config for H100
"""@dataclass
class GPTConfig:
    block_size: int = 2048  # повний довгий контекст

    pitch_size: int = 128
    velocity_size: int = 32
    duration_size: int = 64
    position_size: int = 32

    n_layer: int = 24  # GPT-2 medium рівень
    n_head: int = 16
    n_embd: int = 1024  # 1024 % 4 == 0, 1024 % 16 == 0 ✓
    dropout: float = 0.1
    bias: bool = False"""

class MusicEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % 4 == 0
        emb_each = config.n_embd // 4

        self.pitch = nn.Embedding(config.pitch_size, emb_each)
        self.velocity = nn.Embedding(config.velocity_size, emb_each)
        self.duration = nn.Embedding(config.duration_size, emb_each)
        self.position = nn.Embedding(config.position_size, emb_each) #(B, T, 256) each !


        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, pitch, velocity, duration, position):
        pitch_embedding = self.pitch(pitch)
        velocity_embedding = self.velocity(velocity)
        duration_embedding = self.duration(duration)
        position_embedding = self.position(position)

        x = torch.cat([pitch_embedding, velocity_embedding, duration_embedding, position_embedding], dim=-1) #(B, T, 1024)!

        assert x.size(-1) == self.proj.in_features, \
            f"x last dim = {x.size(-1)}, proj expects {self.proj.in_features}"

        x = self.proj(x)
        return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.pitch_size is not None
        assert config.velocity_size is not None
        assert config.duration_size is not None
        assert config.position_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            music_embeddings = MusicEmbeddings(config),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        #4 seperete heads for different tokens
        self.head_pitch = nn.Linear(config.n_embd, config.pitch_size, bias=False)
        self.head_velocity = nn.Linear(config.n_embd, config.velocity_size, bias=False)
        self.head_duration = nn.Linear(config.n_embd, config.duration_size, bias=False)
        self.head_position = nn.Linear(config.n_embd, config.position_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, pitch, velocity, duration, position, targets=None):
        device = pitch.device
        b, t = pitch.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.music_embeddings(pitch, velocity, duration, position)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)


        if targets is not None:
            p_tgt, v_tgt, d_tgt, pos_tgt = targets  # tuple з 4 тензорів

            loss_p = F.cross_entropy(self.head_pitch(x).view(-1, self.config.pitch_size),
                                     p_tgt.view(-1), ignore_index=-1)
            loss_v = F.cross_entropy(self.head_velocity(x).view(-1, self.config.velocity_size),
                                     v_tgt.view(-1), ignore_index=-1)
            loss_d = F.cross_entropy(self.head_duration(x).view(-1, self.config.duration_size),
                                     d_tgt.view(-1), ignore_index=-1)
            loss_pos = F.cross_entropy(self.head_position(x).view(-1, self.config.position_size),
                                       pos_tgt.view(-1), ignore_index=-1)

            loss = loss_p + loss_v + loss_d + loss_pos  # можна з вагами
            logits = None

        else:
            x_last = x[:, [-1], :]
            logits = (
                self.head_pitch(x_last),
                self.head_velocity(x_last),
                self.head_duration(x_last),
                self.head_position(x_last),
            )
            loss = None
        return logits, loss


    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
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
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    @torch.no_grad()
    def generate(self, pitch, velocity, duration, position, max_new_tokens, temperature=0.85, top_k=None):

        for _ in range(max_new_tokens):
            # crop всі 4 якщо занадто довгі
            p = pitch if pitch.size(1) <= self.config.block_size else pitch[:, -self.config.block_size:]
            v = velocity if velocity.size(1) <= self.config.block_size else velocity[:, -self.config.block_size:]
            d = duration if duration.size(1) <= self.config.block_size else duration[:, -self.config.block_size:]
            pos = position if position.size(1) <= self.config.block_size else position[:, -self.config.block_size:]

            logits, _ = self(p, v, d, pos)  # logits = tuple з 4 елементів

            p_logits, v_logits, d_logits, pos_logits = logits  # кожен (b, 1, size)

            def sample(logits_i, top_k=top_k):
                logits_i = logits_i[:, -1, :] / temperature
                if top_k is not None:
                    v_topk, _ = torch.topk(logits_i, min(top_k, logits_i.size(-1)))
                    logits_i[logits_i < v_topk[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_i, dim=-1)
                return torch.multinomial(probs, num_samples=1)  # (b, 1)

            pitch = torch.cat([pitch, sample(p_logits)], dim=1)
            velocity = torch.cat([velocity, sample(v_logits)], dim=1)
            duration = torch.cat([duration, sample(d_logits)], dim=1)
            position = torch.cat([position, sample(pos_logits)], dim=1)

        return pitch, velocity, duration, position
