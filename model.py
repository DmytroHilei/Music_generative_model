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

    def forward(self, x, cache=None, layer=0):
        """cache (inference only): a KVCache with preallocated k/v buffers. x is either a prefill from position 0
        (T > 1, causal) or one new position at cache.pos that attends to positions <= cache.pos (T == 1).
        Shapes never change, so the one-position step can be captured in a CUDA graph."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        causal, mask = True, None
        if cache is not None:
            ck, cv = cache.k[layer], cache.v[layer]
            if T > 1:
                ck[:, :, :T] = k
                cv[:, :, :T] = v
            else:
                ck.index_copy_(2, cache.pos, k)
                cv.index_copy_(2, cache.pos, v)
                k, v, causal, mask = ck, cv, False, cache.mask  # attend to every filled position

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            elif mask is not None:
                att = att.masked_fill(~mask, float('-inf'))
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

class MoEMLP(nn.Module):
    """
    Top-k routed mixture of expert FFNs (dropless: every token goes to its k experts, no capacity limit).
    Experts are stored as stacked weights and run one after another on the tokens routed to them.
    self.aux holds the load-balancing loss of the last forward (Switch Transformer: E * sum_i f_i * P_i).
    """

    def __init__(self, config):
        super().__init__()
        C, E = config.n_embd, config.moe_experts
        H = int(4 * C * config.moe_hidden_frac)
        self.top_k = config.moe_top_k
        self.router = nn.Linear(C, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, C, H) * 0.02)
        self.w2 = nn.Parameter(torch.randn(E, H, C) * 0.02 / math.sqrt(2 * config.n_layer))
        self.dropout = nn.Dropout(config.dropout)
        self.aux = None

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        probs = F.softmax(self.router(flat).float(), dim=-1)              # (N, E)
        gate, idx = probs.topk(self.top_k, dim=-1)                          # (N, k)
        gate = (gate / gate.sum(-1, keepdim=True)).to(flat.dtype)
        E = probs.size(-1)
        # load balancing: fraction of routed slots per expert * mean router prob per expert
        frac = torch.zeros(E, device=x.device).scatter_add_(0, idx.reshape(-1),
                                                             torch.ones(idx.numel(), device=x.device))
        self.aux = E * (frac / idx.numel() * probs.mean(0)).sum()
        out = torch.zeros_like(flat)
        for e in range(E):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            h = F.gelu(flat[tok] @ self.w1[e].to(flat.dtype)) @ self.w2[e].to(flat.dtype)
            out.index_add_(0, tok, h * gate[tok, slot, None])
        return self.dropout(out.reshape(B, T, C))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MoEMLP(config) if config.moe_experts else MLP(config)

    def forward(self, x, cache=None, layer=0):
        x = x + self.attn(self.ln_1(x), cache, layer)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class MusicConfig:
    block_size: int = 512

    pitch_size: int = 128
    velocity_size: int = 32
    duration_size: int = 512
    delta_time_size: int = 512

    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256   # must be divisible by 4 (embeddings) and n_head
    dropout: float = 0.2
    bias: bool = False
    label_smoothing: float = 0.1
    # True: predict dt -> pitch -> duration -> velocity, each conditioned on the previous ones.
    # False: 4 independent heads (old checkpoints).
    cascade_heads: bool = True
    # True (v2): residual heads, logits = out(LN(z + MLP(LN(z)))) with z = h + cond embeddings; at init this is
    # exactly the independent linear heads plus conditioning. False (v1): plain MLP heads (ab_cascade checkpoint).
    cascade_residual: bool = True
    cond_emb_init_std: float = 0.5  # v2 only: h after ln_f has ~unit scale, 0.02 made the conditioning invisible
    # asymmetric heads (v2 only): the pitch head gains most from capacity, so it can get more/wider blocks
    pitch_head_blocks: int = 1
    pitch_head_mult: int = 1        # hidden width of the pitch head blocks = mult * n_embd
    # mixture of experts in the transformer FFNs (0 = dense MLP). Expert hidden = 4*n_embd*moe_hidden_frac,
    # with top_k=2 and frac=0.5 the active FFN compute equals the dense model.
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_hidden_frac: float = 0.5
    moe_aux_weight: float = 0.01    # Switch-style load-balancing loss


# order in which attributes of the next note are decided by CascadeHeads
CASCADE_ORDER = ('delta_time', 'pitch', 'duration', 'velocity')
# order of the attribute streams everywhere else (inputs, targets, datasets)
STREAM_ORDER = ('pitch', 'velocity', 'duration', 'delta_time')


class ResidualHead(nn.Module):
    """out(LN(z + MLP(LN(z)))): a direct linear path from z (like the independent heads) plus a small MLP
    that can mix in the conditioning non-linearly. c_proj gets the small residual init in GPT.__init__."""

    def __init__(self, C, vocab, bias, n_blocks=1, mult=1):
        super().__init__()
        H = mult * C
        self.ln = LayerNorm(C, bias=bias)
        self.c_fc = nn.Linear(C, H, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(H, C, bias=bias)
        # further residual blocks (asymmetric heads); empty for the default single block -> old checkpoints load
        self.extra = nn.ModuleList(nn.ModuleDict(dict(
            ln=LayerNorm(C, bias=bias), c_fc=nn.Linear(C, H, bias=bias), c_proj=nn.Linear(H, C, bias=bias),
        )) for _ in range(n_blocks - 1))
        self.ln_out = LayerNorm(C, bias=bias)
        self.out = nn.Linear(C, vocab, bias=False)

    def forward(self, z):
        z = z + self.c_proj(self.gelu(self.c_fc(self.ln(z))))
        for blk in self.extra:
            z = z + blk['c_proj'](self.gelu(blk['c_fc'](blk['ln'](z))))
        return self.out(self.ln_out(z))


class CascadeHeads(nn.Module):
    """
    Factorizes P(next note | context) with the chain rule instead of 4 independent heads:
        P(dt|h) * P(pitch|h,dt) * P(dur|h,dt,pitch) * P(vel|h,dt,pitch,dur)
    Each head is a small MLP over h plus the embeddings of the attributes already decided.
    Training uses the true values (teacher forcing), generation feeds back the sampled ones.
    """

    def __init__(self, config):
        super().__init__()
        sizes = dict(pitch=config.pitch_size, velocity=config.velocity_size,
                     duration=config.duration_size, delta_time=config.delta_time_size)
        C = config.n_embd
        # the last attribute is never used as a condition, so it needs no embedding
        self.cond_emb = nn.ModuleDict({
            name: nn.Embedding(sizes[name], C) for name in CASCADE_ORDER[:-1]
        })
        self.residual = config.cascade_residual
        if self.residual:
            self.heads = nn.ModuleDict({
                name: ResidualHead(C, sizes[name], config.bias,
                                   n_blocks=config.pitch_head_blocks if name == 'pitch' else 1,
                                   mult=config.pitch_head_mult if name == 'pitch' else 1)
                for name in CASCADE_ORDER
            })
            return
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                LayerNorm(C, bias=config.bias),
                nn.Linear(C, C, bias=config.bias),
                nn.GELU(),
                nn.Linear(C, sizes[name], bias=False),
            ) for name in CASCADE_ORDER
        })

    def forward(self, h, targets):
        # h: (B, T, C), targets: dict name -> (B, T), -1 = ignore
        logits = {}
        cond = h
        for name in CASCADE_ORDER:
            logits[name] = self.heads[name](cond)
            if name in self.cond_emb:
                cond = cond + self.cond_emb[name](targets[name].clamp(min=0))
        return logits

    def sample(self, h, sample_fn):
        # h: (B, 1, C) -> dict name -> (B, 1); sample_fn(logits, head_name)
        out = {}
        cond = h
        for name in CASCADE_ORDER:
            out[name] = sample_fn(self.heads[name](cond), name)
            if name in self.cond_emb:
                cond = cond + self.cond_emb[name](out[name])
        return out

class MusicEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % 4 == 0
        emb_each = config.n_embd // 4

        self.pitch = nn.Embedding(config.pitch_size, emb_each)
        self.velocity = nn.Embedding(config.velocity_size, emb_each)
        self.duration = nn.Embedding(config.duration_size, emb_each)
        self.delta_time = nn.Embedding(config.delta_time_size, emb_each)


        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, pitch, velocity, duration, delta_time):
        pitch_embedding = self.pitch(pitch)
        velocity_embedding = self.velocity(velocity)
        duration_embedding = self.duration(duration)
        delta_time_embedding = self.delta_time(delta_time)

        x = torch.cat([pitch_embedding, velocity_embedding, duration_embedding, delta_time_embedding], dim=-1)

        assert x.size(-1) == self.proj.in_features, \
            f"x last dim = {x.size(-1)}, proj expects {self.proj.in_features}"

        x = self.proj(x)
        return x


class KVCache:
    """Preallocated keys/values for all layers (inference). pos = position of the next single-note step (a device
    tensor, so a captured CUDA graph reads it at replay time); mask = which cache slots that step may attend to."""

    def __init__(self, n_layer, batch, n_head, block_size, head_size, device, dtype):
        shape = (n_layer, batch, n_head, block_size, head_size)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.slots = torch.arange(block_size, device=device)
        self.mask = torch.zeros(1, 1, 1, block_size, dtype=torch.bool, device=device)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.pitch_size is not None
        assert config.velocity_size is not None
        assert config.duration_size is not None
        assert config.delta_time_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            music_embeddings = MusicEmbeddings(config),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        if config.cascade_heads:
            self.cascade = CascadeHeads(config)
        else:
            self.head_pitch      = nn.Linear(config.n_embd, config.pitch_size,      bias=False)
            self.head_velocity   = nn.Linear(config.n_embd, config.velocity_size,   bias=False)
            self.head_duration   = nn.Linear(config.n_embd, config.duration_size,   bias=False)
            self.head_delta_time = nn.Linear(config.n_embd, config.delta_time_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        if config.cascade_heads and config.cascade_residual:
            for emb in self.cascade.cond_emb.values():
                torch.nn.init.normal_(emb.weight, mean=0.0, std=config.cond_emb_init_std)

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

    def _encode(self, pitch, velocity, duration, delta_time, cache=None):
        """cache=None: plain forward. With a KVCache: T > 1 = prefill from position 0, T == 1 = one step at cache.pos."""
        device = pitch.device
        b, t = pitch.size()
        assert t <= self.config.block_size
        if cache is not None and t == 1:
            seq_pos = cache.pos
            cache.mask.copy_((cache.slots <= cache.pos).view(1, 1, 1, -1))
        else:
            seq_pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.music_embeddings(pitch, velocity, duration, delta_time)
        pos_emb = self.transformer.wpe(seq_pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.h):
            x = block(x, cache, i)
        return self.transformer.ln_f(x)

    def _head_logits(self, x, targets=None):
        # x: (B, T, C) -> dict name -> logits. targets (dict) are only needed for cascade heads.
        if self.config.cascade_heads:
            return self.cascade(x, targets)
        return dict(pitch=self.head_pitch(x), velocity=self.head_velocity(x),
                    duration=self.head_duration(x), delta_time=self.head_delta_time(x))

    def forward(self, pitch, velocity, duration, delta_time, targets=None):
        """
        With targets: returns (parts, loss). loss = sum of the 4 CEs with label smoothing (what we optimize),
        parts = dict name -> CE without label smoothing (detached, for honest logging / comparing runs).
        Without targets (independent heads only): returns ((p, v, d, dt) logits of the last position, None).
        Cascade heads can't give all 4 logits without choosing the earlier attributes, so use generate().
        """
        x = self._encode(pitch, velocity, duration, delta_time)

        if targets is not None:
            tgt = dict(zip(STREAM_ORDER, targets))
            logits = self._head_logits(x, tgt)
            ls = self.config.label_smoothing
            loss = 0.0
            parts = {}
            for name in STREAM_ORDER:
                lg = logits[name].view(-1, logits[name].size(-1))
                t = tgt[name].reshape(-1)
                loss = loss + F.cross_entropy(lg, t, ignore_index=-1, label_smoothing=ls)
                with torch.no_grad():
                    parts[name] = F.cross_entropy(lg.float(), t, ignore_index=-1)
            if self.config.moe_experts:
                aux = sum(b.mlp.aux for b in self.transformer.h) / len(self.transformer.h)
                loss = loss + self.config.moe_aux_weight * aux
                parts['moe_aux'] = aux.detach()
            return parts, loss

        else:
            assert not self.config.cascade_heads, "cascade heads need sequential sampling, use generate()"
            logits = self._head_logits(x[:, [-1], :])
            return tuple(logits[name] for name in STREAM_ORDER), None


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
        # v2 cascade: condition embeddings must stay at the scale of h, so no weight decay on them
        def no_decay(n, p):
            return p.dim() < 2 or (self.config.cascade_residual and 'cond_emb' in n)
        decay_params = [p for n, p in param_dict.items() if not no_decay(n, p)]
        nodecay_params = [p for n, p in param_dict.items() if no_decay(n, p)]
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
    def generate(self, pitch, velocity, duration, delta_time, max_new_tokens, temperature=0.85, top_k=None,
                 anchor=0, slide=None, progress=None, dt_bias=0.0, target_nps=None, dt_seconds=0.02,
                 cuda_graph=True):
        """
        Batched sampling (B rows, same prompt length) with a preallocated KV cache: each new note costs one position
        through the model, not a full re-run. On CUDA the one-note transformer step is captured once as a CUDA graph
        and replayed (cuda_graph=True), which removes the per-kernel launch cost that dominates small-batch decode.
        Positions are learned up to block_size, so the window can't roll one note at a time with a cache (every
        cached position would shift). When the cache is full it is refilled from a shorter window, making room
        for `slide` new notes (default block_size // 4, so the context stays between 3/4 and all of block_size).
        anchor > 0 (optional, off by default): the first `anchor` notes of the sequence (e.g. the prompt's theme)
        stay pinned at the start of every refilled window, in front of the most recent notes.
        progress: optional callback(n_generated) after every note (e.g. a jobstatus.JobStatus update).
        Density control (no retraining; only the delta_time head is touched, every other choice stays the model's):
          dt_bias: fixed logit bias b * log(max(bin, 1)) on delta_time; b > 0 favours longer gaps = fewer notes/s.
          target_nps: notes per second to track (a float, or one per row). A controller nudges each row's bias after
            every note from the density of its last 64 notes, starting at dt_bias. dt_seconds = seconds per bin.
        Nothing here syncs the GPU with the CPU inside the loop.
        """
        L = self.config.block_size
        slide = slide or L // 4
        assert 0 <= anchor and anchor + slide < L, "need anchor + slide < block_size"
        device = pitch.device
        B, n0 = pitch.shape
        total = n0 + max_new_tokens
        # output buffers, filled in place (no growing torch.cat)
        out = [torch.zeros(B, total, dtype=torch.long, device=device) for _ in range(4)]
        for o, s in zip(out, (pitch, velocity, duration, delta_time)):
            o[:, :n0] = s

        # log(bin), with bins 0 and 1 unbiased: longer gaps get likelier, but "same onset" (bin 0 = chord notes)
        # keeps its odds against the shortest gap, so chords aren't broken up into arpeggios
        dt_shape = torch.log(torch.arange(self.config.delta_time_size, dtype=torch.float32,
                                          device=device).clamp(min=1))
        bias = torch.full((B, 1), float(dt_bias), device=device)
        target = None
        if target_nps is not None:
            target = torch.as_tensor(target_nps, dtype=torch.float32, device=device).reshape(-1, 1).expand(B, 1)

        def sample(logits, name=None):  # (B, 1, vocab) → (B, 1)
            logits = logits[:, -1, :].float() / temperature
            if name == 'delta_time' and (dt_bias != 0.0 or target is not None):
                logits = logits + bias * dt_shape
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        dtype = next(self.parameters()).dtype
        C = self.config.n_embd
        cache = KVCache(self.config.n_layer, B, self.config.n_head, L, C // self.config.n_head, device, dtype)

        def prefill(n):
            """Refill the cache from the window ending at note n; returns (last hidden state, cache fill)."""
            if n <= L:
                window = [o[:, :n] for o in out]
            else:
                recent = L - slide - anchor
                window = [torch.cat([o[:, :anchor], o[:, n - recent:n]], dim=1) for o in out]
            w = window[0].size(1)
            if w == 1:
                cache.pos.fill_(0)  # a 1-note window goes through the single-step path at slot 0
            return self._encode(*window, cache=cache)[:, [-1], :], w

        # the one-note step: static inputs -> static output, optionally captured as a CUDA graph
        step_in = [torch.zeros(B, 1, dtype=torch.long, device=device) for _ in range(4)]
        graph = None
        if cuda_graph and device.type == 'cuda':
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm-up outside the graph (allocator, kernel selection)
                for _ in range(2):
                    self._encode(*step_in, cache=cache)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step_out = self._encode(*step_in, cache=cache)
            # the warm-up and capture wrote junk into slot 0; prefill below overwrites every slot it uses

        def step(new, pos):
            for buf, t in zip(step_in, new):
                buf.copy_(t)
            cache.pos.fill_(pos)
            if graph is not None:
                graph.replay()
                return step_out
            return self._encode(*step_in, cache=cache)

        h, filled = prefill(n0)
        for i in range(max_new_tokens):
            if self.config.cascade_heads:
                nxt = self.cascade.sample(h, sample)
            else:
                nxt = {name: sample(lg, name) for name, lg in self._head_logits(h).items()}
            new = [nxt[name] for name in STREAM_ORDER]
            n = n0 + i
            for o, t in zip(out, new):
                o[:, n:n + 1] = t
            n += 1
            if target is not None:
                # per-row integral controller on log density, bias kept in [-2, 6]; stays on the GPU
                w = min(n, 64)
                secs = out[3][:, n - w:n].sum(dim=1, keepdim=True).float() * dt_seconds
                nps = torch.where(secs > 0, w / secs.clamp(min=1e-6), torch.full_like(secs, 1e3))
                bias.add_(0.05 * torch.log(nps.clamp(min=1e-3) / target)).clamp_(-2.0, 6.0)
            if progress is not None:
                progress(i + 1)
            if i == max_new_tokens - 1:
                break
            if filled < L:
                h = step(new, filled)
                filled += 1
            else:
                h, filled = prefill(n)

        return tuple(out)
