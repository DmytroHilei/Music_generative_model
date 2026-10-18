# Transformer architecture inspired by nanoGPT (Karpathy, 2022)
# https://github.com/karpathy/nanoGPT
# Adapted for symbolic music generation


import math
from dataclasses import dataclass
from functools import partial

import torch.nn as nn
import torch
import torch.utils.checkpoint
from torch.nn import functional as F

from musicar.training.memory import sac_context


class LayerNorm(nn.Module):
    def __init__(self, features, bias, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def block_norm(config, features=None):
    """The norm used inside the transformer trunk (blocks, ln_f); the output heads keep LayerNorm."""
    features = features or config.n_embd
    return RMSNorm(features) if config.norm == 'rmsnorm' else LayerNorm(features, bias=config.bias)


def rope_tables(positions, head_size, base):
    """RoPE (Su et al. 2021): cos/sin of the angle pos * theta_i for the head_size/2 rotation pairs, (T, head_size)
    each (the angles repeated for both halves, matching apply_rope's rotate-half layout). fp32, cast by the caller."""
    inv_freq = base ** (-torch.arange(0, head_size, 2, dtype=torch.float32, device=positions.device) / head_size)
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pair i = (x[..., i], x[..., i + hs/2]) by its angle. q·k of two rotated vectors then depends only on the
    distance of their positions. x: (B, nh, T, hs), cos/sin: (T, hs)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # QK-norm: RMSNorm over each head's q and k (before RoPE), keeps attention logits bounded at high LR
        self.q_norm = RMSNorm(config.n_embd // config.n_head) if config.qk_norm else None
        self.k_norm = RMSNorm(config.n_embd // config.n_head) if config.qk_norm else None

    def forward(self, x, cache=None, layer=0, rope=None):
        """cache (inference only): a KVCache with preallocated k/v buffers. x is either a prefill from position 0
        (T > 1, causal) or one new position at cache.pos that attends to positions <= cache.pos (T == 1).
        Shapes never change, so the one-position step can be captured in a CUDA graph.
        rope: (cos, sin) tables of x's positions (pos_emb='rope'); keys are cached already rotated."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        causal, mask = True, None
        if cache is not None:
            ck, cv = cache.k[layer], cache.v[layer]
            if T > 1:
                ck[:, :, :T] = k
                cv[:, :, :T] = v
            else:
                ck.index_copy_(2, cache.pos, k.to(ck.dtype))  # under autocast QK-norm returns fp32
                cv.index_copy_(2, cache.pos, v.to(cv.dtype))
                k, v, causal, mask = ck, cv, False, cache.mask  # attend to every filled position

        # causal self-attention (fused SDPA kernels: flash / cuDNN / memory-efficient)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0,
                                           is_causal=causal)
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


class SwiGLU(nn.Module):
    """Gated MLP (Shazeer 2020): c_proj(silu(c_gate x) * c_fc x). Hidden 8/3 * n_embd (rounded up to 64) keeps the
    parameter count of the 4x GELU MLP."""

    def __init__(self, config):
        super().__init__()
        H = (8 * config.n_embd // 3 + 63) // 64 * 64
        self.c_fc = nn.Linear(config.n_embd, H, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, H, bias=config.bias)
        self.c_proj = nn.Linear(H, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = block_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = block_norm(config)
        self.mlp = SwiGLU(config) if config.mlp == 'swiglu' else MLP(config)

    def forward(self, x, cache=None, layer=0, rope=None):
        x = x + self.attn(self.ln_1(x), cache, layer, rope)
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
    # cascade heads (CascadeHeads): h after ln_f has ~unit scale, 0.02 made the conditioning invisible
    cond_emb_init_std: float = 0.5
    # conditioning: a learned style embedding (genre / artist, index 0 = none) added at every position; 0 = off.
    # pitch_size 130 = special tokens: pitch 128 = BOS (start of piece), 129 = EOS (end of piece)
    n_styles: int = 0
    # window conditioning (architecture.md B2), both added at every position like the style, zero-init, dropped in
    # training (cond_dropout) so 'none' stays usable: cond_inst = the set of instruments in the window (multi-hot over
    # n_programs, via a linear map), n_density = notes-per-second levels of the window (0 = off; 16 = 15 + 'none')
    cond_inst: bool = False
    n_density: int = 0
    # multi-instrument: a 5th note attribute, the instrument (GM program 0-127, DRUM_PROGRAM = drums); 0 = off.
    # Added to the input embedding (zero init, like the style table) and predicted by a cascade head after dt.
    n_programs: int = 0
    # positions: 'learned' = absolute table wpe (block_size rows, added to the input; old checkpoints), 'rope' = rotary
    # embedding of q/k in every attention layer (relative, no parameters, so block_size can change after training)
    pos_emb: str = 'learned'
    rope_base: float = 10000.0
    # trunk block variants (architecture.md A2-A4); the defaults are the original nanoGPT block
    norm: str = 'layernorm'     # 'layernorm' | 'rmsnorm' (blocks and ln_f)
    mlp: str = 'gelu'           # 'gelu' (4x) | 'swiglu' (8/3x, same params)
    qk_norm: bool = False
    # text control (context/claude/text_control.md, T1): a frozen text encoder's caption vector (text_dim wide, 0 =
    # off) -> Linear (zero init) -> added at every position like the style; the zero vector = no caption
    text_dim: int = 0
    # text_center: captions enter centered on the stored mean caption vector (buffer text_mean, set from training
    # captions) and rescaled to norm sqrt(text_dim), so text_proj sees what tells captions apart, not 'a caption is
    # present' (P1: 39% of LaBSE caption energy sat in the common direction); the zero vector stays 'no caption'
    text_center: bool = False
    # T2 'prefix' (text_control.md): text_prefix > 0 = the caption's token states (B, L, text_dim; zero rows =
    # padding, all zero = no caption) -> TextPrefix -> text_prefix vectors in front of the notes (no loss on them),
    # instead of the T1 vector
    text_prefix: int = 0


# order in which attributes of the next note are decided by CascadeHeads
CASCADE_ORDER = ('delta_time', 'pitch', 'duration', 'velocity')
# with n_programs: the instrument is chosen right after the onset, so pitch/duration/velocity are conditioned on it
CASCADE_ORDER_PROGRAMS = ('delta_time', 'program', 'pitch', 'duration', 'velocity')
# order of the attribute streams everywhere else (inputs, targets, datasets)
STREAM_ORDER = ('pitch', 'velocity', 'duration', 'delta_time')


class ResidualHead(nn.Module):
    """out(LN(z + MLP(LN(z)))): a direct linear path from z (like independent linear heads) plus a small MLP
    that can mix in the conditioning non-linearly. c_proj gets the small residual init in GPT.__init__."""

    def __init__(self, C, vocab, bias):
        super().__init__()
        self.ln = LayerNorm(C, bias=bias)
        self.c_fc = nn.Linear(C, C, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(C, C, bias=bias)
        self.ln_out = LayerNorm(C, bias=bias)
        self.out = nn.Linear(C, vocab, bias=False)

    def forward(self, z):
        z = z + self.c_proj(self.gelu(self.c_fc(self.ln(z))))
        return self.out(self.ln_out(z))


class CascadeHeads(nn.Module):
    """
    Factorizes P(next note | context) with the chain rule instead of 4 independent heads:
        P(dt|h) * P(pitch|h,dt) * P(dur|h,dt,pitch) * P(vel|h,dt,pitch,dur)
    Each head is a ResidualHead over h plus the embeddings of the attributes already decided.
    Training uses the true values (teacher forcing), generation feeds back the sampled ones.
    """

    def __init__(self, config):
        super().__init__()
        sizes = dict(pitch=config.pitch_size, velocity=config.velocity_size,
                     duration=config.duration_size, delta_time=config.delta_time_size, program=config.n_programs)
        C = config.n_embd
        self.order = CASCADE_ORDER_PROGRAMS if config.n_programs else CASCADE_ORDER
        # the last attribute is never used as a condition, so it needs no embedding
        self.cond_emb = nn.ModuleDict({
            name: nn.Embedding(sizes[name], C) for name in self.order[:-1]
        })
        self.heads = nn.ModuleDict({name: ResidualHead(C, sizes[name], config.bias) for name in self.order})

    def forward(self, h, targets):
        # h: (B, T, C), targets: dict name -> (B, T), -1 = ignore
        logits = {}
        cond = h
        for name in self.order:
            logits[name] = self.heads[name](cond)
            if name in self.cond_emb:
                cond = cond + self.cond_emb[name](targets[name].clamp(min=0))
        return logits

    def sample(self, h, sample_fn):
        # h: (B, 1, C) -> dict name -> (B, 1); sample_fn(logits, head_name)
        out = {}
        cond = h
        for name in self.order:
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


class TextPrefix(nn.Module):
    """T2: caption token states (B, L, D) -> (B, K, n_embd). K learned queries attend once over the tokens (padding
    = zero rows, masked), then a 2-layer MLP; the output passes a small-init gate (per channel, 0.01). Captions without
    tokens (all zero) get K learned null vectors instead (init like an embedding), the 'no caption' of dropout and
    guidance. Neither starts at exactly zero: all-zero prefix rows gave NaN gradients in the trunk (2026-10-18)."""

    def __init__(self, text_dim, n_embd, k, n_head=12, hidden=2048):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, text_dim) * 0.02)
        self.kv_norm = nn.LayerNorm(text_dim)
        self.attn = nn.MultiheadAttention(text_dim, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_embd))
        self.gate = nn.Parameter(torch.zeros(n_embd))
        self.null = nn.Parameter(torch.zeros(k, n_embd))

    def forward(self, states):
        b = states.size(0)
        tok = states.abs().sum(-1) > 0                      # (B, L) real tokens
        has = tok.any(1)                                    # (B,) rows with a caption
        tok = tok | ~has[:, None]                           # caption-less rows attend to padding (output discarded)
        q = self.queries.to(states.dtype).expand(b, -1, -1)
        kv = self.kv_norm(states)
        z, _ = self.attn(q, kv, kv, key_padding_mask=~tok, need_weights=False)
        p = self.mlp(z + q) * self.gate
        return torch.where(has[:, None, None], p, self.null.to(p.dtype).expand(b, -1, -1))


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
        # activation checkpointing (training only, not saved): the first act_ckpt blocks (-1 = all) keep only their
        # input and recompute their forward during backward (~1.33x compute, activations of those blocks ~gone).
        # act_ckpt_save: selective checkpointing, the checkpointed blocks keep the outputs of these op groups
        # (musicar.training.memory.SAC_OPS: 'attn' = attention output, ~16 MB per block at 4 x 2048 x 1024; 'attn,mm' = also the matmuls)
        self.act_ckpt = 0
        self.act_ckpt_save = ''

        assert config.pos_emb in ('learned', 'rope'), config.pos_emb
        self.transformer = nn.ModuleDict(dict(
            music_embeddings = MusicEmbeddings(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = block_norm(config),
        ))
        if config.pos_emb == 'learned':
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        if config.n_styles:
            self.transformer['style'] = nn.Embedding(config.n_styles, config.n_embd)
        if config.cond_inst:
            assert config.n_programs, 'cond_inst needs n_programs'
            self.transformer['cond_inst'] = nn.Linear(config.n_programs, config.n_embd, bias=False)
        if config.n_density:
            self.transformer['cond_density'] = nn.Embedding(config.n_density, config.n_embd)
        if config.n_programs:
            self.transformer['program'] = nn.Embedding(config.n_programs, config.n_embd)
        if config.text_prefix:
            assert config.text_dim and not config.text_center, 'text_prefix needs text_dim (and replaces T1)'
            assert config.pos_emb == 'rope', 'text_prefix shifts the notes by K positions: RoPE only'
            self.transformer['text_prefix'] = TextPrefix(config.text_dim, config.n_embd, config.text_prefix,
                                                         n_head=config.n_head)
        elif config.text_dim:
            self.transformer['text_proj'] = nn.Linear(config.text_dim, config.n_embd, bias=False)
            if config.text_center:
                self.register_buffer('text_mean', torch.zeros(config.text_dim))

        self.cascade = CascadeHeads(config)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        for emb in self.cascade.cond_emb.values():
            torch.nn.init.normal_(emb.weight, mean=0.0, std=config.cond_emb_init_std)
        if config.n_styles:
            # zero = no effect at the start of a fine-tune: the model begins exactly where the pretrained one was
            torch.nn.init.zeros_(self.transformer['style'].weight)
        if config.n_programs:
            # zero: a piano model grown to programs starts with exactly its old input embedding
            torch.nn.init.zeros_(self.transformer['program'].weight)
        for name in ('cond_inst', 'cond_density', 'text_proj'):
            if name in self.transformer:
                torch.nn.init.zeros_(self.transformer[name].weight)  # starts as the unconditioned model
        if config.text_prefix:  # _init_weights re-drew its Linears; small gate, embedding-like null prefix
            torch.nn.init.constant_(self.transformer['text_prefix'].gate, 0.01)
            torch.nn.init.normal_(self.transformer['text_prefix'].null, std=0.02)

    @property
    def head_names(self):
        """The attributes the model predicts, in STREAM_ORDER (+ 'program'): the order of `targets` in forward()."""
        return STREAM_ORDER + (('program',) if self.config.n_programs else ())

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, pitch, velocity, duration, delta_time, style=None, program=None, cache=None, cond=None,
               text=None, step=None):
        """The trunk: notes -> hidden states after ln_f, (B, T, C).
        cache=None: plain forward. With a KVCache: T > 1 = prefill from position 0, T == 1 = one step at cache.pos.
        text: (B, text_dim) caption vectors for models with text_dim (zeros / None = no caption); text_prefix models:
        (B, L, text_dim) token states (None = no caption), whose prefix goes in front of the notes (positions 0..K-1,
        the notes from K; only the notes' states are returned) in every call but a one-note step.
        step: whether a T == 1 call with a cache is a one-note step (default) or a prefill (text_prefix models)."""
        device = pitch.device
        b, t = pitch.size()
        assert t <= self.config.block_size
        step = cache is not None and t == 1 if step is None else step
        k_pre = 0 if step else self.config.text_prefix
        if step:
            seq_pos = cache.pos
            cache.mask.copy_((cache.slots <= cache.pos).view(1, 1, 1, -1))
        else:
            seq_pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.music_embeddings(pitch, velocity, duration, delta_time)
        rope = None
        if self.config.pos_emb == 'rope':
            rope = rope_tables(seq_pos, self.config.n_embd // self.config.n_head, self.config.rope_base)
        else:
            x = x + self.transformer.wpe(seq_pos)
        if self.config.n_programs:
            # program: (B, T) instrument of each input note; models with programs need it (piano stores send 0)
            x = x + self.transformer['program'](program)
        if style is not None and self.config.n_styles:
            x = x + self.transformer['style'](style).unsqueeze(1)  # style: (B,) -> added at every position
        if cond is not None:  # (instrument multi-hot (B, n_programs), density level (B,)) -> every position
            inst, density = cond
            if self.config.cond_inst:
                x = x + self.transformer['cond_inst'](inst.to(x.dtype)).unsqueeze(1)
            if self.config.n_density:
                x = x + self.transformer['cond_density'](density).unsqueeze(1)
        if text is not None and self.config.text_dim and not self.config.text_prefix:
            if self.config.text_center:
                has = text.abs().sum(-1, keepdim=True) > 0
                centered = F.normalize(text.float() - self.text_mean.float(), dim=-1) * math.sqrt(text.size(-1))
                text = torch.where(has, centered, torch.zeros_like(centered))
            x = x + self.transformer['text_proj'](text.to(x.dtype)).unsqueeze(1)
        if k_pre:
            if text is None:
                text = torch.zeros(b, 1, self.config.text_dim, device=device)
            x = torch.cat([self.transformer['text_prefix'](text.to(x.dtype)).to(x.dtype), x], dim=1)
            rope = rope_tables(torch.arange(k_pre + t, device=device), self.config.n_embd // self.config.n_head,
                               self.config.rope_base)
        x = self.transformer.drop(x)
        n_ckpt = len(self.transformer.h) if self.act_ckpt < 0 else self.act_ckpt
        ckpt_kw = dict(context_fn=partial(sac_context, self.act_ckpt_save)) if self.act_ckpt_save else {}
        for i, block in enumerate(self.transformer.h):
            if i < n_ckpt and self.training and cache is None and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, None, i, rope, use_reentrant=False,
                                                      respect_saved_tensors_hooks=True, **ckpt_kw)
            else:
                x = block(x, cache, i, rope)
        return self.transformer.ln_f(x[:, k_pre:] if k_pre else x)

    def forward(self, pitch, velocity, duration, delta_time, style=None, program=None, targets=None, cond=None,
                text=None):
        """
        Returns (parts, loss). loss = sum of the per-head CEs with label smoothing (what we optimize),
        parts = dict name -> CE without label smoothing (detached, for honest logging / comparing runs).
        targets: one (B, T) tensor per head in head_names order, -1 = ignore.
        style: (B,) style ids when the model has n_styles (0 = none).
        program: (B, T) instrument per input note when the model has n_programs; targets then has a 5th entry, the
        next note's program (see head_names). Duration targets of drum notes are expected as -1 (masked).
        cond: (instrument multi-hot (B, n_programs), density level (B,)) for models with cond_inst / n_density.
        text: (B, text_dim) caption vectors for models with text_dim.
        """
        x = self.encode(pitch, velocity, duration, delta_time, style=style, program=program, cond=cond, text=text)
        assert len(targets) == len(self.head_names), f"{len(targets)} targets for heads {self.head_names}"
        tgt = dict(zip(self.head_names, targets))
        logits = self.cascade(x, tgt)
        ls = self.config.label_smoothing
        loss = 0.0
        parts = {}
        for name in self.head_names:
            lg = logits[name].view(-1, logits[name].size(-1))
            t = tgt[name].reshape(-1)
            # mean over the unmasked targets, 0 (not NaN = 0/0) when there are none, e.g. the duration head
            # of a drums-only window; no data-dependent branch, so no sync / graph break under compile
            n_valid = (t != -1).sum().clamp(min=1)
            loss = loss + F.cross_entropy(lg, t, ignore_index=-1, label_smoothing=ls, reduction='sum') / n_valid
            with torch.no_grad():
                parts[name] = F.cross_entropy(lg.float(), t, ignore_index=-1, reduction='sum') / n_valid
        return parts, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if 'wpe' in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
