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
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
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
    # conditioning: a learned style embedding (genre / artist, index 0 = none) added at every position; 0 = off.
    # pitch_size 130 = special tokens: pitch 128 = BOS (start of piece), 129 = EOS (end of piece)
    n_styles: int = 0
    # multi-instrument: a 5th note attribute, the instrument (GM program 0-127, DRUM_PROGRAM = drums); 0 = off.
    # Added to the input embedding (zero init, like the style table) and predicted by a cascade head after dt.
    n_programs: int = 0
    # positions: 'learned' = absolute table wpe (block_size rows, added to the input; old checkpoints), 'rope' = rotary
    # embedding of q/k in every attention layer (relative, no parameters, so block_size can change after training)
    pos_emb: str = 'learned'
    rope_base: float = 10000.0
    # future prediction (training only, dropped at inference): small heads on h_t predict, for each time horizon after
    # note t's onset ('a-b' seconds), the pitch-class histogram, log note count and mean pitch of the notes sounding
    # there; loss += future_weight * mean over horizons. 0 = off (no parameters)
    future_weight: float = 0.0
    future_horizons: str = '0-2,2-4,4-8'


BOS_PITCH, EOS_PITCH = 128, 129
DRUM_PROGRAM = 128


# order in which attributes of the next note are decided by CascadeHeads
CASCADE_ORDER = ('delta_time', 'pitch', 'duration', 'velocity')
# with n_programs: the instrument is chosen right after the onset, so pitch/duration/velocity are conditioned on it
CASCADE_ORDER_PROGRAMS = ('delta_time', 'program', 'pitch', 'duration', 'velocity')
# order of the attribute streams everywhere else (inputs, targets, datasets)
STREAM_ORDER = ('pitch', 'velocity', 'duration', 'delta_time')


DT_BINS_PER_SEC = 50  # delta_time bins are 20 ms


def future_targets(pitch, dt, valid, real, horizons):
    """Targets of the future heads from the window's own notes (no extra data).
    pitch, dt: (B, N) the window's notes 0..T (N = T + 1: the first input note, then the targets); valid: (B, N) pitched
    real notes (no BOS/EOS, padding or drums); real: (B, N) any real token (not padding). horizons: [(a, b)] in bins.
    Position t (0..T-1) has seen notes 0..t; its horizon (a, b) holds the notes i > t with onset in [on_t + a, on_t + b).
    Returns per horizon (hist (B, T, 12) counts, count (B, T), pitch sum (B, T), ok (B, T)); ok = the horizon ends
    before the window's last real onset (else the window can't tell what is there) and position t is a real token."""
    B, N = pitch.shape
    T = N - 1
    onset = torch.cumsum(dt.clamp(min=0), 1)                                    # (B, N) int bins, non-decreasing
    t_on = onset[:, :T]
    pc = F.one_hot(pitch.clamp(0, 127) % 12, 12).float() * valid.unsqueeze(-1)
    P = F.pad(torch.cumsum(pc, 1), (0, 0, 1, 0))                                # P[i] = sum over notes < i
    Cn = F.pad(torch.cumsum(valid.float(), 1), (1, 0))
    R = F.pad(torch.cumsum(pitch.float() * valid, 1), (1, 0))
    last_on = torch.where(real, onset, torch.zeros_like(onset)).amax(1, keepdim=True)
    nxt = torch.arange(1, T + 1, device=pitch.device).expand(B, T)
    out = []
    for a, b in horizons:
        lo = torch.maximum(torch.searchsorted(onset, t_on + a), nxt)
        hi = torch.maximum(torch.searchsorted(onset, t_on + b), lo)
        hist = P.gather(1, hi.unsqueeze(-1).expand(B, T, 12)) - P.gather(1, lo.unsqueeze(-1).expand(B, T, 12))
        count = Cn.gather(1, hi) - Cn.gather(1, lo)
        psum = R.gather(1, hi) - R.gather(1, lo)
        ok = (t_on + b <= last_on) & real[:, :T]
        out.append((hist, count, psum, ok))
    return out


class FutureHeads(nn.Module):
    """LN -> MLP -> per horizon 12 pitch-class logits + log(1 + count) + (mean pitch - 60) / 12."""

    def __init__(self, config):
        super().__init__()
        self.horizons = [tuple(round(float(x) * DT_BINS_PER_SEC) for x in h.split('-'))
                         for h in config.future_horizons.split(',')]
        C = config.n_embd
        self.ln = LayerNorm(C, bias=config.bias)
        self.fc1 = nn.Linear(C, C, bias=config.bias)
        self.fc2 = nn.Linear(C, 14 * len(self.horizons), bias=config.bias)

    def forward(self, h):
        B, T, _ = h.shape
        return self.fc2(F.gelu(self.fc1(self.ln(h)))).view(B, T, len(self.horizons), 14)

    def loss(self, h, pitch, dt, valid, real):
        """Mean over horizons of pitch-class CE (soft targets) + count MSE + register MSE; also the mean pc CE alone."""
        pred = self(h).float()
        total, pc_ce = 0.0, 0.0
        for k, (hist, count, psum, ok) in enumerate(future_targets(pitch, dt, valid, real, self.horizons)):
            p = pred[:, :, k]
            has = ok & (count > 0)
            n_has, n_ok = has.sum().clamp(min=1), ok.sum().clamp(min=1)
            ce = -(hist / count.clamp(min=1).unsqueeze(-1) * F.log_softmax(p[..., :12], -1)).sum(-1)
            ce = (ce * has).sum() / n_has
            cnt = (((p[..., 12] - torch.log1p(count)) ** 2) * ok).sum() / n_ok
            reg = (((p[..., 13] - (psum / count.clamp(min=1) - 60) / 12) ** 2) * has).sum() / n_has
            total = total + (ce + cnt + reg) / len(self.horizons)
            pc_ce = pc_ce + ce.detach() / len(self.horizons)
        return total, pc_ce


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
                     duration=config.duration_size, delta_time=config.delta_time_size, program=config.n_programs)
        C = config.n_embd
        self.order = CASCADE_ORDER_PROGRAMS if config.n_programs else CASCADE_ORDER
        assert config.cascade_residual or not config.n_programs, "programs need the v2 (residual) cascade heads"
        # the last attribute is never used as a condition, so it needs no embedding
        self.cond_emb = nn.ModuleDict({
            name: nn.Embedding(sizes[name], C) for name in self.order[:-1]
        })
        self.residual = config.cascade_residual
        if self.residual:
            self.heads = nn.ModuleDict({
                name: ResidualHead(C, sizes[name], config.bias,
                                   n_blocks=config.pitch_head_blocks if name == 'pitch' else 1,
                                   mult=config.pitch_head_mult if name == 'pitch' else 1)
                for name in self.order
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

        assert config.pos_emb in ('learned', 'rope'), config.pos_emb
        self.transformer = nn.ModuleDict(dict(
            music_embeddings = MusicEmbeddings(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        if config.pos_emb == 'learned':
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        if config.n_styles:
            self.transformer['style'] = nn.Embedding(config.n_styles, config.n_embd)
        if config.n_programs:
            self.transformer['program'] = nn.Embedding(config.n_programs, config.n_embd)

        if config.cascade_heads:
            self.cascade = CascadeHeads(config)
        else:
            self.head_pitch      = nn.Linear(config.n_embd, config.pitch_size,      bias=False)
            self.head_velocity   = nn.Linear(config.n_embd, config.velocity_size,   bias=False)
            self.head_duration   = nn.Linear(config.n_embd, config.duration_size,   bias=False)
            self.head_delta_time = nn.Linear(config.n_embd, config.delta_time_size, bias=False)
        if config.future_weight:
            self.future = FutureHeads(config)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        if config.cascade_heads and config.cascade_residual:
            for emb in self.cascade.cond_emb.values():
                torch.nn.init.normal_(emb.weight, mean=0.0, std=config.cond_emb_init_std)
        if config.n_styles:
            # zero = no effect at the start of a fine-tune: the model begins exactly where the pretrained one was
            torch.nn.init.zeros_(self.transformer['style'].weight)
        if config.n_programs:
            # zero: a piano model grown to programs starts with exactly its old input embedding
            torch.nn.init.zeros_(self.transformer['program'].weight)

    @property
    def head_names(self):
        """The attributes the model predicts, in STREAM_ORDER (+ 'program'): the order of `targets` in forward()."""
        return STREAM_ORDER + (('program',) if self.config.n_programs else ())

    def load_expanded(self, state_dict):
        """Load a checkpoint into a model that may have grown: extra rows in dim 0 (pitch vocab 128 -> 130) keep
        their fresh init, keys the checkpoint lacks (style embedding) keep theirs. Returns the grown/new keys.
        Exception: a new program condition embedding is zeroed, so the old pitch/duration/velocity heads see exactly
        the conditioning they were trained with (function-preserving growth to multi-instrument), and so is the
        instrument head's output layer."""
        own = self.state_dict()
        # a learned-position checkpoint loaded into a RoPE model: the position table has no place to go
        dropped = [k for k in state_dict if k.startswith('transformer.wpe.') and k not in own]
        for k in dropped:
            del state_dict[k]
        changed = [k for k in own if k not in state_dict] + [f'{k} (dropped)' for k in dropped]
        for k in [k for k in own if k not in state_dict]:
            # the new instrument head's output also starts at zero = a uniform guess (ln 129), not a random one
            if k.startswith('cascade.cond_emb.program.') or k == 'cascade.heads.program.out.weight':
                state_dict[k] = torch.zeros_like(own[k])
        for k, v in state_dict.items():
            if k == 'transformer.wpe.weight' and own[k].shape[0] > v.shape[0]:
                # longer context with learned positions: stretch the table (position interpolation), so position p
                # gets the old embedding at p * old/new, linearly between neighbouring rows
                state_dict[k] = F.interpolate(v.float().t()[None], size=own[k].shape[0], mode='linear',
                                              align_corners=True)[0].t().to(v.dtype)
                changed.append(f'{k} (stretched {v.shape[0]} -> {own[k].shape[0]})')
            elif k in own and own[k].shape != v.shape:
                assert own[k].shape[1:] == v.shape[1:] and own[k].shape[0] >= v.shape[0], \
                    f"{k}: can't grow {tuple(v.shape)} -> {tuple(own[k].shape)}"
                grown = own[k].clone()
                grown[:v.shape[0]] = v
                state_dict[k] = grown
                changed.append(k)
        self.load_state_dict(state_dict, strict=False)
        return changed

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

    def _encode(self, pitch, velocity, duration, delta_time, style=None, program=None, cache=None):
        """cache=None: plain forward. With a KVCache: T > 1 = prefill from position 0, T == 1 = one step at cache.pos."""
        device = pitch.device
        b, t = pitch.size()
        assert t <= self.config.block_size
        if cache is not None and t == 1:
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
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x, cache, i, rope)
        return self.transformer.ln_f(x)

    def _head_logits(self, x, targets=None):
        # x: (B, T, C) -> dict name -> logits. targets (dict) are only needed for cascade heads.
        if self.config.cascade_heads:
            return self.cascade(x, targets)
        return dict(pitch=self.head_pitch(x), velocity=self.head_velocity(x),
                    duration=self.head_duration(x), delta_time=self.head_delta_time(x))

    def forward(self, pitch, velocity, duration, delta_time, style=None, program=None, targets=None):
        """
        With targets: returns (parts, loss). loss = sum of the 4 CEs with label smoothing (what we optimize),
        parts = dict name -> CE without label smoothing (detached, for honest logging / comparing runs).
        Without targets (independent heads only): returns ((p, v, d, dt) logits of the last position, None).
        Cascade heads can't give all 4 logits without choosing the earlier attributes, so use generate().
        style: (B,) style ids when the model has n_styles (0 = none).
        program: (B, T) instrument per input note when the model has n_programs; targets then has a 5th entry, the
        next note's program (see head_names). Duration targets of drum notes are expected as -1 (masked).
        """
        x = self._encode(pitch, velocity, duration, delta_time, style=style, program=program)

        if targets is not None:
            assert len(targets) == len(self.head_names), f"{len(targets)} targets for heads {self.head_names}"
            tgt = dict(zip(self.head_names, targets))
            logits = self._head_logits(x, tgt)
            ls = self.config.label_smoothing
            loss = 0.0
            parts = {}
            for name in self.head_names:
                lg = logits[name].view(-1, logits[name].size(-1))
                t = tgt[name].reshape(-1)
                loss = loss + F.cross_entropy(lg, t, ignore_index=-1, label_smoothing=ls)
                with torch.no_grad():
                    parts[name] = F.cross_entropy(lg.float(), t, ignore_index=-1)
            if self.config.future_weight:
                # the window's notes 0..T: the first input note, then the targets (-1 = padding)
                seq = lambda first, t: torch.cat([first[:, :1], t.clamp(min=0)], 1)
                fp, fdt = seq(pitch, tgt['pitch']), seq(delta_time, tgt['delta_time'])
                real = torch.cat([torch.ones_like(pitch[:, :1], dtype=torch.bool), tgt['pitch'] >= 0], 1)
                valid = real & (fp < BOS_PITCH)
                if self.config.n_programs:
                    valid = valid & (seq(program, tgt['program']) != DRUM_PROGRAM)
                fut, fut_pc = self.future.loss(x, fp, fdt, valid, real)
                loss = loss + self.config.future_weight * fut
                parts['future'] = fut.detach()
                parts['future_pc'] = fut_pc
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
        self.config.block_size = block_size
        if 'wpe' in self.transformer:
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
                 cuda_graph=True, style=None, min_new=0, program=0, top_p=None, cfg_scale=1.0):
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
        style: (B,) style ids (models with n_styles; 0 = none/unconditional).
        program: models with n_programs: every note is played by this one instrument (0 = piano); its head isn't
          sampled. Sampling the instrument per note is to-do 15.
        top_p: nucleus sampling, keep the smallest set of values whose probability reaches top_p (after top_k).
        cfg_scale != 1 (needs a style): classifier-free guidance. The batch runs twice, with the style and with
          style 0 (none, which style dropout trained), and every head samples from uncond + s * (cond - uncond); both
          halves are fed the same note. Not combined with target_nps.
        Special tokens (pitch vocab 130): BOS is never sampled; EOS (end of piece) not before min_new notes. Rows keep
        running after their EOS (batch shapes stay fixed); cut them at the first EOS when writing MIDI.
        Nothing here syncs the GPU with the CPU inside the loop.
        """
        L = self.config.block_size
        slide = slide or L // 4
        assert 0 <= anchor and anchor + slide < L, "need anchor + slide < block_size"
        device = pitch.device
        Bo = pitch.size(0)  # rows asked for; with guidance the model runs 2 * Bo (conditional, then unconditional)
        guided = cfg_scale != 1.0 and style is not None
        assert not (guided and target_nps is not None), "guidance with a density target isn't supported"
        if guided:
            pitch, velocity, duration, delta_time = (torch.cat([t, t]) for t in (pitch, velocity, duration, delta_time))
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
        bias = torch.full((Bo, 1), float(dt_bias), device=device)
        target = None
        if target_nps is not None:
            target = torch.as_tensor(target_nps, dtype=torch.float32, device=device).reshape(-1, 1).expand(B, 1)

        specials = self.config.pitch_size > BOS_PITCH
        step_i = [0]
        if style is not None:
            style = torch.as_tensor(style, dtype=torch.long, device=device).reshape(-1).expand(Bo).contiguous()
            if guided:
                style = torch.cat([style, torch.zeros_like(style)])
        fixed_program = torch.full((B, 1), int(program), dtype=torch.long, device=device) \
            if self.config.n_programs else None

        def sample(logits, name=None):  # (B, 1, vocab) → (B, 1)
            if name == 'program':
                return fixed_program
            logits = logits[:, -1, :].float()
            if guided:
                logits = logits[Bo:] + cfg_scale * (logits[:Bo] - logits[Bo:])
            logits = logits / temperature
            if name == 'pitch' and specials:
                logits[:, BOS_PITCH] = -float('Inf')
                if step_i[0] < min_new:
                    logits[:, EOS_PITCH] = -float('Inf')
            if name == 'delta_time' and (dt_bias != 0.0 or target is not None):
                logits = logits + bias * dt_shape
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if top_p is not None:
                sorted_p, order = torch.sort(probs, dim=-1, descending=True)
                drop = sorted_p.cumsum(-1) - sorted_p > top_p  # mass before this value already reaches top_p
                probs = probs.scatter(-1, order, sorted_p.masked_fill(drop, 0.0))
            nxt = torch.multinomial(probs, num_samples=1)
            return torch.cat([nxt, nxt]) if guided else nxt

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
            prog = None if fixed_program is None else fixed_program.expand(B, w)
            return self._encode(*window, style=style, program=prog, cache=cache)[:, [-1], :], w

        # the one-note step: static inputs -> static output, optionally captured as a CUDA graph
        step_in = [torch.zeros(B, 1, dtype=torch.long, device=device) for _ in range(4)]
        step_style = style  # constant during generation, so the graph can read it directly
        graph = None
        if cuda_graph and device.type == 'cuda':
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm-up outside the graph (allocator, kernel selection)
                for _ in range(2):
                    self._encode(*step_in, style=step_style, program=fixed_program, cache=cache)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step_out = self._encode(*step_in, style=step_style, program=fixed_program, cache=cache)
            # the warm-up and capture wrote junk into slot 0; prefill below overwrites every slot it uses

        def step(new, pos):
            for buf, t in zip(step_in, new):
                buf.copy_(t)
            cache.pos.fill_(pos)
            if graph is not None:
                graph.replay()
                return step_out
            return self._encode(*step_in, style=step_style, program=fixed_program, cache=cache)

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
            step_i[0] = i + 1
            if progress is not None:
                progress(i + 1)
            if i == max_new_tokens - 1:
                break
            if filled < L:
                h = step(new, filled)
                filled += 1
            else:
                h, filled = prefill(n)

        return tuple(o[:Bo] for o in out)
