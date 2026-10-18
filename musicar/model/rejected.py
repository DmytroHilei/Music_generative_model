"""
Architecture options that were tested and rejected, kept for later reuse. Nothing in the training / sampling code
imports this module; model.py has no hooks for these. Each entry lists its result (context/claude/agents.md
experiment log) and what to wire up to use it again.

Last commit where all of them were integrated in GPT (config fields, forward, train.py flags): 22dc6b9
(`git show 22dc6b9:musicar/model/gpt.py`). Their checkpoints (ab_indep, ab_cascade, checkpoints/ckpt.pt,
ladder_L_pitchhead, fut_w0.2_*, fut_w1.0_*) load only with that version.
"""

import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM
from musicar.model.gpt import CASCADE_ORDER, LayerNorm


# ---------------------------------------------------------------------------------------------------------------------
# Mixture of experts FFN (ladder-L-moe8, 2026-09-29, 84a33ed)
# Result: 8 experts top-2 (117M total / 42M active) +0.10 CE vs the dense L per step and about 2.5x slower per step.
# Too few tokens per expert at 100M tokens; unfused expert loop.
# To use again: Block.mlp = MoEMLP(...) instead of MLP / SwiGLU; loss += aux_weight (0.01) * mean of block.mlp.aux
# over the blocks, in GPT.forward after the CE loop.

class MoEMLP(nn.Module):
    """
    Top-k routed mixture of expert FFNs (dropless: every token goes to its k experts, no capacity limit).
    Experts are stored as stacked weights and run one after another on the tokens routed to them.
    self.aux holds the load-balancing loss of the last forward (Switch Transformer: E * sum_i f_i * P_i).
    Expert hidden = 4 * n_embd * hidden_frac; with top_k=2 and frac=0.5 the active FFN compute equals the dense model.
    """

    def __init__(self, n_embd, n_layer, dropout, experts=8, top_k=2, hidden_frac=0.5):
        super().__init__()
        C, E = n_embd, experts
        H = int(4 * C * hidden_frac)
        self.top_k = top_k
        self.router = nn.Linear(C, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, C, H) * 0.02)
        self.w2 = nn.Parameter(torch.randn(E, H, C) * 0.02 / math.sqrt(2 * n_layer))
        self.dropout = nn.Dropout(dropout)
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


# ---------------------------------------------------------------------------------------------------------------------
# Bigger pitch head (ladder-L-pitchhead, 2026-09-29, 84a33ed)
# Result: 2 blocks x 2 width (+1.6M params) +0.04 CE vs L, and the pitch CE itself got worse. Pitch gains come from
# trunk capacity (context), not the head.
# To use again: CascadeHeads builds the 'pitch' head as DeepResidualHead(C, vocab, bias, n_blocks, mult). With
# n_blocks=1, mult=1 it has the same parameters (and state_dict keys) as model.ResidualHead.

class DeepResidualHead(nn.Module):
    """out(LN(z + MLP(LN(z)))) with n_blocks residual MLP blocks of hidden width mult * C."""

    def __init__(self, C, vocab, bias, n_blocks=2, mult=2):
        super().__init__()
        H = mult * C
        self.ln = LayerNorm(C, bias=bias)
        self.c_fc = nn.Linear(C, H, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(H, C, bias=bias)
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


# ---------------------------------------------------------------------------------------------------------------------
# Output head designs before the residual cascade (ab-indep / ab-cascade, 2026-09-28, d5b38ae)
# Result (6M, 4 epochs, summed CE): independent heads 9.824, cascade v1 (plain MLP heads) 9.793, cascade v2
# (model.ResidualHead, the default since) 9.412. v1's pitch head was worse than the independent one despite seeing dt.
# To use again: GPT.cascade = IndependentHeads(...) (its forward ignores targets; sampling then samples every head from
# h alone), or CascadeHeads.heads = v1_cascade_heads(...) without the cond_emb init_std / no-weight-decay rule.

class IndependentHeads(nn.Module):
    """4 linear heads on h, P(note | h) = P(dt|h) P(pitch|h) P(dur|h) P(vel|h). Same call interface as CascadeHeads."""

    def __init__(self, C, sizes):
        super().__init__()
        self.order = CASCADE_ORDER
        self.heads = nn.ModuleDict({name: nn.Linear(C, sizes[name], bias=False) for name in self.order})

    def forward(self, h, targets=None):
        return {name: self.heads[name](h) for name in self.order}

    def sample(self, h, sample_fn):
        return {name: sample_fn(self.heads[name](h), name) for name in self.order}


def v1_cascade_heads(C, sizes, bias):
    """Cascade v1: LN -> Linear -> GELU -> Linear per attribute, no residual path."""
    return nn.ModuleDict({
        name: nn.Sequential(
            LayerNorm(C, bias=bias),
            nn.Linear(C, C, bias=bias),
            nn.GELU(),
            nn.Linear(C, sizes[name], bias=False),
        ) for name in CASCADE_ORDER
    })


# ---------------------------------------------------------------------------------------------------------------------
# Future prediction heads (future heads A/B, 2026-10-02, 9d6c770)
# Result: future_weight 0.2: next-note CE 10.780 vs 10.779 (no effect); 1.0: +0.01 CE (2x seed spread). The future
# pitch-class CE went 2.49 -> 2.14 (floor 1.51): the heads read out the key, little foresight. No structure metric
# changed beyond seed noise. Revisit only with clean, repetitive data (covers) or much longer context.
# To use again: GPT.future = FutureHeads(...); in GPT.forward, after the CE loop:
#     fut, fut_pc = future_loss(self.future, x, pitch, delta_time, program, tgt)
#     loss = loss + future_weight * fut; parts['future'] = fut.detach(); parts['future_pc'] = fut_pc
# and train.py logs ce/future_pc. The heads are training-only: drop the future.* keys before loading for sampling.

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
    """LN -> MLP -> per horizon 12 pitch-class logits + log(1 + count) + (mean pitch - 60) / 12.
    horizons: 'a-b' seconds after each note's onset, comma-separated."""

    def __init__(self, C, bias, horizons='0-2,2-4,4-8'):
        super().__init__()
        self.horizons = [tuple(round(float(x) * DT_BINS_PER_SEC) for x in h.split('-'))
                         for h in horizons.split(',')]
        self.ln = LayerNorm(C, bias=bias)
        self.fc1 = nn.Linear(C, C, bias=bias)
        self.fc2 = nn.Linear(C, 14 * len(self.horizons), bias=bias)

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


def future_loss(heads, h, pitch, delta_time, program, tgt):
    """FutureHeads loss inside GPT.forward. h: ln_f output (B, T, C); pitch, delta_time, program: the inputs (B, T);
    tgt: dict head name -> (B, T) targets (-1 = padding). Returns (loss, pitch-class CE)."""
    # the window's notes 0..T: the first input note, then the targets (-1 = padding)
    seq = lambda first, t: torch.cat([first[:, :1], t.clamp(min=0)], 1)
    fp, fdt = seq(pitch, tgt['pitch']), seq(delta_time, tgt['delta_time'])
    real = torch.cat([torch.ones_like(pitch[:, :1], dtype=torch.bool), tgt['pitch'] >= 0], 1)
    valid = real & (fp < BOS_PITCH)
    if program is not None and 'program' in tgt:
        valid = valid & (seq(program, tgt['program']) != DRUM_PROGRAM)
    return heads.loss(h, fp, fdt, valid, real)


# ---------------------------------------------------------------------------------------------------------------------
# Velocity augmentation (ua-C, 2026-10-01, 7396237)
# Result: B3 x30 recipe on the 448 Ukrainian songs, 600 it: tempo aug 10.479, tempo + velocity shift up to ±2 bins
# 10.497 best Ukrainian val CE (seed noise about ±0.01); val2 (Aria pop+rock) 6.07 -> 6.27. Velocity aug hurts.
# To use again: in NoteWindowDataset.__getitem__, inside `if self.augment and not self.eval_stride:` after the tempo
# stretch: `if self.aug_velocity and store_augmented: velocity = shift_velocity(velocity, notes, self.aug_velocity,
# self.velocity_bins)`, plus the aug_velocity argument of NoteWindowDataset and train.py.

def shift_velocity(velocity, notes, max_shift, velocity_bins=32):
    """Shift the velocity bins of all notes (mask `notes`, i.e. not BOS/EOS) of a window by one random amount in
    ±max_shift, clamped. Symmetric shift, clamped: a no-clip bound would bias it downwards (-0.9 bins on the
    reductions), since most windows hold a few melody notes saturated at the top bin (the melody boost clips at 127)."""
    shift = random.randint(-max_shift, max_shift)
    return torch.where(notes, (velocity + shift).clamp(0, velocity_bins - 1), velocity)
