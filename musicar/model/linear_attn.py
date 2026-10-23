"""Gated DeltaNet linear attention (Yang et al., ICLR 2025, arXiv 2412.06464), as used in the 3 linear : 1 full
hybrid of Qwen3-Next. Each head keeps a (head_size x head_size) state instead of a KV cache:

    S_t = exp(g_t) S_{t-1} + k_t u_t^T,   u_t = beta_t (v_t - (exp(g_t) S_{t-1})^T k_t),   o_t = S_t^T q_t

g_t <= 0 = the gate (log decay: forgetting), beta_t in (0, 1) = the write strength, u_t = the delta-rule write (only
what the state does not already return for k_t). Plain PyTorch: chunked training form (WY / UT transform, derivation
in chunk_gated_delta_rule) and the recurrent reference it is tested against (tests/model/test_linear_attn.py).
KERNEL = 'fla' (environment MUSICAR_GDN_KERNEL=fla) runs the same recurrence through the fused Triton kernels of
flash-linear-attention (fla-core) on CUDA instead; outputs and gradients agree to bf16 precision (~0.5%). Not for
training with fp8 (2026-10-23): with tensorwise fp8 linears the gradient norm from the second step on is ~half the
PyTorch path's (tiny model, eager and compiled), and the 107M 'all' recipe went NaN from step 2.

Conversion from a trained attention layer (context/claude/optimizations.md, linear attention): c_attn / c_proj keep
the attention's names and weights; the short conv starts as the identity, the output gate near 1, so the layer starts
from the attention's projections. Stage 1 (attention transfer, LoLCATs arXiv 2410.10254): attach_teacher() keeps a
frozen copy of the attention; in training the layer returns the teacher's output (the next layers see the original
stream) and sets transfer_loss = its own output's relative MSE to it; in eval it returns its own output.
"""

import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

KERNEL = os.environ.get('MUSICAR_GDN_KERNEL', 'torch')   # 'torch' (chunk_gated_delta_rule) or 'fla' (fused, CUDA only)


def linear_layers(spec, n_layer):
    """MusicConfig.linear_attn -> the set of block indices with linear attention. '' = none; 'hybrid<N>' = every
    layer but each N-th and the last (hybrid4 = Qwen3-Next's 3 : 1); else comma-separated indices."""
    if not spec:
        return set()
    if spec.startswith('hybrid'):
        n = int(spec[len('hybrid'):])
        return {i for i in range(n_layer) if (i + 1) % n and i != n_layer - 1}
    layers = {int(s) for s in spec.split(',') if s.strip()}
    assert all(0 <= i < n_layer for i in layers), f'linear_attn={spec!r} for {n_layer} layers'
    return layers


def recurrent_gated_delta_rule(q, k, v, g, beta, state=None):
    """Step-by-step reference (and the one-note decode form). q, k: (B, H, T, dk), v: (B, H, T, dv), g, beta:
    (B, H, T); state (B, H, dk, dv) or None. Returns o (B, H, T, dv) and the final state, fp32."""
    with torch.autocast(q.device.type, enabled=False):
        q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
        B, H, T, dk = q.shape
        S = q.new_zeros(B, H, dk, v.size(-1)) if state is None else state.float()
        out = []
        for t in range(T):
            S = S * g[:, :, t, None, None].exp()
            u = beta[:, :, t, None] * (v[:, :, t] - torch.einsum('bhkv,bhk->bhv', S, k[:, :, t]))
            S = S + k[:, :, t, :, None] * u[:, :, None, :]
            out.append(torch.einsum('bhkv,bhk->bhv', S, q[:, :, t]))
        return torch.stack(out, 2), S


def chunk_gated_delta_rule(q, k, v, g, beta, chunk=64, state=None):
    """The same recurrence, chunk by chunk (the training form). Inside a chunk with start state S0 and cumulative log
    decay G_t = g_1 + ... + g_t, unrolling gives
        u_t + beta_t sum_{s<t} e^(G_t - G_s) (k_t . k_s) u_s = beta_t v_t - beta_t e^(G_t) S0^T k_t,
    i.e. (I + L) U = beta V - (beta e^G K) S0 with L strictly lower triangular, so U = U0 - W S0 with T = (I + L)^-1,
    U0 = T (beta V), W = T (beta e^G K). Then O = (e^G Q) S0 + (Q K^T * D) U with D[t, s] = e^(G_t - G_s) for s <= t,
    and the next chunk starts from e^(G_C) S0 + (e^(G_C - G) K)^T U. fp32 throughout (autocast off).
    Shapes as in recurrent_gated_delta_rule; T is padded to a multiple of chunk (padded steps: g = beta = 0, k = 0,
    i.e. no decay and no write)."""
    with torch.autocast(q.device.type, enabled=False):
        q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
        B, H, T, dk = q.shape
        dv = v.size(-1)
        pad = (-T) % chunk
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
        N = (T + pad) // chunk
        q, k, v = (t.view(B, H, N, chunk, -1) for t in (q, k, v))
        G = g.view(B, H, N, chunk).cumsum(-1)
        beta = beta.view(B, H, N, chunk)
        i = torch.arange(chunk, device=q.device)
        causal = i[:, None] >= i[None, :]
        D = (G[..., :, None] - G[..., None, :]).masked_fill(~causal, float('-inf')).exp()   # (B, H, N, C, C)
        kb = k * beta[..., None]
        L = ((kb @ k.transpose(-1, -2)) * D).masked_fill(i[:, None] <= i[None, :], 0.0)      # strictly lower
        eye = torch.eye(chunk, device=q.device)
        Tm = torch.linalg.solve_triangular(eye + L, eye.expand_as(L), upper=False, unitriangular=True)
        W = Tm @ (kb * G.exp()[..., None])
        U0 = Tm @ (v * beta[..., None])
        A = (q @ k.transpose(-1, -2)) * D
        S = q.new_zeros(B, H, dk, dv) if state is None else state.float()
        out = []
        for n in range(N):
            U = U0[:, :, n] - W[:, :, n] @ S
            out.append((q[:, :, n] * G[:, :, n, :, None].exp()) @ S + A[:, :, n] @ U)
            last = G[:, :, n, -1]
            S = S * last.exp()[..., None, None] + \
                (k[:, :, n] * (last[..., None] - G[:, :, n]).exp()[..., None]).transpose(-1, -2) @ U
        return torch.stack(out, 2).view(B, H, N * chunk, dv)[:, :, :T], S


class GatedDeltaNet(nn.Module):
    """Drop-in for CausalSelfAttention in a Block (same forward signature; rope is ignored: positions come from the
    order of the recurrence). q, k: projection -> causal depthwise conv (kernel 4) -> SiLU -> L2 norm per head
    (q also * head_size^-0.5); v: projection -> conv. Gate g = -exp(A_log) * softplus(a + dt_bias) (Mamba-2 init:
    per-head half-lives from ~1 to ~700 notes), beta = sigmoid(b). Output: RMSNorm per head * sigmoid(gate) ->
    c_proj. Generation: with a KVCache the layer keeps its own state in cache.lin[layer] (the last conv_size - 1
    projections for the conv, the (H, hs, hs) recurrent state in fp32) and ignores the k / v buffers: a prefill
    (T > 1) starts from zero, a one-note step and an appended chunk (cache.chunk_start, section vectors) continue it;
    a step at slot 0 starts from zero too (no host sync, so the step can be captured in a CUDA graph)."""

    def __init__(self, config, conv_size=4, chunk=64):
        super().__init__()
        from musicar.model.gpt import RMSNorm
        C, H = config.n_embd, config.n_head
        self.n_head, self.n_embd, self.head_size, self.chunk = H, C, C // H, chunk
        self.c_attn = nn.Linear(C, 3 * C, bias=config.bias)
        self.c_proj = nn.Linear(C, C, bias=config.bias)
        self.conv = nn.Conv1d(3 * C, 3 * C, conv_size, groups=3 * C, padding=conv_size - 1, bias=False)
        self.ab_proj = nn.Linear(C, 2 * H, bias=False)       # gate input a and write strength b per head
        self.out_gate = nn.Linear(C, C, bias=True)
        self.A_log = nn.Parameter(torch.zeros(H))
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.o_norm = RMSNorm(self.head_size)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.teacher = None          # stage 1: [frozen CausalSelfAttention] (a list: not a submodule, not saved)
        self.transfer_loss = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """The layer's own parameters (GPT.__init__ calls it again after its generic init)."""
        self.conv.weight.zero_()
        self.conv.weight[:, 0, -1] = 1.0                     # identity: the current note's projection
        self.ab_proj.weight.zero_()                          # a = 0 -> the dt_bias decay; b = 0 -> beta 0.5
        self.out_gate.weight.zero_()
        self.out_gate.bias.fill_(4.0)                        # sigmoid(4) = 0.98: output gate open
        H = self.n_head
        self.A_log.copy_(torch.empty(H).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(H).uniform_(math.log(1e-3), math.log(0.1)))
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # softplus^-1(dt)
        self.o_norm.weight.fill_(1.0)

    def attach_teacher(self, attn):
        """Stage 1: a frozen copy of the attention layer this one replaces (same c_attn / c_proj weights)."""
        self.teacher = [attn.requires_grad_(False)]

    def mix(self, x, cache=None, layer=0):
        """The recurrence's output for x (B, T, C), before c_proj (cache: generation, see the class)."""
        B, T, C = x.size()
        H, hs = self.n_head, self.head_size
        proj = self.c_attn(x).transpose(1, 2)                                                 # (B, 3C, T)
        state = None
        if cache is None:
            qkv = self.conv(proj)[..., :T].transpose(1, 2)
        else:
            K = self.conv.kernel_size[0]
            if layer not in cache.lin:
                cache.lin[layer] = dict(conv=proj.new_zeros(B, 3 * C, K - 1),
                                        S=torch.zeros(B, H, hs, hs, device=x.device, dtype=torch.float32))
            st = cache.lin[layer]
            if cache.chunk_start is not None:
                keep = 1.0
            elif T == 1:
                keep = (cache.pos > 0).float()                                                # (1,)
            else:
                keep = 0.0
            conv_in = torch.cat([st['conv'] * keep, proj.to(st['conv'].dtype)], -1)
            qkv = F.conv1d(conv_in, self.conv.weight.to(conv_in.dtype), groups=3 * C).transpose(1, 2)
            st['conv'].copy_(conv_in[..., -(K - 1):])
            state = st['S'] * keep
        q, k, v = qkv.split(C, dim=2)
        q, k = (F.normalize(F.silu(t).view(B, T, H, hs), dim=-1).to(v.dtype) for t in (q, k))
        v = v.view(B, T, H, hs)
        a, b = self.ab_proj(x).float().chunk(2, dim=-1)                                    # (B, T, H) each
        g = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        beta = torch.sigmoid(b)
        if cache is not None and T == 1:
            o, S = recurrent_gated_delta_rule(*(t.transpose(1, 2) for t in (q * hs ** -0.5, k, v, g, beta)),
                                              state=state)
            o = o.transpose(1, 2)
        elif KERNEL == 'fla' and x.is_cuda:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gated_delta_rule
            o, S = fla_gated_delta_rule(q, k, v, g, beta.to(v.dtype), scale=hs ** -0.5, initial_state=state,
                                        output_final_state=cache is not None)                 # (B, T, H, hs)
        else:
            o, S = chunk_gated_delta_rule(*(t.transpose(1, 2) for t in (q * hs ** -0.5, k, v, g, beta)),
                                          chunk=self.chunk, state=state)
            o = o.transpose(1, 2)
        if cache is not None:
            st['S'].copy_(S)
        o = self.o_norm(o.to(x.dtype)).reshape(B, T, C)
        return o * torch.sigmoid(self.out_gate(x))

    def forward(self, x, cache=None, layer=0, rope=None):
        if cache is not None:
            return self.c_proj(self.mix(x, cache, layer))
        y = self.resid_dropout(self.c_proj(self.mix(x)))
        if self.teacher is None or not self.training:
            self.transfer_loss = None
            return y
        with torch.no_grad():
            y_t = self.teacher[0](x, None, layer, rope)
        self.transfer_loss = (y.float() - y_t.float()).pow(2).mean() / y_t.float().pow(2).mean().clamp(min=1e-8)
        return y_t
