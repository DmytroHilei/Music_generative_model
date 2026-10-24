"""
MHA -> GQA conversion, layer by layer (context/claude/optimizations.md P4; the 107M pilot with mean-pooled K / V
started at held-out CE 10.15 vs 2.77 and was still 0.20 behind after 1,500 it).

Stage 1 of the conversion runs in train.py (attn_transfer=True on a finetune that lowers n_kv_head): every GQA
attention layer keeps a frozen copy of the attention it replaces (the checkpoint's own c_attn / c_proj); in training
the layer returns the teacher's output (the next layers see the original stream) and learns the teacher's output with
the relative MSE (LoLCATs-style attention transfer, arXiv 2410.10254), so each layer is fitted to its own original
layer only. gqa_init='pca' first replaces load_expanded's mean-pool with a data-aware merge from the second moments of
each layer's attention input on calibration batches (collected through the teacher stream):

- values: per group, PCA of its heads' values weighted by what c_proj does with them (V_h R_h^T with Wo_h = Q_h R_h,
  so the error is measured after c_proj), top head_size directions = the shared value head; each query head's c_proj
  columns then take the least-squares map from the shared value back to its own (exact when the group's values span
  head_size dimensions).
- keys: RoPE turns each rotation pair (dims i, i + head_size/2) into a complex number, so a shared key can only mix
  the group's keys per pair with complex weights. Per pair, the top eigenvector of the query-energy-weighted 3 x 3
  (group size) key covariance gives the shared key; each query head's pair is scaled by the conjugate of its
  least-squares complex factor, which commutes with the rotation (RoPE stays exact). Mean pooling is the special
  case weights 1 / group, factor 1.
"""

import dataclasses

import torch


def _complex_rows(w, f, half):
    """Rows (f, f + half) of a (rows, in) block: the real / imaginary part of rotation pair f."""
    return w[f], w[f + half]


@torch.no_grad()
def pca_merge(c_attn, c_proj, cov, n_head, n_kv, eps=1e-8):
    """One attention layer: teacher c_attn (n_embd + 2 n_head hs, n_embd [+1]) and c_proj (n_embd, n_embd) weights
    and the second moment cov (in, in) of its inputs (augmented with a constant 1 when c_attn carries its bias as the
    last column) -> the GQA c_attn (n_embd + 2 n_kv hs, in) and c_proj. float64 inside."""
    dt = c_attn.dtype
    W, Wo, S = c_attn.double(), c_proj.double().clone(), cov.double()
    C = Wo.shape[0]
    hs, r = C // n_head, n_head // n_kv
    half = hs // 2
    Wq = W[:C].clone().view(n_head, hs, -1)
    Wk = W[C:C + n_head * hs].view(n_head, hs, -1)
    Wv = W[C + n_head * hs:].view(n_head, hs, -1)
    Wk_g = torch.zeros(n_kv, hs, W.shape[1], dtype=W.dtype, device=W.device)
    Wv_g = torch.zeros_like(Wk_g)
    for g in range(n_kv):
        heads = range(g * r, (g + 1) * r)
        # values: PCA after c_proj, then each head's least-squares map back
        Rs = [torch.linalg.qr(Wo[:, h * hs:(h + 1) * hs])[1] for h in heads]
        Wz = torch.cat([R @ Wv[h] for R, h in zip(Rs, heads)])          # (r hs, in)
        evals, evecs = torch.linalg.eigh(Wz @ S @ Wz.T)
        U = evecs[:, -hs:]                                              # top hs directions
        Wv_g[g] = U.T @ Wz
        G = Wv_g[g] @ S @ Wv_g[g].T
        G = G + eps * G.diagonal().mean().clamp(min=eps) * torch.eye(hs, dtype=G.dtype, device=G.device)
        for h in heads:
            P = torch.linalg.solve(G, Wv_g[g] @ S @ Wv[h].T)          # V_h ~= v_g P
            Wo[:, h * hs:(h + 1) * hs] = Wo[:, h * hs:(h + 1) * hs] @ P.T
        # keys: per rotation pair, complex rank 1 over the group, weighted by the query energy
        for f in range(half):
            a = torch.stack([Wk[h, f] for h in heads])                    # (r, in) real part
            b = torch.stack([Wk[h, f + half] for h in heads])             # imaginary part
            qa = torch.stack([Wq[h, f] for h in heads])
            qb = torch.stack([Wq[h, f + half] for h in heads])
            w = ((qa @ S) * qa).sum(1) + ((qb @ S) * qb).sum(1)           # E|q_h|^2
            K = (a @ S @ a.T + b @ S @ b.T) + 1j * (b @ S @ a.T - a @ S @ b.T)  # E[k_h conj(k_h')]
            if w.sum() <= eps or K.real.diagonal().sum() <= eps:
                alpha = torch.full((r,), 1.0 / r, dtype=torch.complex128, device=W.device)
            else:
                sw = w.clamp(min=0).sqrt()
                Kw = sw[:, None] * K * sw[None, :]
                Kw = (Kw + Kw.conj().T) / 2
                u = torch.linalg.eigh(Kw)[1][:, -1]
                alpha = u.conj() * sw
            ag = (alpha.real[:, None] * a - alpha.imag[:, None] * b).sum(0)
            bg = (alpha.imag[:, None] * a + alpha.real[:, None] * b).sum(0)
            Wk_g[g, f], Wk_g[g, f + half] = ag, bg
            kg2 = ag @ S @ ag + bg @ S @ bg                               # E|k_g|^2
            for i, h in enumerate(heads):
                # c = E[k_h conj(k_g)] / E|k_g|^2; the query pair takes conj(c)
                num = (a[i] @ S @ ag + b[i] @ S @ bg) + 1j * (b[i] @ S @ ag - a[i] @ S @ bg)
                c = num / kg2 if kg2 > eps else torch.tensor(0j, dtype=torch.complex128)
                gr, gi = c.real, -c.imag
                Wq[h, f], Wq[h, f + half] = gr * qa[i] - gi * qb[i], gi * qa[i] + gr * qb[i]
    out = torch.cat([Wq.reshape(C, -1), Wk_g.reshape(n_kv * hs, -1), Wv_g.reshape(n_kv * hs, -1)])
    return out.to(dt), Wo.to(dt)


def teacher_attention(config, c_attn, c_proj, device):
    """A frozen full-attention copy (n_kv_head = n_head) of one layer from the checkpoint's own weights."""
    from musicar.model.gpt import CausalSelfAttention
    assert not config.qk_norm, 'GQA transfer from a QK-norm model: copy q_norm / k_norm too'
    attn = CausalSelfAttention(dataclasses.replace(config, n_kv_head=0)).to(device)
    attn.c_attn.load_state_dict(c_attn)
    attn.c_proj.load_state_dict(c_proj)
    return attn.eval().requires_grad_(False)


def teacher_weights(state_dict):
    """The attention weights of a checkpoint state dict, before load_expanded pools them: {layer: (c_attn, c_proj)}
    as state dicts of the two Linears."""
    out = {}
    for k, v in state_dict.items():
        parts = k.split('.')
        if len(parts) == 6 and parts[:2] == ['transformer', 'h'] and parts[3] == 'attn' \
                and parts[4] in ('c_attn', 'c_proj'):
            i = int(parts[2])
            out.setdefault(i, ({}, {}))[0 if parts[4] == 'c_attn' else 1][parts[5]] = v.clone()
    return out


class InputMoments:
    """Forward pre-hooks that sum x^T x of each attention layer's input (augmented with 1 for a bias), float64."""

    def __init__(self, layers, bias):
        self.sums, self.n, self.bias, self.handles = {}, 0, bias, []
        for i, attn in layers.items():
            self.handles.append(attn.register_forward_pre_hook(self._hook(i)))

    def _hook(self, i):
        def hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            if self.bias:
                x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
            m = (x.T @ x).double()
            self.sums[i] = self.sums[i] + m if i in self.sums else m
            if i == min(self.sums):
                self.n += x.shape[0]
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def moments(self):
        return {i: s / max(self.n, 1) for i, s in self.sums.items()}


def gqa_layers(model):
    """{block index: attention} of the full-softmax attention layers with fewer key / value heads than query heads."""
    from musicar.model.gpt import CausalSelfAttention
    return {i: b.attn for i, b in enumerate(model.transformer.h)
            if isinstance(b.attn, CausalSelfAttention) and b.attn.n_kv_head < b.attn.n_head}


@torch.no_grad()
def apply_pca_init(model, cov):
    """Replace each GQA layer's c_attn / c_proj by pca_merge of its teacher's weights (attach_gqa_teachers first)."""
    report = []
    for i, attn in gqa_layers(model).items():
        t = attn.teacher[0]
        w = t.c_attn.weight
        if t.c_attn.bias is not None:
            w = torch.cat([w, t.c_attn.bias[:, None]], 1)
        # on the CPU: many small float64 products per rotation pair
        new_attn, new_proj = pca_merge(w.cpu(), t.c_proj.weight.cpu(), cov[i].cpu(), attn.n_head, attn.n_kv_head)
        if t.c_attn.bias is not None:
            attn.c_attn.weight.copy_(new_attn[:, :-1])
            attn.c_attn.bias.copy_(new_attn[:, -1])
        else:
            attn.c_attn.weight.copy_(new_attn)
        attn.c_proj.weight.copy_(new_proj)
        report.append(i)
    return report
