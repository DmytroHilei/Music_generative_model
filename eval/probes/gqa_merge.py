"""
How much a GQA merge (musicar/model/gqa.py) breaks a checkpoint before any training, on the CPU: per layer the
relative output error of the merged attention vs the original (mean-pool and pca) and the spread of the pca layer's
attention logits vs the original's (< 1 = flatter attention), then the held-out CE with every layer merged: full
attention, mean-pool, pca, and pca with each head's queries rescaled by the measured logit-spread ratio (how much of
the pca loss is the temperature). Run it before a run that converts (F1: gqa_init pca on a QK-norm model).

    python eval/probes/gqa_merge.py checkpoints/long_450m_final --kv 6
    python eval/probes/gqa_merge.py checkpoints/v2_A0 --kv 4

Reference (2026-10-25, 6 calibration + 6 test windows x 512 notes; context/claude/caption_450m_2026-10-25.md):
long_450m_final (QK-norm) -> 6 KV heads: CE 2.010 full, 13.898 mean-pool, 6.624 pca, 4.628 pca rescaled; pca logit
spread 0.34-0.72x per layer (single heads down to 0.02x), output error 0.05-0.90. The F1 rehearsal's step 0 (train.py
calibration on 16 batches) was 7.126 on heldout_labeled.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musicar.data.loader import NoteWindowDataset, collate_windows, window_conditions  # noqa: E402
from musicar.model.checkpoint import load_model, pool_kv_heads  # noqa: E402
from musicar.model.gpt import CausalSelfAttention, apply_rope  # noqa: E402
from musicar.model.gqa import pca_merge  # noqa: E402

HEADS = ('pitch', 'velocity', 'duration', 'delta_time')


def merged_layer(cfg, teacher, n_kv, c_attn):
    """A GQA attention layer with this c_attn weight (and the teacher's c_proj / QK-norm)."""
    s = CausalSelfAttention(dataclasses.replace(cfg, n_kv_head=n_kv)).eval()
    s.c_attn.weight.data.copy_(c_attn)
    s.c_proj.load_state_dict(teacher.c_proj.state_dict())
    if cfg.qk_norm:
        s.q_norm.load_state_dict(teacher.q_norm.state_dict())
        s.k_norm.load_state_dict(teacher.k_norm.state_dict())
    return s


def qk(attn, x):
    B, T, C = x.shape
    hs = C // attn.n_head
    q, k, _ = attn.c_attn(x).split([C, attn.n_kv_head * hs, attn.n_kv_head * hs], 2)
    q, k = q.view(B, T, attn.n_head, hs).transpose(1, 2), k.view(B, T, attn.n_kv_head, hs).transpose(1, 2)
    if attn.q_norm is not None:
        q, k = attn.q_norm(q), attn.k_norm(k)
    return q, k.repeat_interleave(attn.n_head // attn.n_kv_head, 1)


def logit_spread(attn, x):
    """(n_head,) std of q.k / sqrt(hs) over all position pairs (RoPE left out: the scale only)."""
    q, k = qk(attn, x)
    return ((q @ k.transpose(-1, -2)) / q.size(-1) ** 0.5).std(dim=(-1, -2)).mean(0)


class RescaledQ(nn.Module):
    """A merged layer whose normalized queries are multiplied per head (full-sequence causal forward only)."""

    def __init__(self, attn, scale):
        super().__init__()
        self.attn, self.scale = attn, scale

    def forward(self, x, cache=None, layer=0, rope=None):
        a = self.attn
        B, T, C = x.shape
        hs = C // a.n_head
        q, k, v = a.c_attn(x).split([C, a.n_kv_head * hs, a.n_kv_head * hs], 2)
        q = q.view(B, T, a.n_head, hs).transpose(1, 2)
        k, v = (t.view(B, T, a.n_kv_head, hs).transpose(1, 2) for t in (k, v))
        if a.q_norm is not None:
            q, k = a.q_norm(q), a.k_norm(k)
        q = q * self.scale.view(1, -1, 1, 1)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return a.c_proj(y.transpose(1, 2).reshape(B, T, C))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--kv', type=int, required=True, help='key / value heads after the merge')
    p.add_argument('--store', default='store:data/cache/gigamidi_heldout')
    p.add_argument('--calib', type=int, default=6, help='calibration windows (the rest of --windows are test)')
    p.add_argument('--windows', type=int, default=12)
    p.add_argument('--notes', type=int, default=512)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.checkpoint)
    model.pad_multiple = 0
    cfg = model.config
    ds = NoteWindowDataset(args.store, root_dir='.', split='validation', block_size=args.notes,
                           eval_stride=args.notes, programs=bool(cfg.n_programs),
                           special_tokens=cfg.pitch_size > 128, pad_short=True)
    idx = torch.linspace(0, len(ds) - 1, args.windows).long().tolist()
    names = ('program',) if cfg.n_programs else ()
    (X, extra), Y = collate_windows([ds[i] for i in idx], extra_names=names)
    cond = None
    if cfg.cond_inst or cfg.n_density:
        own = window_conditions(X[0], X[3], extra.get('program'), cfg.n_programs, cfg.n_density)
        cond = (torch.zeros_like(own[0]), torch.zeros_like(own[1]))    # the trained 'none'
    cal, test = slice(0, args.calib), slice(args.calib, None)

    inputs = {}
    hooks = [b.attn.register_forward_pre_hook(
        lambda m, a, i=i: inputs.setdefault(i, []).append((a[0].detach(), a[3]))) for i, b in enumerate(model.transformer.h)]
    with torch.no_grad():
        for j in range(len(idx)):
            sl = slice(j, j + 1)
            model.encode(*(x[sl] for x in X), **{k: v[sl] for k, v in extra.items()},
                         cond=None if cond is None else tuple(c[sl] for c in cond))
    for h in hooks:
        h.remove()

    hs = cfg.n_embd // cfg.n_head
    layers = {}
    print(f'{args.checkpoint}: {cfg.n_head} -> {args.kv} key / value heads, qk_norm {cfg.qk_norm}')
    print('layer | output error mean-pool / pca | pca logit spread vs original: mean (min, max)')
    for i, b in enumerate(model.transformer.h):
        t = b.attn
        xs = inputs[i]
        flat = torch.cat([x.reshape(-1, x.size(-1)) for x, _ in xs[cal]]).double()
        S = flat.T @ flat / flat.size(0)
        gains = (t.q_norm.weight, t.k_norm.weight) if cfg.qk_norm else (None, None)
        w_pca, proj_pca = pca_merge(t.c_attn.weight, t.c_proj.weight, S, cfg.n_head, args.kv, q_gain=gains[0],
                                    k_gain=gains[1])
        pca = merged_layer(cfg, t, args.kv, w_pca)
        pca.c_proj.weight.data.copy_(proj_pca)
        mean = merged_layer(cfg, t, args.kv, pool_kv_heads(t.c_attn.weight.data.clone(), cfg.n_embd, hs, args.kv))
        err, ratio = {'mean': 0.0, 'pca': 0.0}, 0
        with torch.no_grad():
            for x, rope in xs[test]:
                y = t(x, None, i, rope)
                for name, s in (('mean', mean), ('pca', pca)):
                    err[name] += ((s(x, None, i, rope) - y).pow(2).mean() / y.pow(2).mean()).item() / len(xs[test])
                ratio = ratio + logit_spread(pca, x) / logit_spread(t, x) / len(xs[test])
        layers[i] = dict(mean=mean, pca=pca, ratio=ratio)
        print(f"{i:5d} | {err['mean']:.3f} / {err['pca']:.3f} | {ratio.mean():.2f} ({ratio.min():.2f}, "
              f"{ratio.max():.2f})", flush=True)

    def ce():
        with torch.no_grad():
            parts, _ = model(*(x[test] for x in X), **{k: v[test] for k, v in extra.items()},
                             cond=None if cond is None else tuple(c[test] for c in cond),
                             targets=tuple(y[test] for y in Y))
        return sum(parts[h].item() for h in HEADS)
    original = [b.attn for b in model.transformer.h]
    print(f'\nheld-out CE ({args.windows - args.calib} windows): full attention {ce():.3f}', end='')
    for name in ('mean', 'pca', 'pca rescaled'):
        for i, b in enumerate(model.transformer.h):
            L = layers[i]
            b.attn = L['mean'] if name == 'mean' else L['pca'] if name == 'pca' else RescaledQ(L['pca'], 1 / L['ratio'])
        print(f' | {name} {ce():.3f}', end='', flush=True)
    print()
    for b, a in zip(model.transformer.h, original):
        b.attn = a


if __name__ == '__main__':
    main()
