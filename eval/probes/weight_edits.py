"""
Training-free weight edits, scored by teacher-forced CE (sum of the note heads + program) on held-out windows:
fit / pick on the even windows ('fit'), report on the odd ones ('test'), same windows for every edit (paired).

  temp     per-head logit temperature folded into the head's ln_out weight (best T per head on fit)
  laser    LASER (Sharma et al. 2023): one matrix replaced by its rank-k SVD approximation, k = rho * full rank
  skip     one block removed;  dup: one block applied twice (x -> B_l(B_l(x)))
  ctx      context past the training length: CE of notes block..2*block with the full 2*block context vs the
           sliding training-length window (each note with >= block/2 notes of context); plain RoPE, attention
           temperature (q_norm weight * s) and NTK RoPE base scaling

    python eval/probes/weight_edits.py --checkpoint checkpoints/pilot_mi_grown --modes temp,laser,skip,dup,ctx
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.data.loader import NoteWindowDataset, collate_windows  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', default='checkpoints/pilot_mi_grown')
    p.add_argument('--out', default='results/weight_edits/pilot_mi_grown')
    p.add_argument('--source', default='store:data/cache/gigamidi_heldout')
    p.add_argument('--modes', default='temp,laser,skip,dup,ctx')
    p.add_argument('--windows', type=int, default=32, help='fit + test windows (half each)')
    p.add_argument('--ctx-windows', type=int, default=24)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--rhos', default='0.5,0.2,0.05')
    p.add_argument('--laser-layers', default='', help='comma list, default every layer')
    p.add_argument('--laser-mats', default='attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_gate,mlp.c_proj')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def load_batches(model, src, block, n, batch, device, pad):
    special = model.config.pitch_size >= 130
    ds = NoteWindowDataset(src, root_dir=str(ROOT), split='validation', block_size=block, eval_stride=block,
                           special_tokens=special, programs=bool(model.config.n_programs), pad_short=pad and special)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    extra = ['program'] if model.config.n_programs else []
    out = []
    for b in range(0, len(idx), batch):
        (streams, ex), Y = collate_windows([ds[int(i)] for i in idx[b:b + batch]], extra)
        out.append(([s.to(device) for s in streams], {k: v.to(device) for k, v in ex.items()}, [y.to(device) for y in Y]))
    return out


@torch.no_grad()
def per_note(model, streams, ex, Y, temps=None):
    """(B, T) per-note CE summed over the heads, and per-head (B, T) CE"""
    tgt = dict(zip(model.head_names, Y))
    logits = model.cascade(model.encode(*streams, **ex), tgt)
    heads = {}
    for n, lg in logits.items():
        lg = lg.float() / (temps[n] if temps else 1.0)
        heads[n] = torch.where(tgt[n] != -1, F.cross_entropy(lg.transpose(1, 2), tgt[n].clamp(min=0), reduction='none'), 0.0)
    return sum(heads.values()), heads, tgt['pitch'] != -1


def score(model, batches, temps=None):
    """mean summed CE per note on fit (even windows) and test (odd windows), per-head means on test"""
    s, c = np.zeros(2), np.zeros(2)
    hs = {}
    w = 0
    for streams, ex, Y in batches:
        tot, heads, valid = per_note(model, streams, ex, Y, temps)
        for r in range(tot.size(0)):
            half = w % 2
            s[half] += tot[r][valid[r]].sum().item()
            c[half] += valid[r].sum().item()
            if half:
                for n, v in heads.items():
                    hs[n] = hs.get(n, 0.0) + v[r][valid[r]].sum().item()
            w += 1
    return dict(fit=s[0] / c[0], test=s[1] / c[1], heads={n: v / c[1] for n, v in hs.items()})


def mode_temp(model, batches):
    temps = {n: 1.0 for n in model.head_names}
    base = score(model, batches)
    for n in model.head_names:
        best = (base['fit'], 1.0)
        for T in (0.85, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.15):
            r = score(model, batches, {**temps, n: T})
            if r['fit'] < best[0]:
                best = (r['fit'], T)
        temps[n] = best[1]
    edited = score(model, batches, temps)
    return dict(temps=temps, base=base, edited=edited, gain_test=base['test'] - edited['test'])


def lowrank_(w, rho):
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    k = max(1, int(rho * s.numel()))
    w.copy_(((u[:, :k] * s[:k]) @ vh[:k]).to(w.dtype))


def mode_laser(model, batches, args):
    base = score(model, batches)
    L = len(model.transformer.h)
    layers = [int(x) for x in args.laser_layers.split(',')] if args.laser_layers else list(range(L))
    rows = []
    for l in layers:
        blk = model.transformer.h[l]
        for mat in args.laser_mats.split(','):
            mod = blk.get_submodule(mat)
            orig = mod.weight.data.clone()
            for rho in (float(r) for r in args.rhos.split(',')):
                lowrank_(mod.weight.data, rho)
                r = score(model, batches)
                rows.append(dict(layer=l, mat=mat, rho=rho, d_fit=r['fit'] - base['fit'], d_test=r['test'] - base['test']))
                mod.weight.data.copy_(orig)
            print(f'laser layer {l} {mat}: ' + ', '.join(f"rho {x['rho']} {x['d_fit']:+.4f}" for x in rows[-3:]),
                  flush=True)
    best = sorted(rows, key=lambda x: x['d_fit'])[:10]
    return dict(base=base, best_by_fit=best, n_better_fit=sum(x['d_fit'] < 0 for x in rows), n=len(rows), rows=rows)


def mode_skip_dup(model, batches, dup):
    base = score(model, batches)
    rows = []
    for l, blk in enumerate(model.transformer.h):
        orig = blk.forward
        blk.forward = (lambda x, *a, o=orig, **k: o(o(x, *a, **k), *a, **k)) if dup else (lambda x, *a, **k: x)
        r = score(model, batches)
        blk.forward = orig
        rows.append(dict(layer=l, d_fit=r['fit'] - base['fit'], d_test=r['test'] - base['test']))
        print(f"{'dup' if dup else 'skip'} layer {l}: {rows[-1]['d_fit']:+.4f}", flush=True)
    return dict(base=base, rows=rows)


@torch.no_grad()
def mode_ctx(model, args):
    block = model.config.block_size
    long = load_batches(model, args.source, 2 * block, args.ctx_windows, 1, args.device, pad=False)
    model.config.block_size = 2 * block
    half = block // 2

    def ce_range(streams, ex, Y, a, b, lo, hi):
        """summed CE of notes lo..hi-1 (window coordinates of the long window) using the slice a..b as context"""
        s = [x[:, a:b] for x in streams]
        e = {k: v[:, a:b] for k, v in ex.items()}
        y = [x[:, a:b] for x in Y]
        tot, _, valid = per_note(model, s, e, y)
        m = valid[0, lo - a:hi - a]
        return tot[0, lo - a:hi - a][m].sum().item(), m.sum().item()

    def run(label):
        sums = {'long': [0.0, 0], 'slide': [0.0, 0]}
        for streams, ex, Y in long:
            for lo in range(block, 2 * block, half):  # slide: window [lo + half - block, lo + half)
                s, c = ce_range(streams, ex, Y, lo + half - block, lo + half, lo, lo + half)
                sums['slide'][0] += s; sums['slide'][1] += c
            s, c = ce_range(streams, ex, Y, 0, 2 * block, block, 2 * block)
            sums['long'][0] += s; sums['long'][1] += c
        r = {k: v[0] / max(v[1], 1) for k, v in sums.items()}
        r['gain'] = r['slide'] - r['long']
        print(f"ctx {label}: slide {r['slide']:.4f} long {r['long']:.4f} gain {r['gain']:+.4f}", flush=True)
        return r

    out = {'windows': len(long)}
    out['plain'] = run('plain')
    qn = [blk.attn.q_norm for blk in model.transformer.h if getattr(blk.attn, 'q_norm', None) is not None]
    for s in (1.05, 1.1, 1.2):
        for q in qn:
            q.weight.data *= s
        out[f'attn_temp_{s}'] = run(f'attn temp {s}')
        for q in qn:
            q.weight.data /= s
    base0 = model.config.rope_base
    hs = model.config.n_embd // model.config.n_head
    for f in (2.0, 4.0):
        model.config.rope_base = base0 * f ** (hs / (hs - 2))
        out[f'ntk_{f}'] = run(f'ntk x{f}')
    model.config.rope_base = base0
    model.config.block_size = block
    return out


def main():
    args = parse_args()
    t0 = time.time()
    model, it = load_model(ROOT / args.checkpoint, args.device, torch.bfloat16)
    print(f'{args.checkpoint}: {model.config.n_layer} x {model.config.n_embd}, block {model.config.block_size}, '
          f'qk_norm {model.config.qk_norm}, mlp {model.config.mlp}')
    if model.config.mlp != 'swiglu':
        args.laser_mats = args.laser_mats.replace(',mlp.c_gate', '')
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = args.modes.split(',')
    batches = load_batches(model, args.source, model.config.block_size, args.windows, args.batch, args.device, True) \
        if set(modes) - {'ctx'} else []
    res = {'checkpoint': args.checkpoint, 'iter': it}
    for m in modes:
        if m == 'temp':
            res[m] = mode_temp(model, batches)
            print('temp', res[m]['temps'], f"gain test {res[m]['gain_test']:+.4f}", flush=True)
        elif m == 'laser':
            res[m] = mode_laser(model, batches, args)
        elif m in ('skip', 'dup'):
            res[m] = mode_skip_dup(model, batches, m == 'dup')
        elif m == 'ctx':
            res[m] = mode_ctx(model, args)
        (out_dir / 'edits.json').write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(f'done in {time.time() - t0:.0f} s -> {out_dir}/edits.json')


if __name__ == '__main__':
    main()
