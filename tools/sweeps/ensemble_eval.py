"""
Ensembles of the repeated-data members (scaling plan 2026-11-20, items 3-4, E-E3): K-member mixtures of the 42M
members trained on the replay quarter (checkpoints/members/control_sN) and, as the reference, of the fresh-data
members (fresh_sN), on the held-out windows the runs validate on (1,200 fixed windows of gigamidi_heldout) and on
1,200 unaugmented windows of the quarter they trained on (--train-windows: the memorization gap).

Modes, per K-subset of the members (every subset of each size; L_K = their mean):
    prob    log of the mean of the members' probabilities of the target, per head (the cascade's heads see the true
            earlier attributes, so each head is a mixture of its own)
    logit   softmax of the mean of the members' logits, per head
    joint   log of the mean of the members' probabilities of the whole note (the product over the heads)
CE as train.py logs it: per batch the mean over the valid targets of each head, averaged over the batches, summed
over pitch / velocity / duration / delta_time (prog = the instrument head, apart); joint: the mean over the notes.
Fit (prob and logit): L_K = L_inf + (L_1 - L_inf) / K. With --ref: P = control L_1 - fresh L_1 and
phi = [(L_1 - L_inf)_control - (L_1 - L_inf)_fresh] / P.

Output (--out, default results/aug_sweep/ensemble.json): the top level holds the members' summary that
tools/sweeps/plan_gates.py items34 reads (held-out windows, conditions on, prob mode): L1 .. L<K> (L4 for four
members), mean_member (= L1), g4 (mean_member - L<K>), L_inf, heldout (= L1), train_windows (the members' mean CE on
the quarter windows), gap (heldout - train_windows), plus P and phi with --ref; 'arms' has the same summary and every
mode / condition setting for each member set ('eval'). --save-lp: each member's per-target log-probabilities (fp16, NaN where
masked; 5 heads x windows x 2,048) under <out stem>_lp/.

    .venv/bin/python tools/sweeps/ensemble_eval.py --members control_s3,control_s4,control_s5,control_s6 \\
        --ref fresh_s3,fresh_s4,fresh_s5,fresh_s6 --modes prob,logit,joint --nocond --train-windows 1200 --save-lp
Members are run dirs (model_bf16.pt); a bare name means checkpoints/members/<name>. --micro 16 = the 5090 runs'
evaluation batches (their logged CE is reproduced up to fp8 / compile rounding).
"""

import argparse
import itertools
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from musicar.data.loader import NoteWindowDataset, collate_windows, val_loader, window_conditions  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.model.gpt import STREAM_ORDER  # noqa: E402

DATA = dict(root_dir=str(ROOT), cache_dir=str(ROOT / 'data/cache'), special_tokens=True, programs=True, min_notes=64)
BLOCK = 2048


def member_dir(name):
    p = Path(name)
    return p if p.is_absolute() or '/' in name else ROOT / 'checkpoints/members' / name


def loader(source, split, micro, windows, file_frac=0.25):
    """The fixed windows train.py validates on (split 'validation'), or as many from the train split's file_frac
    share (no augmentation), thinned the same way."""
    if split == 'validation':
        return val_loader(source, micro, windows // micro, extra_names=('program',), block_size=BLOCK, **DATA)
    ds = NoteWindowDataset(source, split='train', block_size=BLOCK, eval_stride=BLOCK, file_frac=file_frac, **DATA)
    if len(ds) > windows:
        ds = Subset(ds, [int(i * len(ds) / windows) for i in range(windows)])
    return DataLoader(ds, batch_size=micro, shuffle=False,
                      collate_fn=lambda b: collate_windows(b, extra_names=('program',)))


def prepare(X, Y, config, drop, device):
    """train.py's prepare for evaluation: the window conditions of its own notes, or 'none' (drop 1)."""
    streams = tuple(t.to(device) for t in X[0])
    extra = {k: v.to(device) for k, v in X[1].items()}
    if config.cond_inst or config.n_density:
        extra['cond'] = window_conditions(streams[0], streams[3], extra.get('program'), config.n_programs,
                                          config.n_density, drop_p=drop)
    return streams, extra, tuple(t.to(device) for t in Y)


@torch.no_grad()
def member_logits(model, streams, extra, Y, ctx):
    """{head: logits (B, T, V) float32} of one member (the cascade's output under teacher forcing) and its parts."""
    out = {}
    hook = model.cascade.register_forward_hook(lambda m, i, o: out.update(o))
    try:
        with ctx:
            parts, _ = model(*streams, **extra, targets=Y)
    finally:
        hook.remove()
    return {k: v.float() for k, v in out.items()}, {k: float(v) for k, v in parts.items()}


def subsets(k_max):
    return {K: list(itertools.combinations(range(k_max), K)) for K in range(1, k_max + 1)}


def run_set(models, batches, drop, modes, ctx, device, save_lp=False):
    """One pass of every member over the batches -> {mode: {subset: per-head CE}} and the members' target
    log-probabilities (when save_lp)."""
    heads = models[0].head_names
    subs = [s for v in subsets(len(models)).values() for s in v]
    acc = {m: {s: ({'joint': 0.0} if m == 'joint' else {h: 0.0 for h in heads}) for s in subs} for m in modes}
    n_batches, lps, member_parts = 0, [[] for _ in models], [{} for _ in models]
    for X, Y in batches:
        out = [member_logits(m, *prepare(X, Y, m.config, drop, device), ctx) for m in models]
        Yd = dict(zip(heads, (t.to(device) for t in Y)))
        lp = {}  # head -> (K, B, T) target log-probs (0 where masked) and its mask
        for h in heads:
            t = Yd[h]
            valid = t != -1
            lp[h] = (torch.stack([torch.log_softmax(o[0][h], -1).gather(-1, t.clamp(min=0)[..., None])[..., 0]
                                  for o in out]) * valid, valid)
        for i, o in enumerate(out):
            for k, v in o[1].items():
                member_parts[i][k] = member_parts[i].get(k, 0.0) + v
            if save_lp:
                lps[i].append(torch.stack([torch.where(lp[h][1], lp[h][0][i], float('nan')) for h in heads])
                              .half().cpu())
        for s in subs:
            idx = list(s)
            for h in heads:
                vals, valid = lp[h]
                n = valid.sum().clamp(min=1)
                if 'prob' in modes:
                    mix = torch.logsumexp(vals[idx], 0) - np.log(len(idx))
                    acc['prob'][s][h] += float(-(mix * valid).sum() / n)
                if 'logit' in modes:
                    mean_logits = torch.stack([out[i][0][h] for i in idx]).mean(0)
                    g = torch.log_softmax(mean_logits, -1).gather(-1, Yd[h].clamp(min=0)[..., None])[..., 0]
                    acc['logit'][s][h] += float(-(g * valid).sum() / n)
            if 'joint' in modes:
                joint = sum(lp[h][0][idx] for h in heads)  # (|s|, B, T): log P(note) per member
                valid = lp['pitch'][1]
                mix = torch.logsumexp(joint, 0) - np.log(len(idx))
                acc['joint'][s]['joint'] += float(-(mix * valid).sum() / valid.sum().clamp(min=1))
        n_batches += 1
    res = {}
    for m in modes:
        res[m] = {}
        for s in subs:
            v = {h: x / n_batches for h, x in acc[m][s].items()}
            if m != 'joint':
                v['total'] = sum(v[h] for h in STREAM_ORDER)
            res[m][s] = v
    parts = [{k: v / n_batches for k, v in p.items()} for p in member_parts]
    return res, parts, [torch.cat(x, 1).numpy() if x else None for x in lps]


def summarize(res, k_max):
    """{mode: dict(L1.., L_inf, subsets, members)} from run_set's CEs (total, or joint)."""
    out = {}
    for mode, per in res.items():
        key = 'joint' if mode == 'joint' else 'total'
        L = {K: float(np.mean([per[s][key] for s in subsets(k_max)[K]])) for K in range(1, k_max + 1)}
        d = {f'L{K}': v for K, v in L.items()}
        if k_max > 1:
            x = np.array([1 / K for K in L])
            slope, inter = np.polyfit(x, np.array(list(L.values())), 1)
            d['L_inf'] = float(inter)
        d['members'] = [per[(i,)][key] for i in range(k_max)]
        if mode != 'joint':
            d['heads'] = {h: float(np.mean([per[(i,)][h] for i in range(k_max)])) for h in per[(0,)] if h != 'total'}
        d['subsets'] = {'+'.join(map(str, s)): v[key] for s, v in per.items()}
        out[mode] = d
    return out


def evaluate_set(names, a, device, ctx, out_stem):
    models = [load_model(member_dir(n), device=device)[0] for n in names]
    k = len(models)
    sets = [('heldout', a.val, 'validation', a.windows)] + (
        [('train_windows', a.train_store, 'train', a.train_windows)] if a.train_windows else [])
    arm, ev = {'members': names}, {}
    for split, source, kind, windows in sets:
        ev[split] = {}
        for cond, drop in [('cond', 0.0)] + ([('nocond', 1.0)] if a.nocond and split == 'heldout' else []):
            t0 = time.time()
            res, parts, lps = run_set(models, loader(source, kind, a.micro, windows, a.train_frac), drop, a.modes, ctx,
                                      device, a.save_lp and cond == 'cond')
            ev[split][cond] = summarize(res, k)
            ev[split][cond]['model_parts_total'] = [sum(p[h] for h in STREAM_ORDER) for p in parts]
            print(f'{names[0]}..: {split} {cond}: ' + ', '.join(
                f'{m} ' + ' '.join(f'L{K} {ev[split][cond][m][f"L{K}"]:.4f}' for K in range(1, k + 1))
                for m in a.modes) + f' ({time.time() - t0:.0f}s)', flush=True)
            if a.save_lp and cond == 'cond':
                d = Path(f'{out_stem}_lp')
                d.mkdir(parents=True, exist_ok=True)
                for n, x in zip(names, lps):
                    np.save(d / f'{Path(n).name}_{split}.npy', x)
    h = ev['heldout']['cond']['prob']
    arm.update(mean_member=h['L1'], **{f'L{K}': h[f'L{K}'] for K in range(1, k + 1)}, heldout=h['L1'],
               g4=h['L1'] - h[f'L{k}'], L_inf=h.get('L_inf'))
    if 'train_windows' in ev:
        arm['train_windows'] = ev['train_windows']['cond']['prob']['L1']
        arm['gap'] = arm['heldout'] - arm['train_windows']
    arm['eval'] = ev  # {heldout | train_windows: {cond | nocond: {mode: L_K, L_inf, members, heads, subsets}}}
    return arm


def arm_name(names):
    return Path(names[0]).name.rsplit('_s', 1)[0]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--members', required=True, help='comma-separated run dirs (bare names: checkpoints/members/)')
    p.add_argument('--ref', default='', help='the fresh-data members, the same way')
    p.add_argument('--modes', default='prob,logit,joint')
    p.add_argument('--nocond', action='store_true', help='also the held-out windows with the conditions set to none')
    p.add_argument('--train-windows', type=int, default=0, help='this many unaugmented windows of the trained files')
    p.add_argument('--train-store', default='store:data/cache/gigamidi_replay')
    p.add_argument('--train-frac', type=float, default=0.25)
    p.add_argument('--val', default='store:data/cache/gigamidi_heldout')
    p.add_argument('--windows', type=int, default=1200)
    p.add_argument('--micro', type=int, default=16, help='windows per evaluation batch (the runs: 16 on the 5090)')
    p.add_argument('--save-lp', action='store_true')
    p.add_argument('--out', default='results/aug_sweep/ensemble.json')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    a.modes = [m for m in a.modes.split(',') if m]
    assert set(a.modes) <= {'prob', 'logit', 'joint'} and 'prob' in a.modes, a.modes
    assert a.windows % a.micro == 0, (a.windows, a.micro)
    ctx = nullcontext() if a.device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    members = [x for x in a.members.split(',') if x]
    arms = {arm_name(members): evaluate_set(members, a, a.device, ctx, out.with_suffix(''))}
    torch.cuda.empty_cache() if a.device == 'cuda' else None
    if a.ref:
        ref = [x for x in a.ref.split(',') if x]
        ref_name = arm_name(ref) if arm_name(ref) != arm_name(members) else arm_name(ref) + '_ref'
        arms[ref_name] = evaluate_set(ref, a, a.device, ctx, Path(f"{out.with_suffix('')}_{ref_name}"))
    ctl = arms[arm_name(members)]
    top = {k: ctl[k] for k in ctl if not isinstance(ctl[k], (dict, list))}
    if a.ref:
        fr = arms[ref_name]
        P = ctl['L1'] - fr['L1']
        top['P'] = P
        if ctl.get('L_inf') is not None and fr.get('L_inf') is not None and P:
            top['phi'] = ((ctl['L1'] - ctl['L_inf']) - (fr['L1'] - fr['L_inf'])) / P
    summary = dict(top, members=members, ref=a.ref.split(',') if a.ref else [], micro=a.micro, windows=a.windows,
                   arms=arms)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(out)
    print(json.dumps(top, indent=1))


if __name__ == '__main__':
    main()
