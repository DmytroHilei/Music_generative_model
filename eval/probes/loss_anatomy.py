"""
Where the loss goes: per-note CE of every head on held-out windows, broken down by position in the window, note type
(chord note dt=0 / onset, drums / pitched, instrument family), target value, plus
  - near misses: the share of the dt / duration CE that is a +-1 bin (20 ms) timing miss,
  - temperature: the CE each head would have with its logits divided by T (fit on even windows, read on odd ones):
    a free gain that folds into the head's ln_out weight,
  - logit lens: the final heads (and ln_f) read from the residual stream after layer l,
  - layer skip: CE with one block removed (the residual passes through).

    python eval/probes/loss_anatomy.py --checkpoint checkpoints/long_450m_final --out results/loss_anatomy/musicar_L
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.data.loader import NoteWindowDataset, collate_windows  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402

TEMPS = [0.8, 0.85, 0.9, 0.93, 0.96, 1.0, 1.04, 1.08, 1.12, 1.2, 1.3]
FAMILIES = ['piano', 'chrom', 'organ', 'guitar', 'bass', 'strings', 'ensemble', 'brass', 'reed', 'pipe', 'synlead',
            'synpad', 'synfx', 'ethnic', 'perc', 'sfx', 'drums']


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', default='checkpoints/long_450m_final')
    p.add_argument('--out', default='results/loss_anatomy/musicar_L')
    p.add_argument('--sources', default='gigamidi=store:data/cache/gigamidi_heldout;aria=store:data/cache/aria')
    p.add_argument('--windows', type=int, default=96)
    p.add_argument('--block', type=int, default=0, help='0 = the model block size')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lens-every', type=int, default=2)
    p.add_argument('--skip-windows', type=int, default=12, help='windows for the layer-skip ablation (0 = off)')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def batches(model, src, args, n_windows):
    ds = NoteWindowDataset(src, root_dir=str(ROOT), split='validation', block_size=args.block,
                           eval_stride=args.block, special_tokens=model.config.pitch_size >= 130,
                           programs=bool(model.config.n_programs), pad_short=model.config.pitch_size >= 130)
    idx = np.linspace(0, len(ds) - 1, min(n_windows, len(ds))).astype(int)
    extra = ['program'] if model.config.n_programs else []
    for b in range(0, len(idx), args.batch):
        (streams, ex), Y = collate_windows([ds[int(i)] for i in idx[b:b + args.batch]], extra)
        yield [s.to(args.device) for s in streams], {k: v.to(args.device) for k, v in ex.items()}, \
            [y.to(args.device) for y in Y], b // args.batch


def head_ce(model, h, tgt):
    """per-head per-note CE (B, T) and logits, teacher forced"""
    logits = model.cascade(h, tgt)
    return {n: F.cross_entropy(lg.float().transpose(1, 2), tgt[n].clamp(min=0), reduction='none')
            for n, lg in logits.items()}, logits


@torch.no_grad()
def run_source(model, name, src, args):
    names = model.head_names
    L = len(model.transformer.h)
    lens_layers = sorted(set(list(range(0, L + 1, args.lens_every)) + [L]))
    resid = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, l=l: resid.__setitem__(l + 1, o))
             for l, blk in enumerate(model.transformer.h)]
    rows = {k: [] for k in ['pos', 'chord', 'drum', 'family', 'win'] + [f'ce_{n}' for n in names]
            + [f't_{n}' for n in names] + ['near_dt', 'near_dur', 'ent_pitch']}
    temp = {n: np.zeros((2, len(TEMPS))) for n in names}
    temp_n = {n: np.zeros(2) for n in names}
    lens = {l: {n: 0.0 for n in names} for l in lens_layers}
    lens_n = {n: 0 for n in names}
    for streams, ex, Y, bi in batches(model, src, args, args.windows):
        tgt = dict(zip(names, Y))
        h = model.encode(*streams, **ex)
        ce, logits = head_ce(model, h, tgt)
        B, T = streams[0].shape
        valid = {n: tgt[n] != -1 for n in names}
        v = valid['pitch']
        pos = torch.arange(T, device=h.device).expand(B, T)
        prog = tgt['program'] if 'program' in tgt else torch.zeros_like(tgt['pitch'])
        win = (torch.arange(B, device=h.device) + bi * args.batch)[:, None].expand(B, T)
        rows['pos'].append(pos[v]); rows['win'].append(win[v])
        rows['chord'].append((tgt['delta_time'] == 0)[v])
        rows['drum'].append((prog == 128)[v])
        rows['family'].append(torch.where(prog == 128, 16, prog.clamp(0, 127) // 8)[v])
        for n in names:
            # masked targets (drum durations) -> nan so the means skip them
            rows[f'ce_{n}'].append(torch.where(valid[n], ce[n], torch.nan)[v])
            rows[f't_{n}'].append(tgt[n][v])
        for n, key in (('delta_time', 'near_dt'), ('duration', 'near_dur')):
            lp = logits[n].float().log_softmax(-1)
            t = tgt[n].clamp(min=0)
            p = sum(lp.gather(-1, (t + d).clamp(0, lp.size(-1) - 1)[..., None]).exp() for d in (-1, 0, 1))[..., 0]
            p = p - (t == 0).float() * lp.gather(-1, t[..., None]).exp()[..., 0]  # no double count at bin 0
            rows[key].append(torch.where(valid[n], -torch.log(p.clamp(min=1e-12)), torch.nan)[v])
        lpp = logits['pitch'].float().log_softmax(-1)
        rows['ent_pitch'].append((-(lpp.exp() * lpp).sum(-1))[v])
        # temperature grid, even / odd windows
        half = (win[:, 0] % 2).long()
        for n in names:
            lg = logits[n].float()
            for j, T_ in enumerate(TEMPS):
                c = F.cross_entropy((lg / T_).transpose(1, 2), tgt[n].clamp(min=0), reduction='none')
                for hh in (0, 1):
                    m = valid[n] & (half[:, None] == hh)
                    temp[n][hh, j] += c[m].sum().item()
            for hh in (0, 1):
                temp_n[n][hh] += (valid[n] & (half[:, None] == hh)).sum().item()
        # logit lens
        resid[0] = None
        for l in lens_layers:
            if l == 0:
                continue
            hl = model.transformer.ln_f(resid[l])
            cl, _ = head_ce(model, hl, tgt)
            for n in names:
                lens[l][n] += cl[n][valid[n]].sum().item()
        for n in names:
            lens_n[n] += valid[n].sum().item()
        print(f'{name}: batch {bi} done', flush=True)
    for hk in hooks:
        hk.remove()
    data = {k: torch.cat(v).float().cpu().numpy() for k, v in rows.items()}
    temp_res = {}
    for n in names:
        ce = temp[n] / temp_n[n][:, None]
        j = int(ce[0].argmin())
        temp_res[n] = dict(best_T=TEMPS[j], ce_T1_odd=float(ce[1][TEMPS.index(1.0)]), ce_bestT_odd=float(ce[1][j]),
                           gain_odd=float(ce[1][TEMPS.index(1.0)] - ce[1][j]), curve_even=ce[0].round(4).tolist())
    lens_res = {l: {n: lens[l][n] / lens_n[n] for n in names} for l in lens_layers if l > 0}
    return data, temp_res, lens_res


@torch.no_grad()
def layer_skip(model, src, args):
    names = model.head_names
    blocks = model.transformer.h
    cache = list(batches(model, src, args, args.skip_windows))

    def total():
        s = {n: 0.0 for n in names}
        cnt = {n: 0 for n in names}
        for streams, ex, Y, _ in cache:
            tgt = dict(zip(names, Y))
            ce, _ = head_ce(model, model.encode(*streams, **ex), tgt)
            for n in names:
                s[n] += ce[n][tgt[n] != -1].sum().item()
                cnt[n] += (tgt[n] != -1).sum().item()
        return {n: s[n] / cnt[n] for n in names}

    base = total()
    out = {'base': base}
    for l in range(len(blocks)):
        orig = blocks[l].forward
        blocks[l].forward = lambda x, *a, **k: x
        out[l] = total()
        blocks[l].forward = orig
        print(f'skip layer {l}: dCE {sum(out[l].values()) - sum(base.values()):+.4f}', flush=True)
    return out


def table(data, names, key, groups):
    lines = []
    for label, mask in groups:
        n = int(mask.sum())
        if n == 0:
            continue
        cells = [f'{np.nanmean(data[f"ce_{h}"][mask]):.3f}' for h in names]
        tot = sum(np.nan_to_num(np.nanmean(data[f'ce_{h}'][mask])) for h in names)
        lines.append(f'| {label} | {n:,} | {100 * n / len(mask):.1f}% | ' + ' | '.join(cells) + f' | {tot:.3f} |')
    return lines


def report(name, data, temp_res, lens_res, skip, names):
    hdr = '| group | notes | share | ' + ' | '.join(names) + ' | sum |'
    sep = '|' + '---|' * (len(names) + 4)
    out = [f'## {name}', '', hdr, sep]
    allm = np.ones(len(data['pos']), bool)
    out += table(data, names, 'all', [('all', allm)])
    edges = [0, 16, 64, 256, 1024, 2048]
    out += table(data, names, 'pos', [(f'pos {a}-{b}', (data['pos'] >= a) & (data['pos'] < b))
                                      for a, b in zip(edges[:-1], edges[1:])])
    out += table(data, names, 'chord', [('chord note (dt=0)', data['chord'] > 0), ('onset (dt>0)', data['chord'] == 0)])
    fam = [(f'family {FAMILIES[f]}', data['family'] == f) for f in range(17)]
    fam = sorted(fam, key=lambda g: -g[1].sum())[:8]
    out += table(data, names, 'fam', fam)
    # share of the total loss per head
    tot = {h: np.nansum(data[f'ce_{h}']) for h in names}
    s = sum(tot.values())
    out += ['', 'Share of the summed loss: ' + ', '.join(f'{h} {100 * tot[h] / s:.1f}%' for h in names)]
    for h, key in (('delta_time', 'near_dt'), ('duration', 'near_dur')):
        ce = np.nanmean(data[f'ce_{h}'])
        near = np.nanmean(data[key])
        out.append(f'{h}: CE {ce:.3f}, CE with a +-1 bin tolerance {near:.3f} -> {ce - near:.3f} nats/note '
                   f'({100 * (ce - near) / ce:.0f}%) is exact-20ms timing')
    # worst target values by total loss
    for h in ('delta_time', 'duration', 'pitch', 'velocity'):
        t = data[f't_{h}']
        c = data[f'ce_{h}']
        ok = ~np.isnan(c)
        vals, inv = np.unique(t[ok], return_inverse=True)
        sums = np.bincount(inv, weights=c[ok])
        cnts = np.bincount(inv)
        top = np.argsort(-sums)[:8]
        out.append(f'{h} top target bins by total loss: ' + ', '.join(
            f'{int(vals[i])} ({100 * sums[i] / sums.sum():.1f}% loss, {100 * cnts[i] / cnts.sum():.1f}% notes, '
            f'CE {sums[i] / cnts[i]:.2f})' for i in top))
    out += ['', '### Temperature (fit on even windows, CE on odd windows)', '',
            '| head | best T | CE at T=1 | CE at best T | gain |', '|---|---|---|---|---|']
    for h in names:
        r = temp_res[h]
        out.append(f'| {h} | {r["best_T"]} | {r["ce_T1_odd"]:.4f} | {r["ce_bestT_odd"]:.4f} | {r["gain_odd"]:+.4f} |')
    out += ['', '### Logit lens (final heads on the residual after layer l)', '',
            '| layer | ' + ' | '.join(names) + ' | sum |', '|' + '---|' * (len(names) + 2)]
    for l, r in lens_res.items():
        out.append(f'| {l} | ' + ' | '.join(f'{r[h]:.3f}' for h in names) + f' | {sum(r.values()):.3f} |')
    if skip:
        base = sum(skip['base'].values())
        out += ['', f'### Layer skip (sum CE base {base:.4f})', '', '| layer | dCE sum | worst head |', '|---|---|---|']
        for l in [k for k in skip if k != 'base']:
            d = {h: skip[l][h] - skip['base'][h] for h in names}
            w = max(d, key=d.get)
            out.append(f'| {l} | {sum(d.values()):+.4f} | {w} {d[w]:+.4f} |')
    return out


def main():
    args = parse_args()
    model, it = load_model(ROOT / args.checkpoint, args.device, torch.bfloat16)
    args.block = args.block or model.config.block_size
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f'# Loss anatomy: {args.checkpoint} (iter {it}, block {args.block}, {args.windows} windows per source)', '']
    js = {}
    for i, spec in enumerate(args.sources.split(';')):
        name, src = spec.split('=', 1)
        data, temp_res, lens_res = run_source(model, name, src, args)
        skip = layer_skip(model, src, args) if args.skip_windows and i == 0 else None
        np.savez_compressed(out_dir / f'{name}_notes.npz', **data)
        md += report(name, data, temp_res, lens_res, skip, model.head_names) + ['']
        js[name] = dict(temperature=temp_res, lens=lens_res, skip=skip)
        (out_dir / 'summary.md').write_text('\n'.join(md))
        (out_dir / 'anatomy.json').write_text(json.dumps(js, indent=1, default=str))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
