"""
Song-structure metrics for whole generated pieces, next to the same metrics on real songs (the reference).

    python structure_eval.py --checkpoints checkpoints/ua_2048_covers checkpoints/ua_2048_mix --style cover \\
        --real data/finetune/ukrainian_covers.csv --samples 8 --device cpu --out-dir samples/structure

Each checkpoint samples --samples pieces from BOS (+ --style), EOS allowed after --min-notes, up to --max-new notes
(one window at 2048, so no sliding). Real reference = the validation songs of the --real CSVs, cropped to the same
--max-new notes (recurrence and key stability grow with length). Per piece:
  sec, notes/s, chord %    length, density, notes on the same onset as the previous note
  ends                     the sample emitted EOS itself (real songs: always)
  key stab                 share of 15 s segments whose Krumhansl-Schmuckler key is the piece's global key
  mel rep                  share of 4-note melody patterns (top note per onset) seen earlier in the piece: hooks,
                           choruses. First 400 notes: real covers 25.6%, real reductions 2.0% (transcription noise
                           breaks every repeat), samples of a reduction model 0.9%
  pit rep                  the same over all pitches in order (chords included): covers 18.4, reductions 2.2
  recur                    per 2 s window, cosine (pitch histogram) to its best match 8-60 s earlier, averaged
  dens drift, reg drift    |log| density ratio and |mean pitch| difference between the first and last third
A model that wanders scores low on the repeat columns, recur and key stab; one that loops a single bar scores
near 100% repeat.
Note level, pooled over all pieces of a set (eval_samples.py features, from tokens on both sides):
  OA                       mean histogram overlap with the reference set (first --real CSV): pitch, pitch class,
                           velocity, duration, onset gap, polyphony, melodic interval (1 = identical)
  dist                     distance to the reference, lower = closer: mean of (1 - OA), |d key|/50, |d melRep|/25,
                           |d pitRep|/20, |d recur|/0.15, |log nps ratio|/0.5, |d chord|/40, |d ends|/100
--style is a preference list ('cover,reduction'): each model gets the first one its style table has, else none.
Compare with the real row, not with 1. Optional --out-dir writes the samples as MIDI for listening.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import load_styles, tokenize_midi
from generate import resolve_checkpoint, tokens_to_midi
from jobstatus import JobStatus
from model import BOS_PITCH, EOS_PITCH, GPT, MusicConfig

DT = 0.02
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PROFILES = np.stack([np.roll(MAJOR, k) for k in range(12)] + [np.roll(MINOR, k) for k in range(12)])  # (24, 12)


def key_of(pitch, dur):
    """Krumhansl-Schmuckler: duration-weighted pitch-class profile vs the 24 rotated key profiles -> key index."""
    h = np.bincount(pitch % 12, weights=dur + 1e-3, minlength=12)
    if h.sum() == 0:
        return -1
    return int(np.argmax([np.corrcoef(h, p)[0, 1] for p in PROFILES]))


def repeats(seq, n=4):
    """% of n-grams of seq that already occurred earlier in it."""
    seen, rep = set(), 0
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        rep += g in seen
        seen.add(g)
    return 100 * rep / max(len(seq) - n + 1, 1)


def metrics(t, ended):
    """t: (n, 4) pitch, velocity bin, duration bin, delta bin of one piece (no BOS/EOS)."""
    pitch, dur, dt = t[:, 0], t[:, 2] * DT, t[:, 3]
    onset = np.cumsum(dt) * DT
    sec = max(onset[-1], 1e-3)
    out = dict(sec=sec, nps=len(t) / sec, chord=100 * (dt[1:] == 0).mean(), ends=100.0 * ended)
    # key stability over 15 s segments (segments with < 8 notes skipped)
    g = key_of(pitch, dur)
    seg = (onset // 15).astype(int)
    keys = [key_of(pitch[seg == s], dur[seg == s]) for s in np.unique(seg) if (seg == s).sum() >= 8]
    out['key'] = 100 * np.mean([k == g for k in keys]) if keys else np.nan
    # repeated 4-note patterns: the melody (top note of each onset) and all pitches in order
    on_ids = np.cumsum(dt > 0)
    melody = np.array([pitch[on_ids == o].max() for o in np.unique(on_ids)])
    out['mel_rep'], out['pit_rep'] = repeats(melody), repeats(pitch)
    # recurrence of 2 s windows (pitch histograms), best match 8-60 s earlier
    win = (onset // 2).astype(int)
    W = np.zeros((win.max() + 1, 128))
    np.add.at(W, (win, pitch.clip(0, 127)), 1)
    norms = np.linalg.norm(W, axis=1)
    keep = norms > 0
    V = W / np.where(keep, norms, 1)[:, None]
    best = [(V[max(0, i - 30):i - 3] @ V[i]).max() for i in range(4, len(V))
            if keep[i] and keep[max(0, i - 30):i - 3].any()]
    out['recur'] = float(np.mean(best)) if best else np.nan
    # drift between the first and last third
    a, b = len(t) // 3, len(t) - len(t) // 3
    nps = lambda s, e: (e - s) / max((onset[e - 1] - onset[s]), 1e-3)
    out['dens_drift'] = abs(np.log(nps(b, len(t)) / nps(0, a))) if a > 2 else np.nan
    out['reg_drift'] = abs(pitch[b:].mean() - pitch[:a].mean()) if a > 0 else np.nan
    return out


FEATURE_BINS = {
    'pitch': np.arange(0, 129), 'pitch_class': np.arange(0, 13), 'velocity': np.arange(0, 33),
    'duration': np.concatenate([[0], np.geomspace(0.02, 10, 40)]),
    'ioi': np.concatenate([[0, 0.01], np.geomspace(0.02, 10, 40)]),
    'polyphony': np.arange(0, 17), 'interval': np.arange(-24, 26),
}


def note_features(t):
    """Per-note feature arrays of one piece from tokens (velocity in bins, times in seconds)."""
    pitch, dur = t[:, 0], np.maximum(t[:, 2] * DT, 0.08)  # 0.08 s minimum, as tokens_to_midi writes it
    start = np.cumsum(t[:, 3]) * DT
    end = start + dur
    poly = np.searchsorted(start, start, side='right') - np.searchsorted(np.sort(end), start, side='right')
    return dict(pitch=pitch, pitch_class=pitch % 12, velocity=t[:, 1], duration=dur, ioi=np.diff(start),
                polyphony=np.clip(poly, 0, 16), interval=np.clip(np.diff(pitch), -24, 24))


def pooled(pieces):
    feats = [note_features(p) for p in pieces if len(p) > 1]
    return {k: np.concatenate([f[k] for f in feats]) for k in FEATURE_BINS}


def overlap(a, b):
    """Mean histogram overlap of two pooled feature sets."""
    oas = []
    for k, bins in FEATURE_BINS.items():
        ha = np.histogram(np.clip(a[k], bins[0], bins[-1]), bins=bins)[0]
        hb = np.histogram(np.clip(b[k], bins[0], bins[-1]), bins=bins)[0]
        oas.append(np.minimum(ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)).sum())
    return float(np.mean(oas))


SCALES = dict(key=50, mel_rep=25, pit_rep=20, recur=0.15, chord=40, ends=100)


def distance(m, ref, oa):
    d = [1 - oa, abs(np.log(m['nps'] / ref['nps'])) / 0.5]
    d += [abs(m[k] - ref[k]) / s for k, s in SCALES.items()]
    return float(np.mean(d))


def pick_style(model, cfg, prefs):
    """First preferred style the model has trained (its row in the style table isn't still all zeros), else none."""
    if not cfg.n_styles:
        return None, '-'
    names = load_styles()
    table = model.transformer['style'].weight
    for name in prefs:
        if name in names and names.index(name) < cfg.n_styles and table[names.index(name)].abs().sum() > 0:
            return names.index(name), name
    return 0, 'none'


def real_pieces(csvs):
    pieces = []
    for csv in csvs:
        df = pd.read_csv(csv)
        for f in df.loc[df['split'] == 'validation', 'midi_filename']:
            toks = tokenize_midi(f)
            if toks is not None:
                pieces.append(np.stack([x.numpy() for x in toks], 1))
    return pieces


def load_model(path, device):
    ck = torch.load(resolve_checkpoint(path), map_location=device)
    cfg = MusicConfig(**{k: v for k, v in ck['model_args'].items() if k in MusicConfig.__dataclass_fields__},
                      dropout=0.0)
    model = GPT(cfg)
    model.load_state_dict({k.replace('_orig_mod.', ''): v.float() for k, v in ck['model'].items()}, strict=False)
    return model.eval().to(device), cfg, ck.get('iter_num', '?')


def sample(model, cfg, args, style, device, job, base):
    B = args.samples
    if cfg.pitch_size > BOS_PITCH:
        seed = [torch.full((B, 1), BOS_PITCH if j == 0 else 0, dtype=torch.long, device=device) for j in range(4)]
    else:  # models without BOS: a silent note, as in generate.py
        seed = [torch.full((B, 1), v, dtype=torch.long, device=device) for v in (0, 0, 10, 0)]
    torch.manual_seed(args.seed)
    with torch.no_grad():
        out = model.generate(*seed, max_new_tokens=args.max_new, temperature=args.temperature,
                             top_p=args.top_p, cfg_scale=args.cfg, dt_bias=args.dt_bias, style=style, min_new=args.min_notes, cuda_graph=False,
                             progress=lambda n: job.update(base + n * B))
    pieces = []
    for r in range(B):
        rows = np.stack([o[r, 1:].cpu().numpy() for o in out], 1)
        ended = EOS_PITCH in rows[:, 0]
        if ended:
            rows = rows[:list(rows[:, 0]).index(EOS_PITCH)]
        pieces.append((rows[rows[:, 0] < BOS_PITCH], ended))
    return pieces


KEYS = ['sec', 'nps', 'chord', 'ends', 'key', 'mel_rep', 'pit_rep', 'recur', 'dens_drift', 'reg_drift']


def summarize(name, ms, pieces, ref=None):
    """One table row; ref = (reference means, reference pooled features) for OA and dist, None for the reference."""
    m = {k: np.nanmean([x[k] for x in ms]) for k in KEYS}
    feats = pooled(pieces)
    oa = overlap(feats, ref[1]) if ref else 1.0
    dist = distance(m, ref[0], oa) if ref else 0.0
    row = (f"{name:40s}{len(ms):4d}{m['sec']:7.0f}{m['nps']:7.2f}{m['chord']:7.1f}{m['ends']:6.0f}"
           f"{m['key']:6.0f}{m['mel_rep']:8.1f}{m['pit_rep']:8.1f}{m['recur']:7.3f}{m['dens_drift']:7.2f}"
           f"{m['reg_drift']:7.1f}{oa:7.3f}{dist:7.3f}")
    return row, (m, feats), dist


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoints', nargs='*', default=[], help='files or run dirs')
    ap.add_argument('--real', nargs='*', default=['data/finetune/ukrainian_covers.csv'],
                    help='CSVs whose validation songs are the reference')
    ap.add_argument('--style', default='cover,reduction',
                    help="style preference list: each model gets the first one in its table (else none)")
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=1600)
    ap.add_argument('--min-notes', type=int, default=200, help='no EOS before this many notes')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=None)
    ap.add_argument('--cfg', type=float, default=1.0, help='classifier-free guidance on the style (1 = off)')
    ap.add_argument('--dt-bias', type=float, default=0.0, help='density bias (generate.py --dt-bias)')
    ap.add_argument('--tag', default='', help='suffix for the row names (e.g. the sampling setting)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out-dir', default=None, help='write the samples as MIDI here')
    ap.add_argument('--out', default=None, help='also write the table here')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    prefs = [x.strip() for x in args.style.split(',') if x.strip()]
    header = (f"{'':40s}{'n':>4s}{'sec':>7s}{'nps':>7s}{'chord%':>7s}{'ends%':>6s}{'key%':>6s}{'melRep':>8s}"
              f"{'pitRep':>8s}{'recur':>7s}{'dDens':>7s}{'dReg':>7s}{'OA':>7s}{'dist':>7s}")
    lines = [f"structure: {args.samples} samples/checkpoint, BOS + style ({args.style}), T={args.temperature}, "
             f"top_p={args.top_p}, cfg={args.cfg}, dt_bias={args.dt_bias}, seed={args.seed}, "
             f"EOS after {args.min_notes}, max {args.max_new} notes; reference = {Path(args.real[0]).stem}", header]
    ref = None
    for csv in args.real:  # the first CSV is the reference, others are scored against it
        reals = [p[:args.max_new] for p in real_pieces([csv])]
        row, stats, _ = summarize(f"real {Path(csv).stem}", [metrics(p, True) for p in reals], reals, ref)
        ref = ref or stats
        lines.append(row)
    print('\n'.join(lines), flush=True)
    ranking = []

    job = JobStatus('structure', max(1, len(args.checkpoints) * args.samples * args.max_new),
                    f'{len(args.checkpoints)} checkpoints')
    for ci, path in enumerate(args.checkpoints):
        model, cfg, it = load_model(path, args.device)
        style, style_name = pick_style(model, cfg, prefs)
        pieces = sample(model, cfg, args, style, args.device, job, ci * args.samples * args.max_new)
        name = f"{Path(path).name} @{it} [{style_name}]{args.tag}"
        row, _, dist = summarize(name, [metrics(p, e) for p, e in pieces], [p for p, _ in pieces], ref)
        lines.append(row)
        ranking.append((dist, name))
        print(row, flush=True)
        if args.out_dir:
            d = Path(args.out_dir) / Path(path).name
            d.mkdir(parents=True, exist_ok=True)
            for i, (p, _) in enumerate(pieces):
                tokens_to_midi(*[p[:, j].tolist() for j in range(4)]).write(str(d / f'sample_{i + 1}.mid'))
        del model
    if ranking:
        lines += ['', 'closest to the reference: ' + ' < '.join(f"{n} ({d:.3f})" for d, n in sorted(ranking))]
        print(lines[-1])
    job.finish('done')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
