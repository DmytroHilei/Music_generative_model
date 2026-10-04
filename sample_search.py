"""
Autotuning of the generation settings: which sampling parameters make the samples closest to real music, overall and
per kind of music. Runs on any checkpoint (e.g. a Hub snapshot) on the laptop GPU.

    python sample_search.py --checkpoint checkpoints/long_450m_snap --out samples/search_50k

Prompts: clean GigaMIDI val files grouped by how they sound (instrument families of their first 1,128 notes):
  rock_pop      drums + bass + guitar               orchestral   >= 2 of strings/brass/winds, no drums
  piano_keys    only keys (piano, organ, chromatic), <= 3 instruments, no drums
  electronic    synth lead/pad + drums               acoustic     guitar or piano + bass, <= 4 instruments, no drums
Each prompt = the file's first 128 notes; its reference = the real next 1,000 notes.

Objective (lower = better): paired distance, every sample against ITS OWN song's real continuation,
    d_row = sum_i w_i |sample_i - real_i| / s_i   (w, s as in sample_eval.distance; s from these real songs)
averaged over rows, so a sparse ballad isn't graded against a dense rock song. Also reported: D of the averages
(sample_eval.distance) and the paired distance per kind of music.

Search: successive halving over random configurations (temperature, instrument temperature, top-p, delta-time bias,
conditioning none / band / band + density): all configs on 2 songs per kind, the best third on 6 per kind, the best
few on 12 per kind. The current default (T 0.9, everything else off, band if the model has it) is always kept.
Writes <out>/results.csv (every config x stage) and <out>/summary.md.
"""

import argparse
import csv
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import BOS_PITCH, DRUM_PROGRAM, GPT, MusicConfig, window_conditions
from sample_eval import DIST_WEIGHTS, SD_FLOOR, STORE, distance, row_metrics

KINDS = ('rock_pop', 'orchestral', 'piano_keys', 'electronic', 'acoustic')
METRICS = list(DIST_WEIGHTS)


def family(program):
    return 'drums' if program == DRUM_PROGRAM else ('keys', 'keys', 'keys', 'guitar', 'bass', 'strings', 'strings',
                                                     'brass', 'winds', 'winds', 'synth', 'synth', 'fx', 'ethnic',
                                                     'perc', 'fx')[program // 8]


def kind_of(programs):
    fam = {family(p) for p in programs}
    n = len(set(programs))
    if {'drums', 'bass', 'guitar'} <= fam:
        return 'rock_pop'
    if 'synth' in fam and 'drums' in fam:
        return 'electronic'
    if 'drums' not in fam and len(fam & {'strings', 'brass', 'winds'}) >= 2:
        return 'orchestral'
    if 'drums' not in fam and fam <= {'keys'} and n <= 3:
        return 'piano_keys'
    if 'drums' not in fam and 'bass' in fam and fam & {'guitar', 'keys'} and n <= 4:
        return 'acoustic'
    return None


def diverse_prompts(per_kind, prompt_notes=128, new_notes=1000, store=STORE, candidates=40000):
    """{kind: [(prompt tokens (T, 4), prompt programs (T,), real continuation tokens, real programs)]}"""
    offsets = np.load(f'{store}/offsets.npy')
    n = int(offsets[-1])
    tokens = np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    programs = np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(n,))
    out = {k: [] for k in KINDS}
    for i in np.linspace(0, len(offsets) - 2, candidates).astype(int):
        a, b = offsets[i], offsets[i + 1]
        if b - a < prompt_notes + new_notes:
            continue
        p = np.asarray(programs[a:a + prompt_notes + new_notes], dtype=np.int64)
        k = kind_of(p.tolist())
        if k is None or len(out[k]) >= per_kind:
            continue
        t = np.asarray(tokens[a:a + prompt_notes + new_notes], dtype=np.int64)
        out[k].append((t[:prompt_notes], p[:prompt_notes], t[prompt_notes:], p[prompt_notes:]))
        if all(len(v) >= per_kind for v in out.values()):
            break
    return out


def config_space(model_cfg, n, seed=0):
    conds = ['none'] + (['band'] if model_cfg.cond_inst else []) + \
            (['band_dens'] if model_cfg.cond_inst and model_cfg.n_density else [])
    grid = dict(temperature=[0.75, 0.85, 0.9, 0.95, 1.05], instrument_temperature=[None, 0.5, 0.7],
                top_p=[None, 0.95, 0.9], dt_bias=[0.0, 0.1, 0.25], cond=conds)
    default = dict(temperature=0.9, instrument_temperature=None, top_p=None, dt_bias=0.0,
                   cond='band' if 'band' in conds else 'none')
    every = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rng = random.Random(seed)
    picked = [c for c in rng.sample(every, min(n, len(every))) if c != default]
    return [default] + picked[:n - 1]


def name(c):
    return (f"T{c['temperature']} iT{c['instrument_temperature'] or '-'} p{c['top_p'] or '-'} "
            f"dt{c['dt_bias']} {c['cond']}")


@torch.no_grad()
def run_config(model, c, rows, new_notes, seed, batch=16):
    """Per-row metrics of one configuration on the given prompt rows (list of (kind, prompt, ...))."""
    cfg = model.config
    out = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        tok = torch.stack([torch.from_numpy(r[1]) for r in chunk]).cuda()
        prog = torch.stack([torch.from_numpy(r[2]) for r in chunk]).cuda()
        streams = [tok[:, :, j] for j in range(4)]
        if cfg.pitch_size > BOS_PITCH:
            bos = [torch.full_like(streams[0][:, :1], BOS_PITCH if j == 0 else 0) for j in range(4)]
            streams = [torch.cat([b, x], 1) for b, x in zip(bos, streams)]
            prog = torch.cat([torch.zeros_like(prog[:, :1]), prog], 1)
        cond = None
        if cfg.cond_inst or cfg.n_density:
            inst, dens = window_conditions(streams[0], streams[3], prog, cfg.n_programs, cfg.n_density or 16)
            none_i, none_d = torch.zeros_like(inst), torch.zeros_like(dens)
            cond = {'none': (none_i, none_d), 'band': (inst, none_d), 'band_dens': (inst, dens)}[c['cond']]
        torch.manual_seed(seed)
        res = model.generate(*streams, max_new_tokens=new_notes, temperature=c['temperature'], top_p=c['top_p'],
                             dt_bias=c['dt_bias'], program=None, programs=prog, return_programs=True, cond=cond,
                             program_temperature=c['instrument_temperature'])
        n0 = streams[0].size(1)
        for r, row in enumerate(chunk):
            m = row_metrics(res[0][r, n0:].cpu(), res[3][r, n0:].cpu(), res[4][r, n0:].cpu(), set(row[2].tolist()))
            out.append((row[0], m))
    return out


def summarize(per_row, real_rows, scale):
    """Paired distance (mean over rows), D of the averages, per-kind paired distance, metric means."""
    paired, by_kind, ms = [], {}, []
    for (kind, m), real in zip(per_row, real_rows):
        if m is None:
            d = float('nan')
        else:
            d = sum(w * abs(m[k] - real[k]) / scale[k] for k, w in DIST_WEIGHTS.items())
            ms.append(m)
        paired.append(d)
        by_kind.setdefault(kind, []).append(d)
    avg = {k: float(np.mean([m[k] for m in ms])) for k in METRICS} if ms else {}
    real_avg = {k: float(np.mean([r[k] for r in real_rows])) for k in METRICS}
    return dict(paired=float(np.nanmean(paired)), D=distance(avg, {**real_avg, 'sd': scale}) if avg else float('nan'),
                **{f'paired_{k}': float(np.nanmean(v)) for k, v in by_kind.items()},
                **{f'mean_{k}': v for k, v in avg.items()})


def write_csv(path, results):
    fields = list(dict.fromkeys(k for r in results for k in r))  # union: per-kind columns can differ by stage
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--configs', type=int, default=36)
    p.add_argument('--stages', default='2,6,12', help='songs per kind in each halving stage')
    p.add_argument('--keep', default='12,4', help='configs kept after each stage but the last')
    p.add_argument('--new-notes', type=int, default=1000)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from generate import resolve_checkpoint
    ck = torch.load(resolve_checkpoint(args.checkpoint), map_location='cuda')
    mcfg = MusicConfig(**{k: v for k, v in ck['model_args'].items() if k in MusicConfig.__dataclass_fields__},
                       dropout=0.0)
    model = GPT(mcfg)
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()})
    model = model.cuda().to(torch.bfloat16).eval()
    print(f"model {args.checkpoint} (iter {ck.get('iter_num')})")

    stages = [int(x) for x in args.stages.split(',')]
    keep = [int(x) for x in args.keep.split(',')]
    prompts = diverse_prompts(max(stages))
    print('prompts per kind: ' + ', '.join(f'{k} {len(v)}' for k, v in prompts.items()))
    # real-song reference: per row, and the spread across all real continuations (the s_i of the distance)
    real_all = {k: [row_metrics(torch.from_numpy(t[:, 0]), torch.from_numpy(t[:, 3]), torch.from_numpy(g),
                                set(pp.tolist())) for (_, pp, t, g) in v] for k, v in prompts.items()}
    flat = [m for v in real_all.values() for m in v]
    scale = {k: max(float(np.std([m[k] for m in flat], ddof=1)), SD_FLOOR[k]) for k in METRICS}

    configs = config_space(mcfg, args.configs)
    default_cfg = configs[0]
    results = []
    for si, per_kind in enumerate(stages):
        rows, real_rows = [], []
        for k, v in prompts.items():
            for j, (tok, prog, _, _) in enumerate(v[:per_kind]):
                rows.append((k, tok, prog))
                real_rows.append(real_all[k][j])
        t0 = time.time()
        stage = []
        for ci, c in enumerate(configs):
            per_row = run_config(model, c, rows, args.new_notes, args.seed + si)
            s = summarize(per_row, real_rows, scale)
            stage.append((s['paired'], c, s))
            results.append(dict(stage=si + 1, rows=len(rows), config=name(c), **c, **s))
            write_csv(out / 'results.csv', results)  # after every evaluation: a crash loses nothing
            print(f'  stage {si + 1} ({len(rows)} songs) {ci + 1}/{len(configs)} {name(c):42s} paired {s["paired"]:.3f} '
                  f'D {s["D"]:.3f}  [{time.time() - t0:.0f} s]', flush=True)
        stage.sort(key=lambda x: x[0])
        if si < len(keep):
            kept = [c for _, c, _ in stage[:keep[si]]]
            if default_cfg not in kept:
                kept.append(default_cfg)  # the current default stays in for comparison
            configs = kept
    final = [r for r in results if r['stage'] == len(stages)]
    final.sort(key=lambda r: r['paired'])
    default = next(r for r in final if r['config'] == name(default_cfg))
    lines = [f'# Generation settings search: {args.checkpoint} (iter {ck.get("iter_num")})', '',
             f'{len(results)} evaluations; final stage {final[0]["rows"]} songs ({stages[-1]} per kind: '
             f'{", ".join(KINDS)}), {args.new_notes:,} new notes each. Lower = closer to the real continuations.', '',
             '| rank | config | paired | D | ' + ' | '.join(KINDS) + ' | notes/s | top | takeover |',
             '|' + '---|' * (7 + len(KINDS))]
    for i, r in enumerate(final):
        mark = ' (default)' if r is default else ''
        lines.append(f"| {i + 1} | {r['config']}{mark} | {r['paired']:.3f} | {r['D']:.3f} | "
                     + ' | '.join(f"{r.get(f'paired_{k}', float('nan')):.2f}" for k in KINDS)
                     + f" | {r['mean_notes_per_s']:.0f} | {r['mean_top_share']:.0%} | {r['mean_takeover']:.0%} |")
    real_mean = {k: float(np.mean([m[k] for m in flat])) for k in METRICS}
    lines += ['', f"Real songs: notes/s {real_mean['notes_per_s']:.0f}, top {real_mean['top_share']:.0%}, "
                  f"takeover {real_mean['takeover']:.0%}.", '',
              '## Best per kind (final stage)']
    for k in KINDS:
        best = min(final, key=lambda r: r.get(f'paired_{k}', 9e9))
        lines.append(f"- {k}: {best['config']} ({best[f'paired_{k}']:.2f}; default "
                     f"{default.get(f'paired_{k}', float('nan')):.2f})")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
