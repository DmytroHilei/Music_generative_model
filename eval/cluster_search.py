"""
Generation settings per seed-song cluster for a multi-track model, validated on every validation song.

    python eval/cluster_search.py --checkpoint checkpoints/multi_control/model_bf16_it1999.pt --out results/multi_router

Clusters: musicar/settings_router.py features of each song's first 128 notes, Gaussian mixtures (EM) fitted on the
CSV's TRAIN songs for K = 1..--max-k; usable K = every cluster holds >= --min-weight of the train songs; K* = the
usable K with the lowest BIC.
Songs: all validation songs long enough for prompt + --new-notes; fold A / B by crc32 of 'artist - title'.
Measure: style_eval's row distance of each sample to its own song's real continuation (spreads over all val songs).
Stage 1: every GRID setting, seed 1, all songs. Stage 2: the shortlist (per fold: the --top best overall and per
cluster of every usable K, responsibility-weighted, + UA_MULTI) with seeds 2, 3.
Validation (shortlist, seed means): every policy's setting(s) are chosen on one fold and scored on the other, so
each song's score comes from settings it did not help choose; policies = UA_MULTI, one global setting, per-cluster
settings (cluster weights = responsibilities, songs routed to their most likely cluster). Differences are paired
per song, +- standard error over songs.
Writes <out>/results.csv (one row per setting x seed x song; reruns skip finished passes), <out>/router.json (K*:
standardisation, mixture, settings per cluster chosen on all val songs) and <out>/summary.md.
"""

import argparse
import csv
import itertools
import json
import os
import time
import zlib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np  # noqa: E402
import torch  # noqa: E402

from musicar import settings_router as R  # noqa: E402
from musicar.checkpoint import load_model  # noqa: E402
from musicar.eval import style_eval  # noqa: E402
from musicar.eval.style_eval import UA_MULTI  # noqa: E402

GRID = dict(temperature=[0.9, 1.0, 1.1], top_p=[0.85, 0.9, 0.95], dt_bias=[-0.2, -0.1, 0.0])
BASE = dict(cond='band', min_new=1000, instrument_temperature=None)
FIELDS = ['config', 'seed', 'song', 'dist', 'nps', 'cons']


def name(c):
    return f"T{c['temperature']} p{c['top_p']} dt{c['dt_bias']}"


def genre(artists):
    """artist -> 'rock' / 'pop' from data/artists/candidates.tsv (uk.wikipedia category)."""
    out = {}
    with open('data/artists/candidates.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out[r['name']] = 'rock' if 'рок' in r['source'] else 'pop'
    return [out.get(a, '?') for a in artists]


def weighted_best(m, configs, w):
    """The config with the lowest w-weighted mean score; m = {config: (n_songs,) scores}."""
    return min(configs, key=lambda c: float((m[c] * w).sum() / max(w.sum(), 1e-9)))


def crossfit(m, configs, resp, fold):
    """(n,) score of every song under the per-cluster policy (resp (n, k); k = 1 is the global policy), settings
    chosen on the other fold; and the choice per fold."""
    score, choice = np.zeros(len(fold)), {}
    route = resp.argmax(1)
    for f in (0, 1):
        train, test = fold != f, fold == f
        choice[f] = [weighted_best(m, configs, resp[:, j] * train) for j in range(resp.shape[1])]
        for i in np.where(test)[0]:
            score[i] = m[choice[f][route[i]]][i]
    return score, choice


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--csv', default='data/finetune/ukrainian_multi_v1.csv')
    p.add_argument('--songs', type=int, default=0, help='validation songs (0 = all long enough; tests: fewer)')
    p.add_argument('--new-notes', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=40, help='songs generated at once (64 does not fit 8 GB)')
    p.add_argument('--top', type=int, default=3, help='shortlist: best settings per fold and cluster')
    p.add_argument('--max-k', type=int, default=6)
    p.add_argument('--min-weight', type=float, default=0.1, help='usable K: every cluster >= this share')
    p.add_argument('--grid', default=None, help="JSON overriding GRID (tests)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # clusters, fitted on the train songs' prompts
    df, _, off, tok, prog = style_eval.csv_store(args.csv, 'train', 128 + args.new_notes)
    ftrain = np.array([R.prompt_features(tok[off[i]:off[i] + 128], prog[off[i]:off[i] + 128])
                       for i in range(len(df)) if off[i + 1] - off[i] >= 128])
    mu, sd = ftrain.mean(0), ftrain.std(0)
    gmms = {k: R.fit_gmm((ftrain - mu) / sd, k) for k in range(1, args.max_k + 1)}
    usable = [k for k in gmms if min(gmms[k]['weights']) >= args.min_weight]
    k_star = min((k for k in usable if k > 1), key=lambda k: gmms[k]['bic'], default=1)
    print(f'mixtures on {len(ftrain)} train prompts: ' + ', '.join(
        f"K{k} BIC {g['bic']:.0f} min weight {min(g['weights']):.2f}" for k, g in gmms.items()) +
        f'; usable {usable}, K* = {k_star}', flush=True)

    # validation songs
    sset = style_eval.csv_set(args.csv, args.songs, new_notes=args.new_notes)
    group = next(iter(sset))
    songs = sset[group]
    n = len(songs)
    fval = np.array([R.prompt_features(s['prompt'], s['prompt_prog']) for s in songs])
    xval = (fval - mu) / sd
    resp = {k: R.responsibilities(xval, gmms[k]) for k in usable}
    fold = np.array([zlib.crc32(s['name'].encode()) % 2 for s in songs])
    genres = genre([s['artist'] for s in songs])
    print(f'{n} validation songs (fold A {int((fold == 0).sum())}, B {int((fold == 1).sum())})', flush=True)

    # results so far
    res_path = out / 'results.csv'
    rows = list(csv.DictReader(open(res_path))) if res_path.exists() else []
    scores = defaultdict(dict)  # (config, seed) -> {song: dist}
    for r in rows:
        scores[(r['config'], int(r['seed']))][int(r['song'])] = float(r['dist'])
    grid = {**GRID, **(json.loads(args.grid) if args.grid else {})}
    configs = {name(c): {**BASE, **c} for c in (dict(zip(grid, v)) for v in itertools.product(*grid.values()))}
    configs.setdefault(name(UA_MULTI), {**BASE, **UA_MULTI})
    model, it = load_model(args.checkpoint, 'cuda', torch.bfloat16)

    def run(cname, seed, stage, i, total):
        if len(scores[(cname, seed)]) == n:
            return
        got = {}
        style_eval.evaluate(model, sset, new_notes=args.new_notes, seed=1000 * seed, chunk=args.chunk,
                            settings=configs[cname], modes=['band'], rows_out=got)
        for _, k, d, nps, dc in got['band']:
            scores[(cname, seed)][k] = d
            rows.append(dict(config=cname, seed=seed, song=k, dist=d, nps=nps, cons=dc))
        with open(res_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        d = np.array([scores[(cname, seed)][k] for k in range(n)])
        print(f'  {stage} {i}/{total} {cname:22s} seed {seed}: {d.mean():.3f} (A {d[fold == 0].mean():.3f} '
              f'B {d[fold == 1].mean():.3f})  [{(time.time() - t0) / 60:.0f} min]', flush=True)

    # stage 1
    for i, c in enumerate(configs, 1):
        run(c, 1, 'S1', i, len(configs))
    m1 = {c: np.array([scores[(c, 1)][k] for k in range(n)]) for c in configs}
    short = {name(UA_MULTI)}
    for f in (0, 1):
        for k in usable:
            for j in range(k):
                w = resp[k][:, j] * (fold == f)
                short |= set(sorted(configs, key=lambda c: (m1[c] * w).sum() / w.sum())[:args.top])
    short = sorted(short)
    print(f'shortlist ({len(short)}): {", ".join(short)}', flush=True)
    # stage 2
    for i, c in enumerate(short, 1):
        for seed in (2, 3):
            run(c, seed, 'S2', i, len(short))

    # validation on every song
    m = {c: np.mean([[scores[(c, s)][k] for k in range(n)] for s in (1, 2, 3)], 0) for c in short}
    seed_sd = float(np.mean([np.std([np.mean(list(scores[(c, s)].values())) for s in (1, 2, 3)], ddof=1)
                             for c in short]))
    base = m[name(UA_MULTI)]
    glob, glob_choice = crossfit(m, short, np.ones((n, 1)), fold)
    lines = [f'# Settings per seed-song cluster: {args.checkpoint} (iter {it})', '',
             f'{n} validation songs of {args.csv} (all long enough), {len(configs)} settings at seed 1, shortlist of '
             f'{len(short)} with seeds 1-3 (scores = 3-seed means). Distance to the real continuation, lower = '
             f'closer. Cross-fitted: settings chosen on one fold, scored on the other, every song scored once. '
             f'Seed sd of a setting\'s mean: {seed_sd:.3f}. Mixtures fitted on {len(ftrain)} train prompts; usable '
             f'K = {usable}, K* = {k_star} (lowest BIC).', '',
             '| policy | mean (cross-fitted) | vs UA_MULTI | vs one global setting | choices (fold A / B) |',
             '|---|---|---|---|---|',
             f'| UA_MULTI ({name(UA_MULTI)}) | {base.mean():.3f} | | | |',
             f'| one global setting | {glob.mean():.3f} | {(glob - base).mean():+.3f} ± {se(glob - base):.3f} | | '
             f'{glob_choice[0][0]} / {glob_choice[1][0]} |']
    policies = {}
    for k in usable:
        if k == 1:
            continue
        sc, ch = crossfit(m, short, resp[k], fold)
        policies[k] = sc
        lines.append(f'| K = {k}{" (K*)" if k == k_star else ""} | {sc.mean():.3f} | {(sc - base).mean():+.3f} ± '
                     f'{se(sc - base):.3f} | {(sc - glob).mean():+.3f} ± {se(sc - glob):.3f} | '
                     f'{"; ".join(ch[0])} / {"; ".join(ch[1])} |')

    # the router: K* with settings chosen on all songs
    rk = resp.get(k_star, np.ones((n, 1)))
    route = rk.argmax(1)
    chosen = [weighted_best(m, short, rk[:, j]) for j in range(rk.shape[1])]
    global_all = weighted_best(m, short, np.ones(n))
    lines += ['', f'## Clusters of K* = {k_star} (settings chosen on all songs)', '',
              '| cluster | train share | val songs | rock / pop / ? | ' + ' | '.join(R.FEATURES) +
              ' | setting | its score | global setting on these songs | UA_MULTI on these |',
              '|' + '---|' * (8 + len(R.FEATURES))]
    for j in range(rk.shape[1]):
        idx = route == j
        g = [genres[i] for i in np.where(idx)[0]]
        fm = fval[idx].mean(0) if idx.any() else np.full(len(R.FEATURES), np.nan)
        lines.append(f'| {j} | {gmms[k_star]["weights"][j]:.2f} | {int(idx.sum())} | {g.count("rock")} / '
                     f'{g.count("pop")} / {g.count("?")} | ' + ' | '.join(f'{v:.2f}' for v in fm) +
                     f' | {chosen[j]} | {m[chosen[j]][idx].mean():.3f} | {m[global_all][idx].mean():.3f} | '
                     f'{base[idx].mean():.3f} |')
    lines += ['', f'Global setting on all songs: {global_all} ({m[global_all].mean():.3f}). Features: log notes/s, '
              'drum share, log instrument count, same-onset share, pitch mean / std (pitched notes), mean velocity bin, '
              'vocal share.', '', '## Shortlist, 3-seed means', '', '| setting | all | A | B |', '|---|---|---|---|']
    for c in sorted(short, key=lambda c: m[c].mean()):
        lines.append(f'| {c} | {m[c].mean():.3f} | {m[c][fold == 0].mean():.3f} | {m[c][fold == 1].mean():.3f} |')
    lines += ['', '## Stage 1 (seed 1)', '', '| setting | all |', '|---|---|']
    for c in sorted(configs, key=lambda c: m1[c].mean()):
        lines.append(f'| {c} | {m1[c].mean():.3f} |')
    lines += ['', f'Total time {(time.time() - t0) / 60:.0f} min.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    router = dict(checkpoint=args.checkpoint, csv=args.csv, features=list(R.FEATURES), mu=mu, sd=sd, k=k_star,
                  gmm={kk: v for kk, v in gmms[k_star].items()}, settings=[configs[c] for c in chosen],
                  setting_names=chosen, global_setting=configs[global_all])
    (out / 'router.json').write_text(R.to_json(router) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
