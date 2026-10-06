"""
Post-run sweep: the best generation settings per style, chosen without the winner's curse.

    python eval/style_search.py --checkpoint checkpoints/long_450m/model_bf16.pt --out results/long_450m_sweep

Measure: style_eval's robust per-style distance (each sample vs its own song's real continuation, spreads per style,
log scale for density / instrument count, terms capped). Lower = closer to real music.
Stage A: every setting of the grid on set A (the 16 songs per style the in-training eval uses).
Stage B: per style, its top --top settings from A (plus the tracking setting) on set B (16 DIFFERENT songs per
style); the reported best per style is the best on B, i.e. judged on songs it was not selected on.
Grid: temperature x top-p x instrument temperature x delta-time bias x conditioning (none / band).
Writes <out>/results.csv (every evaluation), <out>/best_settings.json ({style: settings + scores}) and
<out>/summary.md. Results are written after every setting, so an interrupted sweep keeps what it measured.
"""

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402

from musicar.eval import style_eval  # noqa: E402
from musicar.eval.style_eval import STYLES, TRACK  # noqa: E402

GRID = dict(temperature=[0.8, 0.9, 1.0], top_p=[None, 0.95, 0.9, 0.85], instrument_temperature=[None, 0.6],
            dt_bias=[0.0, 0.1], cond=['none', 'band'])


def name(c):
    return (f"T{c['temperature']} p{c['top_p'] or '-'} iT{c['instrument_temperature'] or '-'} "
            f"dt{c['dt_bias']} {c['cond']}")


def load(path):
    from musicar.model import GPT, MusicConfig
    ck = torch.load(path, map_location='cuda')
    cfg = MusicConfig(**{k: v for k, v in ck['model_args'].items() if k in MusicConfig.__dataclass_fields__},
                      dropout=0.0)
    model = GPT(cfg)
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()})
    return model.cuda().to(torch.bfloat16).eval(), ck.get('iter_num')


def run(model, sset, c, chunk):
    res = style_eval.evaluate(model, sset, settings=c, modes=[c['cond']], chunk=chunk)
    return res[c['cond']]


def write(out, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--per-style', type=int, default=16)
    p.add_argument('--top', type=int, default=5, help='settings per style taken from stage A to stage B')
    p.add_argument('--chunk', type=int, default=40, help='songs generated at once (GPU memory)')
    p.add_argument('--grid', default=None, help="JSON overriding GRID, e.g. '{\"temperature\": [0.9]}' (tests)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = {**GRID, **(json.loads(args.grid) if args.grid else {})}
    model, it = load(args.checkpoint)
    cfg = model.config
    if not cfg.cond_inst:
        grid['cond'] = ['none']
    configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    track = {**TRACK, 'instrument_temperature': None, 'cond': 'none'}
    set_a = style_eval.style_set(args.per_style)
    set_b = style_eval.style_set(args.per_style, skip=args.per_style)
    print(f'model {args.checkpoint} (iter {it}); {len(configs)} settings; sets A and B: {args.per_style} songs per style',
          flush=True)
    rows, a_scores = [], {}
    t0 = time.time()
    for i, c in enumerate(configs):
        r = run(model, set_a, c, args.chunk)
        a_scores[name(c)] = (c, r)
        rows.append(dict(stage='A', config=name(c), **{k: v for k, v in c.items()}, **r))
        write(out, rows)
        print(f"  A {i + 1}/{len(configs)} {name(c):38s} all {r['all']:.3f}  " +
              ' '.join(f"{s[:5]} {r.get(s, float('nan')):.2f}" for s in STYLES) + f"  [{time.time() - t0:.0f} s]",
              flush=True)
    # stage B: per style its best `top` settings from A (+ the tracking setting), measured on held-out songs
    chosen = {name(track): track}
    for s in STYLES:
        ranked = sorted(a_scores.values(), key=lambda cr: cr[1].get(s, 9e9))
        for c, _ in ranked[:args.top]:
            chosen[name(c)] = c
    b_scores = {}
    for i, (n, c) in enumerate(chosen.items()):
        r = run(model, set_b, c, args.chunk)
        b_scores[n] = (c, r)
        rows.append(dict(stage='B', config=n, **{k: v for k, v in c.items()}, **r))
        write(out, rows)
        print(f"  B {i + 1}/{len(chosen)} {n:38s} all {r['all']:.3f}  [{time.time() - t0:.0f} s]", flush=True)
    best = {}
    for s in STYLES:
        n, (c, r) = min(b_scores.items(), key=lambda kv: kv[1][1].get(s, 9e9))
        best[s] = dict(settings=c, config=n, score_b=r[s], score_a=a_scores.get(n, (None, {}))[1].get(s),
                       tracking_b=b_scores[name(track)][1][s])
    n_all, (c_all, r_all) = min(b_scores.items(), key=lambda kv: kv[1][1]['all'])
    best['_overall'] = dict(settings=c_all, config=n_all, score_b=r_all['all'],
                            tracking_b=b_scores[name(track)][1]['all'])
    (out / 'best_settings.json').write_text(json.dumps(best, indent=1))
    lines = [f'# Generation settings per style: {args.checkpoint} (iter {it})', '',
             f'{len(configs)} settings on set A ({args.per_style} songs per style), the top {args.top} per style on '
             f'held-out set B. Scores: robust distance to the real continuations (style_eval.py), lower = better; '
             f'tracking = the in-training setting ({name(track)}).', '',
             '| style | best setting (chosen on A, scored on B) | B score | its A score | tracking on B |',
             '|---|---|---|---|---|']
    for s in list(STYLES) + ['_overall']:
        b = best[s]
        lines.append(f"| {s.strip('_')} | {b['config']} | {b['score_b']:.3f} | "
                     f"{b['score_a']:.3f} | {b['tracking_b']:.3f} |" if b.get('score_a') is not None else
                     f"| {s.strip('_')} | {b['config']} | {b['score_b']:.3f} | | {b['tracking_b']:.3f} |")
    lines += ['', f'Total time {time.time() - t0:.0f} s.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
