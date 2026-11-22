"""
Data files of the two scaling figures (docs/figures/scaling_isoflop.tex, scaling_repeat.tex) from the 5090 runs.

    .venv/bin/python tools/sweeps/scaling_figs.py          # docs/figures/scaling_data/*.dat, then pdflatex the two figures

Iso-FLOP: the main-recipe one-pass arms per budget (isoflop_fit.main_arms), a parabola in ln N per budget, its vertex
with the bootstrap 10-90% band, and ln N* on ln C. Repeat penalty: seed means of results/aug_sweep/runs.json;
penalty = repeated CE - fresh CE at the same size and notes seen; error bar = sqrt(se_rep^2 + se_fresh^2) with
se = |seed 1 - seed 2| / 2 (no bar where a side has one seed).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sweeps.frontier_fit import N_AUG  # noqa: E402
from tools.sweeps.isoflop_fit import budget_fit, drift, main_arms, parabola  # noqa: E402

OUT = ROOT / 'docs/figures/scaling_data'
Q_NOTES = 160_006_144
PLANNED = [('XL', 1e9, 1.2e20, 1.62e20), ('XXL', 1.8e9, 3.78e20, 5.29e20)]
MUSICAR_L = ('musicar-L', 461_070_848, 2.3e19)


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(' '.join(f'{x:.6g}' if isinstance(x, float) else str(x) for x in r) + '\n')


def isoflop():
    runs = json.loads((ROOT / 'results/isoflop/runs.json').read_text())
    arms = main_arms(runs)
    fits = []
    for i, (C, pts) in enumerate(sorted(arms.items())):
        n, ce = (np.array(v, float) for v in zip(*sorted(pts)))
        write(f'iso_points_{i}.dat', 'N ce', zip(n, ce))
        f = parabola(n, ce)
        b = budget_fit(C, n, ce)
        c2, c1, c0 = np.polyfit(np.log(n), ce, 2)
        xs = np.exp(np.linspace(math.log(n.min() / 1.12), math.log(n.max() * 1.12), 60))
        write(f'iso_curve_{i}.dat', 'N ce', zip(xs, c0 + c1 * np.log(xs) + c2 * np.log(xs) ** 2))
        lo, mid, hi = b['n_star_q']
        fits.append(dict(C=C, n_star=f['n_star'], E=f['E'], lo=lo, hi=hi, r=b['r_star']))
        print(f"C {C:.0e}: N* {f['n_star'] / 1e6:.1f}M [{lo / 1e6:.1f}, {hi / 1e6:.1f}], notes/param {b['r_star']:.1f}")
    write('iso_vertex.dat', 'C N lo hi ce', [(f['C'], f['n_star'], f['lo'], f['hi'], f['E']) for f in fits])
    dr = drift([dict(C=f['C'], n_star=f['n_star']) for f in fits])
    a = dr['a']
    b0 = math.log(fits[1]['n_star']) - a * math.log(fits[1]['C'])
    line = lambda C: math.exp(a * math.log(C) + b0)  # noqa: E731
    write('iso_trend_fit.dat', 'C N', [(C, line(C)) for C in np.geomspace(2e16, 4e17, 20)])
    write('iso_trend_extra.dat', 'C N', [(C, line(C)) for C in np.geomspace(3e17, 1e21, 20)])
    write('iso_planned.dat', 'C N Chi label', [(c, n, chi, lab) for lab, n, c, chi in PLANNED])
    write('iso_musicar.dat', 'C N label', [(MUSICAR_L[2], float(MUSICAR_L[1]), MUSICAR_L[0])])
    print(f'N* ~ C^{a:.3f}; on the trend: XL budget {line(1.2e20) / 1e9:.2f}B, XXL {line(3.78e20) / 1e9:.2f}B, '
          f'musicar-L budget {line(2.3e19) / 1e6:.0f}M')


def seeds(runs, size, arm, notes=Q_NOTES, frac=None):
    out = [r['ce'] for r in runs.values() if r.get('ce') is not None and not r.get('failed')
           and r.get('size', '42M') == size and r['arm'] == arm and abs(r['notes'] - notes) < 1e6
           and (frac is None or abs(r.get('frac', 1.0) - frac) < 1e-6)]
    return out


def mean_se(v):
    return float(np.mean(v)), (abs(v[0] - v[1]) / 2 if len(v) == 2 else None)


def penalty(rep, fresh):
    (m1, s1), (m2, s2) = mean_se(rep), mean_se(fresh)
    se = math.sqrt(s1 ** 2 + s2 ** 2) if s1 is not None and s2 is not None else 0.0
    return m1 - m2, se, min(len(rep), len(fresh))


def repeat():
    runs = json.loads((ROOT / 'results/aug_sweep/runs.json').read_text())
    iso = json.loads((ROOT / 'results/isoflop/runs.json').read_text())
    rows = []
    for size in N_AUG:
        c, f = seeds(runs, size, 'control', frac=0.25), seeds(runs, size, 'fresh', frac=1.0)
        if c and f:
            p, se, n = penalty(c, f)
            rows.append((float(N_AUG[size]), p, se, n, size))
            print(f'penalty {size}: {p:.3f} +- {se:.3f} ({n} seed(s))')
    write('rep_size_default.dat', 'N P se seeds label', rows)
    best = []
    for size in ('6M', '42M', '108M'):
        b, fb = seeds(runs, size, 'BEST', frac=0.25), seeds(runs, size, 'fresh_BEST', frac=1.0)
        if b and fb:
            b2 = b[:2] if len(b) > 2 else b  # seeds 1337 / 2 for the error bar; mean over all seeds
            p = float(np.mean(b)) - float(np.mean(fb))
            se = penalty(b2, fb)[1]
            best.append((float(N_AUG[size]), p, se, size))
            print(f'tuned recipe {size}: {p:.3f} +- {se:.3f}')
    write('rep_size_best.dat', 'N P se label', best)
    ep = iso['26']['ce'] - iso['24']['ce']
    write('rep_size_epochs.dat', 'N P', [(float(N_AUG['42M']), ep)])
    print(f'even epochs 42M (#26 - #24): {ep:.3f}')
    axis = []
    for notes in (Q_NOTES, 320_012_288, 639_991_808):
        c, f = seeds(runs, '6M', 'control', notes, frac=0.25), seeds(runs, '6M', 'fresh_census', notes)
        if c and f:
            p, se, n = penalty(c, f)
            axis.append((notes / 41_478_295, p, se, n))
            print(f'6M at {notes / 41_478_295:.1f} passes: {p:.3f} +- {se:.3f} ({n} seed(s))')
    write('rep_passes_6m.dat', 'passes P se seeds', axis)


if __name__ == '__main__':
    isoflop()
    repeat()
