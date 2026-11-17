"""
Fit of the scaling ladder (tools/sweeps/scale_ladder.py): L(N, D) = E + A * N^-a + B * D^-b in nats per note on the
held-out GigaMIDI windows, N = parameters, D = notes seen.

    .venv/bin/python tools/sweeps/scale_fit.py            # results/scale_ladder/fit.md + docs/compute_request/scaling/

For each (a, b) on a grid E, A, B come from non-negative least squares on the arms; the best grid point is the fit.
Three checks say how far it can be trusted: (1) the 160M-note arms predicted from the 40M + 80M ones; (2) musicar-L
(464M, trained on the full data, not in the fit) predicted from the ladder; (3) the leave-one-arm-out range of every
prediction. The anchor's D is ambiguous (8.3B notes seen, 40% of them GigaMIDI), so both are printed.
"""
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/scale_ladder'
DOCS = ROOT / 'docs/compute_request/scaling'
GRID = np.round(np.arange(0.05, 1.501, 0.025), 3)
TARGETS = {'XL: 1B, 20B notes': (1.0e9, 20e9), 'XXL: 1.8B, 35B notes': (1.8e9, 35e9)}
GIGAMIDI_SHARE = 0.4


def nnls(X, y):
    """Least squares with coefficients >= 0, by trying every subset of columns (3 columns)."""
    best = None
    for k in range(X.shape[1], 0, -1):
        for cols in itertools.combinations(range(X.shape[1]), k):
            w, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
            if (w >= 0).all():
                full = np.zeros(X.shape[1])
                full[list(cols)] = w
                sse = float(((X @ full - y) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, full)
    return best


def fit(N, D, y):
    best = None
    for a in GRID:
        for b in GRID:
            X = np.stack([np.ones_like(N), (N / 1e6) ** -a, (D / 1e6) ** -b], 1)
            sse, w = nnls(X, y)
            if best is None or sse < best[0]:
                best = (sse, dict(E=w[0], A=w[1], B=w[2], a=float(a), b=float(b)))
    return best[1], float(np.sqrt(best[0] / len(y)))


def predict(f, n, d):
    return f['E'] + f['A'] * (n / 1e6) ** -f['a'] + f['B'] * (d / 1e6) ** -f['b']


def main():
    runs = json.loads((OUT / 'runs.json').read_text())
    arms = {k: r for k, r in runs.items() if r['kind'] == 'arm'}
    anchors = {k: r for k, r in runs.items() if r['kind'] == 'anchor'}
    names = list(arms)
    N = np.array([arms[k]['params'] for k in names], float)
    D = np.array([arms[k]['notes'] for k in names], float)
    y = np.array([arms[k]['ce'] for k in names], float)
    f, rmse = fit(N, D, y)
    loo = [fit(np.delete(N, i), np.delete(D, i), np.delete(y, i))[0] for i in range(len(y))]

    def row(label, n, d, measured=None):
        p = predict(f, n, d)
        ps = [predict(g, n, d) for g in loo]
        m = f'{measured:.3f}' if measured is not None else ''
        return f'| {label} | {p:.3f} | {min(ps):.3f}–{max(ps):.3f} | {m} |'

    lines = ['# Scaling ladder fit', '',
             f'{len(names)} arms (GigaMIDI replay, under one pass each), held-out GigaMIDI CE in nats per note '
             '(Δt + pitch + duration + velocity).', '',
             '| Parameters | ' + ' | '.join(f'{b} notes' for b in dict.fromkeys(r['budget'] for r in arms.values())) + ' |',
             '|---|' + '---|' * len({r['budget'] for r in arms.values()})]
    for size in dict.fromkeys(r['size'] for r in arms.values()):
        cells = [next((f'{r["ce"]:.3f}' for r in arms.values() if r['size'] == size and r['budget'] == b), '')
                 for b in dict.fromkeys(r['budget'] for r in arms.values())]
        params = next(r['params'] for r in arms.values() if r['size'] == size)
        lines.append(f'| {params / 1e6:.1f}M | ' + ' | '.join(cells) + ' |')
    lines += ['', f'Fit: L = {f["E"]:.3f} + {f["A"]:.2f}·(N/1M)^−{f["a"]:.3f} + {f["B"]:.2f}·(D/1M)^−{f["b"]:.3f}, '
                  f'RMSE {rmse:.3f} on the arms.', '']

    budgets = sorted({r['notes'] for r in arms.values()})
    if len(budgets) >= 3:
        low = D < budgets[-1]
        g, _ = fit(N[low], D[low], y[low])
        err = [predict(g, n, d) - t for n, d, t in zip(N[~low], D[~low], y[~low])]
        lines += [f'Check 1, the largest budget predicted from the smaller ones: error '
                  + ', '.join(f'{e:+.3f}' for e in err) + ' (predicted − measured, by rising size).', '']

    lines += ['| Point | Predicted | Leave-one-out range | Measured |', '|---|---|---|---|']
    for k, r in anchors.items():
        lines.append(row(f'{k} ({r["params"] / 1e6:.0f}M), D = all {r["notes"] / 1e9:.1f}B notes', r['params'], r['notes'],
                         r['ce']))
        lines.append(row(f'{k}, D = its GigaMIDI share {GIGAMIDI_SHARE * r["notes"] / 1e9:.1f}B', r['params'],
                         GIGAMIDI_SHARE * r['notes'], r['ce']))
    for label, (n, d) in TARGETS.items():
        lines.append(row(label, n, d))
        lines.append(row(label + ', GigaMIDI share', n, GIGAMIDI_SHARE * d))
    text = '\n'.join(lines) + '\n'
    (OUT / 'fit.md').write_text(text)
    (OUT / 'fit.json').write_text(json.dumps(dict(fit=f, rmse=rmse, loo=loo), indent=1))
    DOCS.mkdir(parents=True, exist_ok=True)
    for name in ('runs.json', 'fit.md', 'fit.json'):
        shutil.copy(OUT / name, DOCS / name)
    print(text)


if __name__ == '__main__':
    main()
