"""
Iso-FLOP fit: per budget C a parabola CE = E + A (ln N - ln N*)^2 over the main-recipe arms (lr 1e-3, seed 1337,
32,768 notes/step), its vertex N*, D* = C / (6 N*), r* = D*/N* (raw, and with a plateau offset t0: D_eff = D* -
t0 x 32,768), the curvature A, residual-bootstrap bands (10/50/90%), and ln N* regressed on ln C.

    .venv/bin/python tools/sweeps/isoflop_fit.py                 # results/isoflop/runs.json -> results/isoflop/fit.json
    .venv/bin/python tools/sweeps/isoflop_fit.py --t0 500
    .venv/bin/python tools/sweeps/isoflop_fit.py --synthetic     # planted surface, checks the vertex within 2%
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/isoflop'
NPS = 32768


def parabola(n, ce):
    """Least-squares CE = a + b x + c x^2, x = ln N: dict(n_star, A = c, E = CE at the vertex, resid)."""
    x, y = np.log(np.asarray(n, float)), np.asarray(ce, float)
    c, b, a = np.polyfit(x, y, 2)
    xs = -b / (2 * c) if c != 0 else float('nan')
    return dict(n_star=float(np.exp(xs)), A=float(c), E=float(a + b * xs + c * xs * xs),
                resid=(y - (a + b * x + c * x * x)).tolist())


def budget_fit(C, n, ce, t0=0.0, draws=1000, sigma_floor=0.003, seed=0):
    f = parabola(n, ce)
    r = lambda ns: (C / (6 * ns) - t0 * NPS) / ns  # noqa: E731
    rng = np.random.default_rng(seed)
    fitted = np.asarray(ce) - np.asarray(f['resid'])
    res = np.asarray(f['resid'])
    sd = max(float(np.sqrt((res ** 2).sum() / max(len(res) - 3, 1))), sigma_floor)
    boot = []
    for _ in range(draws):
        g = parabola(n, fitted + rng.normal(0, sd, len(fitted)))
        if g['A'] > 0:
            boot.append((g['n_star'], r(g['n_star']), g['A']))
    q = np.percentile(np.array(boot), [10, 50, 90], axis=0) if boot else np.full((3, 3), np.nan)
    return dict(C=C, n_star=f['n_star'], r_star=r(f['n_star']), A=f['A'], E=f['E'], sd=sd, arms=len(n),
                bracketed=bool(f['A'] > 0 and (np.asarray(n) < f['n_star']).sum() >= 2
                               and (np.asarray(n) > f['n_star']).sum() >= 2),
                n_star_q=q[:, 0].tolist(), r_star_q=q[:, 1].tolist(), A_q=q[:, 2].tolist(), ok_draws=len(boot))


def drift(fits):
    """ln N* = a ln C + b: N* ~ C^a, r* ~ C^(1 - 2a)."""
    if len(fits) < 2:
        return {}
    lc = np.log([f['C'] for f in fits])
    ln = np.log([f['n_star'] for f in fits])
    a, b = np.polyfit(lc, ln, 1)
    return dict(a=float(a), r_exponent=float(1 - 2 * a))


def main_arms(runs):
    """{C: [(N, CE)]} of the main-recipe arms (no alternative LR / seed / batch / weight decay)."""
    out = {}
    for r in runs.values():
        extra = [x for x in r.get('extra', []) if not x.startswith('--one_pass_epochs')]
        if r.get('C') and r['lr'] == 1e-3 and r['seed'] == 1337 and r['batch'] == 16 and not extra:
            out.setdefault(r['C'], []).append((r['n_ne'], r['ce']))
    return out


def report(fits, dr):
    for f in fits:
        print(f"C {f['C']:.0e}: N* {f['n_star'] / 1e6:6.2f}M [{f['n_star_q'][0] / 1e6:.2f}, {f['n_star_q'][2] / 1e6:.2f}]"
              f"  r* {f['r_star']:6.1f} [{f['r_star_q'][0]:.1f}, {f['r_star_q'][2]:.1f}]  A {f['A']:.4f}  E {f['E']:.4f}"
              f"  arms {f['arms']}{'' if f['bracketed'] else '  NOT bracketed'}")
    if dr:
        print(f"N* ~ C^{dr['a']:.3f}, r* ~ C^{dr['r_exponent']:+.3f}")
    print('dCE = A (ln f)^2 at f = 1.5 / 2: ' + ', '.join(
        f"{f['C']:.0e} {f['A'] * np.log(1.5) ** 2:.4f} / {f['A'] * np.log(2) ** 2:.4f}" for f in fits))


def synthetic():
    """Planted E + A (ln N - ln N*(C))^2 with N*(C) = 6.93e-2 C^0.5, A 0.05, noise 0.002 on the grid's N; vertex within 2%."""
    grid = {3e16: [5.88e6, 7.49e6, 10.44e6, 14.52e6, 19.83e6, 27.18e6],
            1e17: [10.44e6, 14.52e6, 19.83e6, 27.18e6, 35.69e6, 48.22e6],
            3e17: [19.83e6, 27.18e6, 35.69e6, 48.22e6, 64.85e6, 84.89e6]}
    rng = np.random.default_rng(1)
    fits = []
    for C, ns in grid.items():
        star = 6.93e-2 * C ** 0.5
        ce = [1.7 + 0.05 * np.log(n / star) ** 2 + rng.normal(0, 0.002) for n in ns]
        f = budget_fit(C, ns, ce, draws=200)
        err = abs(f['n_star'] / star - 1)
        print(f'C {C:.0e}: planted N* {star / 1e6:.2f}M, fitted {f["n_star"] / 1e6:.2f}M ({100 * err:.1f}%)')
        assert err < 0.02, err
        fits.append(f)
    dr = drift(fits)
    assert abs(dr['a'] - 0.5) < 0.05, dr
    return fits, dr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=str(OUT / 'runs.json'))
    p.add_argument('--t0', type=float, default=0.0, help='plateau offset in steps (D_eff = (I - t0) x 32,768)')
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    if a.synthetic:
        report(*synthetic())
        return
    arms = main_arms(json.loads(Path(a.runs).read_text()))
    fits = [budget_fit(C, *zip(*sorted(v)), t0=a.t0) for C, v in sorted(arms.items()) if len(v) >= 3]
    dr = drift(fits)
    report(fits, dr)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fit{'_t0_%g' % a.t0 if a.t0 else ''}.json").write_text(json.dumps(dict(fits=fits, drift=dr), indent=1))


if __name__ == '__main__':
    main()
