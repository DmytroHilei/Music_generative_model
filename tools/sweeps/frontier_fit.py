"""
Frontier fit (scaling plan section 7, K7): the fresh-data surface from the iso-FLOP arms, the repetition forms on the
aug-sweep runs, and the tables the compute request needs (compute-optimal N and D, passes over the census U, the
predicted repeat penalty, the data-constrained frontier, floor and ceiling, the recipe layer).

    .venv/bin/python tools/sweeps/frontier_fit.py                    # results/frontier/{fit.json,report.md}
    .venv/bin/python tools/sweeps/frontier_fit.py --draws 1000       # bootstrap draws (default 200)
    .venv/bin/python tools/sweeps/frontier_fit.py --basis poisson    # effective-data basis of the aug runs
    .venv/bin/python tools/sweeps/frontier_fit.py --predictions docs/scaling/frontier_predictions_2026-11-22.json

Data: results/isoflop/runs.json (one-pass pool arms of the main recipe) and results/aug_sweep/runs.json (seed means,
runs of at least MIN_ITERS steps). N = non-embedding parameters incl. the heads (isoflop.n_ne), D = notes seen,
U = unique notes. CE = GigaMIDI held-out, nats per note, 4 heads summed.
Surface: L(N, D) = E + A N^-a + B D^-b on the iso arms, E capped at musicar-L's 1.641. Repetition forms, fitted on the
aug runs' CE with the surface fixed and offsets for the aug sweep's footing (one per store + a slope in ln N):
  a0     D' = U (1 + R (1 - exp(-(D/U - 1) / R)))               Muennighoff, data term only
  a0w    R = R0 (U / 41.5M)^w
  a0wg   R = R0 (U / 41.5M)^w (N / 42M)^g
  a1     a0 + excess parameters: N' = U_N (1 + RN (1 - exp(-(N/U_N - 1) / RN))), U_N = U / 27
  b      Chai & Xiong: L(N, D) + Ac (R_D N / U)^p, R_D = max(D/U - 1, 0)
  P4     Lovelace: L(N, D) + P0 R_D^d (N/U)^k (U / 41.5M)^-y
Validation: leave one group out (217M; R_D >= 6.6; the largest U; 3M); a form passes if every held-out run is within
0.02. The answer is the envelope of the passing forms. Out-of-sample checks on the iso footing (even epochs, no aug
offsets): #26 (4 epochs of the replay quarter at 42M) - #24, #34 (2 epochs plan, 1.33 passes), musicar-L.
Predictions use the epoch basis (the request's runs read the data in even epochs) and the default recipe.
Turn-ups: a repeated run whose held-out CE ends TURN_RISE above its lowest evaluation overfits; it is not fitted (no
form here can climb). Passes cap at N/U: the smaller of 1.5x the most passes fitted at a comparable N/U and half the
earliest turn-up among runs with N/U up to max(2 N/U, 1.05).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/frontier'
ISO_RUNS, AUG_RUNS = ROOT / 'results/isoflop/runs.json', ROOT / 'results/aug_sweep/runs.json'
U_Q, U_STORE, U_POOL = 41.48e6, 167.9e6, 1.897e9   # replay quarter, whole replay store, census GigaMIDI pool (usable)
N_AUG = {'3M': 2_444_288, '6M': 5_881_432, '20M': 19_834_898, '42M': 41_981_440, '108M': 106_615_351,
         '217M': 214_517_760}  # isoflop.n_ne of the aug_sweep shapes 4x192 .. 16x1024
MUSICAR_L = dict(N=461_070_848, D=8_295_677_952, U=4.5e9, ce=1.641)  # 28x1152; U = usable unique notes it read
E_CAP = 1.641
MIN_ITERS = 3000
TOL = 0.02
TURN_RISE = 0.02  # held-out CE this far above its lowest evaluation at the end of a run = a turn-up
K_UN = 27.0  # U_N = U / K_UN: the compute-optimal size for U notes in one pass (r* of the iso fit)
U_C = 8.84e9   # census unique notes after near-duplicate removal (full MinHash pass 2026-11-23, Jaccard >= 0.8)
U_GRID = [2.1e9, 4.5e9, 6.5e9, 7.7e9, U_C, 9.92e9]
# the request's shapes, N = isoflop.n_ne (non-embedding incl. the heads): 28x1664, 37x2112, 38x2304
N_XL, N_XXL, N_25 = 966_734_336, 2_028_593_408, 2_477_551_360
MODELS = {'1B 28x1664': N_XL, '2B 37x2112': N_XXL, '2.5B 38x2304': N_25}
# planned runs: (N, notes per parameter); None = the compute-optimal D of the fitted surface (the low end), 27 = the
# high end of the request
PLANNED = {'XL 1B, D opt': (N_XL, None), 'XL 1B, 27 notes/param': (N_XL, 27.0),
           'XXL 2B, D opt': (N_XXL, None), 'XXL 2B, 27 notes/param': (N_XXL, 27.0)}
C_GRID = [1e18, 3e18, 1e19, 2.3e19, 1.2e20, 3.78e20, 4.1e20, 5e20, 1e21, 1e22]
REP_ARMS = {'control': 'replay', 'fresh': 'replay', 'full': 'replay', 'fresh_census': 'pool', 'sub168_census': 'pool'}
LG = np.array([math.lgamma(k + 1) for k in range(4001)])


# ------------------------------------------------------------------------------------------------------------ data
def iso_points(runs):
    """One-pass pool arms of the main recipe (lr 1e-3, seed 1337, 32,768 notes/step, no extra flags) incl. #13u, #25,
    #27, O1; without #22p (partial) and #34 (1.33 passes: a check)."""
    out = []
    for k, r in runs.items():
        extra = [x for x in r.get('extra', []) if not x.startswith('--one_pass_epochs')]
        if (r['lr'] == 1e-3 and r['seed'] == 1337 and r['batch'] == 16 and r['store'].endswith('pool') and not extra
                and k not in ('22p', '34')):
            out.append(dict(tag=k, N=float(r['n_ne']), D=float(r['notes']), ce=float(r['ce'])))
    return out


def seed_sigma(runs):
    """Per-size seed noise: RMS over arms with two or more seeds of the sample SD (floor 0.003); 217M takes 108M's."""
    by = {}
    for r in runs.values():
        if not r.get('failed') and r.get('ce') is not None:
            key = (r.get('size', '42M'), r['arm'], round(r.get('frac', 1.0), 4), round(r['notes'], -6))
            by.setdefault(key, []).append(r['ce'])
    out = {}
    for size in N_AUG:
        v = [float(np.var(c, ddof=1)) for k, c in by.items() if k[0] == size and len(c) >= 2]
        out[size] = max(math.sqrt(np.mean(v)), 0.003) if v else None
    out['217M'] = out['217M'] or out['108M']
    return out


def aug_points(runs, min_iters=MIN_ITERS):
    """Seed means of the replay / census arms: dict(tag, size, N, D, U, ce, n, sigma, store, kind, use).
    use = False for the store bridge (sub168_census) and for turn-up runs (CE above the same size and U at fewer
    notes: overfitting, outside the saturating forms)."""
    sig, groups = seed_sigma(runs), {}
    for r in runs.values():
        if r.get('failed') or r.get('ce') is None or r.get('arm') not in REP_ARMS or r.get('iters', 0) < min_iters:
            continue
        arm, size = r['arm'], r.get('size', '42M')
        U = r.get('unique_notes') or (U_POOL if REP_ARMS[arm] == 'pool' else U_STORE)
        key = (size, arm, round(U), round(r['notes'], -6))
        groups.setdefault(key, []).append(r['ce'])
    out = []
    for (size, arm, U, D), ces in sorted(groups.items()):
        kind = 'control' if D > 1.05 * U else 'fresh'
        out.append(dict(tag=f'{size} {arm} U{U / 1e6:.0f}M D{D / 1e6:.0f}M', size=size, arm=arm,
                        N=float(N_AUG[size]), D=float(D), U=float(U), ce=float(np.mean(ces)), n=len(ces),
                        sigma=sig[size], store=REP_ARMS[arm], kind=kind, use=arm != 'sub168_census'))
    climbed = {(t['size'], round(t['U']), round(t['D'], -6)) for t in turnups(runs)}
    for p in out:
        if ((p['size'], round(p['U']), round(p['D'], -6)) in climbed
                or any(q['size'] == p['size'] and q['U'] == p['U'] and q['D'] < p['D'] and q['ce'] < p['ce'] for q in out)):
            p['use'], p['turn_up'] = False, True
    return out


def turnups(runs):
    """Repeated runs whose held-out CE ended TURN_RISE or more above its lowest evaluation (overfitting):
    dict(tag, size, N, D, U, nu = N/U, passes at the lowest evaluation, rise)."""
    out = []
    for r in runs.values():
        if r.get('failed') or r.get('arm') not in REP_ARMS or not r.get('curve') or not r.get('iters'):
            continue
        U = r.get('unique_notes') or (U_POOL if REP_ARMS[r['arm']] == 'pool' else U_STORE)
        c = [(s, ce) for s, ce in r['curve'] if s > 0]
        if len(c) < 2 or r['notes'] <= 1.05 * U:
            continue
        s_min, ce_min = min(c, key=lambda x: x[1])
        if c[-1][1] - ce_min >= TURN_RISE:
            size = r.get('size', '42M')
            out.append(dict(tag=f"{size} {r['arm']} U{U / 1e6:.0f}M D{r['notes'] / 1e6:.0f}M s{r['seed']}", size=size,
                            N=float(N_AUG[size]), D=float(r['notes']), U=float(U), nu=N_AUG[size] / U,
                            passes=(s_min + 1) * r['notes'] / r['iters'] / U, rise=c[-1][1] - ce_min))
    return sorted(out, key=lambda t: t['passes'])


def arrays(pts):
    return {k: np.array([p[k] for p in pts], float) for k in ('N', 'D', 'U')}


# ---------------------------------------------------------------------------------------------------------- optimiser
def nelder_mead(f, x0, step=0.3, iters=800, tol=1e-12):
    x0 = np.asarray(x0, float)
    pts = [x0] + [x0 + step * np.eye(len(x0))[i] for i in range(len(x0))]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ----------------------------------------------------------------------------------------------------------- surface
def surf(s, N, D):
    return s['E'] + s['A'] * np.asarray(N, float) ** -s['a'] + s['B'] * np.asarray(D, float) ** -s['b']


def _surface_lin(N, D, y, a, b):
    """E, A, B >= 0 with E <= E_CAP for fixed exponents: (sse, E, A, B) or None."""
    X = np.stack([np.ones_like(N), N ** -a, D ** -b], 1)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    if w[0] > E_CAP or w[0] < 0:
        w0 = min(max(w[0], 0.0), E_CAP)
        w12, *_ = np.linalg.lstsq(X[:, 1:], y - w0, rcond=None)
        w = np.array([w0, *w12])
    if (w[1:] < 0).any():
        return None
    return float(((X @ w - y) ** 2).sum()), *map(float, w)


def fit_surface(N, D, y):
    N, D, y = (np.asarray(v, float) for v in (N, D, y))
    best = None
    for a in np.arange(0.1, 1.21, 0.05):
        for b in np.arange(0.1, 1.21, 0.05):
            r = _surface_lin(N, D, y, a, b)
            if r and (best is None or r[0] < best[0]):
                best = (r[0], a, b)

    def f(x):
        r = _surface_lin(N, D, y, *x) if min(x) > 0.02 else None
        return r[0] if r else 1e9
    (a, b), _ = nelder_mead(f, [best[1], best[2]], step=0.03)
    sse, E, A, B = _surface_lin(N, D, y, a, b)
    return dict(E=E, A=A, B=B, a=float(a), b=float(b), rmse=math.sqrt(sse / len(y)), n=len(y),
                E_capped=bool(abs(E - E_CAP) < 1e-9))


def optimum(s, C):
    """Compute-optimal (N, D) at C = 6 N D on the surface."""
    a, b = s['a'], s['b']
    G = (a * s['A'] / (b * s['B'])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)


def d_opt(s, N):
    """D at which N is compute-optimal on the surface."""
    a, b = s['a'], s['b']
    G = (a * s['A'] / (b * s['B'])) ** (1 / (a + b))
    return (N / G) ** ((a + b) / b) / N


# ---------------------------------------------------------------------------------------------- effective data / N
def dprime(D, U, R, basis='epoch'):
    """Effective notes of D seen over U unique with repetition half-life R (arrays broadcast). epoch: every note D/U
    times; poisson: a note is seen k ~ Poisson(D/U) times (the aug sweep's sampler draws windows with replacement)."""
    D, U, R = np.broadcast_arrays(*(np.asarray(v, float) for v in (D, U, R)))
    if basis == 'epoch':
        x = np.maximum(D / U - 1, 0)
        return np.where(D <= U, D, U * (1 + R * (1 - np.exp(-x / R))))
    lam = (D / U).ravel()
    kmax = int(lam.max() + 12 * math.sqrt(lam.max()) + 30)
    k = np.arange(1, min(kmax, len(LG) - 1))
    logp = k * np.log(lam)[:, None] - lam[:, None] - LG[k]
    val = 1 + R.ravel()[:, None] * (1 - np.exp(-(k - 1) / R.ravel()[:, None]))
    return (U.ravel() * (np.exp(logp) * val).sum(1)).reshape(D.shape)


def nprime(N, U, RN):
    UN = np.asarray(U, float) / K_UN
    N = np.asarray(N, float)
    x = np.maximum(N / UN - 1, 0)
    return np.where(N <= UN, N, UN * (1 + RN * (1 - np.exp(-x / RN))))


def _rd(X):
    return np.maximum(X['D'] / X['U'] - 1, 0)


FORMS = {  # name: (parameter names, init, model CE without footing offsets)
    'a0': (['ln R'], [math.log(5)], lambda t, X, s, bs: surf(s, X['N'], dprime(X['D'], X['U'], math.exp(t[0]), bs))),
    'a0w': (['ln R0', 'w'], [math.log(5), 0.2], lambda t, X, s, bs: surf(
        s, X['N'], dprime(X['D'], X['U'], np.exp(t[0]) * (X['U'] / U_Q) ** t[1], bs))),
    'a0wg': (['ln R0', 'w', 'g'], [math.log(5), 0.2, -0.3], lambda t, X, s, bs: surf(
        s, X['N'], dprime(X['D'], X['U'], np.exp(t[0]) * (X['U'] / U_Q) ** t[1] * (X['N'] / 42e6) ** t[2], bs))),
    'a1': (['ln RD', 'ln RN'], [math.log(5), math.log(10)], lambda t, X, s, bs: surf(
        s, nprime(X['N'], X['U'], math.exp(t[1])), dprime(X['D'], X['U'], math.exp(t[0]), bs))),
    'b': (['ln Ac', 'ln p'], [math.log(0.1), math.log(0.5)], lambda t, X, s, bs: surf(s, X['N'], X['D']) + math.exp(
        t[0]) * (_rd(X) * X['N'] / X['U']) ** math.exp(t[1])),
    'P4': (['ln P0', 'ln d', 'k', 'y'], [math.log(0.05), 0.0, 0.33, 0.0], lambda t, X, s, bs: surf(
        s, X['N'], X['D']) + math.exp(t[0]) * _rd(X) ** math.exp(t[1]) * (X['N'] / X['U']) ** t[2] * (
        X['U'] / U_Q) ** -t[3]),
}


def offsets_design(pts):
    return np.array([[p['store'] == 'replay', p['store'] == 'pool', math.log(p['N'] / 42e6)] for p in pts], float)


def weights(pts):
    return np.array([p['n'] / p['sigma'] ** 2 for p in pts])


def fit_form(name, pts, s, basis, init=None, y=None):
    _, init0, fn = FORMS[name]
    X, Z, w = arrays(pts), offsets_design(pts), weights(pts)
    y = np.array([p['ce'] for p in pts]) if y is None else y
    sw = np.sqrt(w)

    def solve(t):
        with np.errstate(all='ignore'):
            m = fn(t, X, s, basis)
        if not np.all(np.isfinite(m)):
            return 1e9, None, None
        c, *_ = np.linalg.lstsq(Z * sw[:, None], (y - m) * sw, rcond=None)
        r = y - m - Z @ c
        return float((w * r * r).sum()), c, r
    starts = [np.asarray(init if init is not None else init0, float)]
    if init is None:
        starts.append(starts[0] + 0.5)
    best = None
    for x0 in starts:
        t, v = nelder_mead(lambda t: solve(t)[0], x0, step=0.4)
        if best is None or v < best[1]:
            best = (t, v)
    t = best[0]
    chi2, c, r = solve(t)
    n, k = len(pts), len(t) + Z.shape[1]
    return dict(form=name, theta=[float(x) for x in t], offsets=[float(x) for x in c], chi2=chi2,
                rmse=float(np.sqrt((r * r).mean())), resid=[float(x) for x in r],
                aicc=float(n * math.log(chi2 / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)))


def model_ce(fit, s, N, D, U, basis='epoch'):
    """Iso-footing CE (no aug offsets) of a fitted form."""
    X = {k: np.atleast_1d(np.asarray(v, float)) for k, v in (('N', N), ('D', D), ('U', U))}
    X = dict(zip(X, np.broadcast_arrays(*X.values())))
    with np.errstate(all='ignore'):
        return FORMS[fit['form']][2](np.asarray(fit['theta']), X, s, basis)


def predict_aug(fit, s, pts, basis):
    m = model_ce(fit, s, *(np.array([p[k] for p in pts]) for k in ('N', 'D', 'U')), basis=basis)
    return m + offsets_design(pts) @ np.asarray(fit['offsets'])


# -------------------------------------------------------------------------------------------------------- validation
LOGO = {
    '217M': lambda p: p['size'] == '217M',
    'R>=6.6': lambda p: p['D'] / p['U'] - 1 >= 6.6,
    'largest U': lambda p: p['U'] >= 83e6 and p['kind'] == 'control',
    '3M': lambda p: p['size'] == '3M',
}


def logo(name, pts, s, basis, theta):
    """Per held-out group (its repeated runs; fresh runs always stay in the fit): max |error| and the max of
    |error| / max(TOL, 2 SE) (SE = seed noise of the run's mean)."""
    out, ratio = {}, {}
    for g, sel in LOGO.items():
        held = [p for p in pts if sel(p) and p['kind'] == 'control']
        if not held:
            continue
        train = [p for p in pts if not (sel(p) and p['kind'] == 'control')]
        f = fit_form(name, train, s, basis, init=theta)
        err = np.abs(predict_aug(f, s, held, basis) - np.array([p['ce'] for p in held]))
        out[g] = float(err.max())
        ratio[g] = float(max(e / max(TOL, 2 * p['sigma'] / math.sqrt(p['n'])) for e, p in zip(err, held)))
    return out, ratio


def checks(fit, s, iso):
    """Out-of-sample on the iso footing: #26 - #24, #34, musicar-L (epoch basis)."""
    out = {}
    if '26' in iso and '24' in iso:
        p = model_ce(fit, s, 41981440, 160006144, U_Q)[0] - model_ce(fit, s, 41981440, 160006144, 1e15)[0]
        out['#26-#24'] = dict(pred=float(p), meas=round(iso['26']['ce'] - iso['24']['ce'], 4))
    if '34' in iso:
        r = iso['34']
        out['#34'] = dict(pred=float(model_ce(fit, s, r['n_ne'], r['notes'], U_POOL)[0]), meas=r['ce'],
                          fresh=float(surf(s, r['n_ne'], r['notes'])))
    m = MUSICAR_L
    out['musicar-L'] = dict(pred=float(model_ce(fit, s, m['N'], m['D'], m['U'])[0]), meas=m['ce'],
                            fresh=float(surf(s, m['N'], m['D'])))
    return out


def passes_cap(pts, turns, N, U):
    """Passes limit at N/U: the smaller of 1.5 x the most passes fitted at a comparable N/U (within a factor 2) and
    half the earliest turn-up among runs with N/U up to max(2 N/U, 1.05) (a turn-up at a lower N/U binds every larger
    one). dict(cap, fitted, turn_up); None where nothing applies."""
    x = N / U
    got = [p['D'] / p['U'] for p in pts if p['kind'] == 'control' and x / 2 <= p['N'] / p['U'] <= 2 * x]
    t = [q['passes'] for q in turns if q['nu'] <= max(2 * x, 1.05)]
    fitted, turn = (1.5 * max(got) if got else None), (0.5 * min(t) if t else None)
    vals = [v for v in (fitted, turn) if v is not None]
    return dict(cap=min(vals) if vals else None, fitted=fitted, turn_up=turn)


# ------------------------------------------------------------------------------------------------------- predictions
def frontier(fit, s, C, U, eps=0.01):
    """Data-constrained optimum at budget C over U unique notes: N*, D*, passes, L*, the eps band of N."""
    N = np.logspace(6, 12, 600)
    L = model_ce(fit, s, N, C / (6 * N), U)
    i = int(np.nanargmin(L))
    band = N[L <= L[i] + eps]
    return dict(N=float(N[i]), D=float(C / (6 * N[i])), passes=float(C / (6 * N[i]) / U), L=float(L[i]),
                band=[float(band.min()), float(band.max())], edge=bool(i in (0, len(N) - 1)))


def floor(fit, s, U):
    """Lowest CE over N and D for U unique notes (grid), with the N there; edge = still falling at N = 1e12."""
    N = np.logspace(7, 12, 120)[:, None]
    D = U * np.logspace(0, 3, 120)[None, :]
    L = model_ce(fit, s, N, D, U)
    i, j = np.unravel_index(np.nanargmin(L), L.shape)
    return dict(L=float(L[i, j]), N=float(N[i, 0]), D=float(D[0, j]), edge=bool(i == L.shape[0] - 1))


def planned_rows(s):
    """(label, N, D) of the planned runs on the fitted surface s."""
    return [(k, N, d_opt(s, N) if r is None else r * N) for k, (N, r) in PLANNED.items()]


def key_numbers(fits, s):
    """The numbers the bootstrap tracks: per form, per model at U 8.84B and 4.5B: penalty at the compute-optimal D;
    the planned rows' penalties; N* at the XXL budget (6 N D at the compute-optimal D) over 8.84B."""
    out = {}
    c_xxl = 6 * N_XXL * d_opt(s, N_XXL)
    for name, f in fits.items():
        row = {}
        for m, N in MODELS.items():
            D = d_opt(s, N)
            fresh = float(surf(s, N, D))
            for U in (4.5e9, U_C):
                row[f'{m} pen U{U / 1e9:g}'] = float(model_ce(f, s, N, D, U)[0]) - fresh
        for lab, N, D in planned_rows(s):
            row[f'{lab} pen U{U_C / 1e9:g}'] = float(model_ce(f, s, N, D, U_C)[0] - surf(s, N, D))
        row[f'N* XXL budget U{U_C / 1e9:g}'] = frontier(f, s, c_xxl, U_C)['N']
        out[name] = row
    for m, N in MODELS.items():
        out.setdefault('surface', {})[f'{m} D_opt'] = d_opt(s, N)
        out['surface'][f'{m} fresh CE'] = float(surf(s, N, d_opt(s, N)))
    return out


def bootstrap(iso_pts, pts, s, fits, basis, draws, seed=0):
    rng = np.random.default_rng(seed)
    y_iso = np.array([p['ce'] for p in iso_pts])
    y_iso_fit = surf(s, *(np.array([p[k] for p in iso_pts]) for k in ('N', 'D')))
    sd_iso = max(s['rmse'], 0.003)
    y_aug = {n: predict_aug(f, s, pts, basis) for n, f in fits.items()}
    sd_aug = np.array([p['sigma'] / math.sqrt(p['n']) for p in pts])
    rows = []
    for _ in range(draws):
        ys = y_iso_fit + rng.normal(0, sd_iso, len(y_iso))
        sb = fit_surface([p['N'] for p in iso_pts], [p['D'] for p in iso_pts], ys)
        noise = rng.normal(0, 1, len(pts)) * sd_aug
        fb = {n: fit_form(n, pts, sb, basis, init=f['theta'], y=y_aug[n] + noise) for n, f in fits.items()}
        rows.append(key_numbers(fb, sb))
    q = {}
    for name in rows[0]:
        for key in rows[0][name]:
            v = np.array([r[name][key] for r in rows])
            q.setdefault(name, {})[key] = [float(x) for x in np.nanpercentile(v, [10, 50, 90])]
    return q


def recipe_layer(runs):
    """BEST vs control cells: penalties P = control - fresh, P' = BEST - fresh_BEST, their ratio, the fresh cost."""
    by = {}
    for r in runs.values():
        if r.get('failed') or r.get('ce') is None:
            continue
        key = (r.get('size', '42M'), r['arm'], round(r.get('frac', 1.0), 4), round(r['notes'], -6))
        by.setdefault(key, []).append(r['ce'])
    m = {k: float(np.mean(v)) for k, v in by.items()}
    out = []
    for size in N_AUG:
        for frac in (0.25, 0.125):
            g = lambda arm, f: m.get((size, arm, f, 160e6))  # noqa: E731
            c, b, f0, fb = g('control', frac), g('BEST', frac), g('fresh', 1.0), g('fresh_BEST', 1.0)
            if c is None or b is None:
                continue
            row = dict(size=size, frac=frac, control=c, BEST=b, fresh=f0, fresh_BEST=fb, d_best=b - c)
            if f0 is not None and fb is not None:
                row.update(P=c - f0, P_best=b - fb, ratio=(b - fb) / (c - f0), fresh_cost=fb - f0)
            out.append(row)
    return out


# ------------------------------------------------------------------------------------------------------------ report
def fmt_n(x):
    return f'{x / 1e9:.2f}B' if x >= 1e9 else f'{x / 1e6:.1f}M'


def report(res):
    s, L = res['surface'], []
    L.append('# Frontier fit (tools/sweeps/frontier_fit.py)\n')
    L.append(f"Fit basis of the aug runs: {res['basis']}; repeated runs fitted: "
             f"{'N/U <= %g' % res['max_nu'] if res['max_nu'] else 'all'}; predictions: epoch basis (even epochs), default recipe, iso "
             f"footing. Bootstrap: {res['draws']} draws (10 / 50 / 90%).\n")
    L.append('## Fresh surface (iso one-pass arms)\n')
    L.append(f"L = {s['E']:.3f} + {s['A']:.4g} N^-{s['a']:.3f} + {s['B']:.4g} D^-{s['b']:.3f}; RMSE {s['rmse']:.4f} over "
             f"{s['n']} arms{' (E at the cap)' if s['E_capped'] else ''}.\n")
    L.append('| C | N* | D* | notes/param | L* |\n|---|---|---|---|---|')
    for C in (3e16, 1e17, 3e17, 2.3e19, 1.2e20, 3.78e20, 1e21):
        N, D = optimum(s, C)
        L.append(f'| {C:.2g} | {fmt_n(N)} | {fmt_n(D)} | {D / N:.1f} | {float(surf(s, N, D)):.3f} |')
    L.append('\n## Repetition forms (aug runs)\n')
    L.append(f"{res['n_aug']} seed-mean runs ({res['n_rep']} repeated). LOGO = max |error| on the held-out group; "
             f'a form passes if every group is within {TOL}.\n')
    L.append('| form | parameters | RMSE | AICc | ' + ' | '.join(f'{g}: err (/limit)' for g in LOGO) +
             ' | pass 0.02 / noise | #26-#24 (0.054) | #34 | musicar-L |')
    L.append('|' + '---|' * (8 + len(LOGO)))
    for name, f in res['fits'].items():
        th = ', '.join(f'{p} {v:.3g}' for p, v in zip(FORMS[name][0], f['theta']))
        lg = ' | '.join(f"{f['logo'].get(g, float('nan')):.3f} ({f['logo_ratio'].get(g, float('nan')):.2f})" for g in LOGO)
        ck = f['checks']
        L.append(f"| {name} | {th} | {f['rmse']:.4f} | {f['aicc']:.1f} | {lg} | {'yes' if f['pass'] else 'no'} / {'yes' if f['pass_noise'] else 'no'} | "
                 f"{ck.get('#26-#24', {}).get('pred', float('nan')):.3f} | {ck.get('#34', {}).get('pred', float('nan')):.3f}"
                 f" ({ck.get('#34', {}).get('meas', float('nan')):.3f}) | {ck['musicar-L']['pred']:.3f} (1.641) |")
    why = lambda p: ' (turn-up)' if p.get('turn_up') else ' (N/U above the regime)' if p.get('out_of_regime') else ' (store bridge)'  # noqa: E731
    out = [p['tag'] + why(p) for p in res['aug'] if not p['use']]
    L.append(f"\nSeed noise per size: {', '.join(f'{k} {v:.3f}' for k, v in res['sigma'].items())}. Not fitted: "
             f"{'; '.join(out) or 'none'}.\n")
    L.append(f"Passing forms ({res['rule']}): {', '.join(res['passing']) or 'none'}. musicar-L on the fresh surface alone: "
             f"{res['fits'][next(iter(res['fits']))]['checks']['musicar-L']['fresh']:.3f}.\n")
    L.append('## Request table: compute-optimal runs and the repeat penalty\n')
    L.append('D = compute-optimal notes on the fresh surface. Penalty = repeated CE - fresh CE at the same N and D '
             f"(even epochs, default recipe): best-AICc form ({res['best']}), then min..max over the forms used.\n")
    L.append('Cap = passes limit at that N/U (see Turn-ups); OVER = the run needs more passes than the cap.\n')
    L.append('| run | N | D | C (6ND) | fresh CE | ' + ' | '.join(f'U {u / 1e9:g}B: passes (cap) / penalty' for u in
                                                            (4.5e9, U_C, 9.92e9)) + ' |')
    L.append('|---|---|---|---|---|---|---|---|')
    for row in res['request']:
        cells = []
        for u in ('4.5', f'{U_C / 1e9:g}', '9.92'):
            c = row['U'][u]
            cap = c['cap']['cap']
            over = ' OVER' if cap is not None and c['passes'] > cap else ''
            cells.append(f"{c['passes']:.1f} ({f'{cap:.1f}' if cap else '-'}){over} / {c['pen_best']:+.3f} "
                         f"({c['pen_lo']:+.3f}..{c['pen_hi']:+.3f})")
        L.append(f"| {row['run']} | {fmt_n(row['N'])} | {fmt_n(row['D'])} | {row['C']:.2g} | {row['fresh']:.3f} | "
                 + ' | '.join(cells) + ' |')
    L.append('\n## Turn-ups (held-out CE rising during training)\n')
    L.append(f'Runs whose final CE is {TURN_RISE} or more above their lowest evaluation; passes at the lowest one. '
             'They are not fitted. The cap at a target N/U is half the earliest turn-up among runs with N/U up to '
             'max(2 N/U, 1.05), and at most 1.5x the most passes fitted at a comparable N/U.\n')
    L.append('| run | N/U | passes at the lowest CE | rise by the end |\n|---|---|---|---|')
    for t in res['turnups']:
        L.append(f"| {t['tag']} | {t['nu']:.2f} | {t['passes']:.1f} | +{t['rise']:.3f} |")
    if res.get('boot'):
        L.append('\nBootstrap (10 / 50 / 90%) per passing form:\n')
        L.append('| form | quantity | 10% | 50% | 90% |\n|---|---|---|---|---|')
        for name, q in res['boot'].items():
            for k, v in q.items():
                L.append(f'| {name} | {k} | {v[0]:.4g} | {v[1]:.4g} | {v[2]:.4g} |')
    L.append('\n## Data-constrained frontier (per passing form)\n')
    L.append('| form | C | U | N* | D* | passes (cap) | L* | N band (+0.01) |\n|---|---|---|---|---|---|---|---|')
    for r in (r for r in res['frontier'] if r['C'] in (2.3e19, 1.2e20, 5e20, 1e21) and r['U'] in (4.5e9, U_C)):
        cap = f"{r['cap']:.1f}" if r['cap'] else '-'
        L.append(f"| {r['form']} | {r['C']:.2g} | {r['U'] / 1e9:g}B | {fmt_n(r['N'])}{' (edge)' if r['edge'] else ''} | "
                 f"{fmt_n(r['D'])} | {r['passes']:.1f} ({cap}){' OVER' if r['over_cap'] else ''} | {r['L']:.3f} | "
                 f"{fmt_n(r['band'][0])}-{fmt_n(r['band'][1])} |")
    L.append('\n## Floor (lowest CE for U unique notes)\n')
    L.append('| form | U | L floor | at N | at D |\n|---|---|---|---|---|')
    for r in (r for r in res['floor'] if r['U'] in (4.5e9, U_C)):
        L.append(f"| {r['form']} | {r['U'] / 1e9:g}B | {r['L']:.3f} | {fmt_n(r['N'])}{' (still falling)' if r['edge'] else ''}"
                 f" | {fmt_n(r['D'])} |")
    L.append('\n## Recipe layer (BEST = wd 1.0 + dyn + family)\n')
    L.append('| size | frac | BEST - control | P | P under BEST | ratio | fresh cost |\n|---|---|---|---|---|---|---|')
    for r in res['recipe']:
        g = lambda k: f'{r[k]:+.3f}' if r.get(k) is not None else '-'  # noqa: E731
        ratio = f"{r['ratio']:.2f}" if r.get('ratio') is not None else '-'
        L.append(f"| {r['size']} | {r['frac']} | {g('d_best')} | {g('P')} | {g('P_best')} | {ratio} | "
                 f"{g('fresh_cost')} |")
    return '\n'.join(L) + '\n'


# -------------------------------------------------------------------------------------------------------------- main
def run(basis='epoch', draws=200, seed=0, iso_path=ISO_RUNS, aug_path=AUG_RUNS, max_nu=None):
    iso_runs, aug_runs = json.loads(Path(iso_path).read_text()), json.loads(Path(aug_path).read_text())
    iso_pts, all_pts = iso_points(iso_runs), aug_points(aug_runs)
    for p in all_pts:
        if max_nu and p['kind'] == 'control' and p['N'] / p['U'] > max_nu and p['use']:
            p['use'], p['out_of_regime'] = False, True
    pts = [p for p in all_pts if p['use']]
    s = fit_surface([p['N'] for p in iso_pts], [p['D'] for p in iso_pts], [p['ce'] for p in iso_pts])
    fits = {}
    for name in FORMS:
        f = fit_form(name, pts, s, basis)
        f['logo'], f['logo_ratio'] = logo(name, pts, s, basis, f['theta'])
        f['pass'] = bool(f['logo']) and max(f['logo'].values()) <= TOL
        f['pass_noise'] = bool(f['logo']) and max(f['logo_ratio'].values()) <= 1
        f['checks'] = checks(f, s, iso_runs)
        fits[name] = f
    passing = [n for n, f in fits.items() if f['pass']] or [n for n, f in fits.items() if f['pass_noise']]
    use = {n: fits[n] for n in (passing or fits)}
    best = min(use, key=lambda n: use[n]['aicc'])
    turns = turnups(aug_runs)
    request = []
    rows = [(m, N, d_opt(s, N)) for m, N in MODELS.items()] + planned_rows(s)
    for lab, N, D in rows:
        fresh = float(surf(s, N, D))
        cells = {}
        for U in (4.5e9, U_C, 9.92e9):
            pen = {n: float(model_ce(f, s, N, D, U)[0]) - fresh for n, f in use.items()}
            cells[f'{U / 1e9:g}'] = dict(passes=D / U, pen_lo=min(pen.values()), pen_hi=max(pen.values()),
                                         pen_best=pen[best], pen=pen, cap=passes_cap(pts, turns, N, U))
        request.append(dict(run=lab, N=N, D=D, C=6 * N * D, fresh=fresh, U=cells))
    front = [dict(form=n, C=C, U=U, **frontier(f, s, C, U)) for n, f in use.items() for C in C_GRID
             for U in (2.1e9, 4.5e9, U_C, 9.92e9)]
    for r in front:
        r['cap'] = passes_cap(pts, turns, r['N'], r['U'])['cap']
        r['over_cap'] = bool(r['cap'] is not None and r['passes'] > r['cap'])
    flo = [dict(form=n, U=U, **floor(f, s, U)) for n, f in use.items() for U in (2.1e9, 4.5e9, U_C, 9.92e9)]
    boot = bootstrap(iso_pts, pts, s, use, basis, draws, seed) if draws else {}
    rule = 'pre-registered (all groups within 0.02)' if any(f['pass'] for f in fits.values()) else (
        'noise-aware (within max(0.02, 2 SE))' if passing else 'none passes: all forms shown')
    return dict(basis=basis, max_nu=max_nu, draws=draws, surface=s, iso=[p['tag'] for p in iso_pts], n_aug=len(pts),
                n_rep=sum(p['kind'] == 'control' for p in pts), aug=all_pts, fits=fits, passing=passing, rule=rule, best=best, sigma=seed_sigma(aug_runs), turnups=turns,
                request=request, frontier=front, floor=flo, boot=boot, key=key_numbers(use, s),
                recipe=recipe_layer(aug_runs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basis', default='epoch', choices=['epoch', 'poisson'])
    p.add_argument('--draws', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-nu', type=float, help='fit only repeated runs with N/U up to this (the request regime)')
    p.add_argument('--out', default=str(OUT))
    p.add_argument('--predictions', help='also write the pre-registered predictions JSON here')
    a = p.parse_args()
    res = run(a.basis, a.draws, a.seed, max_nu=a.max_nu)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = ('' if a.basis == 'epoch' else f'_{a.basis}') + (f'_nu{a.max_nu:g}' if a.max_nu else '')
    (out / f'fit{tag}.json').write_text(json.dumps(res, indent=1))
    text = report(res)
    (out / f'report{tag}.md').write_text(text)
    print(text)
    if a.predictions:
        pred = dict(written='before any run of the compute request', basis=res['basis'], surface=res['surface'],
                    passing=res['passing'], request=res['request'], key=res['key'], boot=res['boot'],
                    checks={n: f['checks'] for n, f in res['fits'].items()})
        Path(a.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(a.predictions).write_text(json.dumps(pred, indent=1))
        print(f'predictions -> {a.predictions}')


if __name__ == '__main__':
    main()
