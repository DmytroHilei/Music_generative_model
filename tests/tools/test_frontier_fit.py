"""tools/sweeps/frontier_fit.py on planted data: the surface and a repetition form are recovered, the effective-data
function matches plan_gates.dprime, the optimiser, and one run() on synthetic runs.json files.
Run: python tests/tools/test_frontier_fit.py (or pytest tests/)."""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helpers import run  # noqa: E402
from tools.sweeps import frontier_fit as ff  # noqa: E402
from tools.sweeps import plan_gates  # noqa: E402

S0 = dict(E=1.5, A=2600.0, a=0.53, B=2.6e4, b=0.56)
THETA = [math.log(4), 0.5, -0.5]          # a0wg: R0 4, w 0.5, g -0.5
OFFS = {'replay': 0.03, 'pool': 0.01}     # aug footing offsets, plus 0.01 per ln(N / 42M)
GRID_N = [5.9e6, 7.5e6, 10.4e6, 14.5e6, 19.8e6, 27.2e6, 35.7e6, 48.2e6, 64.8e6, 84.9e6]


def iso_runs(noise=0.002, seed=0):
    rng, runs = np.random.default_rng(seed), {}
    for C in (3e16, 1e17, 3e17):
        for N in GRID_N:
            D = C / (6 * N)
            if 2.5e8 <= D <= 1.9e9:
                runs[f'{C:.0e}_{N:.0f}'] = dict(n_ne=N, notes=D, ce=float(ff.surf(S0, N, D) + rng.normal(0, noise)),
                                                lr=1e-3, seed=1337, batch=16, store='data/cache/gigamidi_pool',
                                                extra=[], C=C)
    return runs


def aug_truth(N, D, U, store):
    f = {'form': 'a0wg', 'theta': THETA}
    return float(ff.model_ce(f, S0, N, D, U)[0]) + OFFS[store] + 0.01 * math.log(N / 42e6)


def aug_runs(noise=0.002, seed=0):
    rng, runs = np.random.default_rng(seed), {}
    cells = [('fresh', 1.0, None, 160e6), ('fresh_census', 1.0, None, 160e6), ('fresh_census', 1.0, None, 640e6)]
    cells += [('control', f, u, D) for f, u in ((0.0625, 10.2e6), (0.125, 21e6), (0.25, 41.48e6), (0.5, 83.2e6))
              for D in (160e6, 320e6)]
    for size, N in ff.N_AUG.items():
        for arm, frac, u, D in cells:
            U = u or (ff.U_POOL if arm == 'fresh_census' else ff.U_STORE)
            store = 'pool' if arm == 'fresh_census' else 'replay'
            for s in (1337, 2):
                runs[f'{size}_{arm}_{frac}_{D:.0f}_{s}'] = dict(
                    arm=arm, size=size, frac=frac, notes=D, iters=int(D / 32768), unique_notes=u, seed=s,
                    ce=aug_truth(N, D, U, store) + float(rng.normal(0, noise)))
    return runs


def test_nelder_mead():
    x, v = ff.nelder_mead(lambda t: (t[0] - 1.5) ** 2 + 3 * (t[1] + 0.5) ** 2, [0.0, 0.0])
    assert abs(x[0] - 1.5) < 1e-3 and abs(x[1] + 0.5) < 1e-3 and v < 1e-6


def test_dprime_matches_plan_gates():
    for D, U, R in ((160e6, 41.48e6, 4.0), (640e6, 41.48e6, 9.0), (160e6, 167.9e6, 4.0), (35e9, 8.8e9, 5.0)):
        for basis in ('epoch', 'poisson'):
            a = float(ff.dprime(D, U, R, basis))
            b = plan_gates.dprime(D, U, R, basis)
            assert abs(a - b) / b < 1e-9, (D, U, R, basis, a, b)
    assert float(ff.dprime(1e8, 2e8, 4.0, 'epoch')) == 1e8
    assert abs(float(ff.dprime(4e8, 1e8, 1e6, 'epoch')) - 4e8) / 4e8 < 1e-5


def test_surface_recovered():
    pts = ff.iso_points(iso_runs())
    s = ff.fit_surface([p['N'] for p in pts], [p['D'] for p in pts], [p['ce'] for p in pts])
    assert s['rmse'] < 0.004, s
    for C in (1e17, 1e21):
        n_fit, n_true = ff.optimum(s, C)[0], ff.optimum(S0, C)[0]
        assert abs(n_fit / n_true - 1) < 0.15, (C, n_fit, n_true)
    N, D = 1.8e9, 35e9
    assert abs(float(ff.surf(s, N, D) - ff.surf(S0, N, D))) < 0.02


def test_form_recovered():
    pts = [p for p in ff.aug_points(aug_runs(noise=0.0005)) if p['use']]
    f = ff.fit_form('a0wg', pts, S0, 'epoch')
    assert abs(f['theta'][0] - THETA[0]) < 0.1 and abs(f['theta'][1] - THETA[1]) < 0.05, f['theta']
    assert abs(f['theta'][2] - THETA[2]) < 0.05, f['theta']
    assert abs(f['offsets'][0] - 0.03) < 0.005 and abs(f['offsets'][2] - 0.01) < 0.003, f['offsets']
    assert ff.fit_form('a0', pts, S0, 'epoch')['chi2'] > 20 * f['chi2']


def test_turnups_and_cap():
    runs = {
        'climb': dict(arm='control', size='6M', frac=0.0625, notes=320e6, iters=9766, unique_notes=10.2e6, seed=1,
                      ce=3.65, curve=[[0, 20.0], [2441, 3.9], [4882, 3.6], [7323, 3.54], [9764, 3.65]]),
        'flat': dict(arm='control', size='6M', frac=0.25, notes=640e6, iters=19531, unique_notes=41.48e6, seed=1,
                     ce=2.64, curve=[[0, 20.0], [9764, 2.9], [19528, 2.641], [19530, 2.64]]),
        'big': dict(arm='control', size='108M', frac=0.125, notes=160e6, iters=4883, unique_notes=20.95e6, seed=1,
                    ce=3.2, curve=[[0, 20.0], [3660, 3.06], [4880, 3.2]]),
    }
    t = ff.turnups(runs)
    assert [x['size'] for x in t] == ['108M', '6M'], t
    six = t[1]
    assert abs(six['passes'] - 7324 * 320e6 / 9766 / 10.2e6) < 1e-6 and abs(six['rise'] - 0.11) < 1e-9
    pts = [dict(kind='control', N=5.88e6, U=41.48e6, D=640e6)]
    c = ff.passes_cap(pts, t, 1.8e9, 8.8e9)       # N/U 0.2: the 6M turn-up (0.58 <= 1.05) binds, the 108M one not
    assert abs(c['turn_up'] - six['passes'] / 2) < 1e-9 and c['cap'] == min(c['fitted'], c['turn_up']), c
    c = ff.passes_cap(pts, t, 30e9, 8.8e9)        # N/U 3.4: the 108M turn-up (5.1 <= 6.8) binds too
    assert abs(c['turn_up'] - t[0]['passes'] / 2) < 1e-9 and c['fitted'] is None and c['cap'] == c['turn_up'], c
    assert ff.passes_cap([], [], 1e9, 8.8e9)['cap'] is None


def test_run_on_synthetic_files():
    with tempfile.TemporaryDirectory() as d:
        iso, aug = Path(d) / 'iso.json', Path(d) / 'aug.json'
        iso.write_text(json.dumps(iso_runs()))
        aug.write_text(json.dumps(aug_runs()))
        res = ff.run(draws=2, iso_path=iso, aug_path=aug)
        assert res['fits']['a0wg']['pass'], res['fits']['a0wg']['logo']
        assert 'a0wg' in res['passing'] and len(res['request']) == len(ff.MODELS) + len(ff.PLANNED)
        assert res['boot'] and all(len(v) == 3 for q in res['boot'].values() for v in q.values())
        text = ff.report(res)
        assert '## Request table' in text and '| a0wg |' in text


if __name__ == '__main__':
    run(globals())
