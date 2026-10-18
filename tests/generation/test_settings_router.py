"""Settings router checks (numpy only): python tests/generation/test_settings_router.py  (or pytest tests/)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run  # noqa: E402
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM  # noqa: E402
from musicar.generation import settings_router as R  # noqa: E402


def test_prompt_features():
    tokens = np.array([[BOS_PITCH, 0, 0, 0], [60, 10, 5, 0], [64, 20, 5, 0], [36, 30, 5, 50], [53, 20, 5, 50]])
    programs = np.array([0, 0, 0, DRUM_PROGRAM, 53])
    f = dict(zip(R.FEATURES, R.prompt_features(tokens, programs)))
    assert np.isclose(f['log_nps'], np.log(4 / 2.0)) and f['drum_share'] == 0.25
    assert np.isclose(f['log_instruments'], np.log(3)) and np.isclose(f['same_onset'], 1 / 3)
    assert np.isclose(f['pitch_mean'], (60 + 64 + 53) / 3) and f['velocity'] == 20 and f['vocal_share'] == 0.25


def test_fit_and_route():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 0.3, (60, 2)), rng.normal(3, 0.3, (40, 2))])
    gmm = R.fit_gmm(x, 2, n_init=2)
    assert sorted(np.round(gmm['weights'], 1).tolist()) == [0.4, 0.6]
    r = R.responsibilities(np.array([[-3.0, -3.0], [3.0, 3.0]]), gmm)
    assert np.allclose(r.sum(1), 1) and r[0].argmax() != r[1].argmax() and r.max(1).min() > 0.99
    assert R.fit_gmm(x, 1, n_init=1)['bic'] > gmm['bic'], 'two blobs: k=2 has the lower BIC'


def test_router_json_round_trip():
    rng = np.random.default_rng(1)
    feats = np.stack([R.prompt_features(np.stack([rng.integers(40, 80, 30), rng.integers(0, 32, 30),
                                                  rng.integers(1, 9, 30), rng.integers(0, s, 30)], 1),
                                        np.zeros(30, int)) for s in [3] * 20 + [40] * 20])
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    router = dict(mu=mu, sd=sd, gmm=R.fit_gmm((feats - mu) / sd, 2, n_init=2),
                  settings=[{'temperature': 0.9}, {'temperature': 1.1}])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'router.json'
        path.write_text(R.to_json(router))
        json.loads(path.read_text())
        loaded = R.load(path)
    dense = np.stack([np.full(30, 60), np.full(30, 16), np.full(30, 4), np.ones(30, int)], 1)
    settings, k, r = R.route(loaded, dense, np.zeros(30, int))
    assert settings == router['settings'][k] and np.isclose(r.sum(), 1)


if __name__ == '__main__':
    run(globals())
