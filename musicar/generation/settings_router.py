"""
Generation settings chosen from the seed song: features of the prompt -> soft clusters (Gaussian mixture, EM) ->
the settings of the most likely cluster. Fitted and its settings table chosen by eval/settings/cluster_search.py; numpy only,
so it runs on the demo's CPU.

    router = load('results/multi_router/router.json')
    settings = route(router, prompt_tokens, prompt_programs)     # (T, 4) and (T,) arrays of the seed's notes

Features (from the prompt's notes, BOS/EOS rows dropped): log notes/s, drum share, log instrument count, same-onset
share (chords), mean and std of the pitched notes' pitch, mean velocity bin, vocal share (GM 52-54: the
transcriber's voice track). Standardised with the fit set's mean / std.
"""

import json
import math

import numpy as np

from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM

FEATURES = ('log_nps', 'drum_share', 'log_instruments', 'same_onset', 'pitch_mean', 'pitch_std', 'velocity',
            'vocal_share')
VOCALS = {52, 53, 54}


def prompt_features(tokens, programs, dt_seconds=0.02):
    """Feature vector (len(FEATURES),) of one prompt: tokens (T, 4) = pitch, velocity, duration, delta time bins;
    programs (T,)."""
    tokens, programs = np.asarray(tokens, dtype=np.int64), np.asarray(programs, dtype=np.int64)
    keep = tokens[:, 0] < BOS_PITCH
    tokens, programs = tokens[keep], programs[keep]
    n = len(tokens)
    secs = max(tokens[:, 3].sum() * dt_seconds, 0.5)
    pitched = tokens[programs != DRUM_PROGRAM, 0]
    return np.array([math.log(n / secs), float(np.mean(programs == DRUM_PROGRAM)),
                     math.log(len(set(programs.tolist()))), float(np.mean(tokens[1:, 3] == 0)) if n > 1 else 0.0,
                     float(pitched.mean()) if len(pitched) else 60.0,
                     float(pitched.std()) if len(pitched) > 1 else 0.0, float(tokens[:, 1].mean()),
                     float(np.mean(np.isin(programs, list(VOCALS))))])


def _log_gauss(x, mean, cov):
    """(N,) log density of N(mean, cov) at the rows of x."""
    d = x.shape[1]
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, (x - mean).T)
    return -0.5 * (z ** 2).sum(0) - np.log(np.diag(chol)).sum() - 0.5 * d * math.log(2 * math.pi)


def _log_joint(x, gmm):
    return np.stack([math.log(w) + _log_gauss(x, m, c)
                     for w, m, c in zip(gmm['weights'], gmm['means'], gmm['covs'])], 1)


def _logsumexp(a):
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def fit_gmm(x, k, reg=1e-2, n_init=8, iters=300, tol=1e-6, seed=0):
    """Full-covariance Gaussian mixture by EM (soft assignments), k-means++ starts, the best of n_init by
    log-likelihood. x = standardised features (N, D); reg added to every covariance's diagonal (binary-like features
    such as drum share are constant inside a cluster). Returns {weights, means, covs, loglik, bic}."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    best = None
    for _ in range(n_init):
        centers = [x[rng.integers(n)]]
        for _ in range(k - 1):  # k-means++
            d2 = np.min([((x - c) ** 2).sum(1) for c in centers], 0)
            centers.append(x[rng.choice(n, p=d2 / d2.sum())])
        resp = np.eye(k)[np.argmin([((x - c) ** 2).sum(1) for c in centers], 0)]
        prev = -np.inf
        for _ in range(iters):
            nk = resp.sum(0) + 1e-9  # M step
            means = (resp.T @ x) / nk[:, None]
            covs = [((resp[:, j, None] * (x - means[j])).T @ (x - means[j])) / nk[j] + reg * np.eye(d)
                    for j in range(k)]
            gmm = dict(weights=(nk / n).tolist(), means=means, covs=covs)
            lj = _log_joint(x, gmm)  # E step
            ll = _logsumexp(lj)
            resp = np.exp(lj - ll[:, None])
            if ll.sum() - prev < tol * n:
                break
            prev = ll.sum()
        if best is None or ll.sum() > best['loglik']:
            best = dict(gmm, loglik=float(ll.sum()))
    n_params = (k - 1) + k * d + k * d * (d + 1) // 2
    best['bic'] = float(-2 * best['loglik'] + n_params * math.log(n))
    return best


def responsibilities(x, gmm):
    """(N, k) posterior cluster probabilities of the standardised rows of x."""
    lj = _log_joint(x, gmm)
    return np.exp(lj - _logsumexp(lj)[:, None])


def standardise(f, router):
    return (np.atleast_2d(f) - np.array(router['mu'])) / np.array(router['sd'])


def to_json(router):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v
    return json.dumps(conv(router), indent=1)


def load(path):
    router = json.loads(open(path).read())
    g = router['gmm']
    g['means'] = np.array(g['means'])
    g['covs'] = [np.array(c) for c in g['covs']]
    return router


def route(router, tokens, programs):
    """(settings dict, cluster, responsibilities) for one prompt: the settings of its most likely cluster."""
    r = responsibilities(standardise(prompt_features(tokens, programs), router), router['gmm'])[0]
    k = int(r.argmax())
    return dict(router['settings'][k]), k, r
