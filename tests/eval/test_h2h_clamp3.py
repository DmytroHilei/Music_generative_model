"""h2h_clamp3 maths and patching (CPU, no model): python tests/run.py h2h_clamp3  (or pytest tests/)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval/compare'))

import h2h_clamp3 as hc  # noqa: E402


def test_segment_spans_as_extract_feature():
    assert hc.segment_spans(100, 512) == ([(0, 100)], [100])
    assert hc.segment_spans(512, 512) == ([(0, 512)], [512])
    assert hc.segment_spans(513, 512) == ([(0, 512), (1, 513)], [512, 1])  # the last one overlaps, its weight is 1
    assert hc.segment_spans(1024, 512) == ([(0, 512), (512, 1024)], [512, 512])
    assert hc.segment_spans(1300, 512) == ([(0, 512), (512, 1024), (788, 1300)], [512, 512, 276])


def test_mtf_patches_merge_runs_of_one_type():
    mtf = 'ticks_per_beat 480\nnote_on 0 60 64 0\nnote_on 0 64 64 0\nnote_off 0 60 64 12\nend_of_track 0'
    rows = hc.mtf_patches(mtf)
    text = [''.join(chr(c) for c in r if c > 3) for r in rows]
    assert text == ['', 'ticks_per_beat 480\n', 'note_on 0 60 64 0\t0 64 64 0\n', 'note_off 0 60 64 12\n',
                    'end_of_track 0', '']
    assert all(len(r) == hc.PATCH_SIZE for r in rows)
    assert rows[0] == [1] * hc.PATCH_SIZE and rows[-1] == [1] + [2] * (hc.PATCH_SIZE - 1)
    assert rows[1][:2] == [1, ord('t')] and rows[1][len('ticks_per_beat 480\n') + 1] == 2  # BOS + bar + EOS


def test_mtf_patches_a_full_patch_starts_a_new_one():
    line = 'note_on 0 60 64 0\n'
    text = [''.join(chr(c) for c in r if c > 3) for r in hc.mtf_patches('ticks_per_beat 480\n' + line * 12)[1:-1]]
    assert text[0] == 'ticks_per_beat 480\n' and len(text) == 4  # 5 + 5 + 2 messages: 18 + 10 per merged message
    assert all(len(t) <= hc.PATCH_SIZE - 2 for t in text)
    assert sum(t.count('60 64 0') for t in text) == 12


def test_mrr_and_contrast_on_known_cosines():
    music = np.eye(3)
    captions = np.array([[0.6, 0.8, 0], [0, 1, 0], [0, 0.6, 0.8]])
    others = np.array([[1, 2], [0, 2], [0, 1]])
    s = hc.file_scores(music, captions, np.arange(3), others)
    # cos rows: file 0 = (.6, 0, 0), file 1 = (.8, 1, .6), file 2 = (0, 0, .8): the own caption always leads
    assert np.allclose(s['own'], [0.6, 1.0, 0.8])
    assert np.allclose(s['contrast'], [0.6, 1.0 - 0.7, 0.8])
    assert np.allclose(s['mrr'], [1, 1, 1])
    s = hc.file_scores(music, captions, np.array([1, 2, 0]), others)  # the same files on another prompt's caption
    assert np.allclose(s['own'], [0, 0.6, 0])
    assert np.allclose(s['mrr'], [1 / 3, 1 / 3, 1 / 3])  # a tie counts against the own caption


def test_per_prompt_means_and_missing_prompts():
    v = hc.per_prompt(np.array([1.0, 3.0, 5.0]), np.array([0, 0, 2]), 4)
    assert v[0] == 2 and v[2] == 5 and np.isnan(v[1]) and np.isnan(v[3])


def test_comparators_and_derangement():
    others = hc.comparators(50, 10, 3)
    assert others.shape == (50, 10) and (others != np.arange(50)[:, None]).all()
    assert all(len(set(r.tolist())) == 10 for r in others)
    assert (others == hc.comparators(50, 10, 3)).all()
    assert hc.comparators(5, 100, 3).shape == (5, 4)  # text: all the others of the language
    swap = hc.deranged(50, 3)
    assert sorted(swap.tolist()) == list(range(50)) and (swap != np.arange(50)).all()


def test_bootstrap_shapes_and_shares():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 0.1, 40), rng.normal(0, 0.1, 40)
    res = hc.bootstrap(dict(a=a, b=b, same=a.copy()), 2000, 1)
    assert set(res['rows']) == {'a', 'b', 'same'} and set(res['rows']['a']) == {'mean', 'lo', 'hi'}
    assert res['rows']['a']['lo'] < res['rows']['a']['mean'] < res['rows']['a']['hi']
    assert res['wins']['a']['b'] == 1 and res['wins']['b']['a'] == 0
    assert res['wins']['a']['same'] == 0 and res['wins']['same']['a'] == 0  # paired: equal rows never beat each other
    assert res == hc.bootstrap(dict(a=a, b=b, same=a.copy()), 2000, 1)


def test_bootstrap_skips_missing_units():
    a = np.array([1.0, np.nan, 3.0, np.nan])
    b = np.array([2.0, 2.0, np.nan, np.nan])  # the last unit is nobody's
    res = hc.bootstrap(dict(a=a, b=b), 500, 1)
    assert res['rows']['a']['mean'] == 2 and res['rows']['b']['mean'] == 2
    assert np.isfinite(res['rows']['a']['lo'])


def exact_gaussian(mean, cov, n, seed):
    """n samples whose sample mean and covariance are exactly mean and cov."""
    x = np.random.default_rng(seed).normal(size=(n, len(mean)))
    x -= x.mean(0)
    w, v = np.linalg.eigh(np.cov(x, rowvar=False))
    return (x @ v / np.sqrt(w)) @ np.linalg.cholesky(cov).T + mean


def test_frechet_on_known_gaussians():
    x = exact_gaussian(np.array([3.0, 0.0]), np.diag([1.0, 4.0]), 200, 0)
    y = exact_gaussian(np.array([0.0, 0.0]), np.diag([1.0, 1.0]), 300, 1)
    assert abs(hc.frechet_to(y)(x) - (9 + 0 + 1)) < 1e-9  # ||dm||^2 + (sqrt(4) - sqrt(1))^2
    assert abs(hc.frechet_to(x)(y) - 10) < 1e-9
    assert abs(hc.frechet_to(x)(x)) < 1e-9
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])  # not commuting with diag(1, 4): the closed form of 2-d root traces
    a = exact_gaussian(np.zeros(2), cov, 200, 2)
    b = exact_gaussian(np.zeros(2), np.diag([1.0, 4.0]), 200, 3)
    product = cov @ np.diag([1.0, 4.0])
    root = np.sqrt(np.trace(product) + 2 * np.sqrt(np.linalg.det(product)))
    assert abs(hc.frechet_to(b)(a) - (np.trace(cov) + 5 - 2 * root)) < 1e-9


def test_frechet_boot_centres_each_row_on_its_value():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(400, 8))
    near, far = rng.normal(size=(100, 8)), rng.normal(1, 1, (100, 8))
    res = hc.frechet_boot(dict(near=near, far=far), ref, 200, 1)
    distance = hc.frechet_to(ref)
    assert res['rows']['near']['mean'] == distance(near) and res['rows']['far']['mean'] == distance(far)
    for row in res['rows'].values():
        assert row['lo'] < row['mean'] < row['hi']
    assert res['wins']['near']['far'] == 1 and res['wins']['far']['near'] == 0  # lower is better


def test_disjoint_finds_the_same_file():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(20, 16))
    real = np.concatenate([pool[3:5], rng.normal(size=(3, 16))])
    mask = hc.disjoint(pool, real)
    assert mask.sum() == 18 and not mask[3] and not mask[4]
