"""
Self-test of minhash.py on synthetic files:
    cd /workspace/music && PYTHONPATH=.:data_prep/datasets:data_prep/census .venv/bin/python data_prep/census/test_minhash.py
- a file and a copy with 5% of its notes' pitches changed pair at threshold 0.5 (exact Jaccard ~0.65-0.7 for 4-grams),
  a copy with 1% changed pairs at 0.8, a copy with one track removed pairs at 0.5;
- unrelated files (random, and the same rhythm with other pitches) never pair;
- the signature estimate tracks the exact Jaccard; only EARLIER files are returned as candidates;
- file_notes / shingle_sets read a tiny on-disk Store the same way as the in-memory arrays (and respect --cap).
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

import minhash as mh

rng = np.random.default_rng(7)
N_GRAM = 4


def make_file(n=1200, programs=(0, 33, 128), rhythm=None):
    """Random multi-track file as store tokens (pitch, velocity, duration, delta) + programs, onset-sorted."""
    on = np.sort(rng.integers(0, n * 3, n)) if rhythm is None else rhythm
    prog = rng.choice(np.array(programs, np.uint8), n)
    pitch = np.where(prog == 128, rng.integers(35, 60, n), rng.integers(30, 90, n))
    delta = np.diff(on, prepend=0)
    tok = np.stack([pitch, rng.integers(1, 127, n), rng.integers(1, 20, n), delta], 1).astype(np.uint16)
    return tok, prog, on


def edit(tok, frac):
    t = tok.copy()
    i = rng.choice(len(t), int(round(frac * len(t))), replace=False)
    t[i, 0] = (t[i, 0].astype(int) + rng.choice([-5, -3, -2, 2, 3, 5], len(i))).clip(0, 127)
    return t


def sets_of(files):
    out = []
    for tok, prog in files:
        _, h = mh.shingles(tok, prog, np.array([len(tok)]), N_GRAM)
        out.append(np.unique(h))
    return out


def sigs_of(files):
    tok = np.concatenate([f[0] for f in files])
    prog = np.concatenate([f[1] for f in files])
    L = np.array([len(f[0]) for f in files])
    fid, h = mh.shingles(tok, prog, L, N_GRAM)
    S, cnt = mh.oph(fid, h, len(files))
    assert (S[cnt > 0] < (1 << 32)).all() and (S >= 0).all() and (S[cnt == 0] == mh.EMPTY).all()
    return (S & 0xFFFFFFFF).astype(np.uint32), cnt


def main():
    base, prog, on = make_file()
    drop_track = prog != 33
    files = [
        (base, prog),                                   # 0 original
        (edit(base, 0.05), prog),                       # 1 5% pitches changed
        (edit(base, 0.01), prog),                       # 2 1% changed
        (base[drop_track], prog[drop_track]),           # 3 one track removed (deltas recomputed below)
        make_file()[:2],                                # 4 unrelated
        make_file(rhythm=on)[:2],                       # 5 same onsets, other pitches
        make_file(programs=(0,))[:2],                   # 6 unrelated piano
        (base[:2], prog[:2]),                           # 7 too short for a 4-gram: no shingles
    ]
    t3 = files[3][0].copy()
    t3[:, 3] = np.diff(on[drop_track], prepend=0)
    files[3] = (t3, files[3][1])
    sets = sets_of(files)
    J = np.array([[mh.jaccard(a, b) for b in sets] for a in sets])
    print('exact Jaccard vs file 0:', np.round(J[0], 3))
    assert 0.55 < J[0, 1] < 0.8, J[0, 1]
    assert J[0, 2] >= 0.85 and J[0, 3] >= 0.5
    assert J[0, 4] < 0.05 and J[0, 5] < 0.05 and J[0, 6] < 0.05
    assert len(sets[7]) == 0

    S, cnt = sigs_of(files)
    assert cnt[7] == 0 and (cnt[:7] > 0).all()
    est = (S[0] == S).mean(1)
    print('estimated Jaccard vs file 0:', np.round(est, 3))
    assert np.abs(est[:7] - J[0, :7]).max() < 0.12

    # LSH: query every file against all earlier ones (signature rows as (128, N))
    sigT = np.ascontiguousarray(S.T)
    valid = cnt > 0
    q = np.arange(len(files))
    ql, ref, capped = mh.lsh_candidates(lambda a, b: sigT[a:b], valid, q, 42, 3, 300)
    assert (ref < q[ql]).all(), 'only earlier files may be candidates'
    pairs = set(zip(ql.tolist(), ref.tolist()))
    print('candidate pairs:', sorted(pairs))
    for a in (1, 2, 3):
        assert (a, 0) in pairs, f'file {a} must pair with file 0'
    for a in (4, 5, 6, 7):
        assert not any(x == a or y == a for x, y in pairs if J[x, y] < 0.3), f'file {a} must not pair'
    best = {a: max((J[a, r] for x, r in pairs if x == a), default=0.0) for a in q}
    assert best[1] >= 0.5 and best[2] >= 0.8 and best[3] >= 0.5
    assert best[4] == 0 and best[5] == 0 and best[6] == 0 and best[7] == 0

    # Rows outside the query set never appear, and identical copies pair at J = 1
    S2 = np.concatenate([S, S[[4]]])
    v2 = np.concatenate([valid, [True]])
    ql2, ref2, _ = mh.lsh_candidates(lambda a, b: np.ascontiguousarray(S2.T)[a:b], v2, np.array([len(S2) - 1]), 42, 3,
                                     300)
    assert list(ref2) == [4], ref2

    # on-disk Store round trip + cap
    from common import Store
    with tempfile.TemporaryDirectory(dir=os.environ.get('MINHASH_TEST_DIR', '/workspace/agents/minhash')) as d:
        tok = np.concatenate([f[0] for f in files])
        prg = np.concatenate([f[1] for f in files])
        L = np.array([len(f[0]) for f in files])
        off = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
        tok.astype(np.uint16).tofile(Path(d) / 'tokens.u16')
        prg.astype(np.uint8).tofile(Path(d) / 'programs.u8')
        np.save(Path(d) / 'offsets.npy', off)
        (Path(d) / 'meta.json').write_text(json.dumps(dict(n_files=len(files), n_notes=int(off[-1]))))
        st = Store(d)
        ss = mh.shingle_sets(st, [3, 0, 1], 10**9, N_GRAM)
        for k, i in enumerate([3, 0, 1]):
            assert np.array_equal(ss[k], sets[i])
        capped_sets = mh.shingle_sets(st, [0], 600, N_GRAM)[0]
        ref_cap = sets_of([(files[0][0][:600], files[0][1][:600])])[0]
        assert np.array_equal(capped_sets, ref_cap) and len(capped_sets) < len(sets[0])
    print('test_minhash: OK')


if __name__ == '__main__':
    main()
