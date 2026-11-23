"""
Self-test of minhash_full.py on synthetic stores:
    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/test_minhash_full.py
Two sources (gigamidi_train, discover_train + discover_validation) with 300 unrelated files and planted copies:
- a 1%-edited copy (J > 0.9) and a 5%-edited copy (J ~ 0.65) of an earlier file are dropped at 0.5, only the former at
  0.8 / 0.9; the earlier original is never dropped (only earlier files count);
- a group of 30 near-identical copies (more than --lags) all drop except the first, through chains;
- unrelated files never drop; best_ref always points to an earlier position;
- calibrated drops match the raw drops here (estimates far from the thresholds) and the files npz is written.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

import minhash_full as mf
from test_minhash import edit, make_file


def write_store(d, files):
    d.mkdir(parents=True)
    tok = np.concatenate([f[0] for f in files])
    prg = np.concatenate([f[1] for f in files])
    off = np.concatenate([[0], np.cumsum([len(f[0]) for f in files])]).astype(np.int64)
    tok.astype(np.uint16).tofile(d / 'tokens.u16')
    prg.astype(np.uint8).tofile(d / 'programs.u8')
    np.save(d / 'offsets.npy', off)
    (d / 'meta.json').write_text(json.dumps(dict(n_files=len(files), n_notes=int(off[-1]))))


def main():
    rng = np.random.default_rng(3)
    files = [make_file(n=int(rng.integers(300, 900)))[:2] for _ in range(300)]
    orig = files[10]
    big = files[20]
    giga = files[:100]
    disc_tr = files[100:250] + [(edit(orig[0], 0.01), orig[1]), (edit(orig[0], 0.05), orig[1])]
    disc_va = files[250:] + [(edit(big[0], 0.003), big[1]) for _ in range(30)]
    with tempfile.TemporaryDirectory(dir=os.environ.get('MINHASH_TEST_DIR')) as d:
        d = Path(d)
        write_store(d / 'cache' / 'gigamidi_train', giga)
        write_store(d / 'cache' / 'discover_train', disc_tr)
        write_store(d / 'cache' / 'discover_validation', disc_va)
        out = mf.main(['--cache', str(d / 'cache'), '--work', str(d / 'work'), '--out', str(d / 'r.json'),
                       '--workers', '2', '--chunk', '37', '--verify', '1000'])
        z = np.load(d / 'r_files.npz')
        best, ref = z['best_est'], z['best_ref']
        N = len(best)
        assert N == 100 + 152 + 80 and list(z['stores']) == ['gigamidi_train', 'discover_train', 'discover_validation']
        p1, p5 = 100 + 150, 100 + 151
        assert best[p1] > 0.85 and ref[p1] == 10, (best[p1], ref[p1])
        assert 0.5 < best[p5] < 0.8 and ref[p5] in (10, p1), (best[p5], ref[p5])
        copies = np.arange(N - 30, N)
        assert (best[copies] > 0.9).all()
        planted = set([p1, p5, *copies])
        others = np.array([i for i in range(N) if i not in planted])
        assert best[others].max() < 0.3, best[others].max()
        has = ref >= 0
        assert (ref[has] < np.flatnonzero(has)).all()
        r = out['results']['total']['thresholds']
        assert r['0.5']['raw_drop_files'] == 32 and r['0.9']['raw_drop_files'] == 31, r
        for t in r:
            assert abs(r[t]['cal_drop_files'] - r[t]['raw_drop_files']) < 1e-6, (t, r[t])
        assert out['results']['gigamidi']['thresholds']['0.5']['raw_drop_files'] == 0
        # resume: signatures are not recomputed
        done = list((d / 'work').glob('done_*.txt'))[0].read_text().split()
        assert len(done) == len(set(done))
    print('test_minhash_full: OK')


if __name__ == '__main__':
    main()
