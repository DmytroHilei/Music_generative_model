"""
Per-file hashes and shape stats of token stores, in store order, into census/hashes/<out>.npz.

    PYTHONPATH=.:data_prep/datasets python data_prep/census/hash_store.py --stores gigamidi_train,gigamidi_validation,gigamidi_test --out gigamidi

Arrays (one entry per file, stores concatenated in the given order):
    store (uint8 index into `stores`), idx (index in its store), n_notes, n_programs, n_drum_notes, n_pitched_programs,
    n_piano_notes (notes with GM program 0-7, the piano family), exact, loose, loose_t (uint64).
    Stores without programs.u8 are piano: programs = 0. --workers 1 runs in this process (no pool).
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from common import CACHE, DRUMS, HASHES, Store, all_hashes

_s = {}


def _init(path):
    _s['store'] = Store(path)


def _work(rng):
    st = _s['store']
    out = []
    for i in range(*rng):
        tokens, programs = st.file(i)
        progs = np.unique(programs)
        out.append((len(tokens), len(progs), int((programs == DRUMS).sum()), int((progs != DRUMS).sum()),
                    int((programs < 8).sum()), *all_hashes(tokens, programs)))
    return out


def hash_store(path, workers, chunk=2000):
    n = len(Store(path))
    ranges = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    rows = []
    if workers <= 1:  # in-process
        _init(path)
        for rng in tqdm(ranges, desc=f'hashing {path.name}'):
            rows.extend(_work(rng))
    else:
        with Pool(workers, initializer=_init, initargs=(path,)) as pool:
            for part in tqdm(pool.imap(_work, ranges), total=len(ranges), desc=f'hashing {path.name}'):
                rows.extend(part)
    a = np.array(rows, dtype=np.uint64).reshape(-1, 8)
    return a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stores', required=True, help='comma-separated store dir names under data/cache')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache-dir', default=str(CACHE))
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--out-dir', default=str(HASHES))
    args = ap.parse_args(argv)
    names = args.stores.split(',')
    stats = ('n_notes', 'n_programs', 'n_drum_notes', 'n_pitched_programs', 'n_piano_notes')
    parts = {k: [] for k in ('store', 'idx', *stats, 'exact', 'loose', 'loose_t')}
    for si, name in enumerate(names):
        a = hash_store(Path(args.cache_dir) / name, args.workers)
        parts['store'].append(np.full(len(a), si, np.uint8))
        parts['idx'].append(np.arange(len(a), dtype=np.int64))
        for j, k in enumerate(stats):
            parts[k].append(a[:, j].astype(np.int64))
        for j, k in enumerate(('exact', 'loose', 'loose_t')):
            parts[k].append(a[:, len(stats) + j])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{args.out}.npz'
    np.savez(out, stores=np.array(names), **{k: np.concatenate(v) for k, v in parts.items()})
    print(f'{out}: {sum(len(x) for x in parts["idx"]):,} files')


if __name__ == '__main__':
    main()
