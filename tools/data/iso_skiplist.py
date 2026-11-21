"""
Near-duplicate skip list for the iso-FLOP pool: pool files with MinHash-verified Jaccard >= --j to a scored eval file
(held-out files of >= 2,047 notes, the ones the val windows come from) or to a cont96 source (prompt + real
continuation MIDIs of results/head2head/cont96). Writes their store file indices, one per line (train.py
--one_pass_skip; isoflop.py uses data/cache/gigamidi_pool_skip.txt when it exists).

    PYTHONPATH=.:data_prep/datasets:data_prep/census .venv/bin/python tools/data/iso_skiplist.py \\
        [--pool data/cache/gigamidi_pool_train] [--workers 6] [--out data/cache/gigamidi_pool_skip.txt]

Shingles, signatures and LSH as data_prep/census/minhash.py (4-grams per instrument stream of (pitch, onset delta),
one-permutation MinHash, 42 bands x 3 rows), on the first --cap notes of every file, twice: absolute pitch, and pitch
minus the file's lowest pitched note (a transposed copy). Candidates with estimated J >= --min-est get the exact
Jaccard of their shingle sets.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'data_prep/datasets'), str(ROOT / 'data_prep/census')]
from minhash import DRUMS, band_keys, jaccard, oph, shingles  # noqa: E402
from common import Store  # noqa: E402

BANDS, ROWS = 42, 3
G = {}


def transpose(tok, prog, L):
    """Pitch minus each file's lowest pitched note (drums untouched)."""
    tok = tok.copy()
    fid = np.repeat(np.arange(len(L)), L)
    pitched = prog != DRUMS
    low = np.full(len(L), 255, np.int64)
    np.minimum.at(low, fid[pitched], tok[pitched, 0].astype(np.int64))
    tok[pitched, 0] = tok[pitched, 0] - low[fid[pitched]]
    return tok


def capped(store, idxs, cap):
    o = store.offsets
    a = o[idxs].astype(np.int64)
    L = np.minimum(o[idxs + 1].astype(np.int64) - a, cap)
    pos = np.repeat(a - np.concatenate([[0], np.cumsum(L)[:-1]]), L) + np.arange(int(L.sum()), dtype=np.int64)
    prog = np.asarray(store.programs[pos]) if store.programs is not None else np.zeros(len(pos), np.uint8)
    return np.asarray(store.tokens[pos]), prog, L


def sigs_and_sets(tok, prog, L, n=4, sets=False):
    """Per variant (abs, transposed): signatures (m, 128) and, with sets, the unique shingle arrays per file."""
    out = []
    for t in (tok, transpose(tok, prog, L)):
        fid, h = shingles(t, prog, L, n)
        S, cnt = oph(fid, h, len(L))
        if sets:
            srt = np.lexsort((h, fid))
            f2, h2 = fid[srt], h[srt]
            b = np.searchsorted(f2, np.arange(len(L) + 1))
            out.append((S, cnt, [np.unique(h2[b[j]:b[j + 1]]) for j in range(len(L))]))
        else:
            out.append((S, cnt, None))
    return out


def query_set(args):
    """Queries: (tokens, programs, lengths) of the eval files and the cont96 sources, capped."""
    from musicar.data.loader import tokenize_tracks
    held = Store(args.heldout)
    n = np.diff(held.offsets)
    ev = np.flatnonzero(n >= args.eval_min)
    parts, names = [capped(held, ev, args.cap)], [f'heldout:{i}' for i in ev]
    toks, progs, lens = [], [], []
    assert not args.cont96 or (Path(args.cont96) / 'prompts').is_dir(), f'no {args.cont96}/prompts (--cont96= skips it)'
    for p in sorted((Path(args.cont96) / 'prompts').glob('p*.mid')) if args.cont96 else []:
        segs = [tokenize_tracks(str(p))] + [tokenize_tracks(str(r)) for r in
                                             sorted((Path(args.cont96) / 'real').glob(f'{p.stem}_*.mid'))[:1]]
        segs = [s for s in segs if s is not None]
        t, pr = np.concatenate([s[0] for s in segs])[:args.cap], np.concatenate([s[1] for s in segs])[:args.cap]
        toks.append(t)
        progs.append(pr)
        lens.append(len(t))
        names.append(f'cont96:{p.stem}')
    if toks:
        parts.append((np.concatenate(toks), np.concatenate(progs), np.array(lens, np.int64)))
    tok = np.concatenate([q[0] for q in parts])
    prog = np.concatenate([q[1] for q in parts])
    L = np.concatenate([q[2] for q in parts])
    return tok, prog, L, names


def init(pool, cap, qv, min_est, j):
    G.update(store=Store(pool), cap=cap, qv=qv, min_est=min_est, j=j)


def chunk(rng):
    """Pool files [a, b): (file index, query index, variant, J) of verified near-duplicates."""
    s, cap = G['store'], G['cap']
    idxs = np.arange(*rng)
    tok, prog, L = capped(s, idxs, cap)
    found = []
    for v, (S, cnt, _) in enumerate(sigs_and_sets(tok, prog, L)):
        QS, qsets, qkeys, qmap = G['qv'][v]
        cand = set()
        for b in range(BANDS):
            k = band_keys(S[:, b * ROWS:(b + 1) * ROWS].T, b)
            for i in np.flatnonzero(np.isin(k, qkeys[b]) & (cnt > 0)):
                cand.update((int(i), q) for q in qmap[b][int(k[i])])
        if not cand:
            continue
        pairs = np.array(sorted(cand))
        est = (S[pairs[:, 0]] == QS[pairs[:, 1]]).mean(1)
        pairs = pairs[est >= G['min_est']]
        if len(pairs):
            sets = sigs_and_sets(*capped(s, idxs[np.unique(pairs[:, 0])], cap), sets=True)[v][2]
            pos = {f: k for k, f in enumerate(np.unique(pairs[:, 0]))}
            for i, q in pairs:
                jac = jaccard(sets[pos[i]], qsets[q])
                if jac >= G['j']:
                    found.append((int(idxs[i]), int(q), v, round(jac, 3)))
    return found


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--pool', default='data/cache/gigamidi_pool_train')
    p.add_argument('--heldout', default='data/cache/gigamidi_heldout_validation')
    p.add_argument('--cont96', default='results/head2head/cont96')
    p.add_argument('--eval-min', type=int, default=2047, help='held-out files of >= this many notes are scored')
    p.add_argument('--cap', type=int, default=1024, help='first notes of every file')
    p.add_argument('--j', type=float, default=0.5)
    p.add_argument('--min-est', type=float, default=0.35)
    p.add_argument('--chunk', type=int, default=8000, help='files per job (< 16,384: minhash.shingles)')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--limit', type=int, default=0, help='only the first N pool files (timing)')
    p.add_argument('--out', default='data/cache/gigamidi_pool_skip.txt')
    a = p.parse_args()
    t0 = time.time()
    tok, prog, L, names = query_set(a)
    qv = []
    for S, cnt, sets in sigs_and_sets(tok, prog, L, sets=True):
        keys, maps = [], []
        for b in range(BANDS):
            k = band_keys(S[:, b * ROWS:(b + 1) * ROWS].T, b)
            m = {}
            for q in np.flatnonzero(cnt > 0):
                m.setdefault(int(k[q]), []).append(int(q))
            keys.append(np.array(sorted(m), np.uint64))
            maps.append(m)
        qv.append((S, sets, keys, maps))
    print(f'{len(L):,} queries ({sum(n.startswith("cont96") for n in names)} cont96), {time.time() - t0:.0f}s',
          flush=True)
    n = len(Store(a.pool))
    n = min(n, a.limit) if a.limit else n
    jobs = [(s, min(n, s + a.chunk)) for s in range(0, n, a.chunk)]
    found = []
    with Pool(a.workers, initializer=init, initargs=(a.pool, a.cap, qv, a.min_est, a.j)) as pool:
        for k, f in enumerate(pool.imap_unordered(chunk, jobs)):
            found += f
            if k % 10 == 0:
                print(f'{time.time() - t0:5.0f}s {k + 1}/{len(jobs)} chunks, {len({x[0] for x in found}):,} files',
                      flush=True)
    files = sorted({x[0] for x in found})
    s = Store(a.pool)
    notes = int(sum(s.offsets[i + 1] - s.offsets[i] for i in files))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out + '.tmp')
    tmp.write_text(''.join(f'{i}\n' for i in files))
    tmp.replace(a.out)
    by_kind = {k: len({x[0] for x in found if names[x[1]].startswith(k)}) for k in ('heldout', 'cont96')}
    hit = {k: len({x[1] for x in found if names[x[1]].startswith(k)}) for k in ('heldout', 'cont96')}
    summary = dict(files=len(files), notes=notes, pool_files=n, by_query_kind=by_kind, queries_hit=hit,
                   transposed_only=len({x[0] for x in found} - {x[0] for x in found if x[2] == 0}),
                   cap=a.cap, j=a.j, minutes=round((time.time() - t0) / 60, 1))
    Path(a.out).with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
