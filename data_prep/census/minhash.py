"""
Census level 5: partial-overlap near-duplicates that survive the `loose` dedup, estimated with MinHash on a sample.

    cd /workspace/music
    PYTHONPATH=.:data_prep/datasets:data_prep/census nice -n 10 .venv/bin/python data_prep/census/minhash.py \
        [--index census/hashes/global_index.npz] [--sample 200000] [--seed 20261119] [--workers 3] \
        [--work /workspace/agents/minhash] [--out census/minhash.json]

Population: every file "kept at loose" in the global index (status == kept, or dropped only at loose_t), in the
index's first-copy order (source order, then split train < validation < test except Discover, then row order;
global_index.build_index). Resumable: signatures are written to a memmap in --work, chunk by chunk.

Method
1. Shingles. Per file, its first --cap notes (store order = onset order). The notes are split into streams by program
   (drums = program 128, pitch + 128, as in prepare_discover.note_hashes); each stream is sorted by (onset, pitch).
   A shingle is an n-gram (default n=4) of consecutive notes of one stream: (pitch_0, d_1, pitch_1, ..., d_{n-1},
   pitch_{n-1}), d = onset delta to the previous note of the stream. Not transposition-invariant. The program number
   itself is not hashed (a re-orchestrated copy still matches), but splitting by program keeps a copy with one track
   added or removed close (a single interleaved sequence would change almost every n-gram). Shingles hash to 64 bits.
   Files whose streams all have < n notes have no shingles; they are never counted as near-duplicates (reported).
2. Signatures. One-permutation MinHash (Li, Owen, Zhang 2012): 128 bins by the top 7 hash bits, the minimum of 32
   other hash bits per bin, empty bins filled by rotation densification (Shrivastava & Li 2014). P[bin equal] ~ J.
   Computed for ALL kept files (not just the sample), stored as a (128, N) uint32 memmap, ~3.5 GB for 7M files.
   One-permutation hashing is O(shingles) per file instead of O(128 x shingles) for 128 permutations: that is what
   makes signatures of the whole reference set affordable on ~3 cores.
3. Sample: --sample files drawn uniformly without replacement from the kept files (numpy default_rng(--seed)).
4. LSH over the whole reference set: 42 bands x 3 rows (P[candidate] = 1-(1-J^3)^42: 0.996 at J=0.5, 1.0 at 0.8).
   Per band the band keys of all N files are sorted once; each sampled file collects the files with the same key that
   come EARLIER in the global order (at most --group-cap of them per band, the earliest). So each sampled file is
   compared with the entire corpus before it, not only with the other sampled files: there is no sampling-fraction
   bias, and the per-file indicator "has an earlier near-duplicate" is measured, not extrapolated.
5. Verification: candidates with signature-estimated Jaccard >= --min-est are ranked by it; the top --verify-top get
   the exact Jaccard of the two shingle sets (unique 64-bit shingle hashes). A sampled file is a near-duplicate at
   threshold t if some earlier kept file has exact Jaccard >= t (t = 0.5, 0.8, 0.9). "Earlier" includes files that
   are themselves near-duplicates of something even earlier (connected-component rule: everything after the first
   copy goes), which is how a dedup pass over the corpus in order would also remove chains.
6. Estimates: file rate = share of sampled files that are near-duplicates (Wilson 95% interval); note rate = sum of
   their n_notes (all notes, not just the capped ones) / sum of n_notes of the sample (ratio estimator, 95% bootstrap
   interval, --boot resamples). Per source the same over the sampled files of that source (uniform within source).
   Intervals cover sampling error only, not LSH misses (bounded by the S-curve and the group cap, both reported).

Writes --out (census/minhash.json): parameters, seed, sample sizes and source shares, per-source and total estimates
at each threshold, the LSH diagnostics, runtime, peak RSS (main process and workers), and example pairs.
"""

import argparse
import json
import math
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from common import DRUMS, Store

NBINS = 128
EMPTY = np.int64(1 << 32)
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
K1, K2 = np.uint64(0xbf58476d1ce4e5b9), np.uint64(0x94d049bb133111eb)
GOLD = np.uint64(0x9E3779B97F4A7C15)
DENSE_C = 0x6F4A7C15  # rotation densification offset (odd, < 2^32)
THRESHOLDS = (0.5, 0.8, 0.9)


def mix(x):
    """splitmix64 finalizer, element-wise on uint64 (wrapping)."""
    x = x ^ (x >> np.uint64(30))
    x = x * K1
    x = x ^ (x >> np.uint64(27))
    x = x * K2
    return x ^ (x >> np.uint64(31))


# ------------------------------------------------------------------ shingles and signatures

def shingles(tok, prog, lengths, n):
    """Shingle hashes of several files given as concatenated note arrays (each file's notes in store order).
    tok (T, 4) uint16 pitch/velocity/duration/delta_time, prog (T,) uint8, lengths (m,) notes per file.
    Returns (fid, h): local file index and uint64 hash of every shingle (duplicates included)."""
    m = len(lengths)
    assert m < (1 << 14)
    T = int(lengths.sum())
    if T == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.uint64)
    fid = np.repeat(np.arange(m, dtype=np.uint64), lengths)
    delta = tok[:, 3].astype(np.int64)
    cs = np.cumsum(delta)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    base = np.repeat(cs[starts] - delta[starts], lengths)
    onset = np.minimum(cs - base, (1 << 34) - 1).astype(np.uint64)
    drums = prog == DRUMS
    pitch = (tok[:, 0].astype(np.uint64) + np.uint64(DRUMS) * drums) & np.uint64(0xFF)
    stream = prog.astype(np.uint64) & np.uint64(0xFF)
    key = (fid << np.uint64(50)) | (stream << np.uint64(42)) | (onset << np.uint64(8)) | pitch
    key.sort()
    grp = key >> np.uint64(42)                      # (file, stream)
    p = key & np.uint64(0xFF)
    on = (key >> np.uint64(8)) & np.uint64((1 << 34) - 1)
    d = np.zeros(T, np.uint64)
    d[1:] = np.minimum(on[1:] - on[:-1], np.uint64((1 << 20) - 1))   # garbage across groups, never used
    e = (d << np.uint64(8)) | p
    W = T - n + 1
    if W <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.uint64)
    ok = grp[:W] == grp[n - 1:]
    h = mix(p[:W] + GOLD)
    for k in range(1, n):
        h = mix(h ^ (e[k:k + W] + np.uint64((int(GOLD) * (k + 1)) % 2**64)))
    return (grp[:W][ok] >> np.uint64(8)).astype(np.int64), h[ok]


def oph(fid, h, m):
    """One-permutation MinHash signatures (m, 128) int64 with rotation densification; EMPTY rows = no shingles.
    Also returns the number of shingles per file."""
    S = np.full((m, NBINS), EMPTY, np.int64)
    cnt = np.bincount(fid, minlength=m)
    if len(h):
        b = (h >> np.uint64(57)).astype(np.int64)
        v = ((h >> np.uint64(16)) & np.uint64(0xFFFFFFFF)).astype(np.int64)
        comp = ((fid * NBINS + b).astype(np.uint64) << np.uint64(32)) | v.astype(np.uint64)
        comp.sort()
        kk = (comp >> np.uint64(32)).astype(np.int64)
        first = np.ones(len(kk), bool)
        first[1:] = kk[1:] != kk[:-1]
        S.reshape(-1)[kk[first]] = (comp[first] & np.uint64(0xFFFFFFFF)).astype(np.int64)
    rows = np.flatnonzero(((S == EMPTY).any(1)) & (cnt > 0))
    if len(rows):
        S0 = S[rows]
        out = S0.copy()
        todo = S0 == EMPTY
        for t in range(1, NBINS):
            src = np.roll(S0, -t, axis=1)
            fill = todo & (src != EMPTY)
            out[fill] = (src[fill] + t * DENSE_C) & 0xFFFFFFFF
            todo &= ~fill
            if not todo.any():
                break
        S[rows] = out
    return S, cnt


def file_notes(store, idxs, cap):
    """Concatenated (tokens, programs, lengths) of the first `cap` notes of files idxs (sorted) of a Store."""
    o = store.offsets
    a = o[idxs].astype(np.int64)
    L = np.minimum(o[idxs + 1].astype(np.int64) - a, cap)
    T = int(L.sum())
    pos = np.repeat(a - np.concatenate([[0], np.cumsum(L)[:-1]]), L) + np.arange(T, dtype=np.int64)
    tok = np.asarray(store.tokens[pos])
    prog = np.asarray(store.programs[pos]) if store.programs is not None else np.zeros(T, np.uint8)
    return tok, prog, L


def shingle_sets(store, idxs, cap, n):
    """List of sorted unique shingle-hash arrays, one per file idx (order of idxs, any order)."""
    idxs = np.asarray(idxs, np.int64)
    order = np.argsort(idxs, kind='stable')
    tok, prog, L = file_notes(store, idxs[order], cap)
    fid, h = shingles(tok, prog, L, n)
    out = [None] * len(idxs)
    srt = np.lexsort((h, fid))
    fid, h = fid[srt], h[srt]
    bounds = np.searchsorted(fid, np.arange(len(idxs) + 1))
    for j in range(len(idxs)):
        out[order[j]] = np.unique(h[bounds[j]:bounds[j + 1]])
    return out


def jaccard(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


# ------------------------------------------------------------------ LSH

def band_keys(cols, band):
    """uint64 key of one band from its rows (r, N) uint32/int64."""
    k = mix(np.uint64((int(GOLD) * (band + 1)) % 2**64) + cols[0].astype(np.uint64))
    for r in range(1, len(cols)):
        k = mix(k ^ cols[r].astype(np.uint64))
    return k


def lsh_candidates(sig_rows, valid, queries, bands, rows, group_cap):
    """Candidate pairs (q_local, ref_pos) with ref_pos < queries[q_local] that share a band key in some band.
    sig_rows(b0, b1) -> (b1-b0, N) array of signature rows. valid (N,) bool: files with shingles.
    Returns unique pairs and the number of (query, band) groups truncated by group_cap."""
    N = len(valid)
    qv = valid[queries]
    pairs = []
    capped = 0
    for b in range(bands):
        keys = band_keys(sig_rows(b * rows, (b + 1) * rows), b)
        inval = np.flatnonzero(~valid)
        keys[inval] = mix(inval.astype(np.uint64) ^ np.uint64(0xA5A5A5A5DEADBEEF))  # no shingles: unique keys
        srt = np.argsort(keys, kind='stable')          # stable: positions ascending inside a key group
        ks = keys[srt]
        inv = np.empty(N, np.int64)
        inv[srt] = np.arange(N)
        del keys
        lo = np.searchsorted(ks, ks[inv[queries]], 'left')
        earlier = inv[queries] - lo                    # group members before the query = earlier positions
        earlier[~qv] = 0
        capped += int((earlier > group_cap).sum())
        cnt = np.minimum(earlier, group_cap)
        tot = int(cnt.sum())
        if tot:
            ql = np.repeat(np.arange(len(queries), dtype=np.int64), cnt)
            off = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(tot, dtype=np.int64)
            pairs.append(np.unique(ql * N + srt[off]))
        del srt, ks, inv
        if len(pairs) > 8:
            pairs = [np.unique(np.concatenate(pairs))]
    allp = np.unique(np.concatenate(pairs)) if pairs else np.zeros(0, np.int64)
    return allp // N, allp % N, capped


def est_jaccard(sig_rows_at, qsig, ql, ref, chunk=500_000):
    """Signature agreement of pairs: qsig (Q, 128) of the queries, sig_rows_at(ref) -> (128, len(ref))."""
    out = np.zeros(len(ql), np.float32)
    for a in range(0, len(ql), chunk):
        r = sig_rows_at(ref[a:a + chunk])               # (128, c)
        out[a:a + chunk] = (qsig[ql[a:a + chunk]] == r.T).mean(1)
    return out


def s_curve(j, bands, rows):
    return 1 - (1 - j ** rows) ** bands


# ------------------------------------------------------------------ corpus plumbing

def load_population(index_path):
    """Kept-at-loose files in global first-copy order: dict of arrays per position + source info."""
    sys.path.insert(0, str(Path(__file__).parent))
    from global_index import ARCHIVE_ORDER, KEPT
    from tables import LEVELS
    g = np.load(index_path)
    sources = [str(s) for s in g['sources']]
    li = LEVELS.index('loose')
    status = g['status']
    kept = (status == KEPT) | (status > li)
    tabs = {s: np.load(Path(index_path).parent / f'table_{s}.npz') for s in sources}
    split = np.concatenate([np.zeros(len(tabs[s]['split']), np.uint8) if s in ARCHIVE_ORDER else tabs[s]['split']
                            for s in sources])
    rows = np.arange(len(status), dtype=np.int64)
    order = np.lexsort((rows, split, g['source']))
    del split
    order = order[kept[order]]                        # global rows of kept files, in first-copy order
    src = g['source'][order]
    row = g['row'][order]
    store = np.empty(len(order), np.int16)
    sidx = np.empty(len(order), np.int64)
    nn = np.empty(len(order), np.int64)
    stores = {}
    for si, s in enumerate(sources):
        m = src == si
        t = tabs[s]
        store[m], sidx[m], nn[m] = t['store'][row[m]], t['idx'][row[m]], t['n_notes'][row[m]]
        stores[si] = json.loads(str(t['meta']))['stores']
    assert (store >= 0).all(), 'kept files outside a store'
    return dict(sources=sources, grow=order, src=src, row=row, store=store, sidx=sidx, n_notes=nn,
                store_names=stores, status=status[order])


_W = {}


def _store(path):
    if path not in _W:
        _W[path] = Store(path)
    return _W[path]


def _sign_chunk(job):
    """Worker: signatures of one chunk, written into the shared memmap."""
    cid, path, idxs, pos, sig_path, cnt_path, N, cap, n = job
    st = _store(path)
    tok, prog, L = file_notes(st, idxs, cap)
    fid, h = shingles(tok, prog, L, n)
    S, cnt = oph(fid, h, len(idxs))
    sig = np.memmap(sig_path, dtype=np.uint32, mode='r+', shape=(NBINS, N))
    sig[:, pos] = (S & 0xFFFFFFFF).astype(np.uint32).T
    sig.flush()
    c = np.memmap(cnt_path, dtype=np.int32, mode='r+', shape=(N,))
    c[pos] = np.minimum(cnt, 2**31 - 1)
    c.flush()
    return cid, len(idxs), int(L.sum())


def _verify(job):
    """Worker: exact Jaccard of (query, candidates) groups. job = list of (qpos, [refpos...]) with store info."""
    cap, n, items = job
    out = []
    for qinfo, refs in items:
        qs = shingle_sets(_store(qinfo[0]), [qinfo[1]], cap, n)[0]
        best = (0.0, -1)
        by_store = {}
        for rp, path, ri in refs:
            by_store.setdefault(path, []).append((rp, ri))
        js = []
        for path, lst in by_store.items():
            sets = shingle_sets(_store(path), [ri for _, ri in lst], cap, n)
            js += [(jaccard(qs, s), rp, len(s)) for (rp, _), s in zip(lst, sets)]
        for j, rp, ns in js:
            if j > best[0]:
                best = (j, rp)
        out.append((qinfo[2], best[0], best[1], len(qs), len(js)))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - hw), min(1.0, c + hw)


def boot_ratio(dup, w, B, rng):
    """Note-weighted rate sum(w*dup)/sum(w) with a percentile bootstrap 95% interval."""
    n = len(w)
    if n == 0:
        return (float('nan'),) * 3
    est = float((w * dup).sum() / w.sum())
    res = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        res[b] = (w[i] * dup[i]).sum() / max(w[i].sum(), 1)
    return est, float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def peak_rss_mb():
    return dict(main=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                largest_worker=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--index', default='census/hashes/global_index.npz')
    ap.add_argument('--cache', default='data/cache')
    ap.add_argument('--work', default='/workspace/agents/minhash')
    ap.add_argument('--out', default='census/minhash.json')
    ap.add_argument('--sample', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=20261119)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--ngram', type=int, default=4)
    ap.add_argument('--cap', type=int, default=4000, help='notes per file used for shingles (the first ones)')
    ap.add_argument('--bands', type=int, default=42)
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--group-cap', type=int, default=300, help='max earlier files taken per (query, band) group')
    ap.add_argument('--min-est', type=float, default=0.4, help='min signature-estimated Jaccard to verify')
    ap.add_argument('--verify-top', type=int, default=10)
    ap.add_argument('--chunk', type=int, default=4000, help='max files per signature chunk')
    ap.add_argument('--chunk-notes', type=int, default=2_000_000, help='max capped notes per signature chunk')
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--examples', type=int, default=20)
    args = ap.parse_args(argv)
    assert args.bands * args.rows <= NBINS
    t0 = time.time()
    times = {}
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    pop = load_population(args.index)
    N = len(pop['grow'])
    print(f'population: {N:,} kept-at-loose files, {pop["n_notes"].sum():,} notes', flush=True)
    times['load'] = time.time() - t0

    # ---- signatures of every kept file (resumable)
    tag = f'n{args.ngram}_cap{args.cap}_N{N}_c{args.chunk}x{args.chunk_notes}'
    sig_path, cnt_path, done_path = work / f'sig_{tag}.u32', work / f'cnt_{tag}.i32', work / f'done_{tag}.txt'
    if not sig_path.exists():
        np.memmap(sig_path, dtype=np.uint32, mode='w+', shape=(NBINS, N)).flush()
        np.memmap(cnt_path, dtype=np.int32, mode='w+', shape=(N,)).flush()
        done_path.write_text('')
    done = set(int(x) for x in done_path.read_text().split())
    jobs = []
    pos_all = np.arange(N, dtype=np.int64)
    for si in range(len(pop['sources'])):
        for k, name in enumerate(pop['store_names'][si]):
            m = (pop['src'] == si) & (pop['store'] == k)
            pos = pos_all[m]
            idx = pop['sidx'][m]
            o = np.argsort(idx, kind='stable')
            pos, idx = pos[o], idx[o]
            # chunks of at most --chunk files and --chunk-notes capped notes (bounds worker RAM)
            ln = np.minimum(pop['n_notes'][pos], args.cap)
            a = 0
            while a < len(idx):
                b = a + max(1, min(args.chunk, int(np.searchsorted(np.cumsum(ln[a:a + args.chunk]), args.chunk_notes))))
                cid = len(jobs)
                jobs.append((cid, str(Path(args.cache) / name), idx[a:b], pos[a:b],
                             str(sig_path), str(cnt_path), N, args.cap, args.ngram))
                a = b
    todo = [j for j in jobs if j[0] not in done]
    print(f'signatures: {len(jobs)} chunks, {len(todo)} to do, '
          f'{int(np.minimum(pop["n_notes"], args.cap).sum()):,} capped notes in all', flush=True)
    t1 = time.time()
    if todo:
        with Pool(args.workers) as pool, open(done_path, 'a') as df:
            nd = 0
            for cid, nf, nt in pool.imap_unordered(_sign_chunk, todo):
                df.write(f'{cid}\n')
                df.flush()
                nd += 1
                if nd % 50 == 0 or nd == len(todo):
                    el = time.time() - t1
                    print(f'  {nd}/{len(todo)} chunks, {el / 60:.1f} min, ETA {el / nd * (len(todo) - nd) / 60:.1f} '
                          f'min', flush=True)
    times['signatures'] = time.time() - t1

    sig = np.memmap(sig_path, dtype=np.uint32, mode='r', shape=(NBINS, N))
    cnt = np.array(np.memmap(cnt_path, dtype=np.int32, mode='r', shape=(N,)))
    valid = cnt > 0

    # ---- sample
    rng = np.random.default_rng(args.seed)
    S = min(args.sample, N)
    queries = np.sort(rng.choice(N, S, replace=False))
    qsig = np.ascontiguousarray(sig[:, queries].T)

    # ---- LSH candidates against the whole reference set
    t2 = time.time()
    ql, ref, capped = lsh_candidates(lambda a, b: np.asarray(sig[a:b]), valid, queries, args.bands, args.rows,
                                     args.group_cap)
    times['lsh'] = time.time() - t2
    print(f'LSH: {len(ql):,} candidate pairs, {capped:,} capped (query, band) groups, {times["lsh"]:.0f} s',
          flush=True)
    t3 = time.time()
    est = est_jaccard(lambda r: sig[:, r], qsig, ql, ref)
    keep = est >= args.min_est
    ql, ref, est = ql[keep], ref[keep], est[keep]
    o = np.lexsort((-est, ql))
    ql, ref, est = ql[o], ref[o], est[o]
    first = np.ones(len(ql), bool)
    first[1:] = ql[1:] != ql[:-1]
    rank = np.arange(len(ql)) - np.maximum.accumulate(np.where(first, np.arange(len(ql)), 0))
    top = rank < args.verify_top
    times['estimate'] = time.time() - t3
    print(f'{keep.sum():,} pairs with est >= {args.min_est}, {top.sum():,} to verify '
          f'({len(np.unique(ql)):,} queries)', flush=True)

    # ---- exact verification
    t4 = time.time()
    names = {(si, k): str(Path(args.cache) / nm) for si, lst in pop['store_names'].items() for k, nm in enumerate(lst)}

    def info(p):
        return names[(int(pop['src'][p]), int(pop['store'][p]))], int(pop['sidx'][p])

    groups = {}
    for q, r in zip(ql[top], ref[top]):
        groups.setdefault(int(q), []).append(int(r))
    items = [((*info(queries[q]), q), [(r, *info(r)) for r in rs]) for q, rs in groups.items()]
    batches = [(args.cap, args.ngram, items[a:a + 200]) for a in range(0, len(items), 200)]
    best_j = np.zeros(S, np.float64)
    best_ref = np.full(S, -1, np.int64)
    n_verified = 0
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(_verify, batches):
            for q, j, r, _, nv in res:
                best_j[q], best_ref[q] = j, r
                n_verified += nv
    times['verify'] = time.time() - t4
    print(f'verified {n_verified:,} pairs in {times["verify"]:.0f} s', flush=True)

    # ---- estimates
    qsrc = pop['src'][queries]
    qn = pop['n_notes'][queries].astype(np.float64)
    brng = np.random.default_rng(args.seed + 1)
    res = {}
    for name, m in [(s, qsrc == si) for si, s in enumerate(pop['sources'])] + [('total', np.ones(S, bool))]:
        popm = (pop['src'] == pop['sources'].index(name)) if name != 'total' else np.ones(N, bool)
        r = dict(population_files=int(popm.sum()), population_notes=int(pop['n_notes'][popm].sum()),
                 sampled_files=int(m.sum()), sample_share=float(m.mean()), sampled_notes=int(qn[m].sum()),
                 sampled_without_shingles=int((~valid[queries] & m).sum()), thresholds={})
        for t in THRESHOLDS:
            dup = (best_j >= t) & m
            k = int(dup.sum())
            pf, lo, hi = wilson(k, int(m.sum()))
            pn, nlo, nhi = boot_ratio(dup[m].astype(np.float64), qn[m], args.boot, brng)
            by_first = {}
            for fs in pop['src'][best_ref[dup]]:
                by_first[pop['sources'][fs]] = by_first.get(pop['sources'][fs], 0) + 1
            r['thresholds'][str(t)] = dict(
                dup_files=k, file_rate=pf, file_rate_ci95=[lo, hi], note_rate=pn, note_rate_ci95=[nlo, nhi],
                est_further_drop_files=(int(round(pf * r['population_files'])) if pf == pf else None),
                est_further_drop_notes=(int(round(pn * r['population_notes'])) if pn == pn else None),
                first_copy_source=by_first)
        res[name] = r

    # ---- examples (at 0.8, spread over the range of Jaccard), for spot checks
    ex = []
    cand = np.flatnonzero(best_j >= 0.8)
    if len(cand):
        pick = cand[np.argsort(best_j[cand])][np.linspace(0, len(cand) - 1, min(args.examples, len(cand))).astype(int)]
        for q in pick:
            a, b = queries[q], best_ref[q]
            ex.append(dict(jaccard=round(float(best_j[q]), 4),
                           file=dict(source=pop['sources'][pop['src'][a]], row=int(pop['row'][a]),
                                     store=info(a)[0], store_idx=int(pop['sidx'][a]), n_notes=int(pop['n_notes'][a])),
                           earlier=dict(source=pop['sources'][pop['src'][b]], row=int(pop['row'][b]),
                                        store=info(b)[0], store_idx=int(pop['sidx'][b]),
                                        n_notes=int(pop['n_notes'][b]))))
    hist = np.histogram(best_j[best_j > 0], bins=np.linspace(0, 1, 11))[0].tolist()
    out = dict(
        what='census level 5: partial-overlap near-duplicates surviving loose dedup (MinHash, sampled)',
        index=args.index, seed=args.seed,
        method=dict(
            population='kept at loose (status kept or dup_loose_t), global first-copy order',
            sampling='uniform without replacement over the population, numpy default_rng(seed).choice',
            shingles=f'{args.ngram}-grams (pitch, onset delta) within each program stream (drums pitch+128), '
                     f'first {args.cap} notes per file; program not hashed; not transposition-invariant',
            signature='one-permutation MinHash, 128 bins x 32 bits, rotation densification',
            lsh=f'{args.bands} bands x {args.rows} rows over ALL population files; earlier files only; at most '
                f'{args.group_cap} earlier files per (query, band)',
            verification=f'exact Jaccard of unique shingle sets for the top {args.verify_top} candidates with '
                         f'estimated Jaccard >= {args.min_est}',
            dup_rule='a sampled file is a near-duplicate at t if any earlier population file has Jaccard >= t',
            bias='every sampled file is compared with the entire population before it, so no sampling-fraction '
                 'correction is needed; intervals: Wilson 95% (files), percentile bootstrap 95% (notes, '
                 f'{args.boot} resamples); they cover sampling error only',
            lsh_recall={str(t): s_curve(t, args.bands, args.rows) for t in (0.4, 0.5, 0.6, 0.8, 0.9)}),
        params=vars(args),
        population=dict(files=N, notes=int(pop['n_notes'].sum()), files_without_shingles=int((~valid).sum())),
        sample=dict(files=S, notes=int(qn.sum())),
        lsh=dict(candidate_pairs_after_est_filter=int(len(ql)), verified_pairs=n_verified,
                 capped_query_band_groups=capped, queries_with_candidates=len(groups),
                 best_jaccard_hist_0_1_step_0_1=hist),
        results=res, examples=ex,
        runtime_s={k: round(v, 1) for k, v in times.items()} | dict(total=round(time.time() - t0, 1)),
        peak_rss_mb=peak_rss_mb())
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {t: (v['thresholds'][t]['file_rate'], v['thresholds'][t]['note_rate'])
                          for t in v['thresholds']} for k, v in res.items()}, indent=1))
    print(f'wrote {args.out}; peak RSS {out["peak_rss_mb"]}; {out["runtime_s"]}')
    return out


if __name__ == '__main__':
    main()
