"""
Census level 5, full pass: near-duplicates over EVERY file of the exported loose-level stores (not a sample).

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/minhash_full.py \
        --cache <export>/cache --work <fast disk>/minhash_full --out <dir>/minhash_full.json \
        [--sample-json census/minhash.json] [--workers 12] [--lags 8] [--verify 20000]

Population: the files of the exported stores (`<source>_<split>` dirs from stores_*.tar.zst, README of the census
export), in first-copy order: source order gigamidi, discover, aria, pianocore, piast, beat, irishman; inside a
source train < validation < test; inside a store, store order (= original store order). Together they are the
9.92B-note loose set. Unlike minhash.py this needs no global index: the export stores already hold only the files
kept at `loose`.

Method (shingles and signatures exactly as minhash.py: 4-grams per program stream, first --cap notes, one-permutation
MinHash 128 x 32 bits; same resumable memmap layout, so a work dir can be resumed)
1. Signatures of all files, chunks handed out in store order (sequential reads on a slow USB disk).
2. LSH, 42 bands x 3 rows. Per band, files are sorted by band key (stable: positions ascending inside a key). Each file
   is compared with the --lags files right before it in its key group (the latest earlier ones) and with the group's
   first file (the earliest copy). Signature-estimated Jaccard of each pair; per file the best estimate over earlier
   files and that file's position are kept. No per-group cap on how many files are looked at in total: big groups
   are covered through chains (each member sees its predecessors).
3. Dup rule as minhash.py: a file is dropped at threshold t if some EARLIER file has Jaccard >= t (chains included).
   Raw counts use the signature estimate (sd about 0.035 at J = 0.8).
4. Calibration: --verify files drawn uniformly from those with best estimate >= --min-est get the exact Jaccard to
   their best earlier file (shingle sets read from the stores). Per estimate bin (0.05 wide) the share with exact
   Jaccard >= t gives calibrated drops = sum over bins of the bin's notes x that share. Files below --min-est count as
   kept (minhash.py's sample shows almost no exact pairs >= 0.5 below an estimate of 0.4).
5. If --sample-json (the 200k-sample run of minhash.py) is given, its totals are printed beside these, for the
   overlapping sources.

Writes --out (json summary per source and threshold, raw and calibrated) and <out stem>_files.npz: per file store
name index, store idx, n_notes, best_est (float32), best_ref (position, -1 = none), and the calibration sample.
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import minhash as mh
from common import Store

SOURCES = ['gigamidi', 'discover', 'aria', 'pianocore', 'piast', 'beat', 'irishman']
SPLITS = ['train', 'validation', 'test']


def population(cache):
    """Store dirs in first-copy order and per-file arrays (store index, idx in store, n_notes)."""
    names, store, sidx, nn = [], [], [], []
    for s in SOURCES:
        for sp in SPLITS:
            p = Path(cache) / f'{s}_{sp}'
            if not (p / 'offsets.npy').exists() or json.loads((p / 'meta.json').read_text())['n_files'] == 0:
                continue                                  # absent or empty split (irishman / piast validation)
            st = Store(p)
            n = np.diff(st.offsets.astype(np.int64))
            store.append(np.full(len(n), len(names), np.int16))
            sidx.append(np.arange(len(n), dtype=np.int64))
            nn.append(n)
            names.append(p.name)
    return dict(names=names, store=np.concatenate(store), sidx=np.concatenate(sidx), n_notes=np.concatenate(nn))


def signatures(pop, args):
    """Resumable signatures (128, N) uint32 memmap and shingle counts, as minhash.py."""
    N = len(pop['store'])
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tag = f'full_n{args.ngram}_cap{args.cap}_N{N}_c{args.chunk}x{args.chunk_notes}'
    sig_path, cnt_path, done_path = work / f'sig_{tag}.u32', work / f'cnt_{tag}.i32', work / f'done_{tag}.txt'
    if not sig_path.exists():
        np.memmap(sig_path, dtype=np.uint32, mode='w+', shape=(mh.NBINS, N)).flush()
        np.memmap(cnt_path, dtype=np.int32, mode='w+', shape=(N,)).flush()
        done_path.write_text('')
    done = set(int(x) for x in done_path.read_text().split())
    jobs = []
    for k, name in enumerate(pop['names']):
        pos = np.flatnonzero(pop['store'] == k)
        idx = pop['sidx'][pos]
        ln = np.minimum(pop['n_notes'][pos], args.cap)
        a = 0
        while a < len(idx):
            b = a + max(1, min(args.chunk, int(np.searchsorted(np.cumsum(ln[a:a + args.chunk]), args.chunk_notes))))
            jobs.append((len(jobs), str(Path(args.cache) / name), idx[a:b], pos[a:b], str(sig_path), str(cnt_path),
                         N, args.cap, args.ngram))
            a = b
    todo = [j for j in jobs if j[0] not in done]
    print(f'signatures: {len(jobs)} chunks, {len(todo)} to do', flush=True)
    t1 = time.time()
    if todo:
        with Pool(args.workers) as pool, open(done_path, 'a') as df:
            nd = notes = 0
            for cid, nf, nt in pool.imap_unordered(mh._sign_chunk, todo):   # imap hands chunks out in order
                df.write(f'{cid}\n')
                df.flush()
                nd += 1
                notes += nt
                if nd % 100 == 0 or nd == len(todo):
                    el = time.time() - t1
                    print(f'  {nd}/{len(todo)} chunks, {notes / el / 1e6:.2f}M capped notes/s, {el / 60:.1f} min, '
                          f'ETA {el / nd * (len(todo) - nd) / 60:.1f} min', flush=True)
    return sig_path, cnt_path


def best_earlier(S, valid, bands, rows, lags, chunk=1_000_000):
    """Per file, the best signature-estimated Jaccard to an earlier file sharing a band key, and that file.
    S (N, 128) uint32 in RAM. Compares each file with the `lags` previous members of its key group and the group's
    first member, in every band."""
    N = len(S)
    best = np.zeros(N, np.float32)
    ref = np.full(N, -1, np.int64)
    pairs = 0
    inval = np.flatnonzero(~valid)
    for b in range(bands):
        t = time.time()
        keys = mh.band_keys(S[:, b * rows:(b + 1) * rows].T, b)
        keys[inval] = mh.mix(inval.astype(np.uint64) ^ np.uint64(0xA5A5A5A5DEADBEEF))
        srt = np.argsort(keys, kind='stable')
        ks = keys[srt]
        del keys
        start = np.ones(N, bool)
        start[1:] = ks[1:] != ks[:-1]
        gfirst = srt[np.maximum.accumulate(np.where(start, np.arange(N), 0))]   # first member of each sorted slot's group
        cand = [(srt[j:], srt[:-j], ks[j:] == ks[:-j]) for j in range(1, lags + 1)]
        cand.append((srt, gfirst, (gfirst != srt)))
        nb = 0
        for a_all, p_all, same in cand:
            a, p = a_all[same], p_all[same]
            for c in range(0, len(a), chunk):
                aa, pp = a[c:c + chunk], p[c:c + chunk]
                e = (S[aa] == S[pp]).mean(1, dtype=np.float32)
                # several pairs may share a file: keep the max per file
                o = np.lexsort((e, aa))
                aa, pp, e = aa[o], pp[o], e[o]
                last = np.ones(len(aa), bool)
                last[:-1] = aa[1:] != aa[:-1]
                aa, pp, e = aa[last], pp[last], e[last]
                up = e > best[aa]
                best[aa[up]] = e[up]
                ref[aa[up]] = pp[up]
            nb += len(a)
        pairs += nb
        print(f'  band {b + 1}/{bands}: {nb:,} pairs, {time.time() - t:.0f} s', flush=True)
    return best, ref, pairs


def exact_pairs(pop, cache, q, r, cap, n, workers):
    """Exact Jaccard of position pairs (q[i], r[i]) via minhash._verify."""
    def info(p):
        return str(Path(cache) / pop['names'][pop['store'][p]]), int(pop['sidx'][p])
    items = [((*info(a), i), [(int(b), *info(b))]) for i, (a, b) in enumerate(zip(q, r))]
    batches = [(cap, n, items[a:a + 200]) for a in range(0, len(items), 200)]
    out = np.zeros(len(q), np.float64)
    with Pool(workers) as pool:
        for res in pool.imap_unordered(mh._verify, batches):
            for i, j, _, _, _ in res:
                out[i] = j
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--cache', required=True, help='dir holding the exported <source>_<split> stores')
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample-json', default=None, help="minhash.py's sampled result, for comparison")
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--ngram', type=int, default=4)
    ap.add_argument('--cap', type=int, default=4000)
    ap.add_argument('--bands', type=int, default=42)
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--lags', type=int, default=8)
    ap.add_argument('--min-est', type=float, default=0.4)
    ap.add_argument('--verify', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20261123)
    ap.add_argument('--chunk', type=int, default=4000)
    ap.add_argument('--chunk-notes', type=int, default=2_000_000)
    ap.add_argument('--signatures-only', action='store_true')
    args = ap.parse_args(argv)
    t0 = time.time()
    times = {}

    pop = population(args.cache)
    N = len(pop['store'])
    src_of_store = np.array([SOURCES.index(nm.rsplit('_', 1)[0]) for nm in pop['names']])
    src = src_of_store[pop['store']]
    print(f'population: {N:,} files, {pop["n_notes"].sum():,} notes in {len(pop["names"])} stores', flush=True)

    sig_path, cnt_path = signatures(pop, args)
    times['signatures'] = time.time() - t0
    if args.signatures_only:
        return None
    t1 = time.time()
    sig = np.memmap(sig_path, dtype=np.uint32, mode='r', shape=(mh.NBINS, N))
    S = np.empty((N, mh.NBINS), np.uint32)
    for a in range(0, N, 1_000_000):
        S[a:a + 1_000_000] = np.asarray(sig[:, a:a + 1_000_000]).T
    del sig
    valid = np.array(np.memmap(cnt_path, dtype=np.int32, mode='r', shape=(N,))) > 0
    print(f'signatures in RAM ({S.nbytes / 2**30:.1f} GiB), {(~valid).sum():,} files without shingles', flush=True)

    best, ref, pairs = best_earlier(S, valid, args.bands, args.rows, args.lags)
    del S
    times['lsh'] = time.time() - t1

    # calibration sample: exact Jaccard to the best earlier file
    t2 = time.time()
    rng = np.random.default_rng(args.seed)
    elig = np.flatnonzero(best >= args.min_est)
    cal = np.sort(rng.choice(elig, min(args.verify, len(elig)), replace=False))
    exact = exact_pairs(pop, args.cache, cal, ref[cal], args.cap, args.ngram, args.workers)
    times['verify'] = time.time() - t2
    print(f'verified {len(cal):,} pairs in {times["verify"]:.0f} s', flush=True)

    edges = np.round(np.arange(args.min_est, 1.0001, 0.05), 4)
    edges[-1] = 1.0001
    bin_all = np.digitize(best, edges) - 1           # -1 below min_est
    bin_cal = np.digitize(best[cal], edges) - 1
    share = {t: np.array([(exact[bin_cal == k] >= t).mean() if (bin_cal == k).any() else np.nan
                          for k in range(len(edges) - 1)]) for t in mh.THRESHOLDS}
    calib = {str(t): dict(bins=edges[:-1].tolist(), n=np.bincount(bin_cal, minlength=len(edges) - 1).tolist(),
                          share_exact_ge_t=[None if v != v else round(float(v), 4) for v in share[t]])
             for t in mh.THRESHOLDS}

    nn = pop['n_notes'].astype(np.float64)
    res = {}
    groups = [(s, src == si) for si, s in enumerate(SOURCES) if (src == si).any()] + [('total', np.ones(N, bool))]
    for name, m in groups:
        r = dict(files=int(m.sum()), notes=int(nn[m].sum()), thresholds={})
        for t in mh.THRESHOLDS:
            raw = m & (best >= t)
            sh = np.nan_to_num(share[t], nan=0.0)
            w = np.where(bin_all >= 0, sh[np.clip(bin_all, 0, None)], 0.0) * m
            r['thresholds'][str(t)] = dict(
                raw_drop_files=int(raw.sum()), raw_drop_notes=int(nn[raw].sum()),
                raw_note_rate=float(nn[raw].sum() / nn[m].sum()),
                cal_drop_files=float(w.sum()), cal_drop_notes=float((w * nn).sum()),
                cal_note_rate=float((w * nn).sum() / nn[m].sum()),
                unique_notes_raw=int(nn[m].sum() - nn[raw].sum()),
                unique_notes_cal=float(nn[m].sum() - (w * nn).sum()))
        res[name] = r

    comp = None
    if args.sample_json:
        sj = json.loads(Path(args.sample_json).read_text())['results']
        comp = {s: {t: dict(sample_note_rate=sj[s]['thresholds'][t]['note_rate'],
                            sample_ci95=sj[s]['thresholds'][t]['note_rate_ci95'],
                            full_raw=res[s]['thresholds'][t]['raw_note_rate'],
                            full_cal=res[s]['thresholds'][t]['cal_note_rate'])
                    for t in sj[s]['thresholds']}
                for s in sj if s in res and s != 'total'}

    out = dict(
        what='census level 5, full pass: near-duplicates over every file of the exported loose stores',
        method=dict(order=' > '.join(SOURCES) + '; train < validation < test; store order',
                    lsh=f'{args.bands} bands x {args.rows} rows; per band each file vs its {args.lags} previous key-group '
                        f'members and the group first', dup_rule='dropped at t if some earlier file has J >= t',
                    raw='signature-estimated Jaccard', calibrated=f'{len(cal)} exact checks, est bins of 0.05 from '
                                                               f'{args.min_est}; below that = kept'),
        params=vars(args), stores=pop['names'],
        population=dict(files=N, notes=int(nn.sum()), files_without_shingles=int((~valid).sum())),
        lsh=dict(pairs_compared=int(pairs), files_with_est_ge_min=int(len(elig))),
        calibration=calib, results=res, vs_sample=comp,
        runtime_s={k: round(v, 1) for k, v in times.items()} | dict(total=round(time.time() - t0, 1)),
        peak_rss_mb=mh.peak_rss_mb())
    Path(args.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(args.out).with_name(Path(args.out).stem + '_files.npz'), stores=np.array(pop['names']),
                        store=pop['store'], sidx=pop['sidx'], n_notes=pop['n_notes'], best_est=best, best_ref=ref,
                        cal_pos=cal, cal_exact=exact)
    for name in res:
        print(name, {t: (round(v['raw_note_rate'], 4), round(v['cal_note_rate'], 4),
                         f"{v['unique_notes_cal'] / 1e9:.3f}B") for t, v in res[name]['thresholds'].items()})
    if comp:
        print(json.dumps(comp, indent=1))
    print(f'wrote {args.out}; {out["runtime_s"]}; peak RSS {out["peak_rss_mb"]}')
    return out


if __name__ == '__main__':
    main()
