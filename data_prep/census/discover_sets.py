"""
Census Discover sets (scaling plan 2026-11-20, block B): dense-file stats for every file of the census Discover export
and four token stores, from one stream of stores_discover.tar.zst (12.7 GB; read through zstd -dc, never unpacked).

    python data_prep/census/discover_sets.py build --tar ~/~musicar_export/stores_discover.tar.zst
    python data_prep/census/discover_sets.py check          # the plan's CR-4 checks on the written stores

Dense file ("black MIDI"): more than --dense-onset notes on one onset, or more than --dense-nps notes per second over
its onset span (first to last onset, at least 1 s; 20 ms delta_time bins, gaps capped at 10.22 s by the tokenizer).
Aria-identical: the file's loose hash (sorted (onset, pitch), drums apart) is in census/hashes/aria.npz.
Stores (data/cache/, the repo's store layout plus files.csv: file_id, md5, store, idx of the census store):
    cdisc_t1_train        census Discover train minus dense files, drawn by crc32(md5 + 't1') % 10000 < k, k the
                          smallest that reaches --target notes (the GigaMIDI replay store's 167.9M)
    cdisc_vn_validation   census Discover validation: not dense, not Aria-identical
    cdisc_va_validation   census Discover validation: not dense, Aria-identical
    cdisc_vd_validation   census Discover validation: dense
Stats: data/cache/cdisc_stats.npz, per file in tar order (train, then validation): part, n_notes, span_s, nps,
max_onset, dense, aria (train: by the census hash table discover.npz, validation: hashed here), t1. The written stores
are held in memory until the stream ends (a superset of t1 drawn before the dense flags are known, ~2 GB).
"""

import argparse
import io
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'data_prep/datasets')]
from census_pool import CHUNK, tar_stream  # noqa: E402
from data_prep.census.common import note_hashes  # noqa: E402

CACHE = ROOT / 'data/cache'
PARTS = ('train', 'validation')
VAL_STORES = {'vn': 'not dense, not Aria-identical', 'va': 'not dense, Aria-identical', 'vd': 'dense'}
DT = 0.02  # seconds per delta_time bin


def member(name):
    """'discover/discover_train/tokens.u16' -> ('train', 'tokens.u16')."""
    parts = name.split('/')
    return (parts[-2].removeprefix('discover_'), parts[-1]) if len(parts) >= 3 else (None, None)


def read_small(tar):
    """First pass: offsets.npy, files.csv, meta.json of both parts (they follow the part's tokens and programs)."""
    proc, tf = tar_stream(tar)
    small = {p: {} for p in PARTS}
    for m in tf:
        part, fname = member(m.name)
        if part in small and fname in ('offsets.npy', 'files.csv', 'meta.json'):
            small[part][fname] = tf.extractfile(m).read()
        if all(len(v) == 3 for v in small.values()):
            break
    proc.kill()
    out = {}
    for p, v in small.items():
        off = np.load(io.BytesIO(v['offsets.npy'])).astype(np.int64)
        lines = v['files.csv'].decode().splitlines()
        meta = json.loads(v['meta.json'])
        assert meta['n_notes'] == int(off[-1]) and len(lines) - 1 == len(off) - 1, p
        out[p] = dict(offsets=off, header=lines[0], rows=lines[1:], meta=meta)
    return out


def file_stats(tok, off):
    """Rows of whole files (local offsets off, off[0] = 0) -> (onset span in s, most notes on one onset) per file."""
    n = np.diff(off)
    assert (n > 0).all()
    cs = np.cumsum(tok[:, 3], dtype=np.int64)
    span = (cs[off[1:] - 1] - cs[off[:-1]]) * DT
    new = tok[:, 3] != 0
    new[off[:-1]] = True
    g = np.cumsum(new) - 1
    sizes = np.bincount(g)
    return span, np.maximum.reduceat(sizes, g[off[:-1]])


def crc(md5s, tag='t1'):
    return np.array([zlib.crc32((m + tag).encode()) % 10000 for m in md5s], np.int64)


def smallest_k(c, notes, target):
    """Smallest k with notes[c < k].sum() >= target."""
    per_k = np.bincount(c, weights=notes, minlength=10000).cumsum()
    k = int(np.searchsorted(per_k, target)) + 1
    assert k <= 10000, f'only {per_k[-1]:,.0f} notes'
    return k


def stream(tar, small, keep):
    """Second pass: stats of every file; tokens and programs of the files keep[part] (bool per file) in memory."""
    proc, tf = tar_stream(tar)
    stats = {p: dict(span=np.zeros(len(small[p]['rows'])), max_onset=np.zeros(len(small[p]['rows']), np.int64))
             for p in PARTS}
    kept = {p: {'tokens.u16': [], 'programs.u8': []} for p in PARTS}
    t0, done = time.time(), set()
    for m in tf:
        part, fname = member(m.name)
        if part not in small or fname not in ('tokens.u16', 'programs.u8'):
            continue
        off, f = small[part]['offsets'], tf.extractfile(m)
        width = 8 if fname == 'tokens.u16' else 1
        buf, row0, i = b'', 0, 0  # the buffer starts at row row0; next file i
        while True:
            chunk = f.read(CHUNK)
            buf += chunk
            rows = row0 + len(buf) // width
            j = int(np.searchsorted(off, rows, side='right')) - 1  # files i..j-1 are complete in the buffer
            if j > i:
                a, b = int(off[i]), int(off[j])
                data = np.frombuffer(buf, np.uint16 if width == 8 else np.uint8, (b - a) * (4 if width == 8 else 1),
                                     (a - row0) * width)
                if width == 8:
                    data = data.reshape(-1, 4)
                    span, mx = file_stats(data, off[i:j + 1] - a)
                    stats[part]['span'][i:j], stats[part]['max_onset'][i:j] = span, mx
                sel = keep[part][i:j]
                if sel.any():
                    kept[part][fname].append(data[np.repeat(sel, np.diff(off[i:j + 1]))].copy())
                i = j
            cut = (int(off[i]) - row0) * width
            buf, row0 = buf[cut:], row0 + cut // width
            if not chunk:
                break
        assert i == len(off) - 1, f'{m.name}: stream ended at file {i} of {len(off) - 1}'
        print(f'{time.time() - t0:5.0f}s {m.name} ({m.size / 1e9:.2f} GB)', flush=True)
        done.add((part, fname))
        if len(done) == 4:
            break
    proc.kill()
    for p in PARTS:
        kept[p] = {k: np.concatenate(v) if v else np.zeros((0, 4) if k == 'tokens.u16' else 0, np.uint16
                                                             if k == 'tokens.u16' else np.uint8)
                   for k, v in kept[p].items()}
    return stats, kept


def write_store(out, tokens, programs, n, header, rows, meta):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'meta.json').unlink(missing_ok=True)  # incomplete until written last
    assert len(tokens) == len(programs) == int(n.sum()) and len(rows) == len(n)
    np.ascontiguousarray(tokens, np.uint16).tofile(out / 'tokens.u16')
    np.ascontiguousarray(programs, np.uint8).tofile(out / 'programs.u8')
    np.save(out / 'offsets.npy', np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
    (out / 'files.csv').write_text('\n'.join([header] + list(rows)) + '\n')
    meta = dict(n_files=len(n), n_notes=int(n.sum()), programs=True, usable_notes=int(n[n >= 64].sum()), **meta)
    (out / 'meta.json').write_text(json.dumps(meta))
    print(f'{out.name}: {meta["n_files"]:,} files, {meta["n_notes"]:,} notes ({meta["usable_notes"]:,} usable), '
          f'{(out / "tokens.u16").stat().st_size * 9 / 8 / 1e9:.2f} GB', flush=True)
    return meta


def aria_flags_train(small, census):
    """Aria-identical flags of the train files from the census hash tables (files.csv store/idx -> discover.npz)."""
    d, a = np.load(census / 'discover.npz'), np.load(census / 'aria.npz')
    codes = {s: i for i, s in enumerate(d['stores'])}
    key = d['store'].astype(np.int64) << 40 | d['idx']
    order = np.argsort(key)
    rows = [r.split(',') for r in small['train']['rows']]
    col = small['train']['header'].split(',')
    si, ii = col.index('store'), col.index('idx')
    want = np.array([codes[r[si]] << 40 | int(r[ii]) for r in rows], np.int64)
    pos = order[np.searchsorted(key, want, sorter=order)]
    assert (key[pos] == want).all(), 'files.csv rows missing from discover.npz'
    return np.isin(d['loose'][pos], a['loose']), d['loose'][pos]


def build(args):
    t0 = time.time()
    small = read_small(args.tar)
    print(f'{time.time() - t0:5.0f}s offsets: ' + ', '.join(f'{p} {len(small[p]["rows"]):,} files, '
                                                            f'{int(small[p]["offsets"][-1]):,} notes' for p in PARTS),
          flush=True)
    col = small['train']['header'].split(',')
    md5 = {p: [r.split(',')[col.index('md5')] for r in small[p]['rows']] for p in PARTS}
    n = {p: np.diff(small[p]['offsets']) for p in PARTS}
    c = crc(md5['train'])
    kmax = smallest_k(c, n['train'], args.target * args.margin)
    keep = {'train': c < kmax, 'validation': np.ones(len(n['validation']), bool)}
    print(f'{time.time() - t0:5.0f}s t1 superset: k < {kmax}, {int(keep["train"].sum()):,} files, '
          f'{int(n["train"][keep["train"]].sum()):,} notes', flush=True)
    stats, kept = stream(args.tar, small, keep)
    for p in PARTS:
        s = stats[p]
        s['nps'] = n[p] / np.maximum(s['span'], 1.0)
        s['dense'] = (s['max_onset'] > args.dense_onset) | (s['nps'] > args.dense_nps)
    census = Path(args.census)
    aria_train, _ = aria_flags_train(small, census)
    aria_loose = np.load(census / 'aria.npz')['loose']
    off_v = small['validation']['offsets']
    val_tok, val_prog = kept['validation']['tokens.u16'], kept['validation']['programs.u8']
    loose_v = np.array([note_hashes(val_tok[off_v[i]:off_v[i + 1]], val_prog[off_v[i]:off_v[i + 1]])[1]
                        for i in range(len(n['validation']))], np.uint64)
    aria_val = np.isin(loose_v, aria_loose)

    # t1: the smallest k reaching the target among the non-dense files of the superset
    dense_t = stats['train']['dense']
    k = smallest_k(np.where(dense_t, 10000, c), n['train'], args.target)
    assert k <= kmax, f'k {k} > superset {kmax}: raise --margin'
    t1 = (c < k) & ~dense_t
    in_sup = t1[keep['train']]  # superset files, in order
    rows_sup = np.repeat(in_sup, n['train'][keep['train']])
    tag = dict(source=Path(args.tar).name, dense_rule=f'max_onset > {args.dense_onset} or nps > {args.dense_nps:g}')
    metas = {'t1': write_store(CACHE / 'cdisc_t1_train', kept['train']['tokens.u16'][rows_sup],
                               kept['train']['programs.u8'][rows_sup], n['train'][t1], small['train']['header'],
                               [r for r, x in zip(small['train']['rows'], t1) if x],
                               dict(part='train', selection=f"crc32(md5 + 't1') % 10000 < {k}, not dense", **tag))}
    kept['train'] = None
    dense_v = stats['validation']['dense']
    groups = {'vn': ~dense_v & ~aria_val, 'va': ~dense_v & aria_val, 'vd': dense_v}
    for name, sel in groups.items():
        rows_v = np.repeat(sel, n['validation'])
        metas[name] = write_store(CACHE / f'cdisc_{name}_validation', val_tok[rows_v], val_prog[rows_v],
                                  n['validation'][sel], small['validation']['header'],
                                  [r for r, x in zip(small['validation']['rows'], sel) if x],
                                  dict(part='validation', selection=VAL_STORES[name], **tag))
    np.savez_compressed(CACHE / 'cdisc_stats.npz', part=np.repeat([0, 1], [len(n['train']), len(n['validation'])]).astype(np.uint8),
             n_notes=np.concatenate([n['train'], n['validation']]),
             **{key: np.concatenate([stats['train'][key], stats['validation'][key]])
                for key in ('span', 'nps', 'max_onset', 'dense')},
             aria=np.concatenate([aria_train, aria_val]), t1=np.concatenate([t1, np.zeros(len(n['validation']), bool)]),
             dense_rule=tag['dense_rule'])
    for p, aria in (('train', aria_train), ('validation', aria_val)):
        d, nn = stats[p]['dense'], n[p]
        print(f'{p}: dense {int(d.sum()):,} files ({d.mean():.1%}), {int(nn[d].sum()):,} notes ({nn[d].sum() / nn.sum():.1%}); '
              f'Aria-identical {int(aria.sum()):,} files, {nn[aria].sum() / nn.sum():.1%} of notes; '
              f'onset rule alone {nn[stats[p]["max_onset"] > args.dense_onset].sum() / nn.sum():.1%}, '
              f'nps rule alone {nn[stats[p]["nps"] > args.dense_nps].sum() / nn.sum():.1%} of notes', flush=True)
    print(f'{time.time() - t0:5.0f}s done')
    return metas


def check(args):
    """CR-4: num_notes() = the build's usable notes; each val store gives 1,200 windows; train and val md5 sets
    disjoint; no dense file in vn / va (stats recomputed from the written stores)."""
    from musicar.data.loader import NoteWindowDataset, val_loader
    ok = True
    md5s = {}
    for name, split in (('t1', 'train'), ('vn', 'validation'), ('va', 'validation'), ('vd', 'validation')):
        path = CACHE / f'cdisc_{name}_{split}'
        meta = json.loads((path / 'meta.json').read_text())
        off = np.load(path / 'offsets.npy')
        tok = np.fromfile(path / 'tokens.u16', np.uint16).reshape(-1, 4)
        good = len(tok) == off[-1] == meta['n_notes'] and (path / 'programs.u8').stat().st_size == off[-1]
        lines = (path / 'files.csv').read_text().splitlines()
        col = lines[0].split(',').index('md5')
        md5s[name] = {r.split(',')[col] for r in lines[1:]}
        good &= len(md5s[name]) == len(lines) - 1 == meta['n_files']
        span, mx = file_stats(tok, off)
        dense = (mx > args.dense_onset) | (np.diff(off) / np.maximum(span, 1.0) > args.dense_nps)
        line = f'{path.name}: {meta["n_files"]:,} files, {meta["n_notes"]:,} notes, dense {int(dense.sum()):,}'
        if name in ('vn', 'va'):
            good &= not dense.any()
        if name == 'vd':
            good &= bool(dense.all())
        kw = dict(special_tokens=True, programs=True, min_notes=64)
        if split == 'train':
            ds = NoteWindowDataset(f'store:{CACHE / "cdisc_t1"}', '.', 'train', 2048, pack_short=True, **kw)
            good &= ds.num_notes() == meta['usable_notes']
            line += f'; num_notes() {ds.num_notes():,} vs build {meta["usable_notes"]:,}'
        else:
            dl = val_loader(f'store:{CACHE / f"cdisc_{name}"}', batch_size=4, max_batches=300, block_size=2048,
                            root_dir='.', **kw)
            good &= len(dl.dataset) == 1200
            line += f'; val windows {len(dl.dataset):,}'
        print(f'{"ok  " if good else "BAD "}{line}', flush=True)
        ok &= bool(good)
    overlap = md5s['t1'] & (md5s['vn'] | md5s['va'] | md5s['vd'])
    vals = [md5s['vn'] & md5s['va'], md5s['vn'] & md5s['vd'], md5s['va'] & md5s['vd']]
    print(f'md5 overlap train/val {len(overlap)}, between val stores {sum(map(len, vals))}')
    ok &= not overlap and not any(vals)
    print('CHECK OK' if ok else 'CHECK FAILED')
    return ok


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('cmd', choices=['build', 'check'])
    p.add_argument('--tar', default=str(Path.home() / '~musicar_export/stores_discover.tar.zst'))
    p.add_argument('--census', default='data/cache/census_k4/census/hashes', help='discover.npz and aria.npz (census/hashes/ of census.tar.zst, extracted)')
    p.add_argument('--target', type=float, default=167.9e6, help='t1 notes')
    p.add_argument('--margin', type=float, default=1.6, help='t1 superset drawn before the dense flags')
    p.add_argument('--dense-onset', type=int, default=64)
    p.add_argument('--dense-nps', type=float, default=100.0)
    args = p.parse_args()
    ok = bool(build(args)) if args.cmd == 'build' else check(args)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
