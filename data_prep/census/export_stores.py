"""
Kept-only token stores for the laptop pull: for each source, the files kept at a dedup level (default loose, the
headline level) in a global index, copied into a filtered store per split, then packed as export/stores_<src>.tar.zst.

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/export_stores.py --sources gigamidi,discover

Each filtered store <src>_<split>/ has the repo's layout (tokens.u16, programs.u8 for multi-instrument sources,
offsets.npy, meta.json) plus files.csv: one row per file in store order (file_id, md5, original store, original idx),
so every exported file traces back to the census manifest. meta.json says the level, the index tag and the source.
Later sources never change which earlier files are kept (first copy wins in source order), so sources can be exported
as soon as they are indexed.
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import LEVELS
from global_index import KEPT, index_path, load_tables
from tables import inputs_for, source_rows


def kept_mask(status, level):
    return (status == KEPT) | (status > LEVELS.index(level))


def export_source(src, si, z, tables, metas, args):
    t = {k: np.asarray(tables[src][k]) for k in ('store', 'idx')}  # npz members re-read on every access: load once
    a, b = z['source_start'][si], z['source_start'][si + 1]
    keep = kept_mask(z['status'][a:b], args.level) & (t['store'] >= 0)
    need = set(np.flatnonzero(keep).tolist())
    ids = {}
    for r, row in enumerate(source_rows(src, inputs_for(src, args.census_dir, {}))):  # table order
        if r in need:
            ids[r] = (row['file_id'], row['md5'])
    stage = Path(args.stage) / src
    if stage.exists():
        shutil.rmtree(stage)
    total = dict(files=0, notes=0)
    for k, name in enumerate(metas[src]['stores']):
        rows = np.flatnonzero(keep & (t['store'] == k))
        rows = rows[np.argsort(t['idx'][rows], kind='stable')]   # original store order
        src_dir = Path(args.cache_dir) / name
        off = np.load(src_dir / 'offsets.npy')
        n_all = int(off[-1])
        tok = np.memmap(src_dir / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n_all, 4))
        has_p = (src_dir / 'programs.u8').exists()
        prog = np.memmap(src_dir / 'programs.u8', dtype=np.uint8, mode='r', shape=(n_all,)) if has_p else None
        split = name[len(src) + 1:] if name.startswith(src + '_') else name
        out = stage / f'{src}_{split}'
        out.mkdir(parents=True)
        offsets = [0]
        with open(out / 'tokens.u16', 'wb') as ft, (open(out / 'programs.u8', 'wb') if has_p else open('/dev/null', 'wb')) as fp, \
                open(out / 'files.csv', 'w', newline='', encoding='utf-8') as fc:
            w = csv.writer(fc)
            w.writerow(['file_id', 'md5', 'store', 'idx'])
            for r in rows:
                i = int(t['idx'][r])
                lo, hi = int(off[i]), int(off[i + 1])
                ft.write(np.ascontiguousarray(tok[lo:hi]).tobytes())
                if has_p:
                    fp.write(np.ascontiguousarray(prog[lo:hi]).tobytes())
                offsets.append(offsets[-1] + hi - lo)
                fid, md5 = ids[int(r)]
                w.writerow([fid, md5, name, i])
        np.save(out / 'offsets.npy', np.array(offsets, dtype=np.int64))
        meta = dict(n_files=len(offsets) - 1, n_notes=offsets[-1], source=src, level=args.level,
                    index_tag=args.tag, filtered_from=name)
        if has_p:
            meta['programs'] = True
        (out / 'meta.json').write_text(json.dumps(meta))
        total['files'] += meta['n_files']
        total['notes'] += meta['n_notes']
        print(f'  {out.name}: {meta["n_files"]:,} files, {meta["n_notes"]:,} notes', flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tar = Path(args.out) / f'stores_{src}.tar.zst'
    cmd = f'tar -C {Path(args.stage)} -cf - {src} | zstd -q -f -T{args.threads} -{args.zstd} -o {tar}'
    subprocess.run(cmd, shell=True, check=True)
    shutil.rmtree(stage)
    print(f'{src}: {total["files"]:,} files, {total["notes"]:,} notes -> {tar} ({tar.stat().st_size / 1e9:.2f} GB)',
          flush=True)
    return total


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sources', required=True, help='which sources of the index to export')
    ap.add_argument('--tag', default='', help='global index tag (default: the core-three index)')
    ap.add_argument('--level', default='loose', choices=LEVELS)
    ap.add_argument('--census-dir', default='census')
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--stage', default='/workspace/export_stage')
    ap.add_argument('--out', default='/workspace/export')
    ap.add_argument('--zstd', type=int, default=12)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    z = np.load(index_path(args.census_dir, args.tag))
    sources = [str(s) for s in z['sources']]
    tables, metas = load_tables(sources, args.census_dir, {})
    for src in args.sources.split(','):
        export_source(src, sources.index(src), z, tables, metas, args)


if __name__ == '__main__':
    main()
