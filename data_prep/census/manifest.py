"""
Per-source manifests of the census: <census>/manifests/<source>.csv, one row per listed file.

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/manifest.py --sources gigamidi [--tag T]
    ... manifest.py --sources gigamidi,discover,aria --verify      # counts only; exit 1 on a mismatch

Columns: source, file_id, md5, split, status, n_notes, n_programs, exact, loose, loose_t, licence, then the
source-specific extras (GigaMIDI: category, reason (of a failure); Discover: discover_status (its own csv decision);
Aria / generic: every further csv column, e.g. segment, genre, composer, audio_score; with tiers: in_pruned,
in_deduped, in_unique).
status: kept | failed | excluded (outside the --aria-tier of that index) | dup_<level>_<source of the first copy>.
Hashes are 16-hex, empty when the file was not parsed; n_notes / n_programs empty when unknown.
The status comes from global_index[_<tag>].npz (default: the full-tier index).

--verify checks, per source: manifest row source == table rows == listed files; per store: rows mapped to it ==
meta.json n_files and their notes == meta.json n_notes; every kept file is in a store; kept == stored files - stored
files that the index marks as duplicates (or excluded); and, if the manifest csv exists, its row and kept counts.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from global_index import EXCLUDED, FAILED, KEPT, index_path, load_tables
from tables import LEVELS, TIERS, extras_fields, inputs_for, kv_list, licence_of, source_rows

BASE = ['source', 'file_id', 'md5', 'split', 'status', 'n_notes', 'n_programs', 'exact', 'loose', 'loose_t', 'licence']


def status_names(sources):
    """status code + first source -> string."""
    def name(code, fs):
        if code == KEPT:
            return 'kept'
        if code == FAILED:
            return 'failed'
        if code == EXCLUDED:
            return 'excluded'
        return f'dup_{LEVELS[code]}_{sources[fs]}'
    return name


def write_manifest(src, si, z, t, meta, inp, census_dir, licence):
    a, b = int(z['source_start'][si]), int(z['source_start'][si + 1])
    st, fs = z['status'][a:b], z['first_source'][a:b]
    sources = [str(s) for s in z['sources']]
    name = status_names(sources)
    cols = {k: t[k] for k in ('n_notes', 'n_programs', 'exact', 'loose', 'loose_t', 'parsed', 'tier')}
    extras = extras_fields(src, inp)
    tier_cols = [f'in_{x}' for x in TIERS] if meta['has_tiers'] else []
    out = Path(census_dir) / 'manifests' / f'{src}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.csv.tmp')
    n = 0
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(BASE + extras + tier_cols)
        for i, r in enumerate(source_rows(src, inp)):
            p = bool(cols['parsed'][i])
            nn, npg = int(cols['n_notes'][i]), int(cols['n_programs'][i])
            hx = [f'{int(cols[k][i]):016x}' if p and cols[k][i] else '' for k in ('exact', 'loose', 'loose_t')]
            tb = int(cols['tier'][i])
            w.writerow([src, r['file_id'], r['md5'], r['split'], name(int(st[i]), int(fs[i])),
                        nn if nn >= 0 else '', npg if npg >= 0 else '', *hx, licence]
                       + [r['extras'].get(k, '') for k in extras] + [int(bool(tb >> j & 1)) for j in range(len(tier_cols))])
            n += 1
    assert n == b - a, f'{src}: {n} manifest rows but {b - a} index rows'
    tmp.replace(out)
    return out


def verify(src, si, z, t, meta, census_dir, cache_dir):
    errors = []
    a, b = int(z['source_start'][si]), int(z['source_start'][si + 1])
    st = z['status'][a:b]
    store, nn = t['store'], t['n_notes']
    if len(store) != b - a:
        errors.append(f'table rows {len(store)} != index rows {b - a}')
    for k, sname in enumerate(meta['stores']):
        mp = Path(cache_dir) / sname / 'meta.json'
        if not mp.exists():
            errors.append(f'{sname}: meta.json missing (store incomplete)')
            continue
        m = json.loads(mp.read_text())
        rows = store == k
        if int(rows.sum()) != m['n_files']:
            errors.append(f'{sname}: {int(rows.sum())} rows but meta n_files {m["n_files"]}')
        if int(nn[rows].sum()) != m['n_notes']:
            errors.append(f'{sname}: {int(nn[rows].sum())} notes but meta n_notes {m["n_notes"]}')
    stored = store >= 0
    kept = st == KEPT
    # train-first swaps (Discover): a kept file outside the store is allowed only as the first copy of a stored file
    # the index dropped (see usable.py); anything else is an error
    f = z['first'][a:b] - a
    d = np.flatnonzero(stored & (st >= 0) & (f >= 0) & (f < b - a))
    swapped = np.zeros(b - a, bool)
    swapped[f[d]] = True
    swapped &= kept & ~stored
    if (kept & ~stored & ~swapped).any():
        errors.append(f'{int((kept & ~stored & ~swapped).sum())} kept files are not in a store')
    stored_dups = int((stored & (st != KEPT)).sum())
    if int(kept.sum()) != int(stored.sum()) - stored_dups + int(swapped.sum()):
        errors.append(f'kept {int(kept.sum())} != stored {int(stored.sum())} - stored dups {stored_dups} '
                      f'+ swaps {int(swapped.sum())}')
    out = Path(census_dir) / 'manifests' / f'{src}.csv'
    if out.exists():
        rows = kept_rows = 0
        with open(out, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows += 1
                kept_rows += r['status'] == 'kept'
        if rows != b - a:
            errors.append(f'{out}: {rows} rows but {b - a} listed files')
        if kept_rows != int(kept.sum()):
            errors.append(f'{out}: {kept_rows} kept rows but the index has {int(kept.sum())}')
    summary = dict(listed=b - a, stored=int(stored.sum()), kept=int(kept.sum()), stored_dups=stored_dups,
                   train_first_swaps=int(swapped.sum()),
                   failed=int((st == FAILED).sum()), excluded=int((st == EXCLUDED).sum()))
    return errors, summary


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sources', required=True, help='comma-separated; must be in the index')
    ap.add_argument('--tag', default='', help='global index tag (default: the full-tier index)')
    ap.add_argument('--verify', action='store_true', help='check counts only, exit 1 on a mismatch')
    ap.add_argument('--census-dir', default='census')
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--input', action='append', help='input overrides as in global_index.py')
    ap.add_argument('--licence', action='append', help='licence tag override, e.g. pdmx=pd/cc0')
    args = ap.parse_args(argv)
    z = np.load(index_path(args.census_dir, args.tag))
    index_sources = [str(s) for s in z['sources']]
    inputs = kv_list(args.input)
    tables, metas = load_tables(index_sources, args.census_dir, inputs)
    lic = kv_list(args.licence)
    failed = False
    for src in args.sources.split(','):
        si = index_sources.index(src)
        if args.verify:
            errors, summary = verify(src, si, z, tables[src], metas[src], args.census_dir, args.cache_dir)
            print(f'{src}: {"OK" if not errors else "MISMATCH"} {summary}')
            for e in errors:
                print(f'  {src}: {e}')
            failed |= bool(errors)
        else:
            out = write_manifest(src, si, z, tables[src], metas[src], inputs_for(src, args.census_dir, inputs),
                                 args.census_dir, licence_of(src, lic))
            print(f'{out}: {int(z["source_start"][si + 1] - z["source_start"][si]):,} rows')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
