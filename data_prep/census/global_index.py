"""
Global cross-source deduplication index of the census.

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/global_index.py \
        --sources gigamidi,discover,aria [--aria-tier full|pruned|deduped|unique] [--tag NAME]

Rule: first copy wins, by source order (--sources, = CLAUDE.md processing order), then by split inside a source (train,
then validation, then test: an eval file whose content also occurs in train is the dropped copy; counted per level as
<source>_eval_in_train in counts.json), then by row order inside a source and split (tables.py). For each level (md5, exact, loose, loose_t) the first copy of every hash is found over ALL parsed,
included files at once (np.unique with return_index on the concatenated uint64 hashes: mergesort, so the index is the
first occurrence; no Python sets). A file's status is the FIRST level at which it is not the first copy of its hash:
    kept | dup_<level>_<source of the first copy> | failed (not parsed) | excluded (Aria row outside --aria-tier)
"within" = the first copy is in the same source, "cross" = in an earlier one. Levels nest (same tokens -> same
(onset, pitch) set -> same transposed set), so a file dropped at loose is also a duplicate at loose_t and "kept after
level L" = status not dup at L or any level before it. Files dropped at an earlier level still own their hashes at
the later levels (e.g. an Aria file byte-identical to a GigaMIDI file but tokenized differently by the piano
tokenizer): a later file with the same notes is a duplicate of it.

Later sources never change the status of earlier ones, so the core-three numbers are the first three sources of a
run over all sources. Dropping a source in the middle (e.g. the 'unclear' licences) needs its own run (--tag).
Re-running is cheap: the per-source tables are cached (tables.py) and only rebuilt when their inputs change.

Writes <census>/hashes/global_index[_<tag>].npz (per file, sources concatenated in order: source, row, status
(-3 excluded, -2 failed, -1 kept, 0..3 = dup at LEVELS[i]), first (global row of the first copy, -1), first_source
(255 = none); plus sources, source_start, levels, licences) and <census>/counts[_<tag>].json.
Default tag: '' for --aria-tier full, else 'aria-<tier>'.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tables import LEVELS, TIERS, inputs_for, kv_list, licence_of, load_table

KEPT, FAILED, EXCLUDED = -1, -2, -3


def first_copy(h, valid, order=None):
    """Global row of the first row (in `order`, default row order) with the same hash (-1 where not valid)."""
    idx = np.flatnonzero(valid) if order is None else order[valid[order]]
    out = np.full(len(h), -1, np.int64)
    if len(idx):
        _, fi, inv = np.unique(h[idx], return_index=True, return_inverse=True)
        out[idx] = idx[fi[inv.ravel()]]
    return out


def load_tables(sources, census_dir, inputs, rebuild=()):
    tables, metas = {}, {}
    for src in sources:
        tables[src], metas[src] = load_table(src, inputs_for(src, census_dir, inputs), census_dir, earlier=tables,
                                             rebuild=src in rebuild)
    return tables, metas


def index_path(census_dir, tag):
    return Path(census_dir) / 'hashes' / f'global_index{"_" + tag if tag else ""}.npz'


ARCHIVE_ORDER = {"discover"}  # sources whose within-source first copy is archive order, not train-first


def counts_path(census_dir, tag):
    return Path(census_dir) / f'counts{"_" + tag if tag else ""}.json'


def build_index(sources, tables, aria_tier='full'):
    lens = [len(tables[s]['split']) for s in sources]
    start = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    n = int(start[-1])
    source = np.repeat(np.arange(len(sources), dtype=np.uint8), lens)
    parsed = np.concatenate([tables[s]['parsed'] for s in sources])
    included = np.ones(n, bool)
    if aria_tier != 'full' and 'aria' in sources:
        a = sources.index('aria')
        bit = 1 << TIERS.index(aria_tier)
        included[start[a]:start[a + 1]] = (tables['aria']['tier'] & bit) > 0
    status = np.full(n, KEPT, np.int8)
    status[~parsed] = FAILED
    status[~included] = EXCLUDED
    first = np.full(n, -1, np.int64)
    rows = np.arange(n, dtype=np.int64)
    # first-copy priority: source order, then split (train before validation before test: an eval file that also
    # occurs in train is the dropped copy, counted as <src>_eval_in_train), then row order
    # Discover is the exception: prepare_discover.py already decided its duplicates in archive order and stored only
    # those winners, so its rows keep archive order (its split is an arbitrary 0.5% crc32 cut; nothing to protect)
    split = np.concatenate([np.zeros_like(tables[s]['split']) if s in ARCHIVE_ORDER else tables[s]['split']
                            for s in sources])
    order = np.lexsort((rows, split, source))
    del split
    for li, level in enumerate(LEVELS):
        h = np.concatenate([tables[s][level] for s in sources])
        valid = parsed & included & (h != 0)
        f = first_copy(h, valid, order)
        del h
        dup = (status == KEPT) & valid & (f != rows)
        status[dup] = li
        first[dup] = f[dup]
        del f, dup
    first_source = np.full(n, 255, np.uint8)
    has = first >= 0
    first_source[has] = source[first[has]]
    row = rows - start[source]
    return dict(source=source, row=row, status=status, first=first, first_source=first_source, source_start=start)


def counts(sources, tables, idx, licences, metas=None):
    """Per-source and total file / note counts at every level."""
    out = {}
    tot = None
    for si, src in enumerate(sources):
        a, b = idx['source_start'][si], idx['source_start'][si + 1]
        st, fs = idx['status'][a:b], idx['first_source'][a:b]
        nn = np.maximum(tables[src]['n_notes'], 0)
        c = dict(licence=licences[src], files_listed=int(b - a), files_parsed=int(((st != FAILED) & (st != EXCLUDED)).sum()),
                 files_failed=int((st == FAILED).sum()), files_excluded=int((st == EXCLUDED).sum()),
                 notes_raw=int(nn[(st != FAILED) & (st != EXCLUDED)].sum()), levels={})
        alive = (st != FAILED) & (st != EXCLUDED)
        for li, level in enumerate(LEVELS):
            d = st == li
            within, cross = d & (fs == si), d & (fs != si)
            alive &= ~d
            by_first = {sources[j]: int((d & (fs == j)).sum()) for j in range(si + 1) if (d & (fs == j)).any()}
            c['levels'][level] = dict(
                dropped_within_files=int(within.sum()), dropped_within_notes=int(nn[within].sum()),
                dropped_cross_files=int(cross.sum()), dropped_cross_notes=int(nn[cross].sum()),
                dropped_by_first_source=by_first,
                kept_files=int(alive.sum()), kept_notes=int(nn[alive].sum()))
        # leakage: validation/test files dropped because their first copy is a train file of the same source
        sp, f = tables[src]['split'], idx['first'][a:b]
        in_src = (f >= a) & (f < b)
        fsplit = np.full(b - a, 255, np.uint8)
        fsplit[in_src] = sp[f[in_src] - a]
        if (sp > 0).any() and (sp == 0).any():
            eit = {}
            for li, level in enumerate(LEVELS):
                m = (st == li) & (sp > 0) & (fsplit == 0)
                eit[level] = dict(files=int(m.sum()), notes=int(nn[m].sum()))
            c['eval_in_train'] = eit
            out[f'{src}_eval_in_train'] = eit
        if metas and metas.get(src, {}).get('checks'):
            c['checks'] = metas[src]['checks']
        if src == 'discover':
            ch = c.setdefault('checks', {})
            ch['discover_status_mismatch'], ch['train_first_swaps'] = discover_mismatch(sources, tables, idx, si)
        out[src] = c
        if tot is None:
            tot = json.loads(json.dumps(c))
            tot.pop('licence'), tot.pop('checks', None), tot.pop('eval_in_train', None)
        else:
            for k in ('files_listed', 'files_parsed', 'files_failed', 'files_excluded', 'notes_raw'):
                tot[k] += c[k]
            for level in LEVELS:
                for k, v in c['levels'][level].items():
                    if k == 'dropped_by_first_source':
                        for s2, v2 in v.items():
                            tot['levels'][level][k][s2] = tot['levels'][level][k].get(s2, 0) + v2
                    else:
                        tot['levels'][level][k] += v
    if tot is not None:
        out['total'] = tot
    return out


def discover_mismatch(sources, tables, idx, si):
    """Rows where Discover's own csv decision disagrees with the global index (expected 0).
    Allowed: a csv-kept row that becomes dup_loose_t_*; a csv dup at level L that the index drops at an earlier level
    (e.g. a dup_loose_self row whose tokens equal another dropped row: the index sees every row, Discover only its
    kept ones); a dup_*_giga row that is a byte copy of an earlier Discover row. Excluded and counted as
    train_first_swaps: Discover decided in archive order, the index puts train first, so a kept validation row with a
    later train copy is dropped and that train copy (not in the store) kept; every row sharing that loose hash is
    left out of the comparison."""
    a, b = idx['source_start'][si], idx['source_start'][si + 1]
    st, fs, f = idx['status'][a:b], idx['first_source'][a:b], idx['first'][a:b]
    t = tables['discover']
    sin, sp, loose = t['status_in'], t['split'], t['loose']
    in_src = (f >= a) & (f < b)
    fsplit = np.full(b - a, 255, np.uint8)
    fsplit[in_src] = sp[f[in_src] - a]
    displaced = (sin == 0) & (sp > 0) & (st >= 0) & (fsplit == 0)
    excl = np.isin(loose, loose[displaced]) & (loose != 0)
    g = sources.index('gigamidi') if 'gigamidi' in sources else -1
    ok = np.ones(b - a, bool)
    for code, (lvl, own) in {1: (0, g), 2: (1, g), 3: (1, si), 4: (2, g), 5: (2, si)}.items():
        m = sin == code
        good = (st >= 0) & (st <= lvl) & ((fs == own) | ((own == g) & (fs == si) & (st == 0)))
        ok[m] = good[m]
    m = sin == 0
    ok[m] = ((st == KEPT) | (st == 3))[m]
    m = sin == 6
    ok[m] = (st == FAILED)[m]
    return int((~ok & ~excl).sum()), int(displaced.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sources', default='gigamidi,discover,aria', help='in processing order (first copy wins)')
    ap.add_argument('--aria-tier', default='full', choices=['full', *TIERS])
    ap.add_argument('--tag', default=None, help="output suffix (default '' for full, else aria-<tier>)")
    ap.add_argument('--census-dir', default='census')
    ap.add_argument('--input', action='append', help='input override, e.g. aria.csv=data/aria/aria.csv, '
                                                     'gigamidi.failed_csv=..., aria.tiers=...')
    ap.add_argument('--licence', action='append', help='licence tag override, e.g. pdmx=pd/cc0')
    ap.add_argument('--rebuild', default='', help='comma-separated sources whose cached table to rebuild')
    args = ap.parse_args(argv)
    sources = args.sources.split(',')
    tag = args.tag if args.tag is not None else ('' if args.aria_tier == 'full' else f'aria-{args.aria_tier}')
    tables, metas = load_tables(sources, args.census_dir, kv_list(args.input), set(filter(None, args.rebuild.split(','))))
    if args.aria_tier != 'full' and 'aria' in sources:
        assert metas['aria']['has_tiers'], 'aria table has no tier info: run aria_tiers.py, then --rebuild aria'
    idx = build_index(sources, tables, args.aria_tier)
    lic = {s: licence_of(s, kv_list(args.licence)) for s in sources}
    path = index_path(args.census_dir, tag)
    np.savez(path, sources=np.array(sources), levels=np.array(LEVELS), aria_tier=np.array(args.aria_tier),
             licences=np.array([lic[s] for s in sources]), **idx)
    c = counts(sources, tables, idx, lic, metas)
    c['_meta'] = dict(sources=sources, aria_tier=args.aria_tier, tag=tag, levels=LEVELS)
    counts_path(args.census_dir, tag).write_text(json.dumps(c, indent=1))
    print(f'{path}: {len(idx["status"]):,} files')
    for src in sources + ['total']:
        x = c[src]
        lv = x['levels']
        print(f"{src:>10}: listed {x['files_listed']:,} parsed {x['files_parsed']:,} failed {x['files_failed']:,} "
              f"excluded {x['files_excluded']:,} | notes raw {x['notes_raw']:,} -> "
              + ' -> '.join(f"{l} {lv[l]['kept_notes']:,}" for l in LEVELS))
    return c


if __name__ == '__main__':
    main()
