"""
Usable notes at context 2048, as the training loader counts them, per source and scenario, from the global index.

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/usable.py [--tags ,aria-pruned,aria-deduped]

For every store of every source a real NoteWindowDataset is built (csv_path='store:<cache>/<prefix>', split, block_size
2048, pack_short=True, min_notes=64, special_tokens=True, programs = the store has programs.u8). The loader's own kept
files (ds.file_ids, ds.lengths) give each stored file's usable notes = ds.lengths - 2 (BOS/EOS are not notes; the sum
over a store equals ds.num_notes()); files the loader drops count 0. No rule is re-implemented here.
Usable notes of a scenario = the sum over the stored files that survive dedup at the level (loose = headline,
loose_t next to it). Only levels >= loose are reported: Discover's files dropped at exact/loose are not in its store.

Shape stats over the notes of the files kept at the level (raw note counts, not usable):
    long_share_gt_{2048,4096,8192}: share of notes in files with more than 2k / 4k / 8k notes
    mix (share of notes; from hash_store's per-file stats, piano stores = all program 0):
        piano_only    every note has GM program 0-7 (piano family) -- all of Aria and the other piano stores
        drums_only    every note is a drum note (program 128)
        single_other  one pitched program outside 0-7, no drums (e.g. solo guitar, a score in one GM program)
        multi         anything else: >= 2 pitched programs, or pitched + drums
        unknown       no stats (should be 0)

Writes <census>/usable[_<tag>].json per tag.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from global_index import KEPT, index_path, load_tables
from tables import LEVELS, kv_list

REPORT_LEVELS = ['loose', 'loose_t']
LONG = (2048, 4096, 8192)


def store_usable(cache_dir, store_name, block_size=2048, min_notes=64):
    """Per-file usable notes of one store (0 = dropped by the loader), from a real NoteWindowDataset."""
    from musicar.data.loader import NoteWindowDataset
    path = Path(cache_dir) / store_name
    prefix, split = store_name.rsplit('_', 1)
    meta = json.loads((path / 'meta.json').read_text())
    u = np.zeros(meta['n_files'], np.int64)
    try:
        ds = NoteWindowDataset(csv_path=f'store:{Path(cache_dir) / prefix}', root_dir='.', split=split,
                               block_size=block_size, pack_short=True, min_notes=min_notes, special_tokens=True,
                               programs=(path / 'programs.u8').exists(), cache_dir=str(cache_dir))
    except ValueError:  # "No MIDI sequences longer than block_size": nothing usable
        return u
    u[ds.file_ids] = ds.lengths - 2 * ds.pad
    assert int(u.sum()) == ds.num_notes()
    return u


def mix_of(t, m):
    nn, dr = t['n_notes'][m], t['n_drum_notes'][m]
    pi, npp = t['n_piano_notes'][m], t['n_pitched_programs'][m]
    unknown = (dr < 0) | (pi < 0)
    drums = ~unknown & (dr == nn)
    piano = ~unknown & ~drums & (pi == nn)
    single = ~unknown & ~drums & ~piano & (dr == 0) & (npp == 1)
    multi = ~unknown & ~drums & ~piano & ~single
    tot = max(int(nn.sum()), 1)
    return {k: round(int(nn[v].sum()) / tot, 6) for k, v in
            dict(piano_only=piano, drums_only=drums, single_other=single, multi=multi, unknown=unknown).items()}


def usable_for_tag(tag, census_dir, cache_dir, inputs, cache=None):
    z = np.load(index_path(census_dir, tag))
    sources = [str(s) for s in z['sources']]
    tables, metas = load_tables(sources, census_dir, inputs)
    cache = {} if cache is None else cache
    out = {}
    tot = {lv: dict(kept_files=0, kept_notes=0, usable_notes=0) for lv in REPORT_LEVELS}
    tot_stores = dict(stored_files=0, stored_notes=0, usable_in_stores=0, loader_files=0)
    for si, src in enumerate(sources):
        t = tables[src]
        a, b = z['source_start'][si], z['source_start'][si + 1]
        st = z['status'][a:b]
        store, idx = t['store'], t['idx']
        u = np.zeros(b - a, np.int64)
        for k, name in enumerate(metas[src]['stores']):
            if name not in cache:
                cache[name] = store_usable(cache_dir, name)
            rows = np.flatnonzero(store == k)
            u[rows] = cache[name][idx[rows]]
        stored = store >= 0
        nn = np.maximum(t['n_notes'], 0)
        c = dict(stored_files=int(stored.sum()), stored_notes=int(nn[stored].sum()),
                 usable_in_stores=int(u.sum()), loader_files=int((u > 0).sum()), levels={})
        for lv in REPORT_LEVELS:
            li = LEVELS.index(lv)
            kept = (st == KEPT) | (st > li)
            ul, borrowed = u, 0
            missing = kept & ~stored
            if missing.any():
                # train-first swaps (Discover): a kept train file that Discover itself dropped (not in its store) as a
                # copy of an earlier validation file; the index now drops that stored file with this one as its first
                # copy. Same notes at exact/loose (same n_notes), so its loader count is borrowed from the stored file.
                f = z['first'][a:b] - a
                d = np.flatnonzero(stored & (st >= 0) & (st <= li) & (f >= 0) & (f < b - a))
                d = d[missing[f[d]]]
                ul = u.copy()
                ul[f[d]] = u[d]
                covered = np.zeros(b - a, bool)
                covered[f[d]] = True
                borrowed = int(missing.sum())
                assert not (missing & ~covered).any(), f'{src}: kept files outside the store at level {lv}'
            kn = nn[kept]
            c['levels'][lv] = dict(kept_files=int(kept.sum()), kept_notes=int(kn.sum()), usable_notes=int(ul[kept].sum()),
                                   usable_borrowed_files=borrowed,
                                   **{f'long_share_gt_{L}': round(float(kn[kn > L].sum()) / max(int(kn.sum()), 1), 6)
                                      for L in LONG},
                                   mix=mix_of(t, kept))
            for k in tot[lv]:
                tot[lv][k] += c['levels'][lv][k]
        for k in tot_stores:
            tot_stores[k] += c[k]
        out[src] = c
    out['total'] = dict(**tot_stores, levels=tot)
    out['_meta'] = dict(tag=tag, sources=sources, aria_tier=str(z['aria_tier']), block_size=2048, min_notes=64,
                        pack_short=True)
    p = Path(census_dir) / f'usable{"_" + tag if tag else ""}.json'
    p.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--tags', default='', help="comma-separated global index tags ('' = the default index)")
    ap.add_argument('--census-dir', default='census')
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--input', action='append', help='input overrides as in global_index.py')
    args = ap.parse_args(argv)
    cache, res = {}, {}
    for tag in args.tags.split(','):
        out = usable_for_tag(tag, args.census_dir, args.cache_dir, kv_list(args.input), cache)
        res[tag] = out
        for src in out['_meta']['sources'] + ['total']:
            lv = out[src]['levels']
            print(f"[{tag or 'default'}] {src:>10}: " + ' | '.join(
                f"{l}: kept {lv[l]['kept_notes']:,} usable@2048 {lv[l]['usable_notes']:,}" for l in REPORT_LEVELS))
    return res


if __name__ == '__main__':
    main()
