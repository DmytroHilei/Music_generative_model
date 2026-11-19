"""
Per-source file tables of the census: one row per LISTED file of a source, in a fixed order, with the hashes and stats
needed by global_index.py / usable.py / manifest.py. Cached in <census>/hashes/table_<source>.npz and rebuilt when an
input is newer than the cache (or with --rebuild), so adding a later source never re-reads the earlier ones.

Row order (the row order of the manifest; first-copy-wins inside a source uses split first (train, validation, test),
then this order, see global_index.py):
    gigamidi  rows of data/gigamidi/gigamidi.csv (validation, test, train: the processing order), then the rows of
              gigamidi_failed.csv (parse failures and oversized members)
    discover  rows of data/discover/discover.csv (archive order; kept, dup_* and failed rows)
    generic   (aria and sources 4-8) rows of the census csv: file_id[, segment], split, status (kept | failed), n_notes,
              md5, any further columns (kept as manifest extras). Without a status column every row is a kept file.
In every csv the kept / stored rows of a split are in store order, so the k-th stored row of split s is file k of the
store <source>_<s>; its hashes and shape stats come from <census>/hashes/<source>.npz (hash_store.py).

Files not in a store (Discover's dropped rows) get what their csv row has (n_notes, n_programs, exact, loose); their
loose_t is looked up via the exact (dup_exact_*) or loose (dup_loose_*) hash of their first copy, which has the same
loose_t by construction. Discover's dup_md5_giga rows were never tokenized: they are byte-identical to a GigaMIDI file,
so all their hashes and stats are taken from that file (looked up by md5). They count as parsed.

Hash value 0 = unknown (not parsed). md5 is reduced to its first 64 bits (16 hex digits) for the md5 level.
"""

import csv
import json
from array import array
from pathlib import Path

import numpy as np

SPLITS = ['train', 'validation', 'test']
LEVELS = ['md5', 'exact', 'loose', 'loose_t']  # = common.LEVELS (not imported: common pulls in torch)
STAT_COLS = ['n_notes', 'n_programs', 'n_drum_notes', 'n_pitched_programs', 'n_piano_notes']
TIERS = ['pruned', 'deduped', 'unique']  # bit i of `tier` = in that Aria tier

# licence tags (CLAUDE.md): nc-sa, nc, cc-by, pd/cc0, research-only, unclear. Override with --licence src=tag.
LICENCES = {'gigamidi': 'nc', 'discover': 'nc-sa', 'aria': 'nc-sa'}

DISCOVER_STATUS = ['kept', 'dup_md5_giga', 'dup_exact_giga', 'dup_exact_self', 'dup_loose_giga', 'dup_loose_self',
                   'failed']
GENERIC_CORE = {'file_id', 'split', 'status', 'n_notes', 'md5'}


def kv_list(items):
    """['a.b=c', ...] -> {'a': {'b': 'c'}} (input overrides) / ['a=c'] -> {'a': 'c'} (licences)."""
    out = {}
    for it in items or []:
        k, _, v = it.partition('=')
        if '.' in k:
            src, key = k.split('.', 1)
            out.setdefault(src, {})[key] = v
        else:
            out[k] = v
    return out


def default_inputs(src, census_dir):
    if src == 'gigamidi':
        return dict(kind='gigamidi', csv='data/gigamidi/gigamidi.csv', failed_csv='data/gigamidi/gigamidi_failed.csv')
    if src == 'discover':
        return dict(kind='discover', csv='data/discover/discover.csv')
    d = dict(kind='generic', csv=f'data/{src}/{src}.csv')
    if src == 'aria':
        d['tiers'] = str(Path(census_dir) / 'aria_tiers.csv')
    return d


def inputs_for(src, census_dir, overrides):
    d = default_inputs(src, census_dir)
    d.update(overrides.get(src, {}))
    return d


def licence_of(src, overrides):
    return overrides.get(src, LICENCES.get(src, 'unclear'))


def _int(x):
    return int(x) if x not in (None, '') else -1


def _hex(x):
    return int(x, 16) if x else 0


def source_rows(src, inp):
    """Yield one dict per listed file, in table order: file_id, md5 (hex or ''), split, status_in, n_notes,
    n_programs (-1 = unknown), exact, loose (int, 0 = unknown), extras (dict of str)."""
    kind = inp['kind']
    if kind == 'gigamidi':
        with open(inp['csv'], newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                yield dict(file_id=r['md5'], md5=r['md5'], split=r['split'], status_in='stored',
                           n_notes=_int(r['n_notes']), n_programs=_int(r['n_programs']), exact=0, loose=0,
                           extras=dict(category=r['category'], reason=''))
        fc = inp.get('failed_csv')
        if fc and Path(fc).exists():
            with open(fc, newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    yield dict(file_id=r['md5'], md5=r['md5'], split=r['split'], status_in='failed', n_notes=-1,
                               n_programs=-1, exact=0, loose=0, extras=dict(category=r['category'], reason=r['reason']))
    elif kind == 'discover':
        with open(inp['csv'], newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                yield dict(file_id=r['md5'], md5=r['md5'], split=r['split'], status_in=r['status'],
                           n_notes=_int(r['n_notes']), n_programs=_int(r['n_programs']), exact=_hex(r['exact']),
                           loose=_hex(r['loose']), extras=dict(discover_status=r['status']))
    else:
        with open(inp['csv'], newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                fid = r['file_id'] + (f"_{r['segment']}" if 'segment' in r else '')
                st = r.get('status', 'kept')
                assert st in ('kept', 'failed'), f'{src}: unknown status {st!r}'
                yield dict(file_id=fid, md5=r.get('md5', ''), split=r['split'],
                           status_in='stored' if st == 'kept' else 'failed',
                           n_notes=_int(r['n_notes']) if st == 'kept' else -1, n_programs=-1, exact=0, loose=0,
                           extras={k: v for k, v in r.items() if k not in GENERIC_CORE})


def extras_fields(src, inp):
    kind = inp['kind']
    if kind == 'gigamidi':
        return ['category', 'reason']
    if kind == 'discover':
        return ['discover_status']
    with open(inp['csv'], newline='', encoding='utf-8') as f:
        header = next(csv.reader(f))
    return [k for k in header if k not in GENERIC_CORE]


def read_tiers(path):
    """census/aria_tiers.csv -> {stem: bits}."""
    tiers = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tiers[r['stem']] = sum(1 << i for i, t in enumerate(TIERS) if r.get(f'in_{t}') in ('1', 'True', 'true'))
    return tiers


def _lookup(keys, query):
    """Index of the first entry of keys equal to each query value, -1 if absent (keys need not be sorted)."""
    if len(keys) == 0:
        return np.full(len(query), -1, np.int64)
    order = np.argsort(keys, kind='stable')
    ks = keys[order]
    pos = np.minimum(np.searchsorted(ks, query), len(ks) - 1)
    return np.where(ks[pos] == query, order[pos], -1)


class LazyTable:
    """A cached table: each column is read from the .npz when accessed (not kept), to bound RAM."""

    def __init__(self, path):
        self.z = np.load(path, allow_pickle=False)

    def __getitem__(self, k):
        return self.z[k]

    def __len__(self):
        return len(self.z['split'])


def build_table(src, inp, hashes_dir, earlier=None):
    """Build the table of one source (dict of numpy arrays). earlier: {source: table} of the sources before it
    (Discover's dup_md5_giga / loose_t lookups use gigamidi)."""
    split_c, stored, md5, n_notes, n_prog, exact, loose, status_in = (
        array('b'), array('b'), array('Q'), array('q'), array('q'), array('Q'), array('Q'), array('b'))
    status_names = DISCOVER_STATUS if inp['kind'] == 'discover' else ['stored', 'failed']
    tiers = read_tiers(inp['tiers']) if inp.get('tiers') and Path(inp['tiers']).exists() else None
    tier = array('B')
    for r in source_rows(src, inp):
        split_c.append(SPLITS.index(r['split']))
        stored.append(r['status_in'] in ('stored', 'kept'))
        md5.append(_hex(r['md5'][:16]))
        n_notes.append(r['n_notes'])
        n_prog.append(r['n_programs'])
        exact.append(r['exact'])
        loose.append(r['loose'])
        status_in.append(status_names.index(r['status_in']))
        if tiers is not None:
            tier.append(tiers.get(r['file_id'], 0))
    n = len(split_c)
    def arr(a, dt):
        return np.array(a, dtype=dt)

    t = dict(split=arr(split_c, np.uint8), stored=arr(stored, bool), md5=arr(md5, np.uint64),
             exact=arr(exact, np.uint64), loose=arr(loose, np.uint64), loose_t=np.zeros(n, np.uint64),
             n_notes=arr(n_notes, np.int64), n_programs=arr(n_prog, np.int64),
             n_drum_notes=np.full(n, -1, np.int64), n_pitched_programs=np.full(n, -1, np.int64),
             n_piano_notes=np.full(n, -1, np.int64), store=np.full(n, -1, np.int16), idx=np.full(n, -1, np.int64),
             status_in=arr(status_in, np.int8),
             tier=arr(tier, np.uint8) if tiers is not None else np.zeros(n, np.uint8))
    csv_notes, csv_exact, csv_loose = t['n_notes'].copy(), t['exact'].copy(), t['loose'].copy()
    checks = {}

    # stored rows <- hash_store npz, matched per split in order
    stores = []
    if t['stored'].any():
        z = np.load(Path(hashes_dir) / f'{src}.npz')
        stores = [str(s) for s in z['stores']]
        zstore = z['store']
        for s in np.unique(t['split'][t['stored']]):
            split = SPLITS[s]
            cand = [i for i, name in enumerate(stores) if name.endswith(f'_{split}')]
            assert len(cand) == 1, f'{src}: no unique store for split {split} in {stores}'
            si = cand[0]
            rows = np.flatnonzero(t['stored'] & (t['split'] == s))
            sel = np.flatnonzero(zstore == si)
            sel = sel[np.argsort(z['idx'][sel], kind='stable')]
            assert len(rows) == len(sel), f'{src} {split}: {len(rows)} stored csv rows but {len(sel)} files in the npz'
            t['store'][rows] = si
            t['idx'][rows] = z['idx'][sel]
            for k in ('exact', 'loose', 'loose_t'):
                t[k][rows] = z[k][sel]
            for k in STAT_COLS:
                if k in z.files:
                    t[k][rows] = z[k][sel]
        st = t['stored']
        known = st & (csv_notes >= 0)
        bad = int((csv_notes[known] != t['n_notes'][known]).sum())
        assert bad == 0, f'{src}: {bad} stored rows whose csv n_notes differs from the store (misaligned)'
        known = st & (csv_exact != 0)
        bad = int(((csv_exact[known] != t['exact'][known]) | (csv_loose[known] != t['loose'][known])).sum())
        assert bad == 0, f'{src}: {bad} stored rows whose csv hashes differ from the store (misaligned)'

    if inp['kind'] == 'discover':
        giga = (earlier or {}).get('gigamidi')
        st = t['stored']
        # dup_md5_giga: everything from the GigaMIDI file with the same md5
        m = t['status_in'] == DISCOVER_STATUS.index('dup_md5_giga')
        if m.any():
            assert giga is not None, 'discover needs the gigamidi table for its dup_md5_giga rows'
            gp = np.flatnonzero(giga['n_notes'] >= 0)
            j = _lookup(giga['md5'][gp], t['md5'][m])
            ok = j >= 0
            rows = np.flatnonzero(m)[ok]
            g = gp[j[ok]]
            for k in ('exact', 'loose', 'loose_t', *STAT_COLS):
                t[k][rows] = giga[k][g]
            checks['md5_lookup_missing'] = int((~ok).sum())
        # dup_exact_* / dup_loose_*: loose_t of the first copy
        pools = [(t['exact'][st], t['loose'][st], t['loose_t'][st])]
        if giga is not None:
            gp = np.flatnonzero(giga['n_notes'] >= 0)
            pools.insert(0, (giga['exact'][gp], giga['loose'][gp], giga['loose_t'][gp]))
        pe, pl, pt = (np.concatenate([p[i] for p in pools]) for i in range(3))
        missing = 0
        for name, keys in (('exact', pe), ('loose', pl)):
            m = np.isin(t['status_in'], [DISCOVER_STATUS.index(f'dup_{name}_giga'), DISCOVER_STATUS.index(f'dup_{name}_self')])
            if m.any():
                j = _lookup(keys, t[name][m])
                t['loose_t'][np.flatnonzero(m)[j >= 0]] = pt[j[j >= 0]]
                missing += int((j < 0).sum())
        checks['loose_t_lookup_missing'] = missing
    t['parsed'] = t['n_notes'] >= 0
    meta = dict(source=src, stores=stores, status_names=status_names, has_tiers=tiers is not None, n_rows=n,
                checks=checks)
    return t, meta


def table_path(census_dir, src):
    return Path(census_dir) / 'hashes' / f'table_{src}.npz'


def load_table(src, inp, census_dir, earlier=None, rebuild=False):
    """Cached table (dict of arrays) + meta; rebuilt if missing, stale (an input is newer) or rebuild."""
    path = table_path(census_dir, src)
    hashes_dir = Path(census_dir) / 'hashes'
    deps = [Path(inp[k]) for k in ('csv', 'failed_csv', 'tiers') if inp.get(k)] + [hashes_dir / f'{src}.npz']
    newest = max((p.stat().st_mtime for p in deps if p.exists()), default=0)
    if not rebuild and path.exists() and path.stat().st_mtime >= newest:
        t = LazyTable(path)
        return t, json.loads(str(t['meta']))
    t, meta = build_table(src, inp, hashes_dir, earlier)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, meta=np.array(json.dumps(meta)), **t)
    tmp.replace(path)
    del t
    t = LazyTable(path)
    return t, meta
