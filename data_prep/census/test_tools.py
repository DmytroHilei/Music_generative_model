"""
End-to-end test of the census tools on tiny synthetic stores and csvs (in the formats the prepare_* scripts write):
    cd /workspace/music && PYTHONPATH=.:data_prep/datasets:data_prep/census .venv/bin/python data_prep/census/test_tools.py
Cases: within-source duplicates at md5 / exact / loose, cross-source duplicates at each level, a transposed copy
(loose_t only), failed files, Discover's own dup rows (not in its store), Aria tiers, a file the loader drops (< 64
notes), long files. Runs aria_tiers, hash_store, global_index (3 tiers), usable, manifest (+ --verify) in-process.
"""

import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

from common import DRUMS, all_hashes
import aria_tiers
import global_index
import hash_store
import manifest
import usable

ROOT = Path(os.environ.get('CENSUS_TEST_DIR', '/workspace/agents/tools_test'))
CACHE, DATA, CENSUS = ROOT / 'cache', ROOT / 'data', ROOT / 'census'
rng = np.random.default_rng(1)


def piece(n, programs):
    tok = np.zeros((n, 4), np.uint16)
    tok[:, 0] = rng.integers(40, 80, n)
    tok[:, 1] = rng.integers(0, 32, n)
    tok[:, 2] = rng.integers(1, 50, n)
    tok[:, 3] = rng.integers(0, 20, n)
    prog = np.asarray(programs, np.uint8) if np.ndim(programs) else np.full(n, programs, np.uint8)
    return tok, prog


def revel(f):  # same (onset, pitch), other velocity: loose duplicate, not exact
    tok, prog = f
    t = tok.copy()
    t[:, 1] = (t[:, 1] + 1) % 32
    return t, prog


def transpose(f, k):
    tok, prog = f
    t = tok.copy()
    t[prog != DRUMS, 0] = (t[prog != DRUMS, 0].astype(np.int64) + k).astype(np.uint16)
    return t, prog


def write_store(name, files, programs=True):
    p = CACHE / name
    p.mkdir(parents=True)
    toks = [f[0] for f in files]
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)
    (np.concatenate(toks) if toks else np.zeros((0, 4), np.uint16)).tofile(p / 'tokens.u16')
    if programs:
        (np.concatenate([f[1] for f in files]) if files else np.zeros(0, np.uint8)).tofile(p / 'programs.u8')
    np.save(p / 'offsets.npy', offsets)
    (p / 'meta.json').write_text(json.dumps(dict(n_files=len(files), n_notes=int(offsets[-1]), programs=programs)))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def md5(i):
    return hashlib.md5(str(i).encode()).hexdigest()


def write_tar(path, stems):
    with tarfile.open(path, 'w:gz') as tf:
        for s in stems:
            data = b'MThd'
            ti = tarfile.TarInfo(f'aria-midi/data/aa/{s}.mid')
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def build():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    # ---------------- GigaMIDI (multi-track); csv order = validation, test, train; first copy wins train-first
    P0 = piece(300, rng.choice([0, 33, DRUMS], 300))
    P1 = piece(200, 0)
    P2 = piece(150, DRUMS)
    P3 = piece(3000, 40)
    P4 = piece(30, 0)
    G = dict(validation=[('G0', md5(1), P0), ('G1', md5(2), P1)],
             test=[('G2', md5(3), P2)],
             train=[('G3', md5(4), P0),            # exact copy of G0: G0 (validation) is dropped (eval_in_train)
                    ('G4', md5(5), revel(P1)),     # loose copy of G1: G1 dropped at loose
                    ('G5', md5(6), P3),            # long
                    ('G6', md5(7), P4),            # 30 notes: the loader drops it
                    ('G7', md5(3), P2)])           # same bytes as G2: G2 (test) dropped at md5
    rows = []
    for split in ('validation', 'test', 'train'):
        write_store(f'gigamidi_{split}', [f for _, _, f in G[split]])
        rows += [dict(md5=m, split=split, category='c', n_notes=len(f[0]), n_programs=len(np.unique(f[1])))
                 for _, m, f in G[split]]
    write_csv(DATA / 'gigamidi/gigamidi.csv', ['md5', 'split', 'category', 'n_notes', 'n_programs'], rows)
    write_csv(DATA / 'gigamidi/gigamidi_failed.csv', ['md5', 'split', 'category', 'reason'],
              [dict(md5=md5(90), split='train', category='c', reason='parse'),
               dict(md5=md5(91), split='test', category='c', reason='oversized:3000000')])

    # ---------------- Discover: csv has every row; the store only the kept ones
    P5, P6, P7 = piece(400, 0), piece(500, 1), piece(250, 33)
    D = [('D0', md5(20), 'train', 'kept', P5),
         ('D1', md5(2), 'train', 'dup_md5_giga', None),          # bytes of G1
         ('D2', md5(22), 'train', 'dup_exact_giga', P2),         # notes of G2
         ('D3', md5(23), 'train', 'dup_loose_giga', revel(P0)),  # loose of G0
         ('D4', md5(24), 'validation', 'kept', P6),
         ('D5', md5(25), 'train', 'dup_exact_self', P5),         # = D0
         ('D6', md5(26), 'train', 'dup_loose_self', revel(P6)),  # loose of D4: Discover keeps archive order (D4 kept)
         ('D7', md5(27), 'train', 'failed', None),
         ('D8', md5(28), 'train', 'kept', transpose(P3, 3)),     # G5 in another key: loose_t only
         ('D9', md5(29), 'train', 'kept', P7),
         ('D10', md5(30), 'train', 'dup_loose_self', revel(P5)),  # loose of D0
         ('D11', md5(31), 'train', 'dup_loose_self', revel(P5))]  # = D10's tokens: the index says dup_exact (ok)
    rows = []
    for _, m, split, status, f in D:
        r = dict(md5=m, split=split, status=status)
        if f is not None:
            e, l, _ = all_hashes(*f)
            r.update(n_notes=len(f[0]), n_programs=len(np.unique(f[1])), exact=f'{e:016x}', loose=f'{l:016x}')
        rows.append(r)
    write_csv(DATA / 'discover/discover.csv', ['md5', 'split', 'status', 'n_notes', 'n_programs', 'exact', 'loose'], rows)
    for split in ('train', 'validation'):
        write_store(f'discover_{split}', [f for _, _, s, st, f in D if st == 'kept' and s == split])

    # ---------------- Aria (piano store: no programs.u8)
    P8, P11 = piece(180, 0), piece(5000, 0)
    A = [('r000', md5(40), 'train', 'kept', (P1[0], None)),                 # = G1 (program 0): cross exact
         ('r001', md5(41), 'train', 'kept', (P8[0], None)),
         ('r002', md5(41), 'train', 'kept', (P8[0], None)),                 # same bytes as r001: within md5
         ('r003', md5(43), 'train', 'failed', None),
         ('r004', md5(44), 'train', 'kept', (transpose(P7, -2)[0], None)),  # D9 in another key: cross loose_t
         ('r005', md5(45), 'train', 'kept', (revel(P6)[0], None)),          # loose of D4: cross loose
         ('r006', md5(46), 'validation', 'kept', (P11[0], None))]
    rows = [dict(file_id=fid, segment='0', split=split, genre='classical', composer='', audio_score='0.9',
                 n_notes=0 if f is None else len(f[0]), md5=m, status=st) for fid, m, split, st, f in A]
    write_csv(DATA / 'aria/aria.csv', ['file_id', 'segment', 'split', 'genre', 'composer', 'audio_score', 'n_notes',
                                       'md5', 'status'], rows)
    for split in ('train', 'validation'):
        write_store(f'aria_{split}', [f for _, _, s, st, f in A if st == 'kept' and s == split], programs=False)
    tars = ROOT / 'tars'
    tars.mkdir()
    write_tar(tars / 'pruned.tar.gz', ['r000_0', 'r001_0', 'r002_0', 'r004_0', 'r006_0'])
    write_tar(tars / 'deduped.tar.gz', ['r001_0', 'r006_0'])
    write_tar(tars / 'unique.tar.gz', ['r006_0'])
    return dict(G=G, D=D, A=A)


def lv(c, src, level):
    return c[src]['levels'][level]


def main():
    data = build()
    inputs = ['--input', f'gigamidi.csv={DATA}/gigamidi/gigamidi.csv',
              '--input', f'gigamidi.failed_csv={DATA}/gigamidi/gigamidi_failed.csv',
              '--input', f'discover.csv={DATA}/discover/discover.csv',
              '--input', f'aria.csv={DATA}/aria/aria.csv']
    common_args = ['--census-dir', str(CENSUS)] + inputs
    tars = ROOT / 'tars'
    aria_tiers.main(['--census-dir', str(CENSUS), '--pruned', str(tars / 'pruned.tar.gz'),
                     '--deduped', str(tars / 'deduped.tar.gz'), '--unique', str(tars / 'unique.tar.gz')])
    tiers = list(csv.DictReader(open(CENSUS / 'aria_tiers.csv')))
    assert len(tiers) == 5 and {r['stem'] for r in tiers if r['in_deduped'] == '1'} == {'r001_0', 'r006_0'}
    for src, stores in (('gigamidi', 'gigamidi_train,gigamidi_validation,gigamidi_test'),
                        ('discover', 'discover_train,discover_validation'), ('aria', 'aria_train,aria_validation')):
        hash_store.main(['--stores', stores, '--out', src, '--cache-dir', str(CACHE), '--workers', '1',
                         '--out-dir', str(CENSUS / 'hashes')])

    # ---------------- global index, full tier
    c = global_index.main(['--sources', 'gigamidi,discover,aria'] + common_args)
    g = c['gigamidi']
    assert (g['files_listed'], g['files_parsed'], g['files_failed']) == (10, 8, 2), g
    assert lv(c, 'gigamidi', 'md5')['dropped_within_files'] == 1        # G7
    assert lv(c, 'gigamidi', 'exact')['dropped_within_files'] == 1      # G3
    assert lv(c, 'gigamidi', 'loose')['dropped_within_files'] == 1      # G4
    assert lv(c, 'gigamidi', 'loose_t')['dropped_within_files'] == 0
    giga_kept = sum(len(f[0]) for name, _, f in sum(data['G'].values(), []) if name in ('G3', 'G4', 'G5', 'G6', 'G7'))
    assert lv(c, 'gigamidi', 'loose')['kept_notes'] == giga_kept and lv(c, 'gigamidi', 'loose')['kept_files'] == 5
    # train-first leakage counts: G2 (md5 of G7), G0 (exact of G3), G1 (loose of G4)
    assert c['gigamidi_eval_in_train'] == dict(md5=dict(files=1, notes=150), exact=dict(files=1, notes=300),
                                               loose=dict(files=1, notes=200), loose_t=dict(files=0, notes=0)), c
    assert g['eval_in_train'] == c['gigamidi_eval_in_train'] and 'eval_in_train' not in c['total']
    d = c['discover']
    assert (d['files_listed'], d['files_parsed'], d['files_failed']) == (12, 11, 1), d
    assert d['checks'] == dict(md5_lookup_missing=0, loose_t_lookup_missing=0, discover_status_mismatch=0,
                               train_first_swaps=0), d['checks']
    assert c['discover_eval_in_train']['loose'] == dict(files=0, notes=0)      # archive order: no train-first for Discover
    assert c['aria_eval_in_train'] == {l: dict(files=0, notes=0) for l in ('md5', 'exact', 'loose', 'loose_t')}
    exp = dict(md5=(0, 1), exact=(2, 1), loose=(2, 1), loose_t=(0, 1))  # (within, cross)
    for level, (w, x) in exp.items():
        L = lv(c, 'discover', level)
        assert (L['dropped_within_files'], L['dropped_cross_files']) == (w, x), (level, L)
    assert lv(c, 'discover', 'md5')['dropped_cross_notes'] == 200     # D1 takes G1's n_notes
    assert lv(c, 'discover', 'loose_t')['kept_files'] == 3            # D0, D4, D9
    assert lv(c, 'discover', 'loose_t')['dropped_by_first_source'] == {'gigamidi': 1}
    a = c['aria']
    assert (a['files_listed'], a['files_parsed'], a['files_failed'], a['files_excluded']) == (7, 6, 1, 0), a
    exp = dict(md5=(1, 0), exact=(0, 1), loose=(0, 1), loose_t=(0, 1))
    for level, (w, x) in exp.items():
        L = lv(c, 'aria', level)
        assert (L['dropped_within_files'], L['dropped_cross_files']) == (w, x), (level, L)
    assert lv(c, 'aria', 'loose')['kept_files'] == 3 and lv(c, 'aria', 'loose_t')['kept_files'] == 2
    assert lv(c, 'aria', 'loose_t')['kept_notes'] == 180 + 5000
    assert lv(c, 'total', 'loose')['kept_notes'] == sum(lv(c, s, 'loose')['kept_notes'] for s in ('gigamidi', 'discover', 'aria'))

    # ---------------- tiers
    cd = global_index.main(['--sources', 'gigamidi,discover,aria', '--aria-tier', 'deduped'] + common_args)
    assert (cd['aria']['files_excluded'], lv(cd, 'aria', 'loose_t')['kept_files']) == (5, 2), cd['aria']
    assert cd['gigamidi'] == c['gigamidi'] and cd['discover'] == c['discover']   # earlier sources unchanged
    cp = global_index.main(['--sources', 'gigamidi,discover,aria', '--aria-tier', 'pruned'] + common_args)
    A = cp['aria']
    assert A['files_excluded'] == 2 and A['files_failed'] == 0                      # r003 (failed) and r005 out
    assert lv(cp, 'aria', 'md5')['dropped_within_files'] == 1 and lv(cp, 'aria', 'exact')['dropped_cross_files'] == 1
    assert lv(cp, 'aria', 'loose')['dropped_cross_files'] == 0 and lv(cp, 'aria', 'loose_t')['dropped_cross_files'] == 1
    assert lv(cp, 'aria', 'loose')['kept_files'] == 3 and lv(cp, 'aria', 'loose_t')['kept_files'] == 2

    # ---------------- usable @2048
    res = usable.main(['--tags', ',aria-deduped', '--census-dir', str(CENSUS), '--cache-dir', str(CACHE)] + inputs)
    u = res['']
    U = u['gigamidi']['levels']['loose']
    assert U['kept_notes'] == giga_kept and U['usable_notes'] == giga_kept - 30, U          # G6 < 64 notes
    assert abs(U['long_share_gt_2048'] - 3000 / giga_kept) < 1e-5 and U['long_share_gt_4096'] == 0
    mix = U['mix']
    assert abs(mix['multi'] - 300 / giga_kept) < 1e-5 and abs(mix['drums_only'] - 150 / giga_kept) < 1e-5
    assert abs(mix['piano_only'] - 230 / giga_kept) < 1e-5 and abs(mix['single_other'] - 3000 / giga_kept) < 1e-5
    assert mix['unknown'] == 0
    assert u['discover']['levels']['loose']['usable_notes'] == 400 + 500 + 3000 + 250    # D4 is stored
    assert u['discover']['levels']['loose']['usable_borrowed_files'] == 0
    assert u['discover']['levels']['loose_t']['usable_notes'] == 400 + 500 + 250
    ua = u['aria']['levels']
    assert ua['loose']['usable_notes'] == 180 + 250 + 5000 and ua['loose_t']['usable_notes'] == 5180
    assert ua['loose_t']['mix']['piano_only'] == 1.0
    assert u['aria']['usable_in_stores'] == 200 + 180 + 180 + 250 + 500 + 5000
    assert u['total']['levels']['loose']['usable_notes'] == sum(u[s]['levels']['loose']['usable_notes']
                                                                 for s in ('gigamidi', 'discover', 'aria'))
    assert res['aria-deduped']['aria']['levels']['loose']['usable_notes'] == 5180

    # ---------------- manifests + verify
    margs = ['--sources', 'gigamidi,discover,aria', '--census-dir', str(CENSUS), '--cache-dir', str(CACHE)] + inputs
    manifest.main(margs)
    man = {s: list(csv.DictReader(open(CENSUS / 'manifests' / f'{s}.csv'))) for s in ('gigamidi', 'discover', 'aria')}
    assert [len(man[s]) for s in man] == [10, 12, 7]
    st = [r['status'] for r in man['gigamidi']]
    assert st == ['dup_exact_gigamidi', 'dup_loose_gigamidi', 'dup_md5_gigamidi', 'kept', 'kept', 'kept', 'kept',
                  'kept', 'failed', 'failed'], st
    st = [r['status'] for r in man['discover']]
    assert st == ['kept', 'dup_md5_gigamidi', 'dup_exact_gigamidi', 'dup_loose_gigamidi', 'kept',
                  'dup_exact_discover', 'dup_loose_discover', 'failed', 'dup_loose_t_gigamidi', 'kept', 'dup_loose_discover',
                  'dup_exact_discover'], st
    assert man['discover'][1]['n_notes'] == '200' and man['discover'][1]['loose_t'] != ''
    assert all(r['loose_t'] for r in man['discover'] if r['status'] != 'failed')
    st = [r['status'] for r in man['aria']]
    assert st == ['dup_exact_gigamidi', 'kept', 'dup_md5_aria', 'failed', 'dup_loose_t_discover',
                  'dup_loose_discover', 'kept'], st
    r = man['aria'][0]
    assert (r['file_id'], r['segment'], r['genre'], r['in_pruned'], r['in_deduped'], r['licence']) == \
        ('r000_0', '0', 'classical', '1', '0', 'nc-sa'), r
    assert man['gigamidi'][9]['reason'] == 'oversized:3000000' and man['gigamidi'][0]['licence'] == 'nc'
    assert man['aria'][3]['exact'] == '' and man['aria'][3]['n_notes'] == ''
    manifest.main(margs + ['--verify'])                              # must pass (no SystemExit)
    with open(CENSUS / 'manifests' / 'aria.csv', 'a') as f:           # tamper: one row too many
        f.write(','.join(['aria'] + [''] * 10) + '\n')
    try:
        manifest.main(margs + ['--verify'])
        raise AssertionError('verify should fail on a tampered manifest')
    except SystemExit as e:
        assert e.code == 1
    meta = CACHE / 'aria_train' / 'meta.json'                          # tamper: store meta
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps(m | {'n_files': m['n_files'] + 1}))
    manifest.main(margs)
    try:
        manifest.main(margs + ['--verify'])
        raise AssertionError('verify should fail on a store meta mismatch')
    except SystemExit as e:
        assert e.code == 1
    meta.write_text(json.dumps(m))
    manifest.main(margs + ['--verify'])
    print('census tools tests ok')


if __name__ == '__main__':
    sys.exit(main())
