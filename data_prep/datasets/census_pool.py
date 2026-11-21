"""
Census GigaMIDI pool: the census GigaMIDI train + validation stores in one store, minus the held-out twins.

    python data_prep/datasets/census_pool.py build --tar stores_gigamidi.tar.zst     # -> data/cache/gigamidi_pool_train
    python data_prep/datasets/census_pool.py verify
    python data_prep/datasets/census_pool.py count --tar stores_gigamidi.tar.zst     # files / notes only, writes nothing

Held-out twin = a pool file whose loose_t hash (data_prep/census/common.py: sorted (onset, pitch), drums apart, lowest
pitched note moved to 0) equals that of a file of the held-out store (gigamidi_heldout_validation, 10% of the clean
GigaMIDI test split). exact and loose matches are loose_t matches too. The full pool: 967 twins dropped (864 train,
103 validation), 1,574,692 files, 1,910,871,953 notes, 1,897,348,864 usable notes (files of >= 64 notes, the loader's
pack_short rule), 17.21 GB.

The tar is read as a stream (zstd -dc | tarfile): tokens.u16 and programs.u8 of the wanted parts are appended to the
output, then the twins are cut out in place, so the peak disk use is the tar + the pool. Candidates for a twin share
the held-out file's onset sequence (cumsum of delta_time), which every hash level fixes; only those are hashed in full.
Output: tokens.u16, programs.u8, offsets.npy, files.csv (file_id, md5, store, idx of the census store), meta.json last.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from data_prep.census.common import Store, all_hashes  # noqa: E402

EXPECT = {  # parts: (files, notes)
    ('train', 'validation'): (1_574_692, 1_910_871_953),
    ('train',): (1_421_356, 1_720_203_274),
    ('validation',): (153_336, 190_668_679),
}
EXPECT_USABLE = {('train', 'validation'): 1_897_348_864}
CHUNK = 1 << 26


def onset_key(tokens):
    onset = np.cumsum(np.asarray(tokens[:, 3], dtype=np.int64))
    return hashlib.blake2b(onset.tobytes(), digest_size=8).digest()


def heldout_index(path):
    """{onset key: True} and the (exact, loose, loose_t) hash sets of the held-out store."""
    s = Store(path)
    keys, sets = set(), (set(), set(), set())
    for i in range(len(s)):
        tok, prog = s.file(i)
        keys.add(onset_key(tok))
        for k, h in zip(sets, all_hashes(tok, prog)):
            k.add(h)
    return keys, sets


def tar_stream(tar):
    proc = subprocess.Popen(['zstd', '-dc', str(tar)], stdout=subprocess.PIPE, bufsize=CHUNK)
    return proc, tarfile.open(fileobj=proc.stdout, mode='r|')


def member_name(name):
    """'gigamidi/gigamidi_train/tokens.u16' -> ('train', 'tokens.u16')."""
    parts = name.split('/')
    return (parts[-2].removeprefix('gigamidi_'), parts[-1]) if len(parts) >= 3 else (None, None)


def read_small(tar, parts):
    """First pass: offsets.npy, files.csv, meta.json of each part (tokens and programs come first in the tar)."""
    proc, tf = tar_stream(tar)
    small = {p: {} for p in parts}
    for m in tf:
        part, fname = member_name(m.name)
        if part in small and fname in ('offsets.npy', 'files.csv', 'meta.json'):
            small[part][fname] = tf.extractfile(m).read()
        if all(len(v) == 3 for v in small.values()):
            break
    proc.kill()
    return {p: dict(offsets=np.load(io.BytesIO(v['offsets.npy'])), files=v['files.csv'],
                    meta=json.loads(v['meta.json'])) for p, v in small.items()}


def twins(n_files, tok_of, prog_of, keys, loose_t, cand=None):
    """(n_files,) bool: files whose loose_t hash is a held-out file's (cand = precomputed onset candidates)."""
    out = np.zeros(n_files, bool)
    for i in (range(n_files) if cand is None else cand):
        tok = tok_of(i)
        if cand is not None or onset_key(tok) in keys:
            out[i] = all_hashes(tok, prog_of(i))[2] in loose_t
    return out


def count(args):
    """Files / notes of the pool from the tar without writing it: offsets first, then the onset candidates while
    the tokens stream past (their rows kept in memory), then their programs."""
    parts = tuple(args.parts.split(','))
    keys, sets = heldout_index(args.heldout)
    small = read_small(args.tar, parts)
    proc, tf = tar_stream(args.tar)
    cand_tok, cand_prog = {p: {} for p in parts}, {p: {} for p in parts}
    for m in tf:
        part, fname = member_name(m.name)
        if part not in small or fname not in ('tokens.u16', 'programs.u8'):
            continue
        off, f = small[part]['offsets'], tf.extractfile(m)
        width = 8 if fname == 'tokens.u16' else 1
        buf, row0, i = b'', 0, 0  # buffer starts at row row0; next file i
        while True:
            chunk = f.read(CHUNK)
            buf += chunk
            rows = row0 + len(buf) // width
            while i < len(off) - 1 and (off[i + 1] <= rows):
                a, b = (int(off[i]) - row0) * width, (int(off[i + 1]) - row0) * width
                if fname == 'tokens.u16':
                    tok = np.frombuffer(buf[a:b], np.uint16).reshape(-1, 4)
                    if onset_key(tok) in keys:
                        cand_tok[part][i] = tok.copy()
                elif i in cand_tok[part]:
                    cand_prog[part][i] = np.frombuffer(buf[a:b], np.uint8).copy()
                i += 1
            cut = (int(off[i]) - row0) * width if i < len(off) - 1 else len(buf)
            buf, row0 = buf[cut:], row0 + cut // width
            if not chunk:
                break
        assert i == len(off) - 1, f'{m.name}: stream ended at file {i} of {len(off) - 1}'
        if fname == 'programs.u8' and part == parts[-1]:
            break
    proc.kill()
    files = notes = 0
    for p in parts:
        off = small[p]['offsets']
        c = sorted(cand_tok[p])
        drop = twins(len(off) - 1, cand_tok[p].get, cand_prog[p].get, keys, sets[2], cand=c)
        n = np.diff(off)
        print(f'{p}: {len(n):,} files, {int(n.sum()):,} notes; {len(c):,} onset candidates, {int(drop.sum()):,} '
              f'twins ({int(n[drop].sum()):,} notes)')
        files += int((~drop).sum())
        notes += int(n[~drop].sum())
        usable = int(n[~drop][n[~drop] >= 64].sum())
        print(f'  kept {int((~drop).sum()):,} files, {int(n[~drop].sum()):,} notes, {usable:,} usable')
    print(f'pool: {files:,} files, {notes:,} notes')
    return files, notes


def compact(path, width, offsets, keep):
    """Cut the dropped files out of a flat file in place (the write position never passes the read position)."""
    runs, start = [], None
    for i, k in enumerate(np.append(keep, False)):
        if k and start is None:
            start = i
        elif not k and start is not None:
            runs.append((int(offsets[start]), int(offsets[i])))
            start = None
    w = 0
    with open(path, 'r+b') as f:
        for a, b in runs:
            if a == w:
                w = b
                continue
            for s in range(a, b, CHUNK // width):
                e = min(b, s + CHUNK // width)
                f.seek(s * width)
                data = f.read((e - s) * width)
                f.seek(w * width)
                f.write(data)
                w += e - s
        f.truncate(w * width)
    return w


def build(args):
    parts = tuple(args.parts.split(','))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'meta.json').unlink(missing_ok=True)  # incomplete until written last
    t0 = time.time()
    keys, sets = heldout_index(args.heldout)
    proc, tf = tar_stream(args.tar)
    small, done = {p: {} for p in parts}, set()
    with open(out / 'tokens.u16', 'wb') as ft, open(out / 'programs.u8', 'wb') as fp:
        for m in tf:
            part, fname = member_name(m.name)
            if part not in small:
                continue
            f = tf.extractfile(m)
            if fname in ('tokens.u16', 'programs.u8'):
                dst = ft if fname == 'tokens.u16' else fp
                while chunk := f.read(CHUNK):
                    dst.write(chunk)
                print(f'{time.time() - t0:5.0f}s {m.name} ({m.size / 1e9:.2f} GB)', flush=True)
            else:
                small[part][fname] = f.read()
            if len(small[part]) == 3:
                done.add(part)
            if done == set(parts):
                break
    proc.kill()
    offs, rows, csv_lines = [], 0, []
    for p in parts:  # tar order is train, validation: the same order as the appended bytes
        off = np.load(io.BytesIO(small[p]['offsets.npy']))
        assert json.loads(small[p]['meta.json'])['n_notes'] == int(off[-1])
        offs.append(off[:-1] + rows)
        rows += int(off[-1])
        lines = small[p]['files.csv'].decode().splitlines()
        csv_lines += lines[1:] if csv_lines else lines
    offsets = np.append(np.concatenate(offs), rows).astype(np.int64)
    assert (out / 'tokens.u16').stat().st_size == rows * 8 and (out / 'programs.u8').stat().st_size == rows
    tokens = np.memmap(out / 'tokens.u16', dtype=np.uint16, mode='r', shape=(rows, 4))
    programs = np.memmap(out / 'programs.u8', dtype=np.uint8, mode='r', shape=(rows,))
    drop = twins(len(offsets) - 1, lambda i: tokens[offsets[i]:offsets[i + 1]],
                 lambda i: programs[offsets[i]:offsets[i + 1]], keys, sets[2])
    tokens = programs = None
    print(f'{time.time() - t0:5.0f}s {int(drop.sum()):,} held-out twins of {len(drop):,} files', flush=True)
    keep = ~drop
    n = np.diff(offsets)[keep]
    new_off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    assert compact(out / 'tokens.u16', 8, offsets, keep) == new_off[-1]
    assert compact(out / 'programs.u8', 1, offsets, keep) == new_off[-1]
    np.save(out / 'offsets.npy', new_off)
    header, body = csv_lines[0], csv_lines[1:]
    assert len(body) == len(keep)
    (out / 'files.csv').write_text('\n'.join([header] + [r for r, k in zip(body, keep) if k]) + '\n')
    meta = dict(n_files=int(keep.sum()), n_notes=int(new_off[-1]), programs=True, source=Path(args.tar).name,
                parts=list(parts), heldout=str(args.heldout), twin_level='loose_t',
                twins_dropped=int(drop.sum()), twin_notes=int(np.diff(offsets)[drop].sum()))
    (out / 'meta.json').write_text(json.dumps(meta))
    print(f'{time.time() - t0:5.0f}s {meta["n_files"]:,} files, {meta["n_notes"]:,} notes -> {out}')
    return meta['n_files'], meta['n_notes']


def check(parts, files, notes, source):
    want = EXPECT.get(parts) if Path(source).name == 'stores_gigamidi.tar.zst' else None
    if want is None:
        print('no expected counts for this tar / these parts')
        return True
    ok = (files, notes) == want
    print(f'expected {want[0]:,} files, {want[1]:,} notes: {"OK" if ok else "MISMATCH"}')
    return ok


def verify(args):
    s = Store(args.out)
    meta, off = s.meta, s.offsets
    ok = bool((np.diff(off) > 0).all()) and (Path(args.out) / 'programs.u8').stat().st_size == off[-1]
    lines = (Path(args.out) / 'files.csv').read_text().splitlines()
    ok &= len(lines) - 1 == len(s)
    print(f'{len(s):,} files, {int(off[-1]):,} notes, layout {"ok" if ok else "BAD"}')
    keys, sets = heldout_index(args.heldout)
    hits = [0, 0, 0]
    for i in range(len(s)):
        tok, prog = s.file(i)
        if onset_key(tok) in keys:
            for j, (h, st) in enumerate(zip(all_hashes(tok, prog), sets)):
                hits[j] += h in st
    print('held-out exact / loose / loose_t matches: {} / {} / {}'.format(*hits))
    ok &= hits == [0, 0, 0]
    step = max(1, int(off[-1]) // 2_000_000)
    sample = np.asarray(s.tokens[::step])
    ok &= bool((sample[:, 0] < 128).all() and (np.asarray(s.programs[::step]) <= 128).all())
    from musicar.data.loader import NoteWindowDataset
    prefix = str(Path(args.out)).removesuffix('_train')
    ds = NoteWindowDataset(f'store:{prefix}', '.', 'train', 2048, special_tokens=True, pack_short=True,
                           min_notes=64, programs=True)
    usable = ds.num_notes()
    parts = tuple(meta.get('parts', ()))
    print(f'loader num_notes(): {usable:,} usable notes in {ds.n_files:,} files')
    ok &= check(parts, len(s), int(off[-1]), meta.get('source', ''))
    if parts in EXPECT_USABLE and meta.get('source') == 'stores_gigamidi.tar.zst':
        ok &= usable == EXPECT_USABLE[parts]
        print(f'expected {EXPECT_USABLE[parts]:,} usable notes')
    print('VERIFY OK' if ok else 'VERIFY FAILED')
    return ok


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('cmd', choices=['build', 'verify', 'count'])
    p.add_argument('--tar', default='stores_gigamidi.tar.zst')
    p.add_argument('--heldout', default='data/cache/gigamidi_heldout_validation')
    p.add_argument('--out', default='data/cache/gigamidi_pool_train')
    p.add_argument('--parts', default='train,validation', help='census parts, in tar order')
    args = p.parse_args()
    if args.cmd == 'verify':
        ok = verify(args)
    else:
        ok = check(tuple(args.parts.split(',')), *{'build': build, 'count': count}[args.cmd](args), args.tar)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
