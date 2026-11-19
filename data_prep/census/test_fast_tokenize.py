"""
Check data_prep/census/fast_tokenize.py against the original tokenizers on real MIDI files from zip archives.

For every sampled file, tokenize_tracks and tokenize_midi (musicar.data.loader, pretty_midi) are compared with
fast_tokenize_tracks / fast_tokenize_midi: equal when both are None, or tokens (and programs) are identical arrays.
pretty_midi parses each file once (loader._read_midi is cached per file inside this script) so both original
functions run on the same parse; --speed N times the unpatched originals and the fast versions on N files.

--fuzz K also tests K corrupted copies per file (random byte changes, truncation, insertion), to exercise the
broken-file paths (which exceptions mean None).

Run from the repo root, single process:
  .venv/bin/python -I data_prep/census/test_fast_tokenize.py --zip a.zip --zip b.zip --per-category 2500
"""

import argparse
import random
import sys
import time
import warnings
import zipfile
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402

from musicar.data import loader  # noqa: E402
from fast_tokenize import STATS, fast_tokenize_midi, fast_tokenize_tracks  # noqa: E402

warnings.filterwarnings('ignore')
_orig_read_midi = loader._read_midi


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, tuple):
        return all(x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def describe(a, b):
    """Short description of how two results differ."""
    if a is None or b is None:
        return f'none-ness: original {"None" if a is None else len(a[0] if isinstance(a, tuple) else a)} ' \
               f'fast {"None" if b is None else len(b[0] if isinstance(b, tuple) else b)}'
    ta, tb = (a[0], b[0]) if isinstance(a, tuple) else (a, b)
    if ta.shape != tb.shape:
        return f'n_notes {len(ta)} vs {len(tb)}'
    rows = np.nonzero((ta != tb).any(1))[0]
    if len(rows):
        cols = sorted(set(np.nonzero(ta[rows] != tb[rows])[1].tolist()))
        return f'{len(rows)} token rows differ (cols {cols}), first row {rows[0]}: {ta[rows[0]]} vs {tb[rows[0]]}'
    return 'programs differ'


def sample_members(paths, per_category, macosx, seed):
    rng = random.Random(seed)
    out = []
    for p in paths:
        z = zipfile.ZipFile(p)
        groups = {}
        for i in z.infolist():
            if not i.filename.lower().endswith('.mid'):
                continue
            parts = i.filename.split('/')
            cat = '__MACOSX' if parts[0] == '__MACOSX' else (parts[1] if len(parts) > 2 else 'root')
            groups.setdefault(cat, []).append(i)
        for cat, infos in sorted(groups.items()):
            k = macosx if cat == '__MACOSX' else per_category
            for i in rng.sample(infos, min(k, len(infos))):
                out.append((p, cat, i))
    rng.shuffle(out)
    return out


def corrupt(data, rng):
    b = bytearray(data)
    kind = rng.randrange(4)
    if kind == 0 and len(b) > 14:                     # flip a few bytes
        for _ in range(rng.randint(1, 4)):
            b[rng.randrange(14, len(b)) if rng.random() < .9 else rng.randrange(len(b))] = rng.randrange(256)
    elif kind == 1:                                   # truncate
        b = b[:rng.randrange(len(b) + 1)]
    elif kind == 2:                                   # insert bytes
        at = rng.randrange(len(b) + 1)
        b[at:at] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 3)))
    else:                                             # delete bytes
        at = rng.randrange(max(1, len(b)))
        del b[at:at + rng.randint(1, 3)]
    return bytes(b)


def _var(v):
    out = [v & 0x7f]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7f))
        v >>= 7
    return bytes(reversed(out))


def synth(rng):
    """A small random raw MIDI file that hits the edge cases: overlapping same-pitch notes, same-tick on/off,
    velocity-0 note-ons, program changes between on and off, drums, running status (also after sysex/system
    messages), tempo maps (incl. repeats, tick-0 resets, tempo 0), odd/negative/zero ticks_per_beat, meta messages
    of every decodable type with good and bad payloads, sysex with bytes > 127, wrong chunk sizes, junk."""
    def meta(t, payload):
        return b'\xff' + bytes([t]) + _var(len(payload)) + payload

    def track():
        ev = bytearray()
        last = None
        for _ in range(rng.randint(0, 40)):
            r = rng.random()
            delta = rng.choice([0, 0, 0, 1, 2, 5, 30, 240]) if rng.random() < .98 else rng.randrange(1 << 24)
            ev += _var(delta)
            ch = rng.choice([0, 0, 1, 9, 9, 15])
            if r < .55:
                st = (0x90 if rng.random() < .65 else 0x80) | ch
                body = bytes([rng.choice([60, 60, 62, 64]), rng.choice([0, 1, 64, 100, 127])])
            elif r < .62:
                st, body = 0xc0 | ch, bytes([rng.randrange(128)])
            elif r < .66:
                st, body = rng.choice([0xb0, 0xe0, 0xa0]) | ch, bytes([rng.randrange(128), rng.randrange(128)])
            elif r < .68:
                st, body = 0xd0 | ch, bytes([rng.randrange(128)])
            elif r < .86:
                t = rng.choice([0x51, 0x51, 0x51, 0x58, 0x58, 0x59, 0x59, 0x54, 0x00, 0x20, 0x21, 0x2f, 0x03, 0x7f, 0x08])
                n = rng.choice({0x51: [3, 3, 3, 2, 4], 0x58: [4, 4, 3], 0x59: [2, 2, 1], 0x54: [5, 5, 4],
                                0x00: [0, 1, 2], 0x20: [0, 1]}.get(t, [0, 3]))
                if t == 0x51:
                    payload = rng.choice([500000, 500000, 600000, 250000, 1, 0xffffff, 0]).to_bytes(3, 'big')
                    payload = (payload + b'\x00')[:n]
                elif t == 0x58:
                    payload = bytes([rng.choice([0, 4, 3, 255]), rng.choice([2, 3, 0, 29, 31, 255, 64]), 24, 8])[:n]
                elif t == 0x59:
                    payload = bytes([rng.choice([0, 3, 7, 8, 0xf9, 0xf8, 0x80]), rng.choice([0, 1, 2])])[:n]
                elif t == 0x54:
                    payload = bytes([rng.choice([0, 0x61, 0x80, 0xe0]), rng.choice([0, 59, 60]),
                                     rng.choice([0, 59, 60]), 0, rng.choice([0, 99, 100])])[:n]
                else:
                    payload = bytes(rng.randrange(256) for _ in range(n))
                ev += meta(t, payload)
                continue  # meta: no running status change
            elif r < .9:
                st = rng.choice([0xf0, 0xf7])
                payload = bytes(rng.choice([0xf0, 0x10, 0x7f, 0xf7, 0x80]) for _ in range(rng.randint(0, 4)))
                body = _var(len(payload)) + payload
            elif r < .93:
                st = rng.choice([0xf1, 0xf2, 0xf3, 0xf6, 0xf8, 0xfe, 0xf4, 0xf9])
                body = bytes(rng.randrange(128) for _ in range({0xf1: 1, 0xf2: 2, 0xf3: 1}.get(st, 0)))
            else:
                ev += bytes([rng.randrange(256)])  # junk byte as status/data
                continue
            if st == last and rng.random() < .7:
                ev += body           # running status
            else:
                ev += bytes([st]) + body
            last = st
        if rng.random() < .8:
            ev += b'\x00\xff\x2f\x00'
        size = len(ev) + (0 if rng.random() < .95 else rng.choice([-2, -1, 1, 2]))
        return b'MTrk' + max(0, size).to_bytes(4, 'big') + bytes(ev)

    ntr = rng.choice([1, 1, 2, 2, 3, 4])
    res = rng.choice([480, 480, 96, 1, 7, 960, 0x7fff, -1, -5, 0]) if rng.random() < .97 else 0
    hdr_ntr = ntr if rng.random() < .95 else rng.choice([0, ntr + 1, -1, ntr - 1])
    out = b'MThd' + (6).to_bytes(4, 'big') + rng.choice([0, 1, 1, 2]).to_bytes(2, 'big') + \
        hdr_ntr.to_bytes(2, 'big', signed=True) + res.to_bytes(2, 'big', signed=True)
    out += b''.join(track() for _ in range(ntr))
    if rng.random() < .05:
        out += bytes(rng.randrange(256) for _ in range(rng.randint(1, 10)))
    return out


def check_store(args):
    """fast_tokenize_tracks on every file of a (possibly still growing) prepare_gigamidi store, against the rows the
    original tokenize_tracks wrote there: <store>/{tokens.u16, programs.u8, offsets.npy} + the metadata CSV (md5 per
    stored file, in order) + the --failed-csv list (files the original returned None for). Only checkpointed files
    (those in offsets.npy and the CSVs) are compared. Members are looked up by md5 file stem in --zip."""
    import csv
    store = Path(args.store)
    offsets = np.load(store / 'offsets.npy')
    tokens = np.memmap(store / 'tokens.u16', dtype=np.uint16, mode='r').reshape(-1, 4)
    programs = np.memmap(store / 'programs.u8', dtype=np.uint8, mode='r')
    rows = [r for r in csv.DictReader(open(args.store_csv)) if r['split'] == args.split][:len(offsets) - 1]
    failed = [r['md5'] for r in csv.DictReader(open(args.store_failed_csv))
              if r['split'] == args.split and r['reason'] == 'parse'] if args.store_failed_csv else []
    members = {}
    for p in args.zip:
        z = zipfile.ZipFile(p)
        for i in z.infolist():
            if i.filename.lower().endswith('.mid') and not i.filename.startswith('__MACOSX'):
                members[Path(i.filename).stem] = (z, i)
    jobs = [(r['md5'], k) for k, r in enumerate(rows)] + [(m, None) for m in failed]
    jobs.sort(key=lambda j: members[j[0]][1].header_offset if j[0] in members else -1)
    stats, bad, t_fast, t0 = Counter(), [], 0.0, time.time()
    for n, (md5, k) in enumerate(jobs):
        if md5 not in members:
            stats['missing'] += 1
            continue
        z, info = members[md5]
        data = z.read(info)
        t = time.perf_counter()
        res = fast_tokenize_tracks(data)
        t_fast += time.perf_counter() - t
        want = None if k is None else (np.asarray(tokens[offsets[k]:offsets[k + 1]]),
                                       np.asarray(programs[offsets[k]:offsets[k + 1]]))
        ok = same(want, res)
        stats['files'] += 1
        stats['none'] += want is None
        stats['ok'] += ok
        if not ok:
            bad.append(f'{info.filename}: {describe(want, res)}')
        if (n + 1) % 20000 == 0:
            print(f'{n + 1}/{len(jobs)} {stats["ok"]}/{stats["files"]} ok, {time.time() - t0:.0f}s', flush=True)
    print(f'\n== against store {store} ({args.split}) ==')
    print(f'{stats["files"]} files ({stats["none"]} None in the original), {stats["ok"]} identical = '
          f'{stats["ok"] / max(1, stats["files"]):.6%}; {stats["missing"]} not found in the zips')
    print(f'fallback to pretty_midi: {STATS["fallback"]} of {STATS["fast"] + STATS["fallback"]} calls; '
          f'fast {stats["files"] / t_fast:.0f} files/s (tokenize only, single process)')
    for b in bad[:50]:
        print('  ', b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', action='append', required=True)
    ap.add_argument('--per-category', type=int, default=2500)
    ap.add_argument('--macosx', type=int, default=500, help='AppleDouble ._*.mid junk files (broken-file tests)')
    ap.add_argument('--fuzz', type=int, default=0)
    ap.add_argument('--synthetic', type=int, default=0, help='random synthetic edge-case files')
    ap.add_argument('--speed', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-mismatch', default=None, help='directory to write mismatching files to')
    ap.add_argument('--store', help='compare against a prepare_gigamidi store instead (see check_store)')
    ap.add_argument('--store-csv')
    ap.add_argument('--store-failed-csv')
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()
    if args.store:
        return check_store(args)

    members = sample_members(args.zip, args.per_category, args.macosx, args.seed)
    rng = random.Random(args.seed + 1)
    zips = {p: zipfile.ZipFile(p) for p in args.zip}
    stats = Counter()
    kinds = Counter()
    t_orig = t_fast = 0.0
    examples = []
    t0 = time.time()
    jobs = [(p, cat, info) for p, cat, info in members] + [(None, 'synthetic', None)] * args.synthetic
    for k, (p, cat, info) in enumerate(jobs):
        if p is None:
            data, name = synth(rng), f'synthetic_{k}.mid'
        else:
            data, name = zips[p].read(info), info.filename
        variants = [(cat, data)] + [(cat + '+fuzz', corrupt(data, rng)) for _ in range(args.fuzz)]
        for label, blob in variants:
            t = time.perf_counter()
            pm = _orig_read_midi(blob)
            loader._read_midi = lambda _m: pm
            try:
                o_tr, o_mi = loader.tokenize_tracks(blob), loader.tokenize_midi(blob)
            finally:
                loader._read_midi = _orig_read_midi
            t_orig += time.perf_counter() - t
            t = time.perf_counter()
            f_tr, f_mi = fast_tokenize_tracks(blob), fast_tokenize_midi(blob)
            t_fast += time.perf_counter() - t
            group = 'fuzz' if label.endswith('+fuzz') else 'synthetic' if label == 'synthetic' else 'real'
            stats[group, 'files'] += 1
            stats[group, 'none'] += o_tr is None
            ok_tr, ok_mi = same(o_tr, f_tr), same(o_mi, f_mi)
            stats[group, 'tracks_ok'] += ok_tr
            stats[group, 'midi_ok'] += ok_mi
            stats[group, 'both_ok'] += ok_tr and ok_mi
            if label != group:
                stats[label, 'files'] += 1
                stats[label, 'both_ok'] += ok_tr and ok_mi
            for fn, ok, a, b in (('tracks', ok_tr, o_tr, f_tr), ('midi', ok_mi, o_mi, f_mi)):
                if not ok:
                    d = describe(a, b)
                    kinds[fn, d.split(':')[0]] += 1
                    if len(examples) < 50:
                        examples.append(f'{label} {name} [{fn}] {d}')
                    if args.save_mismatch:
                        out = Path(args.save_mismatch)
                        out.mkdir(parents=True, exist_ok=True)
                        (out / f'{group}_{k}_{Path(name).name}').write_bytes(blob)
        if (k + 1) % 2000 == 0:
            r = stats['real', 'both_ok'] / max(1, stats['real', 'files'])
            print(f'{k + 1}/{len(jobs)} files, real match {r:.6f}, {time.time() - t0:.0f}s', flush=True)

    print('\n== equality (tokens + programs + None-ness) ==')
    for group in ('real', 'fuzz', 'synthetic'):
        n = stats[group, 'files']
        if n:
            print(f'{group}: {n} files ({stats[group, "none"]} None in the original): tokenize_tracks '
                  f'{stats[group, "tracks_ok"]}/{n}, tokenize_midi {stats[group, "midi_ok"]}/{n}, '
                  f'both {stats[group, "both_ok"]}/{n} = {stats[group, "both_ok"] / n:.6%}')
    for (label, key), v in sorted(stats.items()):
        if key == 'files' and label not in ('real', 'fuzz', 'synthetic'):
            print(f'  {label}: {stats[label, "both_ok"]}/{v}')
    print(f'fallback to pretty_midi: {STATS["fallback"]} of {STATS["fast"] + STATS["fallback"]} calls')
    print('mismatch kinds:', dict(kinds) or 'none')
    for e in examples:
        print('  ', e)
    print(f'(comparison loop: original {t_orig:.1f}s incl. one shared parse, fast {t_fast:.1f}s for both functions)')

    if args.speed:   # unpatched end-to-end speed, single process, tokenize_tracks on the first N real files
        blobs = [zips[p].read(i) for p, _, i in members[:args.speed]]
        t = time.perf_counter()
        for b in blobs:
            loader.tokenize_tracks(b)
        to = time.perf_counter() - t
        t = time.perf_counter()
        for b in blobs:
            fast_tokenize_tracks(b)
        tf = time.perf_counter() - t
        mb = sum(map(len, blobs)) / 1e6
        print(f'\n== speed, tokenize_tracks, {len(blobs)} files ({mb:.1f} MB), single process ==')
        print(f'original {len(blobs) / to:.1f} files/s, fast {len(blobs) / tf:.1f} files/s, speed-up {to / tf:.1f}x')


if __name__ == '__main__':
    main()
