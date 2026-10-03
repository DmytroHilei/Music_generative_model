"""
Tokenize Discover MIDI straight from its .tar.gz into token stores, deduplicating against GigaMIDI and within itself
in the same pass.

    python data/prepare_discover.py --tar /data/discover/Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz --name discover
    python data/prepare_discover.py --tar <partial .incomplete> --name discover_probe --allow-truncated  # probe

Discover (projectlosangeles/Discover-MIDI-Dataset, CC-BY-NC-SA) is ./MIDIs/<x>/<y>/<md5>.mid with the md5 of the file
bytes as its name, the same naming as GigaMIDI. Every file goes through three duplicate checks, the first copy wins:
    md5    the file name is a GigaMIDI file name: byte-identical, dropped before tokenizing
    exact  hash of the tokens + programs (the notes exactly as the loader sees them; catches re-saved files)
    loose  hash of the sorted (onset, pitch) pairs, drums kept apart from pitched notes: the same notes with other
           instruments, velocities or durations
GigaMIDI's hashes are computed once from its stores (no re-tokenizing) and cached in data/gigamidi/<giga>_hashes.npz;
building that cache also prints GigaMIDI's own duplicate counts.

Writes data/cache/<name>_{train,validation}/ (tokens.u16 + programs.u8 like the GigaMIDI stores, tokenized by the same
function) and data/discover/<name>.csv with one row per MIDI file in the --keep-permille subset: md5, split, status
(kept | dup_md5 | dup_exact | dup_loose | failed; dup_* also says whether the earlier copy was in GigaMIDI or Discover),
n_notes, n_programs, exact, loose. Split: --val-permille of the files by md5 go to validation.

Resumable: a .tar.gz can't seek, so a restart streams the archive again and skips the files already done without
parsing them; the seen hashes are rebuilt from the csv.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from prepare_gigamidi import ResumableStore, tokenize_tracks

CSV_FIELDS = ['md5', 'split', 'status', 'n_notes', 'n_programs', 'exact', 'loose']
EXPECTED_FILES = 6_740_000
DRUMS = 128


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tar', required=True)
    parser.add_argument('--name', default='discover')
    parser.add_argument('--giga', default='gigamidi', help='name of the GigaMIDI stores/csv to dedupe against')
    parser.add_argument('--cache-dir', default='data/cache')
    parser.add_argument('--work-dir', default='data/discover', help='csv, progress and stats files')
    parser.add_argument('--val-permille', type=int, default=5)
    parser.add_argument('--keep-permille', type=int, default=1000,
                        help='random subset by md5 (decided before parsing), so the store fits on disk')
    parser.add_argument('--max-files', type=int, default=None, help='stop after this many MIDI files (probe)')
    parser.add_argument('--max-bytes', type=int, default=2_000_000, help='skip larger MIDI files (pathological)')
    parser.add_argument('--allow-truncated', action='store_true', help='end cleanly at the end of a partial archive')
    parser.add_argument('--min-free-gb', type=float, default=3.0, help='stop at a checkpoint when free disk is lower')
    parser.add_argument('--checkpoint', type=int, default=50_000, help='files between checkpoints')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    return parser.parse_args()


def h64(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little')


def note_hashes(tokens, programs):
    """(exact, loose) 64-bit hashes of one file's notes. tokens (n, 4) pitch/velocity/duration/delta_time."""
    tokens = np.ascontiguousarray(tokens, dtype=np.uint16)
    programs = np.ascontiguousarray(programs, dtype=np.uint8)
    exact = h64(tokens.tobytes() + programs.tobytes())
    onset = np.cumsum(tokens[:, 3], dtype=np.int64)
    pitch = tokens[:, 0].astype(np.int64) + DRUMS * (programs == DRUMS)
    order = np.lexsort((pitch, onset))
    loose = h64(np.stack([onset[order], pitch[order]]).tobytes())
    return exact, loose


# ---------------------------------------------------------------- GigaMIDI hash index

_giga = {}


def _giga_init(store):
    meta = json.loads((Path(store) / 'meta.json').read_text())
    _giga['offsets'] = np.load(Path(store) / 'offsets.npy')
    _giga['tokens'] = np.memmap(Path(store) / 'tokens.u16', dtype=np.uint16, mode='r', shape=(meta['n_notes'], 4))
    _giga['programs'] = np.memmap(Path(store) / 'programs.u8', dtype=np.uint8, mode='r', shape=(meta['n_notes'],))


def _giga_hash(rng):
    o = _giga['offsets']
    return [note_hashes(_giga['tokens'][o[i]:o[i + 1]], _giga['programs'][o[i]:o[i + 1]]) for i in range(*rng)]


def giga_index(args):
    """md5 names + exact/loose hashes of every GigaMIDI file (all splits), cached in an .npz."""
    path = Path('data/gigamidi') / f'{args.giga}_hashes.npz'
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    md5 = {}
    with open(Path('data/gigamidi') / f'{args.giga}.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            md5.setdefault(row['split'], []).append(row['md5'])
    parts = {k: [] for k in ('md5', 'split', 'exact', 'loose')}
    for split_id, split in enumerate(('train', 'validation', 'test')):
        store = Path(args.cache_dir) / f'{args.giga}_{split}'
        n = len(np.load(store / 'offsets.npy')) - 1
        assert n == len(md5[split]), f'{store}: {n} files but {len(md5[split])} csv rows'
        ranges = [(a, min(a + 2000, n)) for a in range(0, n, 2000)]
        hashes = []
        with Pool(args.workers, initializer=_giga_init, initargs=(store,)) as pool:
            for chunk in tqdm(pool.imap(_giga_hash, ranges), total=len(ranges), desc=f'hashing {args.giga} {split}'):
                hashes.extend(chunk)
        h = np.array(hashes, dtype=np.uint64)
        parts['md5'].append(np.array([bytes.fromhex(m) for m in md5[split]], dtype='S16'))
        parts['split'].append(np.full(n, split_id, dtype=np.uint8))
        parts['exact'].append(h[:, 0])
        parts['loose'].append(h[:, 1])
    index = {k: np.concatenate(v) for k, v in parts.items()}
    np.savez(path, **index)

    # GigaMIDI's own duplicates, for the record
    stats = {}
    for kind in ('exact', 'loose'):
        _, first, counts = np.unique(index[kind], return_index=True, return_counts=True)
        dup = len(index[kind]) - len(first)
        # files in validation/test whose notes also occur in train (leakage into GigaMIDI's own eval split)
        train = set(index[kind][index['split'] == 0].tolist())
        leak = sum(1 for h in index[kind][index['split'] > 0].tolist() if h in train)
        stats[kind] = dict(duplicate_files=int(dup), share=round(dup / len(index[kind]), 4), val_test_in_train=leak)
    print(f'{args.giga}: {len(index["md5"]):,} files, duplicates {json.dumps(stats)}', flush=True)
    (Path('data/gigamidi') / f'{args.giga}_dupes.json').write_text(json.dumps(stats, indent=1))
    return index


# ---------------------------------------------------------------- Discover pass

def work(item):
    k, data = item
    result = tokenize_tracks(data)
    if result is None:
        return k, None
    tokens, programs = result
    return k, (tokens, programs, *note_hashes(tokens, programs))


def work_or_skip(item):
    """Files already decided in the reader (item[1] is None) pass through, so the results stay in archive order."""
    return item if item[1] is None else work(item)


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    giga = giga_index(args)
    giga_md5 = set(giga['md5'].tolist())
    seen = {kind: dict.fromkeys(giga[kind].tolist(), 'giga') for kind in ('exact', 'loose')}
    del giga

    progress_path = work_dir / f'{args.name}_progress.json'
    csv_path = work_dir / f'{args.name}.csv'
    state = json.loads(progress_path.read_text()) if progress_path.exists() else {'files': 0, 'done': False}
    if state['done']:
        print(f'{args.name} is already done')
        return
    counts = state.pop('counts', {})
    if state['files']:  # resume: rebuild the seen hashes from the csv
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['status'] == 'kept':
                    seen['exact'][int(row['exact'], 16)] = 'self'
                    seen['loose'][int(row['loose'], 16)] = 'self'
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def split_of(md5):
        return 'validation' if zlib.crc32(md5.encode()) % 1000 < args.val_permille else 'train'

    stores = {s: ResumableStore(Path(args.cache_dir) / f'{args.name}_{s}', resume=state['files'] > 0)
              for s in ('train', 'validation')}
    start = state['files']
    total = min(EXPECTED_FILES, args.max_files or EXPECTED_FILES)
    pbar = tqdm(desc=args.name, unit='file', total=total, initial=start, smoothing=0.05)
    names, rows = {}, []

    def save():
        for store in stores.values():
            store.checkpoint()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerows(rows)
        rows.clear()
        progress_path.write_text(json.dumps(state | {'counts': counts}, indent=1))

    def items():
        """(k, bytes) of the files to tokenize; md5 duplicates of GigaMIDI and oversized files are decided here."""
        k = -1
        try:
            with tarfile.open(args.tar, 'r|gz') as tf:
                for m in tf:
                    if not m.isfile() or not m.name.endswith('.mid'):
                        continue
                    k += 1
                    if args.max_files is not None and k >= args.max_files:
                        return
                    if k < start:
                        continue
                    md5 = Path(m.name).stem
                    if zlib.crc32(md5.encode() + b'keep') % 1000 >= args.keep_permille:
                        names[k] = (md5, 'skipped')
                    elif md5 in giga_md5:
                        names[k] = (md5, 'dup_md5_giga')
                    elif m.size > args.max_bytes:
                        names[k] = (md5, 'failed')
                    else:
                        names[k] = (md5, None)
                        yield k, tf.extractfile(m).read()
                        continue
                    yield k, None
        except (EOFError, tarfile.ReadError, zlib.error) as e:
            if not args.allow_truncated:
                raise
            print(f'\nend of the partial archive after {k + 1:,} files ({type(e).__name__})', flush=True)

    with Pool(args.workers) as pool:
        for k, result in pool.imap(work_or_skip, items(), chunksize=64):  # imap keeps archive order
            md5, status = names.pop(k)
            split = split_of(md5)
            row = dict(md5=md5, split=split, status=status)
            if status == 'skipped':
                pass
            elif status is None and result is None:
                row['status'] = 'failed'
            elif status is None:
                tokens, programs, exact, loose = result
                row.update(n_notes=len(tokens), n_programs=len(np.unique(programs)),
                           exact=f'{exact:016x}', loose=f'{loose:016x}')
                if exact in seen['exact']:
                    row['status'] = f'dup_exact_{seen["exact"][exact]}'
                elif loose in seen['loose']:
                    row['status'] = f'dup_loose_{seen["loose"][loose]}'
                else:
                    row['status'] = 'kept'
                    seen['exact'][exact] = seen['loose'][loose] = 'self'
                    stores[split].add(tokens, programs)
            counts[row['status']] = counts.get(row['status'], 0) + 1
            if status != 'skipped':
                rows.append(row)
            state['files'] = k + 1
            pbar.update(1)
            if state['files'] % args.checkpoint == 0:
                save()
                kept = counts.get('kept', 0)
                notes = sum(s.offsets[-1] for s in stores.values())
                pbar.set_postfix(kept=f'{kept / state["files"]:.0%}', notes=f'{notes / 1e6:.0f}M',
                                 proj_gb=f'{notes * 9 / 1e9 * total / state["files"]:.1f}')
                if shutil.disk_usage(args.cache_dir).free / 1e9 < args.min_free_gb:
                    print(f'\nSTOP: less than {args.min_free_gb} GB free; rerun to resume', flush=True)
                    return
    pbar.close()
    state['done'] = True
    save()
    for split, store in stores.items():
        meta = store.close(counts.get('failed', 0) if split == 'train' else 0)
        print(f"{split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes -> {store.path}")
    n = state['files']
    print(f'{n:,} MIDI files: ' + ', '.join(f'{s} {c:,} ({c / n:.1%})' for s, c in sorted(counts.items())))
    print(f'metadata: {csv_path}')


if __name__ == '__main__':
    main()
