"""
Tokenize GigaMIDI straight from its downloaded archive into token stores, without extracting anything to disk.

    python data/prepare_gigamidi.py --zip ~/Downloads/Final_GigaMIDI_V2.0_Final.zip --name gigamidi
    python data/prepare_gigamidi.py --zip ... --name gigamidi_probe --max-files 20000 --splits validation  # size probe

The archive (HF repo file Final_GigaMIDI_V2.0_Final.zip, content V1.1) holds three deflated inner zips
(training-80% / validation-10% / test-10%), each with drums-only/, no-drums/, all-instruments-with-drums/ folders.
An inner zip is compressed inside the outer one, so it can't be seeked cheaply: its central directory is read once,
then the inner zip is streamed forward once and every member is sliced out at its header offset. Decompression and
tokenization run in worker processes from memory.

Writes data/cache/<name>_{train,validation,test}/ with
    tokens.u16    (n_notes, 4) pitch, velocity, duration, delta_time -- same layout as the other stores
    programs.u8   (n_notes,)   GM program 0-127, 128 = drums (all tracks are kept, unlike the piano stores)
    offsets.npy, meta.json (written last: a store without meta.json is incomplete)
and data/gigamidi/<name>.csv with one row per file: md5 (the file name), split, category, n_notes, n_programs.
Notes of all tracks are merged and sorted by (onset, program, pitch); delta_time is onset-to-onset as usual.

Resumable: progress (files done per split) is checkpointed every --checkpoint files and the stores are truncated
back to the last checkpoint on restart. Uses GigaMIDI's own split. GigaMIDI is CC BY-NC 4.0: non-commercial research.
"""

import argparse
import csv
import io
import json
import os
import shutil
import struct
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pretty_midi
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import quantize_time, quantize_velocity  # noqa: E402

SPLITS = {'training': 'train', 'validation': 'validation', 'test': 'test'}  # inner zip prefix -> store split
CATEGORIES = ('drums-only', 'no-drums', 'all-instruments-with-drums')
CSV_FIELDS = ['md5', 'split', 'category', 'n_notes', 'n_programs']


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--zip', required=True, help='the downloaded outer archive')
    parser.add_argument('--name', default='gigamidi')
    parser.add_argument('--cache-dir', default='data/cache')
    parser.add_argument('--work-dir', default='data/gigamidi', help='csv and progress file')
    parser.add_argument('--splits', default='validation,test,training', help='inner zips to process, in order')
    parser.add_argument('--categories', default=','.join(CATEGORIES))
    parser.add_argument('--max-files', type=int, default=None, help='per split (size probe)')
    parser.add_argument('--max-bytes', type=int, default=2_000_000, help='skip larger MIDI files (pathological)')
    parser.add_argument('--min-free-gb', type=float, default=1.0, help='stop at a checkpoint when free disk is lower')
    parser.add_argument('--checkpoint', type=int, default=20_000, help='files between checkpoints')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    return parser.parse_args()


def tokenize_tracks(data, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """MIDI bytes -> ((n, 4) uint16 tokens, (n,) uint8 programs), all tracks incl. drums; None if empty/broken."""
    try:
        midi = pretty_midi.PrettyMIDI(io.BytesIO(data))
    except Exception:
        return None
    notes = [(n.start, 128 if inst.is_drum else inst.program, n.pitch, n.velocity, n.end - n.start)
             for inst in midi.instruments for n in inst.notes]
    if not notes:
        return None
    notes.sort(key=lambda n: n[:3])
    tokens = np.empty((len(notes), 4), dtype=np.uint16)
    programs = np.empty(len(notes), dtype=np.uint8)
    prev = notes[0][0]
    for i, (start, program, pitch, velocity, length) in enumerate(notes):
        tokens[i] = (pitch, quantize_velocity(velocity, velocity_bins),
                     quantize_time(max(0.0, length), time_resolution, max_duration_bin),
                     quantize_time(max(0.0, start - prev), time_resolution, max_delta_bin))
        programs[i] = program
        prev = start
    return tokens, programs


def work(item):
    key, method, raw = item
    try:
        data = zlib.decompress(raw, -15) if method == zipfile.ZIP_DEFLATED else raw
    except zlib.error:
        return key, None
    return key, tokenize_tracks(data)


def iter_inner(outer, member, infos):
    """Stream the deflated inner zip forward once; yield (info, compress_type, raw bytes) in header-offset order."""
    with outer.open(member) as f:
        pos = 0
        for info in infos:
            while pos < info.header_offset:  # forward skip = read-through; never seek backwards
                pos += len(f.read(min(1 << 24, info.header_offset - pos)))
            header = f.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError(f'bad local header for {info.filename}')
            name_len, extra_len = struct.unpack('<HH', header[26:30])
            f.read(name_len + extra_len)
            raw = f.read(info.compress_size)
            pos = info.header_offset + 30 + name_len + extra_len + info.compress_size
            yield info, info.compress_type, raw


class ResumableStore:
    """tokens.u16 + programs.u8 + offsets.npy, checkpointed so a crash loses one checkpoint interval at most."""

    def __init__(self, path, resume):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / 'meta.json').unlink(missing_ok=True)  # incomplete until close()
        self.offsets = np.load(self.path / 'offsets.npy').tolist() if resume else [0]
        n = self.offsets[-1]
        self.tokens = open(self.path / 'tokens.u16', 'a+b')
        self.programs = open(self.path / 'programs.u8', 'a+b')
        self.tokens.truncate(n * 4 * 2)  # drop anything written after the last checkpoint
        self.programs.truncate(n)

    def add(self, tokens, programs):
        self.tokens.write(tokens.tobytes())
        self.programs.write(programs.tobytes())
        self.offsets.append(self.offsets[-1] + len(tokens))

    def checkpoint(self):
        self.tokens.flush()
        self.programs.flush()
        np.save(self.path / 'offsets.npy', np.array(self.offsets, dtype=np.int64))

    def close(self, n_failed):
        self.checkpoint()
        self.tokens.close()
        self.programs.close()
        meta = dict(n_files=len(self.offsets) - 1, n_notes=self.offsets[-1], programs=True, n_empty_or_failed=n_failed)
        (self.path / 'meta.json').write_text(json.dumps(meta))
        return meta


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work_dir / f'{args.name}_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    csv_path = work_dir / f'{args.name}.csv'
    if not progress:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    categories = set(args.categories.split(','))

    outer = zipfile.ZipFile(os.path.expanduser(args.zip))
    members = {p: m for p in SPLITS for m in outer.namelist()
               if m.endswith('.zip') and not m.startswith('__MACOSX') and Path(m).name.startswith(p)}

    with Pool(args.workers) as pool:
        for prefix in args.splits.split(','):
            split = SPLITS[prefix]
            state = progress.get(split, {'files': 0, 'failed': 0, 'done': False})
            if state['done']:
                continue
            print(f'{split}: reading the central directory of {members[prefix]} ...', flush=True)
            with outer.open(members[prefix]) as f:
                infos = zipfile.ZipFile(f).infolist()
            infos = sorted((i for i in infos if i.filename.lower().endswith('.mid')
                            and not i.filename.startswith('__MACOSX') and i.filename.split('/')[1] in categories
                            and i.file_size <= args.max_bytes), key=lambda i: i.header_offset)
            infos = infos[:args.max_files]
            store = ResumableStore(Path(args.cache_dir) / f'{args.name}_{split}', resume=state['files'] > 0)
            start = state['files']
            pbar = tqdm(desc=split, unit='file', total=len(infos), initial=start, smoothing=0.05)
            rows = []

            def save():
                store.checkpoint()
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writerows(rows)
                rows.clear()
                progress[split] = state
                progress_path.write_text(json.dumps(progress, indent=1))

            items = ((k, m, raw) for k, (info, m, raw) in enumerate(iter_inner(outer, members[prefix], infos))
                     if k >= start)
            for k, result in pool.imap(work, items, chunksize=64):  # imap keeps archive order
                info = infos[k]
                if result is None:
                    state['failed'] += 1
                else:
                    store.add(*result)
                    rows.append(dict(md5=Path(info.filename).stem, split=split, category=info.filename.split('/')[1],
                                     n_notes=len(result[0]), n_programs=len(np.unique(result[1]))))
                state['files'] = k + 1
                pbar.update(1)
                if state['files'] % args.checkpoint == 0:
                    save()
                    notes = store.offsets[-1]
                    pbar.set_postfix(notes=f'{notes / 1e6:.0f}M', per_file=f'{notes / (len(store.offsets) - 1):.0f}',
                                     proj_gb=f'{notes * 9 / 1e9 * len(infos) / state["files"]:.1f}')
                    if shutil.disk_usage(work_dir).free / 1e9 < args.min_free_gb:
                        print(f'\nSTOP: less than {args.min_free_gb} GB free; rerun to resume', flush=True)
                        return
            pbar.close()
            state['done'] = True
            save()
            meta = store.close(state['failed'])
            print(f"{split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes, {state['failed']} failed "
                  f"-> {store.path}", flush=True)
    print(f'metadata: {csv_path}')


if __name__ == '__main__':
    main()
