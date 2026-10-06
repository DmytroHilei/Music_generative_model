"""
Tokenize Aria-MIDI straight from the .tar.gz into token stores, without extracting 371k files to disk.

    python data/prepare_aria.py --tar data/aria/aria-midi-v1-deduped-ext.tar.gz --name aria
    python data/prepare_aria.py --tar ... --name aria_poprock --genres pop,rock     # fine-tune stage subset

Writes data/cache/<name>_train/ and data/cache/<name>_validation/ (same format MaestroDataset builds:
tokens.u16 + offsets.npy + meta.json) and data/aria/<name>.csv with one row per MIDI segment:
file_id, segment, split, genre, composer, audio_score, n_notes.
Use in training with csv_path='data/combined.csv,store:data/cache/aria'.

Split: 1% of recordings (by file_id, so all segments of a recording stay together) go to validation.
Aria-MIDI is CC-BY-NC-SA 4.0: non-commercial research use.
"""

import argparse
import csv
import io
import json
import os
import tarfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from musicar.data_loader import _tokenize_to_array


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tar', required=True)
    parser.add_argument('--name', default='aria')
    parser.add_argument('--cache-dir', default='data/cache')
    parser.add_argument('--genres', default=None, help="comma-separated genres to keep, e.g. 'pop,rock'")
    parser.add_argument('--min-audio-score', type=float, default=0.0)
    parser.add_argument('--val-percent', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4096, help='files read from the tar per parallel batch')
    return parser.parse_args()


def tokenize_bytes(item):
    name, data = item
    return name, _tokenize_to_array(io.BytesIO(data))


class StoreWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / 'meta.json').unlink(missing_ok=True)  # incomplete until close()
        self.f = open(self.path / 'tokens.u16', 'wb')
        self.offsets = [0]

    def add(self, arr):
        self.f.write(arr.tobytes())
        self.offsets.append(self.offsets[-1] + len(arr))

    def close(self, n_failed):
        self.f.close()
        np.save(self.path / 'offsets.npy', np.array(self.offsets, dtype=np.int64))
        meta = dict(n_files=len(self.offsets) - 1, n_notes=self.offsets[-1], n_empty_or_failed=n_failed)
        (self.path / 'meta.json').write_text(json.dumps(meta))
        return meta


def main():
    args = parse_args()
    genres = set(args.genres.split(',')) if args.genres else None

    # metadata.json comes before data/ in the archive; read it in a first quick pass
    with tarfile.open(args.tar, 'r|gz') as tf:
        for m in tf:
            if m.name.endswith('metadata.json'):
                metadata = json.load(tf.extractfile(m))
                break

    def keep(file_id, segment):
        entry = metadata.get(file_id, {})
        if genres is not None and entry.get('metadata', {}).get('genre') not in genres:
            return False
        return entry.get('audio_scores', {}).get(segment, 1.0) >= args.min_audio_score

    def split_of(file_id):
        return 'validation' if zlib.crc32(file_id.encode()) % 100 < args.val_percent else 'train'

    writers = {s: StoreWriter(Path(args.cache_dir) / f'{args.name}_{s}') for s in ('train', 'validation')}
    csv_path = Path(args.tar).parent / f'{args.name}.csv'
    rows, failed = [], {'train': 0, 'validation': 0}

    def flush(batch, pool):
        for name, arr in pool.imap(tokenize_bytes, batch, chunksize=32):  # imap keeps tar order
            file_id, segment = Path(name).stem.rsplit('_', 1)
            split = split_of(file_id)
            if arr is None:
                failed[split] += 1
                continue
            writers[split].add(arr)
            md = metadata.get(file_id, {})
            rows.append(dict(file_id=file_id, segment=segment, split=split,
                             genre=md.get('metadata', {}).get('genre', ''),
                             composer=md.get('metadata', {}).get('composer', ''),
                             audio_score=md.get('audio_scores', {}).get(segment, ''),
                             n_notes=len(arr)))

    with tarfile.open(args.tar, 'r|gz') as tf, Pool(os.cpu_count()) as pool:
        batch, pbar = [], tqdm(desc='Tokenizing Aria', unit='file', total=None)
        for m in tf:
            if not m.name.endswith('.mid'):
                continue
            file_id, segment = Path(m.name).stem.rsplit('_', 1)
            if not keep(file_id, segment):
                continue
            batch.append((m.name, tf.extractfile(m).read()))
            if len(batch) >= args.batch:  # bounded memory: the tar is read sequentially, parsed in parallel
                flush(batch, pool)
                pbar.update(len(batch))
                batch = []
        flush(batch, pool)
        pbar.update(len(batch))
        pbar.close()

    for split, w in writers.items():
        meta = w.close(failed[split])
        print(f"{split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes, {failed[split]} failed -> {w.path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'metadata: {csv_path}')


if __name__ == '__main__':
    main()
