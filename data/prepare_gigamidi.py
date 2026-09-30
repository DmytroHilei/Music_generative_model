"""
Tokenize GigaMIDI straight from its Parquet shards into token stores: one shard (<=1.1 GB) is downloaded, tokenized
from the in-memory MIDI bytes, then deleted, so the raw dataset never sits on disk.

    .venv/bin/huggingface-cli login     # once; the dataset is gated (accept the terms on its HF page first)
    python data/prepare_gigamidi.py --name gigamidi
    python data/prepare_gigamidi.py --name gigamidi --max-shards 1   # measure notes/file first, then rerun to resume

Writes data/cache/<name>_{train,validation,test}/ with
    tokens.u16    (n_notes, 4) pitch, velocity, duration, delta_time -- same layout as the other stores
    programs.u8   (n_notes,)   GM program 0-127, 128 = drums (all tracks are kept, unlike the piano stores)
    offsets.npy, meta.json (written last: a store without meta.json is incomplete)
and data/gigamidi/<name>.csv with one row per file: md5, split, n_notes, n_tracks, styles, nomml, artist, title.
Notes of all tracks are merged and sorted by (onset, program, pitch); delta_time is onset-to-onset as usual.

Resumable: finished shards are listed in data/gigamidi/<name>_progress.json and the stores are truncated back to
the last finished shard on restart. Uses GigaMIDI's own train/validation/test split.
GigaMIDI is CC BY-NC 4.0: non-commercial research use.
"""

import argparse
import csv
import io
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')  # no hidden multi-GB download cache

import numpy as np
import pretty_midi
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import quantize_time, quantize_velocity  # noqa: E402

REPO = 'Metacreation/GigaMIDI'
SPLITS = ('train', 'validation', 'test')
META_COLUMNS = ['md5', 'num_tracks', 'music_styles_curated', 'NOMML', 'artist', 'title']
CSV_FIELDS = ['md5', 'split', 'n_notes', 'n_tracks', 'styles', 'nomml', 'artist', 'title']


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', default='gigamidi')
    parser.add_argument('--version', default='v2.0.0', help='dataset folder in the HF repo (sharded per split)')
    parser.add_argument('--cache-dir', default='data/cache')
    parser.add_argument('--work-dir', default='data/gigamidi', help='csv, progress file and the one shard in flight')
    parser.add_argument('--max-shards', type=int, default=None, help='stop after this many new shards')
    parser.add_argument('--max-bytes', type=int, default=2_000_000, help='skip larger MIDI files (pathological)')
    parser.add_argument('--min-free-gb', type=float, default=3.0, help='stop before a shard when free disk is lower')
    parser.add_argument('--batch', type=int, default=2048, help='Parquet rows per parallel batch')
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


class ResumableStore:
    """tokens.u16 + programs.u8 + offsets.npy, checkpointed after every shard so a crash loses one shard at most."""

    def __init__(self, path, resume):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / 'meta.json').unlink(missing_ok=True)  # incomplete until close()
        if resume and (self.path / 'offsets.npy').exists():
            self.offsets = np.load(self.path / 'offsets.npy').tolist()
        else:
            self.offsets = [0]
        n = self.offsets[-1]
        self.tokens = open(self.path / 'tokens.u16', 'a+b')
        self.programs = open(self.path / 'programs.u8', 'a+b')
        self.tokens.truncate(n * 4 * 2)  # drop a half-written shard
        self.programs.truncate(n)
        self.n_failed = 0

    def add(self, tokens, programs):
        self.tokens.write(tokens.tobytes())
        self.programs.write(programs.tobytes())
        self.offsets.append(self.offsets[-1] + len(tokens))

    def checkpoint(self):
        self.tokens.flush()
        self.programs.flush()
        np.save(self.path / 'offsets.npy', np.array(self.offsets, dtype=np.int64))

    def close(self):
        self.checkpoint()
        self.tokens.close()
        self.programs.close()
        meta = dict(n_files=len(self.offsets) - 1, n_notes=self.offsets[-1], programs=True,
                    n_empty_or_failed_last_run=self.n_failed)
        (self.path / 'meta.json').write_text(json.dumps(meta))
        return meta


def work(item):
    idx, data = item
    return idx, tokenize_tracks(data)


def as_text(value):
    if value is None:
        return ''
    return ';'.join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work_dir / f'{args.name}_progress.json'
    done = json.loads(progress_path.read_text())['done'] if progress_path.exists() else []
    csv_path = work_dir / f'{args.name}.csv'
    if not done:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    sizes = {f.rfilename: f.size or 0 for f in HfApi().dataset_info(REPO, files_metadata=True).siblings
             if f.rfilename.startswith(args.version + '/') and f.rfilename.endswith('.parquet')}
    shards = sorted(sizes)
    todo = [s for s in shards if s not in done][:args.max_shards]
    print(f'{len(shards)} shards, {len(done)} done, {len(todo)} to do now')
    stores = {s: ResumableStore(Path(args.cache_dir) / f'{args.name}_{s}', resume=bool(done)) for s in SPLITS}
    shard_dir = work_dir / 'shard'

    with Pool(os.cpu_count()) as pool:
        for shard in todo:
            free_gb = shutil.disk_usage(work_dir).free / 1e9
            if free_gb < args.min_free_gb:
                print(f'STOP: {free_gb:.1f} GB free < {args.min_free_gb} GB; rerun to resume')
                break
            split = next(s for s in SPLITS if f'/{s}/' in shard or shard.endswith(f'/{s}.parquet'))
            store = stores[split]
            local = hf_hub_download(REPO, shard, repo_type='dataset', local_dir=shard_dir)
            try:
                pf = pq.ParquetFile(local)
                columns = ['music'] + [c for c in META_COLUMNS if c in pf.schema_arrow.names]
                files_before, notes_before = len(store.offsets), store.offsets[-1]
                rows = []
                pbar = tqdm(desc=shard, unit='file', total=pf.metadata.num_rows)
                for batch in pf.iter_batches(batch_size=args.batch, columns=columns):
                    table = batch.to_pydict()
                    items = [(i, m) for i, m in enumerate(table['music']) if m and len(m) <= args.max_bytes]
                    store.n_failed += len(table['music']) - len(items)
                    for i, result in pool.imap(work, items, chunksize=16):  # imap keeps shard order
                        if result is None:
                            store.n_failed += 1
                            continue
                        store.add(*result)
                        meta = {c: as_text(table[c][i]) if c in table else '' for c in META_COLUMNS}
                        rows.append(dict(md5=meta['md5'], split=split, n_notes=len(result[0]),
                                         n_tracks=meta['num_tracks'], styles=meta['music_styles_curated'],
                                         nomml=meta['NOMML'], artist=meta['artist'], title=meta['title']))
                    pbar.update(len(table['music']))
                pbar.close()
            finally:
                Path(local).unlink(missing_ok=True)  # the raw shard never outlives its tokenization
            store.checkpoint()
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerows(rows)
            done.append(shard)
            progress_path.write_text(json.dumps({'done': done}, indent=1))
            n_files, n_notes = len(store.offsets) - files_before, store.offsets[-1] - notes_before
            gb = n_notes * 9 / 1e9  # 8 bytes tokens + 1 byte program per note
            projected = gb * sum(sizes.values()) / max(sizes[shard], 1)
            print(f'{shard}: {n_files:,} files, {n_notes:,} notes ({n_notes / max(n_files, 1):,.0f}/file), '
                  f'{gb:.2f} GB tokenized, whole dataset ~{projected:.0f} GB at this rate; '
                  f'failed so far in {split}: {store.n_failed}', flush=True)
    shutil.rmtree(shard_dir, ignore_errors=True)

    finished = len(done) == len(shards)
    for split, store in stores.items():
        if finished:
            meta = store.close()
            print(f"{split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes -> {store.path}")
        else:
            store.checkpoint()
    print(f'metadata: {csv_path}; {len(done)}/{len(shards)} shards done' + ('' if finished else ' (rerun to resume)'))


if __name__ == '__main__':
    main()
