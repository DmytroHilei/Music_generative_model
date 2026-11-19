"""
Token stores for the census's smaller sources (4-9), in the same layout as the repo's stores, plus a csv per source that
the census tools read as a generic source (file_id, split, status kept|failed, n_notes, md5, extras...).

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/prepare_extra.py --source pianocore \
        --input /workspace/raw/pianocore

Sources and how they are read:
    pianocore  parquet shards (data/*.parquet); one row per performance: performance_midi_bytes (raw performance MIDI);
               extras keep performance_dataset (origin), tier_a / tier_a_star / tier_b, is_transcription, is_duplicate
    piast      every .mid / .midi file under the input dir
    beat       a .tar.gz of MusicXML scores, converted to MIDI with music21 in the workers (30 s cap per file)
    dir        generic: every .mid / .midi under --input (for small extras), tokenized like --piano says
Piano sources (pianocore, piast, beat) are tokenized with tokenize_midi (no programs.u8, as the Aria stores);
--multi switches to tokenize_tracks. Both through the bit-identical C port (fast_tokenize.py).

Writes data/cache/<source>_{train,validation}/ and data/<source>/<source>.csv. Split: 1% of file ids (crc32) go to
validation, as prepare_aria.py. Not resumable (the sources are small); a rerun starts over.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import signal
import sys
import tarfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_tokenize import fast_tokenize_midi, fast_tokenize_tracks  # noqa: E402

PIANO = {'pianocore', 'piast', 'beat'}
MAX_BYTES = 2_000_000


class Writer:
    def __init__(self, path, programs):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / 'meta.json').unlink(missing_ok=True)
        self.t = open(self.path / 'tokens.u16', 'wb')
        self.p = open(self.path / 'programs.u8', 'wb') if programs else None
        self.offsets = [0]

    def add(self, tokens, programs=None):
        self.t.write(np.ascontiguousarray(tokens, np.uint16).tobytes())
        if self.p is not None:
            self.p.write(np.ascontiguousarray(programs, np.uint8).tobytes())
        self.offsets.append(self.offsets[-1] + len(tokens))

    def close(self, n_failed):
        self.t.close()
        if self.p is not None:
            self.p.close()
        np.save(self.path / 'offsets.npy', np.array(self.offsets, dtype=np.int64))
        meta = dict(n_files=len(self.offsets) - 1, n_notes=self.offsets[-1], n_empty_or_failed=n_failed)
        if self.p is not None:
            meta['programs'] = True
        (self.path / 'meta.json').write_text(json.dumps(meta))
        return meta


# ------------------------------------------------------------------ readers: yield (file_id, bytes, kind, extras)

def read_pianocore(root):
    import pyarrow.parquet as pq
    cols = ['id', 'performance_dataset', 'tier_a', 'tier_a_star', 'tier_b', 'is_transcription', 'is_duplicate',
            'split', 'performance_midi_bytes']
    for shard in sorted(Path(root).glob('data/*.parquet')):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d['id'])):
                extras = {k: d[k][i] for k in cols if k not in ('id', 'performance_midi_bytes')}
                extras['source_split'] = extras.pop('split')
                yield str(d['id'][i]), d['performance_midi_bytes'][i], 'midi', extras


def read_dir(root):
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.mid', '.midi'):
            yield str(p.relative_to(root)), p.read_bytes(), 'midi', {}


def read_beat(root):
    tars = sorted(Path(root).glob('*.tar.gz'))
    assert tars, f'no .tar.gz under {root}'
    with tarfile.open(tars[0], 'r|gz') as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(('.musicxml', '.xml', '.mxl')):
                yield m.name, tf.extractfile(m).read(), 'musicxml', {}


READERS = {'pianocore': read_pianocore, 'piast': read_dir, 'beat': read_beat, 'dir': read_dir}


# ------------------------------------------------------------------ workers

class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


def xml_to_midi(data, timeout=30):
    """MusicXML bytes -> MIDI bytes via music21 (None on failure or after `timeout` s)."""
    import music21
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        score = music21.converter.parseData(data.decode('utf-8', errors='replace'), format='musicxml')
        mf = music21.midi.translate.streamToMidiFile(score)
        return mf.writestr()
    except Exception:
        return None
    finally:
        signal.alarm(0)


MULTI = {'flag': False}


def work(item):
    fid, data, kind, extras = item
    if data is None or len(data) > (MAX_BYTES * 10 if kind == 'musicxml' else MAX_BYTES):
        return fid, None, '', extras
    md5 = hashlib.md5(data).hexdigest()
    if kind == 'musicxml':
        data = xml_to_midi(data)
        if data is None:
            return fid, None, md5, extras
    res = fast_tokenize_tracks(data) if MULTI['flag'] else fast_tokenize_midi(data)
    return fid, res, md5, extras


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--source', required=True, help='pianocore | piast | beat | dir')
    ap.add_argument('--input', required=True)
    ap.add_argument('--name', default=None, help='store/csv name (default: --source)')
    ap.add_argument('--multi', action='store_true', help='tokenize_tracks (all instruments, programs.u8)')
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--cache-dir', default='data/cache')
    args = ap.parse_args()
    name = args.name or args.source
    MULTI['flag'] = args.multi or args.source not in PIANO and args.source != 'dir'
    writers = {s: Writer(Path(args.cache_dir) / f'{name}_{s}', MULTI['flag']) for s in ('train', 'validation')}
    failed = {'train': 0, 'validation': 0}
    out_dir = Path('data') / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, fields = [], None

    def items():
        for k, it in enumerate(READERS[args.source](args.input)):
            if args.max_files is not None and k >= args.max_files:
                return
            yield it

    with Pool(args.workers) as pool:
        for fid, res, md5, extras in tqdm(pool.imap(work, items(), chunksize=16), desc=name, unit='file'):
            split = 'validation' if zlib.crc32(fid.encode()) % 100 < 1 else 'train'
            row = dict(file_id=fid, split=split, status='kept', n_notes=0, md5=md5, **extras)
            if res is None:
                row['status'] = 'failed'
                failed[split] += 1
            elif MULTI['flag']:
                writers[split].add(*res)
                row['n_notes'] = len(res[0])
            else:
                writers[split].add(res)
                row['n_notes'] = len(res)
            rows.append(row)
    for s, w in writers.items():
        meta = w.close(failed[s])
        print(f"{s}: {meta['n_files']:,} files, {meta['n_notes']:,} notes, {failed[s]} failed -> {w.path}", flush=True)
    fields = list(rows[0])
    with open(out_dir / f'{name}.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f'{len(rows):,} files listed -> {out_dir / (name + ".csv")}')


if __name__ == '__main__':
    main()
