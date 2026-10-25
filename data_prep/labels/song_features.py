"""
Per-song features of every file in a token store, for the caption builder (text-control plan 15h, step 1.3).

    python data_prep/labels/song_features.py --stores data/cache/gigamidi_replay_train,data/cache/gigamidi_heldout_validation

Writes data/labels/<store>.features.npz, one row per store file (labeled or not):
    seconds, n_notes, nps (notes/s), ops (onsets/s), notes_per_onset, pitch_mean, pitch_std, pitch_lo, pitch_hi
    (5th / 95th percentile), vel_mean, vel_std (velocity bins 0-31), key (0-11 major on that tonic, 12-23 minor,
    -1 = none), key_conf (correlation margin over the second-best key), tempo (BPM, 0 = none), tempo_conf,
    drums (share of drum notes), programs (n, 17) packed bits: GM programs 0-127 + drums (128) with >= 2% of the
    notes and >= 8 notes.
Pitch statistics, key and tempo leave out drum notes. Key: Krumhansl-Schmuckler over the whole song
(musicar/eval/structure_eval.py). Tempo: autocorrelation of the onset signal (20 ms frames, onsets weighted by
note count x velocity) over the first 120 s, lags 0.3-1.5 s (40-200 BPM), with a log-normal prior around 110 BPM
against octave errors (librosa-style); conf = peak height relative to lag 0.

--check compares key and tempo with MidiCaps (data/labels/<store>.jsonl from build_labels.py) on the labeled files.

--hub reads the stores from the data repo on the Hub (cloud/sync/hub.py) instead of data/cache: --stores are then store
names, read by HTTP range requests in chunks of whole files (--chunk-mb of tokens, the next chunk fetched while this
one is computed) into a small temporary store each, so nothing large lands on the disk (gigamidi_train is 17.7 GB and
not on the laptop). Each chunk's rows go to <out-dir>/<store>.features.parts/ first, so a stopped run resumes there.

    python data_prep/labels/song_features.py --hub --stores gigamidi_train --workers 8
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from musicar.data.features import DT, FIELDS, PITCH_CLASSES, file_features, key_estimate, tempo_estimate  # noqa: F401

_store = {}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stores', required=True, help='comma-separated store directories')
    p.add_argument('--out-dir', default='data/labels')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--check', action='store_true', help='compare key / tempo with MidiCaps on the labeled files')
    p.add_argument('--hub', action='store_true', help='read the stores from the Hub data repo by range requests')
    p.add_argument('--chunk-mb', type=int, default=256, help='--hub: token bytes per chunk')
    p.add_argument('--tmp-dir', default='data/cache/tmp_features', help='--hub: the chunks\' temporary stores')
    p.add_argument('--max-files', type=int, default=0, help='--hub: only the first N files (a test; no parts kept)')
    return p.parse_args()


def _init(store):
    _store['path'] = str(store)
    offsets = np.load(Path(store) / 'offsets.npy')
    n = int(offsets[-1])
    _store['offsets'] = offsets
    _store['tokens'] = np.memmap(Path(store) / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    path = Path(store) / 'programs.u8'
    _store['programs'] = np.memmap(path, dtype=np.uint8, mode='r', shape=(n,)) if path.exists() else None


def _work(job):
    """job: (store directory, first file, end file); a worker opens a store the first time it sees it."""
    store, *rng = job
    if _store.get('path') != str(store):
        _init(store)
    o, rows, bits = _store['offsets'], [], []
    for i in range(*rng):
        a, b = int(o[i]), int(o[i + 1])
        prog = (_store['programs'][a:b].astype(np.int64) if _store['programs'] is not None
                else np.zeros(b - a, np.int64))
        f, pb = file_features(np.asarray(_store['tokens'][a:b]), prog)
        rows.append(f)
        bits.append(pb)
    return rng[0], np.stack(rows), np.stack(bits)


def check(out, store, labels_dir):
    path = Path(labels_dir) / f'{Path(store).name}.jsonl'
    if not path.exists():
        print(f'{path}: no label table, skipping the check')
        return
    z = np.load(out)
    key_hit = mode_hit = rel = n_key = 0
    tempo_err, octave_ok = [], 0
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        m = r.get('midicaps')
        if not m:
            continue
        i = r['file']
        if m.get('key'):
            tonic, mode = m['key'].rsplit(' ', 1)
            ref = PITCH_CLASSES.index(tonic.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#')
                                      .replace('Ab', 'G#').replace('Bb', 'A#')) + (12 if mode == 'minor' else 0)
            k = int(z['key'][i])
            n_key += 1
            key_hit += k == ref
            mode_hit += (k >= 12) == (ref >= 12)
            rel += k >= 0 and ((k < 12 and ref == 12 + (k + 9) % 12) or (k >= 12 and ref == (k - 12 + 3) % 12))
        if m.get('tempo') and z['tempo'][i] > 0:
            ratio = z['tempo'][i] / m['tempo']
            tempo_err.append(abs(np.log2(ratio)))
            octave_ok += min(abs(np.log2(ratio) - o) for o in (-1, 0, 1)) < np.log2(1.08)
    e = np.array(tempo_err)
    print(f'{Path(store).name} vs MidiCaps: key exact {key_hit / n_key:.1%}, mode {mode_hit / n_key:.1%}, '
          f'relative major/minor confusions {rel / n_key:.1%} ({n_key} files); tempo within 8% '
          f'{(e < np.log2(1.08)).mean():.1%}, within 8% allowing x2 / x0.5 {octave_ok / len(e):.1%} ({len(e)} files)')


def features(pool, store, n, desc=None):
    """(n, len(FIELDS)) features and (n, 17) program bits of a store's files, in 500-file jobs over the pool."""
    jobs = [(str(store), a, min(a + 500, n)) for a in range(0, n, 500)]
    feats = np.zeros((n, len(FIELDS)))
    bits = np.zeros((n, 17), np.uint8)
    for a, f, b in tqdm(pool.imap_unordered(_work, jobs), total=len(jobs), desc=desc, disable=desc is None):
        feats[a:a + len(f)], bits[a:a + len(f)] = f, b
    return feats, bits


def save(out, feats, bits):
    arrays = {name: feats[:, j].astype(np.int16 if name in ('key',) else np.float32) for j, name in enumerate(FIELDS)}
    arrays['n_notes'] = feats[:, FIELDS.index('n_notes')].astype(np.int32)
    np.savez(out, programs=bits, **arrays)
    print(f'{out}: {len(feats):,} files, median {np.median(arrays["seconds"]):.0f} s, {np.median(arrays["nps"]):.1f} '
          f'notes/s, tempo found {(arrays["tempo"] > 0).mean():.0%}, key found {(arrays["key"] >= 0).mean():.0%}')


class HubStore:
    """A token store in the Hub data repo, read by HTTP range requests: meta.json and offsets.npy downloaded,
    tokens.u16 / programs.u8 fetched per chunk of whole files into a temporary store directory."""

    def __init__(self, name, tmp_dir):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from huggingface_hub import get_token, hf_hub_download, hf_hub_url
        from cloud.sync.hub import api, data_repo
        hf = api()
        self.repo, self.name, self.token, self.url = data_repo(hf), name, get_token(), hf_hub_url
        local = Path(tmp_dir) / 'hub'
        for f in ('meta.json', 'offsets.npy'):
            hf_hub_download(self.repo, f'{name}/{f}', repo_type='dataset', local_dir=local)
        self.offsets = np.load(local / name / 'offsets.npy')
        self.programs = hf.file_exists(self.repo, f'{name}/programs.u8', repo_type='dataset')

    def _get(self, file, a, b, tries=6):
        import requests
        url = self.url(self.repo, f'{self.name}/{file}', repo_type='dataset')
        for attempt in range(tries):
            try:
                r = requests.get(url, headers={'Authorization': f'Bearer {self.token}', 'Range': f'bytes={a}-{b - 1}'},
                                 timeout=600)
                if r.status_code == 206 and len(r.content) == b - a:
                    return r.content
                print(f'{file} bytes {a}-{b}: HTTP {r.status_code}, {len(r.content)} bytes; retry', flush=True)
            except requests.RequestException as e:
                print(f'{file} bytes {a}-{b}: {e}; retry', flush=True)
            time.sleep(2 ** attempt)
        raise RuntimeError(f'{self.name}/{file}: bytes {a}-{b} failed {tries} times')

    def fetch(self, file, a, b, parts=4):
        """Bytes [a, b) of one store file, in `parts` parallel requests."""
        if b <= a:
            return b''
        cuts = np.linspace(a, b, parts + 1).astype(np.int64)
        with ThreadPoolExecutor(parts) as ex:
            return b''.join(ex.map(lambda ab: self._get(file, int(ab[0]), int(ab[1])), zip(cuts[:-1], cuts[1:])))

    def chunks(self, n, chunk_bytes):
        """[(first file, end file)] of whole files with about chunk_bytes of tokens each (8 bytes per note)."""
        out, i = [], 0
        while i < n:
            j = int(np.searchsorted(self.offsets, self.offsets[i] + chunk_bytes // 8, side='right')) - 1
            j = min(max(j, i + 1), n)
            out.append((i, j))
            i = j
        return out

    def stage(self, i, j, directory):
        """Files i..j-1 as a store of their own in `directory` (offsets from 0)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        a, b = int(self.offsets[i]), int(self.offsets[j])
        np.save(directory / 'offsets.npy', self.offsets[i:j + 1] - a)
        (directory / 'tokens.u16').write_bytes(self.fetch('tokens.u16', 8 * a, 8 * b))
        if self.programs:
            (directory / 'programs.u8').write_bytes(self.fetch('programs.u8', a, b))
        return directory


def hub_features(pool, name, args):
    """Features of a Hub store chunk by chunk (parts saved per chunk unless --max-files), assembled in file order."""
    hub = HubStore(name, args.tmp_dir)
    n = len(hub.offsets) - 1 if not args.max_files else min(args.max_files, len(hub.offsets) - 1)
    parts = Path(args.out_dir) / f'{name}.features.parts'
    parts.mkdir(parents=True, exist_ok=True)
    chunks = hub.chunks(n, args.chunk_mb << 20)
    part = lambda i: parts / f'{i:08d}.npz'  # noqa: E731
    todo = [c for c in chunks if args.max_files or not part(c[0]).exists()]
    print(f'{name}: {n:,} files in {len(chunks)} chunks of ~{args.chunk_mb} MB, {len(todo)} to do', flush=True)
    t0, done_bytes = time.time(), 0
    with ThreadPoolExecutor(1) as prefetch:
        nxt = prefetch.submit(hub.stage, *todo[0], Path(args.tmp_dir) / f'chunk_{todo[0][0]}') if todo else None
        for k, (i, j) in enumerate(todo):
            d = nxt.result()
            if k + 1 < len(todo):
                nxt = prefetch.submit(hub.stage, *todo[k + 1], Path(args.tmp_dir) / f'chunk_{todo[k + 1][0]}')
            f, b = features(pool, d, j - i)
            np.savez(part(i), feats=f, bits=b)
            shutil.rmtree(d)
            done_bytes += 8 * int(hub.offsets[j] - hub.offsets[i])
            rate = done_bytes / (time.time() - t0) / 1e6
            print(f'chunk {k + 1}/{len(todo)}: files {i:,}-{j:,}, {rate:.1f} MB/s of tokens, '
                  f'~{(8 * int(hub.offsets[n] - hub.offsets[j]) / 1e6) / max(rate, 1e-9) / 60:.0f} min left', flush=True)
    feats = np.zeros((n, len(FIELDS)))
    bits = np.zeros((n, 17), np.uint8)
    for i, j in chunks:
        z = np.load(part(i))
        feats[i:j], bits[i:j] = z['feats'], z['bits']
    if args.max_files:
        shutil.rmtree(parts)
    return feats, bits


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for store in args.stores.split(','):
            if args.hub:
                feats, bits = hub_features(pool, store, args)
            else:
                n = len(np.load(Path(store) / 'offsets.npy')) - 1
                feats, bits = features(pool, store, n, desc=Path(store).name)
            out = out_dir / f'{Path(store).name}.features.npz'
            save(out, feats, bits)
            if args.check:
                check(out, store, args.out_dir)


if __name__ == '__main__':
    main()
