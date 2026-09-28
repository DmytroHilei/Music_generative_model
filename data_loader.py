from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MaestroDataset(Dataset):
    """
    Dataset of MIDI files listed in one or more CSVs (columns: split, midi_filename).

    Each note is represented by 4 token streams:
        pitch:      MIDI pitch, 0..127
        velocity:   quantized velocity, 0..velocity_bins-1
        duration:   quantized note duration
        delta_time: quantized time shift from previous note start

    Tokenized notes of all files are stored once in a flat on-disk array (cache_dir/<name>/tokens.u16,
    shape (total_notes, 4) uint16) plus per-file offsets, and memory-mapped, so datasets much larger than RAM work.

    Train mode: every item is a random window, with the start drawn uniformly over all valid positions of all
    files (long files are sampled proportionally more often than short fragments).
    Eval mode (eval_stride): deterministic windows every eval_stride notes of every file.

    Output:
        x = (pitch, velocity, duration, delta_time)
        y = same streams shifted by 1 token
    """

    def __init__(
        self,
        csv_path,
        root_dir,
        split,
        block_size,
        velocity_bins=32,
        time_resolution=0.02,
        max_duration_bin=511,
        max_delta_bin=511,
        debug=False,
        augment=False,
        cache_dir="data/cache",
        eval_stride=None,
    ):
        """
        csv_path:    one or more sources separated by ',': a CSV (paths inside relative to root_dir or absolute),
                     or 'store:<prefix>' = a prebuilt token store <prefix>_<split>/ (e.g. from data/prepare_aria.py)
        eval_stride: if set, the dataset is deterministic (no randomness, no augmentation): use it for validation
        """
        sources = [s.strip() for s in str(csv_path).split(',') if s.strip()]
        self.csv_paths = [Path(s) for s in sources if not s.startswith('store:')]
        prebuilt = [Path(f"{s[len('store:'):]}_{split}") for s in sources if s.startswith('store:')]
        self.root_dir = Path(root_dir)
        self.split = split
        self.block_size = block_size

        self.velocity_bins = velocity_bins
        self.time_resolution = time_resolution
        self.max_duration_bin = max_duration_bin
        self.max_delta_bin = max_delta_bin
        self.debug = debug
        self.augment = augment
        self.cache_dir = Path(cache_dir)
        self.eval_stride = eval_stride

        self.store_dirs = []
        if self.csv_paths:
            self.midi_paths = self._load_midi_paths()
            self.store_dirs.append(self._build_store())
        for store in prebuilt:
            if not (store / "meta.json").exists():
                raise FileNotFoundError(f"Prebuilt token store {store} missing or incomplete")
            meta = json.loads((store / "meta.json").read_text())
            print(f"Loaded tokenized {self.split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes from {store}")
            self.store_dirs.append(store)
        self._tokens = None  # memmaps, opened lazily in each DataLoader worker

        store_ids, starts, lengths = [], [], []
        for i, store in enumerate(self.store_dirs):
            offsets = np.load(store / "offsets.npy")
            n = np.diff(offsets)
            keep = n > self.block_size  # need block_size + 1 notes because targets are shifted by one
            store_ids.append(np.full(int(keep.sum()), i, dtype=np.int32))
            starts.append(offsets[:-1][keep])
            lengths.append(n[keep])
        self.store_ids, self.starts, self.lengths = map(np.concatenate, (store_ids, starts, lengths))
        if len(self.lengths) == 0:
            raise ValueError("No MIDI sequences longer than block_size. Reduce block_size or check the data.")
        self.n_files = len(self.lengths)
        # number of valid window starts per file, cumulative (for uniform sampling over all positions)
        self.cum_valid = np.cumsum(self.lengths - self.block_size)

        if eval_stride:
            # (store, start) pairs; order = sources in the given order, files in store order
            self.windows = np.concatenate([
                np.stack([np.full(len(r), sid), r], axis=1)
                for sid, s, n in zip(self.store_ids, self.starts, self.lengths)
                for r in [np.arange(s, s + n - self.block_size, eval_stride)]
            ])

    # ------------------------------------------------------------------ storage

    def _load_midi_paths(self):
        paths = []
        for csv_path in self.csv_paths:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)
            if "split" not in df.columns or "midi_filename" not in df.columns:
                raise ValueError(f"{csv_path} must contain 'split' and 'midi_filename' columns.")
            df = df[df["split"] == self.split]
            if self.debug:
                df = df.head(5)
            paths += [self.root_dir / f for f in df["midi_filename"]]

        if len(paths) == 0:
            raise ValueError(f"No MIDI files found for split='{self.split}'.")

        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"WARNING [{self.split}]: {len(missing)}/{len(paths)} MIDI files are missing "
                  f"and will be skipped (first: {missing[0]})")
            paths = [p for p in paths if p.exists()]
            if not paths:
                raise FileNotFoundError(f"All MIDI files for split='{self.split}' are missing.")
        return paths

    def _build_store(self):
        key = repr((
            [str(p) for p in self.midi_paths],
            self.velocity_bins, self.time_resolution, self.max_duration_bin, self.max_delta_bin,
        ))
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        name = "+".join(p.stem for p in self.csv_paths)  # unchanged for a single CSV -> old stores still match
        store = self.cache_dir / f"{name}_{self.split}_{digest}"
        if (store / "meta.json").exists():
            meta = json.loads((store / "meta.json").read_text())
            print(f"Loaded tokenized {self.split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes from {store}")
            return store

        store.mkdir(parents=True, exist_ok=True)
        parse = partial(_tokenize_to_array, velocity_bins=self.velocity_bins,
                        time_resolution=self.time_resolution,
                        max_duration_bin=self.max_duration_bin, max_delta_bin=self.max_delta_bin)
        workers = min(len(self.midi_paths), os.cpu_count() or 1)
        offsets, n_notes, n_failed = [0], 0, 0
        # streamed to disk in order, so memory stays flat no matter how large the dataset is
        with open(store / "tokens.u16", "wb") as f, ProcessPoolExecutor(max_workers=workers) as pool:
            for arr in tqdm(pool.map(parse, self.midi_paths, chunksize=16), total=len(self.midi_paths),
                            desc=f"Tokenizing {self.split} ({workers} workers)"):
                if arr is None:
                    n_failed += 1
                else:
                    f.write(arr.tobytes())
                    n_notes += len(arr)
                offsets.append(n_notes)
        np.save(store / "offsets.npy", np.array(offsets, dtype=np.int64))
        meta = dict(n_files=len(self.midi_paths), n_notes=n_notes, n_empty_or_failed=n_failed)
        (store / "meta.json").write_text(json.dumps(meta))  # written last = store is complete
        print(f"Tokenized {self.split}: {n_notes:,} notes, {n_failed} empty/broken files skipped -> {store}")
        return store

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = []
            for store in self.store_dirs:
                n = int(np.load(store / "offsets.npy")[-1])
                self._tokens.append(np.memmap(store / "tokens.u16", dtype=np.uint16, mode="r", shape=(n, 4)))
        return self._tokens

    def __getstate__(self):
        # never pickle the memmap into DataLoader workers, each worker opens its own
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state

    # ------------------------------------------------------------------ sampling

    def num_notes(self):
        return int(self.lengths.sum())

    def __len__(self):
        if self.eval_stride:
            return len(self.windows)
        # one "epoch" = as many windows as fit without overlap; items are random anyway
        return max(1, int(self.lengths.sum()) // self.block_size)

    def __getitem__(self, idx):
        if self.eval_stride:
            store_id, start = (int(v) for v in self.windows[idx])
        else:
            pos = random.randrange(int(self.cum_valid[-1]))
            file_idx = int(np.searchsorted(self.cum_valid, pos, side="right"))
            prev = int(self.cum_valid[file_idx - 1]) if file_idx > 0 else 0
            store_id = int(self.store_ids[file_idx])
            start = int(self.starts[file_idx]) + (pos - prev)

        tokens = self.tokens[store_id]
        window = torch.from_numpy(tokens[start:start + self.block_size + 1].astype(np.int64))
        pitch, velocity, duration, delta_time = window.unbind(1)

        if self.augment and not self.eval_stride:
            # transpose by up to ±5 semitones, limited so no note leaves 0..127 (clamping would bend them)
            lo, hi = int(pitch.min()), int(pitch.max())
            shift = random.randint(max(-5, -lo), min(5, 127 - hi))
            pitch = pitch + shift

        return self._split_xy((pitch, velocity, duration, delta_time))

    def _split_xy(self, tokens):
        x = tuple(t[:-1] for t in tokens)
        y = tuple(t[1:] for t in tokens)
        return x, y


def tokenize_midi(midi_path, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """
    MIDI file -> (pitch, velocity, duration, delta_time) LongTensors, one entry per note, or None if no notes.
    All non-drum instruments are merged, notes sorted by (start, pitch).
    delta_time = quantized onset difference to the previous note, duration = quantized note length.
    """
    # a path, or a file-like object (e.g. BytesIO read straight from a tar archive)
    midi = pretty_midi.PrettyMIDI(str(midi_path) if isinstance(midi_path, (str, Path)) else midi_path)

    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        notes.extend(instrument.notes)

    if len(notes) == 0:
        return None

    notes.sort(key=lambda note: (note.start, note.pitch))

    pitches = []
    velocities = []
    durations = []
    delta_times = []

    prev_start = notes[0].start

    for i, note in enumerate(notes):
        pitches.append(note.pitch)
        velocities.append(quantize_velocity(note.velocity, velocity_bins))
        durations.append(quantize_time(max(0.0, note.end - note.start), time_resolution, max_duration_bin))
        delta_sec = 0.0 if i == 0 else max(0.0, note.start - prev_start)
        delta_times.append(quantize_time(delta_sec, time_resolution, max_delta_bin))
        prev_start = note.start

    return (
        torch.tensor(pitches, dtype=torch.long),
        torch.tensor(velocities, dtype=torch.long),
        torch.tensor(durations, dtype=torch.long),
        torch.tensor(delta_times, dtype=torch.long),
    )


def _tokenize_to_array(midi_path, **kwargs):
    # worker side of the parallel preload: returns (n_notes, 4) uint16 in STREAM order, None if empty/broken.
    # Broken files are skipped instead of failing: large crawled datasets always contain a few.
    try:
        tokens = tokenize_midi(midi_path, **kwargs)
    except Exception:
        return None
    return None if tokens is None else np.stack([t.numpy() for t in tokens], axis=1).astype(np.uint16)


def quantize_time(time_sec, time_resolution, max_bin):
    token = round(time_sec / time_resolution)
    return min(max_bin, max(0, token))


def quantize_velocity(velocity, velocity_bins):
    token = velocity * velocity_bins // 128
    return min(velocity_bins - 1, max(0, token))
