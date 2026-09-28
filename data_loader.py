from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import hashlib
import os
import pickle
import random

import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MaestroDataset(Dataset):
    """
    Dataset for MAESTRO MIDI files.

    Each note is represented by 4 token streams:
        pitch:      MIDI pitch, 0..127
        velocity:   quantized velocity, 0..velocity_bins-1
        duration:   quantized note duration
        delta_time: quantized time shift from previous note start

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
        preload=True,
        augment=False,
        cache_dir="data/cache",
        eval_stride=None,
    ):
        """
        cache_dir:   tokenized files are pickled here (keyed by file list + tokenizer params), None = no cache
        eval_stride: if set, the dataset becomes deterministic: items are all windows of block_size+1 notes
                     taken every eval_stride notes from every file (no randomness, no augmentation).
                     Use it for validation so val loss is comparable between runs.
        """
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.split = split
        self.block_size = block_size

        self.velocity_bins = velocity_bins
        self.time_resolution = time_resolution
        self.max_duration_bin = max_duration_bin
        self.max_delta_bin = max_delta_bin
        self.preload = preload
        self.debug = debug
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.eval_stride = eval_stride

        self.midi_paths = self._load_midi_paths()

        if preload or eval_stride:
            self.data = self._preload_data()
        else:
            self.data = None

        if eval_stride:
            self.windows = [
                (i, start)
                for i, tokens in enumerate(self.data)
                for start in range(0, len(tokens[0]) - self.block_size, eval_stride)
            ]

    def num_notes(self):
        return sum(len(t[0]) for t in self.data) if self.data is not None else None

    def __len__(self):
        if self.eval_stride:
            return len(self.windows)
        if self.preload:
            return len(self.data)
        return len(self.midi_paths)

    def __getitem__(self, idx):
        if self.eval_stride:
            i, start = self.windows[idx]
            return self._split_xy(tuple(t[start:start + self.block_size + 1] for t in self.data[i]))
        if self.preload:
            tokens = self.data[idx]
        else:
            tokens = self._load_tokens_from_path(self.midi_paths[idx])

        return self._sample_block(tokens)

    def _load_midi_paths(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        debug = self.debug

        df = pd.read_csv(self.csv_path)

        if "split" not in df.columns:
            raise ValueError("CSV must contain a 'split' column.")

        if "midi_filename" not in df.columns:
            raise ValueError("CSV must contain a 'midi_filename' column.")

        df = df[df["split"] == self.split].reset_index(drop=True)

        if debug:
            df = df.head(5)

        if len(df) == 0:
            raise ValueError(f"No MIDI files found for split='{self.split}'.")

        midi_paths = [
            self.root_dir / filename
            for filename in df["midi_filename"]
        ]

        missing = [path for path in midi_paths if not path.exists()]
        if missing:
            print(f"WARNING [{self.split}]: {len(missing)}/{len(midi_paths)} MIDI files are missing "
                  f"and will be skipped (first: {missing[0]})")
            midi_paths = [path for path in midi_paths if path.exists()]
            if not midi_paths:
                raise FileNotFoundError(f"All MIDI files for split='{self.split}' are missing.")

        return midi_paths

    def _cache_path(self):
        key = repr((
            [str(p) for p in self.midi_paths],
            self.velocity_bins, self.time_resolution, self.max_duration_bin, self.max_delta_bin,
        ))
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{self.csv_path.stem}_{self.split}_{digest}.pkl"

    def _preload_data(self):
        cache_path = self._cache_path() if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            with open(cache_path, "rb") as f:
                all_tokens = pickle.load(f)
            print(f"Loaded {len(all_tokens)} tokenized files from {cache_path}")
        else:
            parse = partial(_tokenize_to_numpy, velocity_bins=self.velocity_bins,
                            time_resolution=self.time_resolution,
                            max_duration_bin=self.max_duration_bin, max_delta_bin=self.max_delta_bin)
            workers = min(len(self.midi_paths), os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(tqdm(pool.map(parse, self.midi_paths, chunksize=8), total=len(self.midi_paths),
                                    desc=f"Tokenizing {self.split} ({workers} workers)"))
            all_tokens = [None if r is None else tuple(torch.from_numpy(a) for a in r) for r in results]
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(all_tokens, f)
                print(f"Cached {len(all_tokens)} tokenized files to {cache_path}")

        data = []

        for tokens in all_tokens:
            if tokens is None:
                continue

            pitch, velocity, duration, delta_time = tokens

            # Need block_size + 1 because targets are shifted by one token
            if len(pitch) > self.block_size:
                data.append(tokens)

        if len(data) == 0:
            raise ValueError(
                "No valid MIDI sequences found. "
                "Try reducing block_size or checking MIDI parsing."
            )

        return data

    def _load_tokens_from_path(self, midi_path):
        try:
            return self._parse_midi_delta_time(midi_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse MIDI file: {midi_path}") from e

    def _sample_block(self, tokens):
        pitch, velocity, duration, delta_time = tokens

        if len(pitch) <= self.block_size:
            raise ValueError(
                f"Sequence too short: length={len(pitch)}, block_size={self.block_size}"
            )

        start = random.randint(0, len(pitch) - self.block_size - 1)
        end = start + self.block_size + 1

        pitch = pitch[start:end]
        velocity = velocity[start:end]
        duration = duration[start:end]
        delta_time = delta_time[start:end]

        if self.augment:
            shift = random.randint(-5, 5)
            pitch = torch.clamp(pitch + shift, 0, 127)

        return self._split_xy((pitch, velocity, duration, delta_time))

    def _split_xy(self, tokens):
        pitch, velocity, duration, delta_time = tokens
        x = (
            pitch[:-1],
            velocity[:-1],
            duration[:-1],
            delta_time[:-1],
        )

        y = (
            pitch[1:],
            velocity[1:],
            duration[1:],
            delta_time[1:],
        )

        return x, y

    def _parse_midi_delta_time(self, midi_path):
        return tokenize_midi(midi_path, self.velocity_bins, self.time_resolution,
                             self.max_duration_bin, self.max_delta_bin)


def tokenize_midi(midi_path, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """
    MIDI file -> (pitch, velocity, duration, delta_time) LongTensors, one entry per note, or None if no notes.
    All non-drum instruments are merged, notes sorted by (start, pitch).
    delta_time = quantized onset difference to the previous note, duration = quantized note length.
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

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


def _tokenize_to_numpy(midi_path, **kwargs):
    # worker side of the parallel preload: numpy arrays pickle cheaply between processes, tensors don't
    try:
        tokens = tokenize_midi(midi_path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to parse MIDI file: {midi_path}") from e
    return None if tokens is None else tuple(t.numpy() for t in tokens)


def quantize_time(time_sec, time_resolution, max_bin):
    token = round(time_sec / time_resolution)
    return min(max_bin, max(0, token))


def quantize_velocity(velocity, velocity_bins):
    token = velocity * velocity_bins // 128
    return min(velocity_bins - 1, max(0, token))
