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


BOS_PITCH, EOS_PITCH = 128, 129       # special "notes" when special_tokens=True (pitch vocab 130)
DRUM_PROGRAM = 128                    # programs.u8 value of drum notes (GM programs are 0-127)
DEFAULT_CSV_STYLE = {'combined': 'classical'}   # style of CSV sources without a style/artist column
DEFAULT_STORE_STYLE = 'none'


def load_styles(path='data/styles.json'):
    return json.loads(Path(path).read_text())


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
        x = (pitch, velocity, duration, delta_time) [+ program] [+ style]
        y = (pitch, velocity, duration, delta_time) [+ program], shifted by 1 token
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
        source_weights=None,
        special_tokens=False,
        styles=None,
        style_dropout=0.0,
        boundary_frac=0.0,
        aug_tempo=0.0,
        aug_velocity=0,
        aug_stores=None,
        programs=False,
    ):
        """
        csv_path:    one or more sources separated by ',': a CSV (paths inside relative to root_dir or absolute),
                     or 'store:<prefix>' = a prebuilt token store <prefix>_<split>/ (e.g. from data/prepare_aria.py)
        eval_stride: if set, the dataset is deterministic (no randomness, no augmentation): use it for validation
        source_weights: optional sampling weight per store (the CSV sources form one store, first; then each
                     'store:' source in order). Default: uniform over all note positions, i.e. by size. Use it to mix a
                     small fine-tune set with replay, e.g. [0.8, 0.2]
        special_tokens: every file is read as [BOS] + notes + [EOS] (pitch BOS_PITCH / EOS_PITCH, other attributes 0),
                     without touching the token stores
        styles:      a style vocabulary (list of names, index 0 = 'none'): each window then also returns the style id of
                     its file (CSV column 'style', else 'artist', else DEFAULT_CSV_STYLE; store metadata 'genre').
                     style_dropout: probability (training only) of replacing it with 0 = unconditional
        boundary_frac: training only, with special_tokens: this fraction of windows is placed exactly at a piece's
                     start (half) or end (half), so openings and EOS are seen often (otherwise ~0.2% of windows)
        aug_tempo:   training only: stretch time (duration and delta_time) by a log-uniform factor in
                     [1/(1+aug_tempo), 1+aug_tempo] per window, with stochastic rounding (0 = off)
        aug_velocity: training only: shift all velocity bins of a window by up to ±aug_velocity, clamped to the
                     bin range (0 = off)
        aug_stores:  optional 0/1 per store (same order as source_weights): which stores get the tempo/velocity
                     augmentation (e.g. [1, 0] = only the fine-tune CSVs, the replay store stays real). Default: all.
                     Transposition applies to every store regardless
        programs:    multi-instrument: also return the instrument stream (x and y), from the store's programs.u8
                     (data/prepare_gigamidi.py); stores without one are piano = program 0. Drum notes are never
                     transposed and their duration is set to 0 in x and masked (-1) in y. False = piano-only: stores
                     with programs.u8 are refused, since their drum notes would be read as pitched notes
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
        self.aug_tempo = aug_tempo
        self.aug_velocity = aug_velocity
        self.aug_stores = aug_stores
        self.cache_dir = Path(cache_dir)
        self.eval_stride = eval_stride
        self.programs = programs

        self.pad = 1 if special_tokens else 0
        self.style_dropout = style_dropout
        self.boundary_frac = boundary_frac if special_tokens else 0.0
        self.style_ids = {name: i for i, name in enumerate(styles)} if styles else None
        self.store_dirs, file_styles = [], []
        if self.csv_paths:
            self.midi_paths = self._load_midi_paths()
            file_styles.append(self.midi_styles)
            self.store_dirs.append(self._build_store())
        for store in prebuilt:
            if not (store / "meta.json").exists():
                raise FileNotFoundError(f"Prebuilt token store {store} missing or incomplete")
            meta = json.loads((store / "meta.json").read_text())
            print(f"Loaded tokenized {self.split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes from {store}")
            self.store_dirs.append(store)
            file_styles.append(self._store_styles(store, meta['n_files']))
        self._tokens = None  # memmaps, opened lazily in each DataLoader worker
        self._programs = None

        store_ids, starts, lengths, styles_kept = [], [], [], []
        for i, store in enumerate(self.store_dirs):
            offsets = np.load(store / "offsets.npy")
            n = np.diff(offsets) + 2 * self.pad  # virtual length incl. BOS/EOS
            keep = n > self.block_size  # need block_size + 1 notes because targets are shifted by one
            store_ids.append(np.full(int(keep.sum()), i, dtype=np.int32))
            starts.append(offsets[:-1][keep])
            lengths.append(n[keep])
            if self.style_ids is not None:
                ids = np.array([self.style_ids.get(x, 0) for x in file_styles[i]], dtype=np.int64)
                assert len(ids) == len(n), f"{store}: {len(ids)} style entries for {len(n)} files"
                styles_kept.append(ids[keep])
        self.store_ids, self.starts, self.lengths = map(np.concatenate, (store_ids, starts, lengths))
        self.file_style = np.concatenate(styles_kept) if self.style_ids is not None else None
        if len(self.lengths) == 0:
            raise ValueError("No MIDI sequences longer than block_size. Reduce block_size or check the data.")
        self.n_files = len(self.lengths)
        self.source_weights = None
        if source_weights is not None and not eval_stride:
            assert len(source_weights) == len(self.store_dirs), \
                f"{len(source_weights)} source_weights for {len(self.store_dirs)} stores"
            ids = np.arange(len(self.store_dirs))
            # files of one store are contiguous (stores are concatenated in order)
            self.store_file_range = [(int(np.searchsorted(self.store_ids, i, 'left')),
                                      int(np.searchsorted(self.store_ids, i, 'right'))) for i in ids]
            self.source_weights = [float(w) for w in source_weights]
        if aug_stores is not None:
            assert len(aug_stores) == len(self.store_dirs), f"{len(aug_stores)} aug_stores for {len(self.store_dirs)} stores"
        # number of valid window starts per file, cumulative (for uniform sampling over all positions)
        self.cum_valid = np.cumsum(self.lengths - self.block_size)

        if eval_stride:
            # (file index, offset in the file's virtual sequence); order = sources in order, files in store order
            self.windows = np.concatenate([
                np.stack([np.full(len(r), fi), r], axis=1)
                for fi, n in enumerate(self.lengths)
                for r in [np.arange(0, n - self.block_size, eval_stride)]
            ])

    # ------------------------------------------------------------------ storage

    def _store_styles(self, store, n_files):
        """Per-file style names of a prebuilt store: 'genre' of the <split> rows of data/aria/<name>.csv."""
        if self.style_ids is None:
            return None
        name = store.name[:-len(f"_{self.split}")]
        meta_csv = Path("data/aria") / f"{name}.csv"
        if not meta_csv.exists():
            return [DEFAULT_STORE_STYLE] * n_files
        df = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
        return list(df.loc[df["split"] == self.split, "genre"])

    def _load_midi_paths(self):
        paths, styles = [], []
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
            col = "style" if "style" in df.columns else "artist" if "artist" in df.columns else None
            styles += list(df[col]) if col else [DEFAULT_CSV_STYLE.get(csv_path.stem, "none")] * len(df)

        if len(paths) == 0:
            raise ValueError(f"No MIDI files found for split='{self.split}'.")

        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"WARNING [{self.split}]: {len(missing)}/{len(paths)} MIDI files are missing "
                  f"and will be skipped (first: {missing[0]})")
            styles = [st for p, st in zip(paths, styles) if p.exists()]
            paths = [p for p in paths if p.exists()]
            if not paths:
                raise FileNotFoundError(f"All MIDI files for split='{self.split}' are missing.")
        self.midi_styles = styles
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
            self._tokens, self._programs = [], []
            for store in self.store_dirs:
                has_programs = (store / "programs.u8").exists()
                # multi-instrument stores (data/prepare_gigamidi.py) keep drums in the stream: not for the piano model
                assert self.programs or not has_programs, f"{store} has instrument programs: pass programs=True"
                n = int(np.load(store / "offsets.npy")[-1])
                self._tokens.append(np.memmap(store / "tokens.u16", dtype=np.uint16, mode="r", shape=(n, 4)))
                self._programs.append(np.memmap(store / "programs.u8", dtype=np.uint8, mode="r", shape=(n,))
                                      if has_programs else None)  # None = piano, program 0
        return self._tokens

    def __getstate__(self):
        # never pickle the memmap into DataLoader workers, each worker opens its own
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_programs"] = None
        return state

    # ------------------------------------------------------------------ sampling

    def num_notes(self):
        return int(self.lengths.sum()) - 2 * self.pad * len(self.lengths)

    def __len__(self):
        if self.eval_stride:
            return len(self.windows)
        # one "epoch" = as many windows as fit without overlap; items are random anyway
        return max(1, int(self.lengths.sum()) // self.block_size)

    def _read(self, file_idx, v):
        """block_size + 1 rows of the file's virtual sequence ([BOS] + notes + [EOS] with special tokens) from v:
        (rows, 4) tokens and, with programs, (rows,) instruments (BOS/EOS and piano stores = 0), else None."""
        store_id, start = int(self.store_ids[file_idx]), int(self.starts[file_idx])
        n = int(self.lengths[file_idx]) - 2 * self.pad
        tokens = self.tokens[store_id]
        lo, hi = v - self.pad, v - self.pad + self.block_size + 1  # real note range
        a, b = start + max(lo, 0), start + min(hi, n)
        rows = tokens[a:b].astype(np.int64)
        programs = None
        if self.programs:
            store_programs = self._programs[store_id]
            programs = store_programs[a:b].astype(np.int64) if store_programs is not None else np.zeros(b - a, np.int64)
        if lo < 0:
            rows = np.concatenate([np.array([[BOS_PITCH, 0, 0, 0]]), rows])
            programs = None if programs is None else np.concatenate([[0], programs])
        if hi > n:
            rows = np.concatenate([rows, np.array([[EOS_PITCH, 0, 0, 0]])])
            programs = None if programs is None else np.concatenate([programs, [0]])
        return torch.from_numpy(rows), None if programs is None else torch.from_numpy(programs)

    def __getitem__(self, idx):
        if self.eval_stride:
            file_idx, v = (int(x) for x in self.windows[idx])
        else:
            if self.source_weights is None:
                pos = random.randrange(int(self.cum_valid[-1]))
            else:
                sid = random.choices(range(len(self.source_weights)), weights=self.source_weights)[0]
                lo, hi = self.store_file_range[sid]
                base = int(self.cum_valid[lo - 1]) if lo > 0 else 0
                pos = base + random.randrange(int(self.cum_valid[hi - 1]) - base)
            file_idx = int(np.searchsorted(self.cum_valid, pos, side="right"))
            prev = int(self.cum_valid[file_idx - 1]) if file_idx > 0 else 0
            v = pos - prev
            if self.boundary_frac:
                r = random.random()
                if r < self.boundary_frac / 2:
                    v = 0                                                   # starts with BOS
                elif r < self.boundary_frac:
                    v = int(self.lengths[file_idx]) - self.block_size - 1  # ends with EOS

        rows, program = self._read(file_idx, v)
        pitch, velocity, duration, delta_time = rows.unbind(1)
        drums = program == DRUM_PROGRAM if program is not None else None
        if drums is not None:
            duration = torch.where(drums, 0, duration)  # drum hits have no meaningful length

        if self.augment and not self.eval_stride:
            # transpose by up to ±5 semitones, limited so no note leaves 0..127 (clamping would bend them);
            # BOS/EOS (pitch >= 128) and drums (pitch = drum sound) stay as they are
            notes = pitch < 128
            pitched = notes if drums is None else notes & ~drums
            if pitched.any():
                lo, hi = int(pitch[pitched].min()), int(pitch[pitched].max())
                shift = random.randint(max(-5, -lo), min(5, 127 - hi))
                pitch = torch.where(pitched, pitch + shift, pitch)
            store_augmented = self.aug_stores is None or bool(self.aug_stores[int(self.store_ids[file_idx])])
            if self.aug_tempo and store_augmented:
                s = (1 + self.aug_tempo) ** random.uniform(-1, 1)
                duration = self._stretch(duration, s, self.max_duration_bin)
                delta_time = self._stretch(delta_time, s, self.max_delta_bin)
            if self.aug_velocity and store_augmented:
                # symmetric shift, clamped: a no-clip bound would bias it downwards (-0.9 bins on the reductions), since
                # most windows hold a few melody notes saturated at the top bin (the melody boost clips at 127)
                shift = random.randint(-self.aug_velocity, self.aug_velocity)
                velocity = torch.where(notes, (velocity + shift).clamp(0, self.velocity_bins - 1), velocity)

        streams = (pitch, velocity, duration, delta_time) + ((program,) if program is not None else ())
        x, y = self._split_xy(streams)
        if drums is not None:
            y = (y[0], y[1], torch.where(drums[1:], -1, y[2])) + y[3:]  # no duration loss on drum notes
        if self.file_style is not None:
            style = int(self.file_style[file_idx])
            if self.augment and not self.eval_stride and random.random() < self.style_dropout:
                style = 0
            x = x + (torch.tensor(style),)
        return x, y

    @staticmethod
    def _stretch(bins, s, max_bin):
        # stochastic rounding keeps the stretched bins unbiased and avoids a lattice of empty bins; 0 (chords) stays 0
        scaled = bins.double() * s
        low = scaled.floor()
        out = low + (torch.rand_like(scaled) < scaled - low).double()
        return out.long().clamp(max=max_bin)

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
