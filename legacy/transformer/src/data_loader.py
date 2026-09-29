from pathlib import Path
import random

import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset


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
    ):
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

        self.midi_paths = self._load_midi_paths()

        if preload:
            self.data = self._preload_data()
        else:
            self.data = None

    def __len__(self):
        if self.preload:
            return len(self.data)
        return len(self.midi_paths)

    def __getitem__(self, idx):
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
            raise FileNotFoundError(f"Missing MIDI file: {missing[0]}")

        return midi_paths

    def _preload_data(self):
        data = []

        for midi_path in self.midi_paths:
            tokens = self._load_tokens_from_path(midi_path)

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
            pitch_token = note.pitch
            velocity_token = self._quantize_velocity(note.velocity)

            duration_sec = max(0.0, note.end - note.start)
            duration_token = self._quantize_time(
                duration_sec,
                max_bin=self.max_duration_bin,
            )

            if i == 0:
                delta_sec = 0.0
            else:
                delta_sec = max(0.0, note.start - prev_start)

            delta_token = self._quantize_time(
                delta_sec,
                max_bin=self.max_delta_bin,
            )

            pitches.append(pitch_token)
            velocities.append(velocity_token)
            durations.append(duration_token)
            delta_times.append(delta_token)

            prev_start = note.start

        return (
            torch.tensor(pitches, dtype=torch.long),
            torch.tensor(velocities, dtype=torch.long),
            torch.tensor(durations, dtype=torch.long),
            torch.tensor(delta_times, dtype=torch.long),
        )

    def _quantize_time(self, time_sec, max_bin):
        token = round(time_sec / self.time_resolution)
        token = max(0, token)
        token = min(max_bin, token)
        return token

    def _quantize_velocity(self, velocity):
        token = velocity * self.velocity_bins // 128
        token = max(0, token)
        token = min(self.velocity_bins - 1, token)
        return token

