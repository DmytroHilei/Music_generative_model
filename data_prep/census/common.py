"""
Shared pieces of the data census (data_prep/census/): note hashes, store access, paths.

Hash levels (first copy wins, in SOURCES order):
    md5      file bytes (from the file name for GigaMIDI / Discover, computed for the rest)
    exact    tokens + programs                          (prepare_discover.note_hashes)
    loose    sorted (onset, pitch), drums kept apart    (prepare_discover.note_hashes)
    loose_t  loose after moving the lowest pitched (non-drum) note to 0: the same piece in another key

Piano stores (no programs.u8) are hashed with programs = 0, as CLAUDE.md asks.
Run from the repo root with PYTHONPATH=.:data_prep/datasets (prepare_discover imports prepare_gigamidi by name).
"""

import json
from pathlib import Path

import numpy as np

from prepare_discover import DRUMS, h64, note_hashes  # noqa: F401  (re-exported)

CACHE = Path('data/cache')
CENSUS = Path('census')
HASHES = CENSUS / 'hashes'
MANIFESTS = CENSUS / 'manifests'

# processing order = deduplication order (first copy wins)
SOURCES = ['gigamidi', 'discover', 'aria', 'pianocore', 'pijama', 'piast', 'pdmx', 'beat']
LEVELS = ['md5', 'exact', 'loose', 'loose_t']


def loose_t_hash(tokens, programs):
    """loose hash after transposing the pitched notes so the lowest is 0 (drums untouched, kept apart)."""
    tokens = np.ascontiguousarray(tokens, dtype=np.uint16)
    programs = np.ascontiguousarray(programs, dtype=np.uint8)
    onset = np.cumsum(tokens[:, 3], dtype=np.int64)
    pitch = tokens[:, 0].astype(np.int64)
    drums = programs == DRUMS
    if (~drums).any():
        pitch = np.where(drums, pitch, pitch - pitch[~drums].min())
    pitch = pitch + DRUMS * drums
    order = np.lexsort((pitch, onset))
    return h64(np.stack([onset[order], pitch[order]]).tobytes())


def all_hashes(tokens, programs):
    """(exact, loose, loose_t) of one file."""
    exact, loose = note_hashes(tokens, programs)
    return exact, loose, loose_t_hash(tokens, programs)


class Store:
    """Read-only view of a token store dir (tokens.u16, offsets.npy, optional programs.u8, meta.json)."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'meta.json').read_text())
        self.offsets = np.load(self.path / 'offsets.npy')
        n = int(self.offsets[-1])
        assert n == self.meta['n_notes'] and len(self.offsets) - 1 == self.meta['n_files'], f'{path}: meta mismatch'
        self.tokens = np.memmap(self.path / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
        p = self.path / 'programs.u8'
        self.programs = np.memmap(p, dtype=np.uint8, mode='r', shape=(n,)) if p.exists() else None

    def __len__(self):
        return len(self.offsets) - 1

    def file(self, i):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        prog = self.programs[a:b] if self.programs is not None else np.zeros(b - a, np.uint8)
        return self.tokens[a:b], prog
