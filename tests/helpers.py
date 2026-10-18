"""Shared pieces of the CPU tests: tiny models, random batches, synthetic token stores."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from musicar.model.gpt import GPT, MusicConfig  # noqa: E402

TINY = dict(n_layer=2, n_head=2, n_embd=32, block_size=64, dropout=0.0)
VOCAB = dict(pitch=128, velocity=32, duration=512, delta_time=512)


def tiny(**kw):
    torch.manual_seed(0)
    return GPT(MusicConfig(**{**TINY, **kw})).eval()


def batch(B=3, T=16, pitch_size=128, n_programs=0, seed=1):
    """(4 input streams, program or None, targets) of random notes."""
    g = torch.Generator().manual_seed(seed)
    streams = [torch.randint(0, n, (B, T), generator=g) for n in (128, 32, 512, 512)]
    program = torch.randint(0, n_programs, (B, T), generator=g) if n_programs else None
    targets = [torch.randint(0, n, (B, T), generator=g) for n in (pitch_size, 32, 512, 512)]
    if n_programs:
        targets.append(torch.randint(0, n_programs, (B, T), generator=g))
    return streams, program, targets


def make_store(path, files, programs=None):
    """A token store at `path` (the dataset reads it as 'store:<prefix>' with path = <prefix>_<split>): files = list of
    (n, 4) token arrays, programs = optional list of (n,) instrument arrays."""
    path.mkdir(parents=True)
    tokens = np.concatenate(files).astype(np.uint16)
    tokens.tofile(path / 'tokens.u16')
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in files])]).astype(np.int64)
    np.save(path / 'offsets.npy', offsets)
    meta = dict(n_files=len(files), n_notes=int(offsets[-1]))
    if programs is not None:
        np.concatenate(programs).astype(np.uint8).tofile(path / 'programs.u8')
        meta['programs'] = True
    (path / 'meta.json').write_text(json.dumps(meta))


def notes(n, seed=0, pitch=(40, 80)):
    """(n, 4) random note tokens with pitches in [pitch[0], pitch[1])."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(*pitch, n), rng.integers(0, 32, n), rng.integers(1, 50, n),
                     rng.integers(0, 20, n)], 1)


def run(namespace):
    """The __main__ runner of a test file: every test_* function in order."""
    for name, fn in list(namespace.items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
