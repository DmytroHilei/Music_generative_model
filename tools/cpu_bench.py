"""
CPU inference benchmark (baseline for a Raspberry Pi 5 estimate; no decode optimisation): fp32, batch 1, eager
decoding with the KV cache, a 128-note prompt and a cache sized for a full 2,048-note window (a long song: every
one-note step attends over 2,048 slots). Per model and thread count: time to the first new note (prefill of the
prompt + one step) and decode notes/s over --notes notes after a short warm-up. Also measures the memory bandwidth
(tensor copy), since one-note decoding at batch 1 reads every weight once per note and is mostly bandwidth-bound.

    .venv/bin/python tools/cpu_bench.py                                  # 108M + 450M, 1 and 4 threads
    .venv/bin/python tools/cpu_bench.py --checkpoints checkpoints/ua_2048_v2 --threads 4 8
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from musicar.checkpoint import load_model
from musicar.data_loader import BOS_PITCH, tokenize_midi
from musicar.sampling import generate, with_bos


class Stop(Exception):
    pass


def bandwidth(threads, mb=512, reps=5):
    """GB/s of a tensor copy (read + write counted)."""
    torch.set_num_threads(threads)
    a = torch.ones(mb * 2**20 // 4)
    b = torch.empty_like(a)
    b.copy_(a)
    t = time.perf_counter()
    for _ in range(reps):
        b.copy_(a)
    return 2 * a.numel() * 4 * reps / (time.perf_counter() - t) / 1e9


def prompt(n=128):
    """The first n notes of the first MAESTRO / GiantMIDI validation file with enough notes (piano, program 0); without
    the dataset (e.g. on a Raspberry Pi) a fixed random prompt: the timing doesn't depend on which notes they are."""
    try:
        df = pd.read_csv('data/combined.csv')
    except FileNotFoundError:
        g = torch.Generator().manual_seed(0)
        return torch.stack([torch.randint(40, 90, (n,), generator=g), torch.randint(8, 24, (n,), generator=g),
                            torch.randint(5, 40, (n,), generator=g), torch.randint(0, 15, (n,), generator=g)], 1)[None]
    for f in df.loc[df['split'] == 'validation', 'midi_filename']:
        tokens = tokenize_midi(f)
        if tokens is not None and len(tokens) >= n:
            return torch.from_numpy(tokens[:n].astype(np.int64))[None]


def run(model, tokens, threads, notes, warmup=8, window=2048):
    torch.set_num_threads(threads)
    streams = tuple(tokens[:, :, j] for j in range(4))
    programs = torch.zeros_like(streams[0])
    if model.config.pitch_size > BOS_PITCH:
        streams, programs = with_bos(streams, programs)
    times = []

    def progress(n):
        times.append(time.perf_counter())
        if n == warmup + notes:
            raise Stop

    torch.manual_seed(1)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            generate(model, *streams, max_new_tokens=window - streams[0].size(1), temperature=0.9, cuda_graph=False,
                     program=None if model.config.n_programs else 0,
                     programs=programs if model.config.n_programs else None, progress=progress)
    except Stop:
        pass
    first = times[0] - t0
    rate = notes / (times[warmup + notes - 1] - times[warmup - 1])
    return first, rate


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoints', nargs='+', default=['checkpoints/ua_2048_v2', 'checkpoints/long_450m_final'])
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--notes', type=int, default=48, help='decoded notes timed after the warm-up')
    args = ap.parse_args()
    print(f'torch {torch.__version__}, {torch.get_num_threads()} threads available')
    for t in args.threads:
        print(f'memory bandwidth (copy), {t} thread(s): {bandwidth(t):.1f} GB/s')
    tokens = prompt()
    print(f'{"checkpoint":32s}{"params":>9s}{"fp32 GB":>9s}{"threads":>9s}{"first note s":>14s}{"notes/s":>9s}'
          f'{"GB/s read":>11s}')
    for path in args.checkpoints:
        model, _ = load_model(path, 'cpu', torch.float32)
        n = sum(p.numel() for p in model.parameters())
        for t in args.threads:
            first, rate = run(model, tokens, t, args.notes)
            # weights read per note (the KV cache reads are extra, ~0.1-0.3 GB at a 2,048-note window)
            print(f'{path:32s}{n / 1e6:8.0f}M{n * 4 / 1e9:9.2f}{t:9d}{first:14.2f}{rate:9.1f}{rate * n * 4 / 1e9:11.1f}',
                  flush=True)
        del model


if __name__ == '__main__':
    main()
