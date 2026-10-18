"""Validation loss checks (CPU, tiny model): python tests/eval/test_val_loss.py  (or pytest tests/)."""

import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import batch, run, tiny  # noqa: E402
from musicar.eval.val_loss import estimate_val_loss  # noqa: E402


def test_estimate_val_loss():
    m = tiny(n_programs=129).train()
    batches = []
    for seed in (1, 2):
        streams, program, targets = batch(n_programs=129, seed=seed)
        batches.append(((streams, {'program': program}), targets))
    out = estimate_val_loss(m, batches, contextlib.nullcontext())
    assert m.training, 'back in train mode'
    with torch.no_grad():
        runs = [m.eval()(*s, **e, targets=t) for (s, e), t in batches]
    for name in m.head_names:
        assert abs(out[f'ce/{name}'] - sum(p[name].item() for p, _ in runs) / 2) < 1e-5, name
    assert abs(out['loss'] - sum(l.item() for _, l in runs) / 2) < 1e-5
    total = sum(out[f'ce/{k}'] for k in ('pitch', 'velocity', 'duration', 'delta_time'))
    assert abs(out['ce/total'] - total) < 1e-6 and abs(out['ce/total_all'] - total - out['ce/program']) < 1e-6


if __name__ == '__main__':
    run(globals())
