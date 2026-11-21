"""synth_leak: a planted transposed copy of held-out music is found, unrelated music is not (CPU):
python tests/tools/test_synth_leak.py  (or pytest tests/)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/sweeps'), str(ROOT / 'tests')]

import synth_leak as S  # noqa: E402
from helpers import make_store, notes  # noqa: E402


def test_planted_copy_found():
    held = [notes(3000, seed=i) for i in range(3)]
    copy = held[1][1000:2024].copy()
    copy[:, 0] += 5  # transposed
    tok = np.concatenate([notes(200, seed=50), copy, notes(1224, seed=51), notes(1224, seed=52)])
    spans = [(200, 1224), (1224, 2448), (2448, 3672)]
    e = S.embed(tok, spans)
    h_off = np.concatenate([[0], np.cumsum([len(f) for f in held])])
    h = S.embed(np.concatenate(held), S.heldout_spans(h_off, 1024))
    share, best = S.near_share(e, h, 0.95)
    assert best[0] > 0.95 and best[1] < 0.5 and best[2] < 0.5 and abs(share - 1 / 3) < 1e-9


def test_cli_on_stores():
    d = Path(tempfile.mkdtemp())
    held = [notes(3000, seed=i) for i in range(4)]
    make_store(d / 'h_validation', held)
    make_store(d / 'r_train', [notes(1500, seed=100 + i) for i in range(40)])
    syn = [np.concatenate([notes(128, seed=200 + i), notes(1024, seed=300 + i)]) for i in range(9)]
    syn.append(np.concatenate([notes(128, seed=9), held[2][:1024]]))
    make_store(d / 's_train', syn)
    (d / 's_train' / 'files.jsonl').write_text(''.join(json.dumps(dict(prompt_notes=128)) + '\n' for _ in syn))
    argv, sys.argv = sys.argv, ['synth_leak.py', '--synth', str(d / 's_train'), '--heldout', str(d / 'h_validation'),
                                '--real', str(d / 'r_train'), '--real-frac', '1', '--out', str(d / 'out.json')]
    try:
        S.main()
    finally:
        sys.argv = argv
    res = json.loads((d / 'out.json').read_text())
    assert abs(res['synth_share'] - 0.1) < 1e-9 and res['real_share'] == 0.0 and not res['g4c_pass']


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
