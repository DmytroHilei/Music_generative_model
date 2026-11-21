"""discover_sets' per-file stats and k choice, and scale_ladder.parse's extra CE sets (CPU):
python tests/data/test_discover_sets.py  (or pytest tests/)."""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'data_prep/census'), str(ROOT / 'tools/sweeps')]

import discover_sets as D  # noqa: E402
from scale_ladder import parse  # noqa: E402

OLD = ('step 4882: train loss 2.3011, val loss 2.5012 | gigamidi_heldout CE 2.426 (pit 0.811 vel 0.512 dur 0.701 '
       'del 0.402) | prog CE 0.091\n')
PANEL = ('step 4882: train loss 2.3011, val loss 2.5012 | cdisc_vn CE 2.601 (pit 0.9 vel 0.5 dur 0.7 del 0.5) | prog CE '
         '0.090 | nocond CE 2.700 | disc_aria CE 2.111 | disc_dense CE 1.234 | giga CE 2.430\n')


def test_parse_keeps_extras():
    d = Path(tempfile.mkdtemp())
    (d / 'old.log').write_text('Model parameters : 43,060,992 (43.06M)\n' + OLD)
    (d / 'panel.log').write_text(PANEL)
    (old,), params = parse(d / 'old.log')
    heads = {'pit': 0.811, 'vel': 0.512, 'dur': 0.701, 'del': 0.402}
    assert params == 43_060_992 and old == dict(step=4882, ce=2.426, heads=heads, prog=0.091)  # as before
    (new,), _ = parse(d / 'panel.log')
    assert new['ce'] == 2.601 and new['prog'] == 0.09
    assert new['extras'] == {'nocond': 2.7, 'disc_aria': 2.111, 'disc_dense': 1.234, 'giga': 2.43}


def test_file_stats():
    # file 0: 3 notes on one onset then one 1 s later; file 1: 70 notes on one onset (dense by the onset rule)
    f0 = [(60, 10, 5, 0), (64, 10, 5, 0), (67, 10, 5, 0), (60, 10, 5, 50)]
    f1 = [(40 + i % 50, 10, 5, 7 if i == 0 else 0) for i in range(70)]
    tok = np.array(f0 + f1, np.uint16)
    span, mx = D.file_stats(tok, np.array([0, 4, 74]))
    assert np.allclose(span, [1.0, 0.0]) and list(mx) == [3, 70]


def test_smallest_k():
    c = np.array([0, 5, 5, 9, 9999])
    notes = np.array([10, 20, 30, 40, 50])
    assert D.smallest_k(c, notes, 10) == 1 and D.smallest_k(c, notes, 11) == 6 and D.smallest_k(c, notes, 100) == 10


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
