"""Sanity checks of the census hashes: python data_prep/census/test_hashes.py (PYTHONPATH=.:data_prep/datasets)."""
import numpy as np

from common import DRUMS, all_hashes

rng = np.random.default_rng(0)
n = 300
tok = np.zeros((n, 4), np.uint16)
tok[:, 0] = rng.integers(40, 80, n)
tok[:, 1] = rng.integers(0, 32, n)
tok[:, 2] = rng.integers(1, 50, n)
tok[:, 3] = rng.integers(0, 20, n)
prog = rng.choice([0, 33, DRUMS], n).astype(np.uint8)

e, l, t = all_hashes(tok, prog)
up = tok.copy(); up[prog != DRUMS, 0] += 5                      # same piece, 5 semitones up, drums untouched
e2, l2, t2 = all_hashes(up, prog)
assert e != e2 and l != l2 and t == t2, 'transposition must change exact/loose but not loose_t'
vel = tok.copy(); vel[:, 1] = 7; vel[:, 2] = 3                  # other velocities/durations
assert all_hashes(vel, prog)[1] == l and all_hashes(vel, prog)[0] != e, 'loose ignores velocity/duration'
p2 = prog.copy(); p2[p2 == 33] = 0                               # other pitched instrument
assert all_hashes(tok, p2)[1] == l, 'loose ignores the pitched program'
d = tok.copy(); d[prog == DRUMS, 0] += 1                         # drums moved: loose_t must see it
assert all_hashes(d, prog)[2] != t, 'drums are not transposed away'
print('hash tests ok')
