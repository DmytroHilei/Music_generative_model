"""musicar/generation/repetition.py checks (CPU): python tests/generation/test_repetition.py  (or tests/run.py)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run, tiny  # noqa: E402
from musicar.data.loader import DRUM_PROGRAM  # noqa: E402
from musicar.generation.repetition import RepeatPenalty, chords, copies, copy_share  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402


def code(p, q, key):
    return p % 12 if key == 'pitch_class' else q * 128 + p


def brute_penalized(seq, dt_c, prog_c, allowed, key):
    """Pitches the penalty must lower for a note (dt_c, prog_c) after seq [(pitch, dt, program)], from the
    definition: chords = sets of codes per onset with their gap to the previous chord."""
    onset, groups = 0, {}
    for p, d, q in seq:
        onset += d
        if q != DRUM_PROGRAM:
            groups.setdefault(onset, set()).add(code(p, q, key))
    times = list(groups)
    chord = [(min(t - (times[k - 1] if k else 0), 511), frozenset(groups[t])) for k, t in enumerate(times)]
    t_c = onset + dt_c
    joins = bool(times) and t_c == times[-1]
    if joins:  # the open chord is the last one: copies run over the chords before it
        passage, played, gap = chord[:-1], chord[-1][1], chord[-1][0]
    else:
        passage, played, gap = chord, frozenset(), min(t_c - (times[-1] if times else 0), 511)
    out = set()
    n = len(passage)
    for j in range(1, n if joins else n):
        m = 0
        while m < j and passage[j - 1 - m] == passage[n - 1 - m]:
            m += 1
        g, codes = passage[j]
        if m >= allowed and g == gap and played <= codes and prog_c != DRUM_PROGRAM:
            for c in codes - played:
                if key == 'pitch_class':
                    out |= {p for p in range(128) if p % 12 == c}
                elif c // 128 == prog_c:
                    out.add(c % 128)
    return out


def random_song(rng):
    """A loop of chords (random note order within each onset, drums in between) played a few times, cut anywhere."""
    loop = []
    for _ in range(rng.integers(3, 10)):
        notes = [(int(rng.integers(40, 80)), int(rng.choice([0, 33]))) for _ in range(rng.integers(1, 4))]
        loop.append((int(rng.integers(1, 4)), notes))
    seq = []
    for _ in range(rng.integers(2, 5)):
        for gap, notes in loop:
            order = rng.permutation(len(notes))
            seq.append((notes[order[0]][0], gap, notes[order[0]][1]))
            seq += [(notes[k][0], 0, notes[k][1]) for k in order[1:]]
            if rng.random() < 0.3:
                seq.append((36, 0, DRUM_PROGRAM))
    return seq[:rng.integers(len(seq) // 2, len(seq) + 1)]


def test_penalty_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(60):
        seq = random_song(rng)
        key = 'pitch_class' if trial % 3 == 0 else 'notes'
        allowed = int(rng.integers(1, 6))
        pitch, dt, prog = (torch.tensor([[x[k] for x in seq]]) for k in range(3))
        rp = RepeatPenalty(1, 400, 1.0, 2.0, allowed, key=key)
        rp.prime(pitch, dt, prog)
        for dt_c in range(4):
            for prog_c in (0, 33, DRUM_PROGRAM):
                rp.dt.fill_(dt_c)
                rp.prog.fill_(prog_c)
                got = set((rp.penalize(torch.zeros(1, 130)) < 0)[0].nonzero().flatten().tolist())
                assert got == brute_penalized(seq, dt_c, prog_c, allowed, key), (trial, key, dt_c, prog_c)


def test_penalty_grows_with_the_copy():
    loop = [(40 + k, 1, 0) for k in range(20)]  # 20 one-note chords, then the same again: 19 copied
    seq = loop + loop[:19]
    pitch, dt, prog = (torch.tensor([[x[k] for x in seq]]) for k in range(3))
    rp = RepeatPenalty(1, 100, 1.0, 2.0, allowed=16)
    rp.prime(pitch, dt, prog)
    rp.dt.fill_(1)
    pen = -rp.penalize(torch.zeros(1, 128))[0]
    assert pen[59] == 2.0 ** (19 - 16) and (pen > 0).sum() == 1, 'only the copy continuation, 1 x 2^3'
    rp.dt.fill_(2)  # another timing: the note would not continue the copy
    assert not (rp.penalize(torch.zeros(1, 128)) < 0).any()


def test_chords_copies_and_share():
    assert copies([1, 2, 3, 1, 2, 3, 4])[0].tolist() == [0, 0, 0, 1, 2, 3, 0]
    assert copies([1, 2, 3, 1, 2, 3, 4])[1].tolist() == [-1, -1, -1, 0, 1, 2, -1]
    # a 3-note chord, then the same chord with its notes in another order: one key, order-blind
    keys = chords([60, 64, 67, 67, 60, 64], [1, 0, 0, 5, 0, 0], [0] * 6)[0]
    assert len(keys) == 2 and keys[0] != keys[1], 'gaps differ (1 vs 5)'
    keys = chords([60, 64, 67, 67, 60, 64], [5, 0, 0, 5, 0, 0], [0] * 6)[0]
    assert keys[0] == keys[1]
    pitch = list(range(40, 56)) * 2  # 16 one-note chords played twice: the second time is a copy of 16
    dt, prog = [1] * 32, [0] * 32
    assert copy_share(pitch, dt, prog, 16) == 0.5 and copy_share(pitch, dt, prog, 17) == 0.0
    assert copy_share(pitch, dt, prog, 16, skip=16) == 1.0, 'shares count the notes from skip on'
    assert copy_share(pitch, dt, [DRUM_PROGRAM] * 32, 1) == 0.0, 'drums are left out'
    octave = pitch[:16] + [p + 12 for p in pitch[16:]]  # the repeat an octave up: a pitch-class copy only
    assert copy_share(octave, dt, prog, 16, key='notes') == 0.0
    assert copy_share(octave, dt, prog, 2, key='pitch_class') > 0.4


class Prefer(torch.nn.Module):
    """A cascade head stub: logits (noise + 20 on one value), so sampling always picks that value unless it's
    penalized."""

    def __init__(self, size, value):
        super().__init__()
        self.size, self.value = size, value

    def forward(self, z):
        logits = torch.randn(*z.shape[:-1], self.size, generator=torch.Generator().manual_seed(z.shape[1]))
        logits[..., self.value] += 20.0
        return logits.to(z.device)


def test_generate_with_penalty():
    """Heads that always prefer one note (pitch 60, delta time 1, piano) loop forever; with the penalty no passage is
    copied for more than dry_allowed chords. Off (multiplier 0) = the plain sampler."""
    m = tiny(n_programs=129)
    for name, value in (('pitch', 60), ('delta_time', 1), ('program', 0)):
        m.cascade.heads[name] = Prefer(m.cascade.heads[name].out.out_features, value)
    pitch = torch.full((2, 4), 60)
    prompt = (pitch, torch.full_like(pitch, 20), torch.full_like(pitch, 10), torch.ones_like(pitch))
    kw = dict(max_new_tokens=60, temperature=1.0, program=None, programs=torch.zeros_like(pitch),
              return_programs=True, cuda_graph=False)
    torch.manual_seed(0)
    plain = generate(m, *prompt, **kw)
    torch.manual_seed(0)
    off = generate(m, *prompt, dry_multiplier=0.0, **kw)
    assert all(torch.equal(a, b) for a, b in zip(plain, off))
    longest = lambda out, r: copies(chords(out[0][r], out[3][r], out[4][r])[0])[0].max()  # noqa: E731
    assert min(longest(plain, r) for r in range(2)) > 50, 'the stub loops'
    torch.manual_seed(0)
    dry = generate(m, *prompt, dry_multiplier=100.0, dry_allowed=8, **kw)
    assert max(longest(dry, r) for r in range(2)) <= 8
    assert (dry[0][:, 4:] == 60).float().mean() > 0.5, 'other choices stay the model\'s: mostly pitch 60 still'


if __name__ == '__main__':
    run(globals())
