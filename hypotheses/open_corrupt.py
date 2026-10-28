"""Opening input corruption. The first `notes` input notes after a BOS of the captioned windows have the attributes in
`streams` replaced, each note with probability p, by those of a random real note of the batch; the targets stay. The
in-context evidence about the opening's instruments / dynamics / harmony gets weaker, so the caption is worth more
exactly where it acts: the word-dropout fix for a strong decoder that ignores its conditioning (Bowman et al. 2016,
posterior collapse), here against the 450M hiding the caption prefix because the notes already tell it everything
(probes of 2026-10-25). For the caption stage only (a train / test mismatch at generation, as with any dropout).
Expected: early-notes text gain above plain512's (+0.015 at 600 it, +0.02 plateau) at some CE cost.
    python train.py config/f1/f1.py config/f1/f1_local.py ... --hypothesis=open_corrupt:p=0.3:streams=pitch,program
"""

import torch

from hypotheses import Hypothesis as Base
from musicar.data.loader import BOS_PITCH
from musicar.model.gpt import GPT

STREAMS = ('pitch', 'velocity', 'duration', 'delta_time', 'program')


class Hypothesis(Base):
    name = 'open_corrupt'
    p = 0.3                               # per input note of the opening: the probability of being replaced
    notes = 128                           # the opening: this many notes after a BOS (the BOS itself stays)
    streams = 'pitch,velocity,program'    # the attributes replaced, of pitch / velocity / duration / delta_time / program
    captioned = True                      # only the windows with a caption (the others keep their notes)
    rows = 1.0                            # the share of the eligible windows corrupted

    def __init__(self, **args):
        super().__init__(**args)
        self.streams = tuple(s.strip() for s in str(self.streams).split(',') if s.strip())
        unknown = set(self.streams) - set(STREAMS)
        assert not unknown, f"open_corrupt: unknown streams {sorted(unknown)}"
        assert 0 <= self.p <= 1 and 0 <= self.rows <= 1, (self.p, self.rows)

    def batch(self, X, Y, iter_num):
        if self.p <= 0:
            return X, Y
        streams, extra = X
        pitch = streams[0]
        B, T = pitch.shape
        dev = pitch.device
        real = pitch < BOS_PITCH                                        # BOS / EOS positions are never replaced
        hit = GPT.opening_mask(pitch, self.notes) & real
        rows = torch.ones(B, dtype=torch.bool, device=dev)
        if self.captioned and 'text' in extra:
            rows &= extra['text'].reshape(B, -1).abs().sum(1) > 0
        if self.rows < 1:
            rows &= torch.rand(B, device=dev) < self.rows
        hit &= rows[:, None] & (torch.rand(B, T, device=dev) < self.p)
        # donors: random real notes of the batch (the real positions first in a random order, then random picks)
        key = torch.rand(B * T, device=dev).masked_fill(~real.reshape(-1), 2.0)
        order = key.argsort()
        donor = order[(torch.rand(B, T, device=dev) * real.sum()).long().clamp(max=B * T - 1)]

        def replace(t):
            return torch.where(hit, t.reshape(-1)[donor], t)

        streams = tuple(replace(s) if name in self.streams else s for name, s in zip(STREAMS, streams))
        if 'program' in self.streams and 'program' in extra:
            extra = dict(extra, program=replace(extra['program']))
        return (streams, extra), Y
