"""Whole-window dynamics shift. Every note of a window (input and target velocity) moves by the same random number of
velocity bins, drawn per window in [-shift, shift] and limited so no note leaves 0..bins-1 (no clamping, as with the
loader's transposition); BOS / EOS and masked targets stay. A repeated window is then heard at another overall
loudness while its phrasing (the velocity contour) stays intact. It differs from the rejected per-note jitter (±2 bins,
10.497 vs 10.479, docs/experiments.md §5), which blurred the contour. Data augmentation list of the census doc
(2026-11-19), idea 4: for training that repeats the ~10B unique notes (20B tokens ≈ 2 epochs, 35B ≈ 3.5).
Windows with a caption are left alone by default (their words may name the dynamics: 'soft', 'loud').
Expected: on a repeated-data run (unique data capped so the run sees ~4 epochs) the late-run val CE below the control's,
no worse velocity CE on real (unshifted) validation data. Null if the gap is within seed noise (±0.013).
    python train.py config/pretrain/scale_ladder.py ... --hypothesis=aug_dyn:shift=4
"""

import torch

from hypotheses import Hypothesis as Base
from musicar.data.loader import BOS_PITCH


class Hypothesis(Base):
    name = 'aug_dyn'
    shift = 4            # the largest shift either way, in velocity bins
    bins = 32            # velocity bins of the tokenizer (NoteWindowDataset velocity_bins)
    rows = 1.0           # the share of the eligible windows shifted
    captioned = False    # also shift windows that have a caption

    def __init__(self, **args):
        super().__init__(**args)
        assert 0 <= self.shift < self.bins and 0 <= self.rows <= 1, (self.shift, self.rows)

    def batch(self, X, Y, iter_num):
        if self.shift <= 0 or self.rows <= 0:
            return X, Y
        streams, extra = X
        pitch, vel = streams[0], streams[1]
        B = pitch.size(0)
        dev = pitch.device
        real_x = pitch < BOS_PITCH
        real_y = (Y[0] >= 0) & (Y[0] < BOS_PITCH) & (Y[1] >= 0)
        lo = torch.minimum(vel.masked_fill(~real_x, self.bins).amin(1), Y[1].masked_fill(~real_y, self.bins).amin(1))
        hi = torch.maximum(vel.masked_fill(~real_x, -1).amax(1), Y[1].masked_fill(~real_y, -1).amax(1))
        down = lo.clamp(max=self.shift)
        up = (self.bins - 1 - hi).clamp(max=self.shift)
        s = (torch.rand(B, device=dev) * (down + up + 1)).long().clamp(max=down + up) - down
        keep = hi < 0                                                   # a window without notes
        if not self.captioned and 'text' in extra:
            keep |= extra['text'].reshape(B, -1).abs().sum(1) > 0
        if self.rows < 1:
            keep |= torch.rand(B, device=dev) >= self.rows
        s = s.masked_fill(keep, 0)[:, None]
        vel = torch.where(real_x, vel + s, vel)
        y_vel = torch.where(real_y, Y[1] + s, Y[1])
        return (streams[:1] + (vel,) + streams[2:], extra), Y[:1] + (y_vel,) + Y[2:]
