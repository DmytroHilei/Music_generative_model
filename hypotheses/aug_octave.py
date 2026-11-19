"""Single-track octave shift. In a window with several pitched instruments, the notes of one of them (a random program
present; drums never) move up or down 12 semitones, input and target alike, if they all stay in 0..127 (the other
direction is tried when the drawn one doesn't fit; neither = no shift). The other parts stay where they are, so the
window holds a voicing the corpus may not have: the bass an octave lower, the melody an octave higher. Data
augmentation list of the census doc (2026-11-19), idea 7: for training that repeats the ~10B unique notes.
Windows with a caption are left alone by default (section captions name the register).
Expected: on a repeated-data run (unique data capped so the run sees ~4 epochs) the late-run val CE below the control's.
Risk: register clashes the model then reproduces; listen to samples before promoting.
    python train.py config/pretrain/scale_ladder.py ... --hypothesis=aug_octave:rows=0.5
"""

import torch

from hypotheses import Hypothesis as Base
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM


class Hypothesis(Base):
    name = 'aug_octave'
    rows = 0.5           # the share of the eligible windows shifted
    min_tracks = 2       # eligible: windows with at least this many pitched programs (1 = solo windows too)
    captioned = False    # also shift windows that have a caption

    def __init__(self, **args):
        super().__init__(**args)
        assert 0 <= self.rows <= 1 and self.min_tracks >= 1, (self.rows, self.min_tracks)

    def batch(self, X, Y, iter_num):
        streams, extra = X
        if self.rows <= 0 or 'program' not in extra:
            return X, Y
        pitch, prog = streams[0], extra['program']
        B = pitch.size(0)
        dev = pitch.device
        real_x = (pitch < BOS_PITCH) & (prog != DRUM_PROGRAM)
        real_y = (Y[0] >= 0) & (Y[0] < BOS_PITCH) & (Y[4] >= 0) & (Y[4] != DRUM_PROGRAM)
        present = torch.zeros(B, DRUM_PROGRAM + 1, device=dev)
        present.scatter_(1, prog.masked_fill(~real_x, DRUM_PROGRAM), 1.0)
        present.scatter_(1, Y[4].clamp(min=0).masked_fill(~real_y, DRUM_PROGRAM), 1.0)
        present = present[:, :DRUM_PROGRAM] > 0
        pick = torch.rand(B, DRUM_PROGRAM, device=dev).masked_fill(~present, -1.0).argmax(1)
        sel_x = real_x & (prog == pick[:, None])
        sel_y = real_y & (Y[4] == pick[:, None])
        lo = torch.minimum(pitch.masked_fill(~sel_x, 127).amin(1), Y[0].masked_fill(~sel_y, 127).amin(1))
        hi = torch.maximum(pitch.masked_fill(~sel_x, 0).amax(1), Y[0].masked_fill(~sel_y, 0).amax(1))
        up_ok, down_ok = hi + 12 <= 127, lo - 12 >= 0
        want_up = torch.rand(B, device=dev) < 0.5
        s = torch.where(want_up, torch.where(up_ok, 12, torch.where(down_ok, -12, 0)),
                        torch.where(down_ok, -12, torch.where(up_ok, 12, 0)))
        keep = present.sum(1) < self.min_tracks
        if not self.captioned and 'text' in extra:
            keep |= extra['text'].reshape(B, -1).abs().sum(1) > 0
        if self.rows < 1:
            keep |= torch.rand(B, device=dev) >= self.rows
        s = s.masked_fill(keep, 0)[:, None]
        new_x = torch.where(sel_x, pitch + s, pitch)
        new_y = torch.where(sel_y, Y[0] + s, Y[0])
        return ((new_x,) + streams[1:], extra), (new_y,) + Y[1:]
