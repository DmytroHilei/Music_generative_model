"""Instrument swap within its GM family. Per window, every pitched program is mapped through a random permutation of
its family of 8 (pianos, chromatic percussion, organs, guitars, basses, strings, ensembles, brass, reeds, pipes, synth
leads, synth pads, synth effects, ethnic, percussive, sound effects): input and target program streams and the
instrument-set condition alike, so distinct instruments stay distinct and the condition matches the notes. Drums
stay. The notes don't change, only their labels: the GM program choice in the corpora is noisy anyway (a guitar part
as 'acoustic' in one file, 'clean electric' in the next). Data augmentation list of the census doc (2026-11-19),
idea 6: for training that repeats the ~10B unique notes.
Skipped by default: windows with a caption (they name instruments) and piano-only windows (program 0 alone: the
piano stores; family 0 also holds harpsichord and clavinet).
Expected: on a repeated-data run (unique data capped so the run sees ~4 epochs) the late-run val CE below the control's,
no worse instrument CE on real validation data. Risk: blurs timbre-specific style (an e-piano part played as a
grand piano part).
    python train.py config/pretrain/scale_ladder.py ... --hypothesis=aug_family:rows=0.5
"""

import torch

from hypotheses import Hypothesis as Base
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM


class Hypothesis(Base):
    name = 'aug_family'
    rows = 0.5           # the share of the eligible windows remapped
    piano_only = False   # also remap windows whose only pitched program is 0
    captioned = False    # also remap windows that have a caption

    def __init__(self, **args):
        super().__init__(**args)
        assert 0 <= self.rows <= 1, self.rows

    def batch(self, X, Y, iter_num):
        streams, extra = X
        if self.rows <= 0 or 'program' not in extra:
            return X, Y
        pitch, prog = streams[0], extra['program']
        B = pitch.size(0)
        dev = prog.device
        # map[b, p]: program p of row b -> its family's base + a per-row permutation of the 8 slots; drums -> drums
        perm = torch.rand(B, DRUM_PROGRAM // 8, 8, device=dev).argsort(-1)
        base = torch.arange(0, DRUM_PROGRAM, 8, device=dev)[None, :, None]
        mapping = torch.cat([(base + perm).reshape(B, DRUM_PROGRAM),
                             torch.full((B, 1), DRUM_PROGRAM, device=dev, dtype=perm.dtype)], 1)
        real_x = pitch < BOS_PITCH
        pitched = real_x & (prog != DRUM_PROGRAM)
        keep = torch.zeros(B, dtype=torch.bool, device=dev)
        if not self.piano_only:
            keep |= ~(pitched & (prog != 0)).any(1)                    # no pitched program but 0 (or none at all)
        if not self.captioned and 'text' in extra:
            keep |= extra['text'].reshape(B, -1).abs().sum(1) > 0
        if self.rows < 1:
            keep |= torch.rand(B, device=dev) >= self.rows
        ident = torch.arange(DRUM_PROGRAM + 1, device=dev, dtype=mapping.dtype).expand(B, -1)
        mapping = torch.where(keep[:, None], ident, mapping)

        def remap(p, notes):  # BOS / EOS rows (program 0) and masked targets keep theirs
            return torch.where(notes & (p >= 0), mapping.gather(1, p.clamp(min=0).long()).to(p.dtype), p)

        extra = dict(extra, program=remap(prog, real_x))
        if 'cond' in extra:
            inst, density = extra['cond']
            if inst.size(1) == DRUM_PROGRAM + 1:
                extra['cond'] = (torch.zeros_like(inst).scatter_(1, mapping.long(), inst), density)
        return (streams, extra), Y[:4] + (remap(Y[4], (Y[0] >= 0) & (Y[0] < BOS_PITCH)),) + Y[5:]
