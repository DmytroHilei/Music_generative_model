"""Counterfactual hinge on the opening (file captions). The own caption must beat another window's caption by eta nats
per note over the first `notes` notes after BOS: loss += weight * max(0, eta - (CE_other - CE_own)), the two CEs from
forwards of the batch cropped to those notes (the prefix and positions 0..notes-1; causal, so the own-caption logits
equal the main forward's), the other caption = the batch's captions rolled by one row; only the windows that start with
a BOS and carry a caption count (targets -1 elsewhere). A direct gradient to read the caption whatever its share of the
loss: the file-caption version of the section counterfactual loss (MuseBarControl's; our section arm of 2026-10-22).
Costs two cropped forwards (about 2 x notes / block_size of a step).
    python train.py ... --hypothesis=open_cf:weight=1.0:eta=0.05:notes=128
"""

import torch
import torch.nn.functional as F

from hypotheses import Hypothesis as Base
from musicar.data.loader import BOS_PITCH

PER_POSITION = ('program', 'remain')      # the extra inputs laid out per note, cropped with the streams


class Hypothesis(Base):
    name = 'open_cf'
    weight = 1.0
    eta = 0.05
    notes = 128

    def forward(self, net, X, Y, train_kw, iter_num):
        parts, loss = net(*X[0], **X[1], targets=Y, **train_kw)
        streams, extra = X
        text = extra.get('text')
        if text is None:
            return parts, loss
        pitch = streams[0]
        B, T = pitch.shape
        n = min(self.notes, T)
        rows = (pitch[:, 0] == BOS_PITCH) & (text.reshape(B, -1).abs().sum(1) > 0)
        xs = tuple(s[:, :n] for s in streams)
        ex = {k: (v[:, :n] if k in PER_POSITION else v) for k, v in extra.items() if k != 'cf'}
        ys = tuple(torch.where(rows[:, None], y[:, :n], torch.full_like(y[:, :n], -1)) for y in Y)
        _, ce_own = net(*xs, **ex, targets=ys)
        _, ce_other = net(*xs, **dict(ex, text=text.roll(1, 0)), targets=ys)
        margin = ce_other - ce_own
        hinge = F.relu(self.eta - margin)
        parts['ocf_margin'] = margin.detach()
        parts['ocf_hinge'] = hinge.detach()
        parts['ocf_rows'] = rows.float().mean()
        return parts, loss + self.weight * hinge
