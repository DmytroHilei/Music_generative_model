"""Proxy distillation of the 107M's caption reading into the 450M, on the opening. The 107M T2 model reads captions
(early-notes text gain +0.031: text_t2_v3_long); the 450M does not. Proxy tuning (Liu et al. 2024, arXiv 2401.08565)
adds a small tuned model's logit shift to a large model at decoding: logits_450(x | null) + alpha * (logits_107(x | c)
- logits_107(x | null)) is a distribution with the 450M's quality and the 107M's caption dependence. Here that
distribution is the training target of the 450M on the first `notes` notes after a BOS of the captioned windows:
loss += weight * KL(q || p_450(x | c)) per head, q from three no-grad forwards of the batch cropped to the opening
(the 450M with the null caption, the 107M with the own and the null caption). A dense caption-dependent target at
every opening position, far lower in variance than the one-hot CE where the caption is worth ~0.03 nats: the gradient
toward reading the caption does not depend on the loss share. Needs the same note vocabulary and heads (it is).
    python train.py ... --hypothesis=open_proxy:teacher=checkpoints/text_t2_v3_long:alpha=1.0:weight=1.0
With compile: the forward hook on the cascade heads is traced by torch.compile; --compile=False --fp8=False if the arm
fails to compile.
"""

import torch
import torch.nn.functional as F

from hypotheses import Hypothesis as Base
from musicar.data.loader import BOS_PITCH
from musicar.model.checkpoint import load_model

PER_POSITION = ('program', 'remain')


class Hypothesis(Base):
    name = 'open_proxy'
    teacher = 'checkpoints/text_t2_v3_long'   # the small model that reads captions (a T2 prefix model)
    alpha = 1.0                                # scale of the 107M's caption logit shift in the target
    weight = 1.0
    notes = 128

    def __init__(self, **args):
        super().__init__(**args)
        self.net = None
        self.logits = None

    def model(self, model):
        assert model.config.text_prefix, 'open_proxy needs a T2 text prefix model'
        self.net, _ = load_model(self.teacher, dtype=torch.bfloat16)
        self.net.pad_multiple = 0
        assert self.net.config.text_prefix and self.net.config.text_dim == model.config.text_dim, \
            'the teacher must read the same caption token states'
        for name in model.head_names:
            assert getattr(self.net.config, {'pitch': 'pitch_size', 'velocity': 'velocity_size',
                                             'duration': 'duration_size', 'delta_time': 'delta_time_size',
                                             'program': 'n_programs'}[name]) == \
                getattr(model.config, {'pitch': 'pitch_size', 'velocity': 'velocity_size', 'duration': 'duration_size',
                                       'delta_time': 'delta_time_size', 'program': 'n_programs'}[name]), name
        for p in self.net.parameters():
            p.requires_grad_(False)
        self._model = model
        model.cascade.register_forward_hook(self._capture)   # the logits of every forward, {head: (B, T, V)}

    def _capture(self, module, inputs, output):
        self.logits = output

    def optimizer(self, optimizer):
        self.net.to(next(self._model.parameters()).device)

    def forward(self, net, X, Y, train_kw, iter_num):
        parts, loss = net(*X[0], **X[1], targets=Y, **train_kw)
        streams, extra = X
        text = extra.get('text')
        if text is None:
            return parts, loss
        own = {k: v for k, v in self.logits.items()}             # the main forward's logits, with grad
        pitch = streams[0]
        B, T = pitch.shape
        n = min(self.notes, T)
        rows = (pitch[:, 0] == BOS_PITCH) & (text.reshape(B, -1).abs().sum(1) > 0)
        xs = tuple(s[:, :n] for s in streams)
        ex = {k: (v[:, :n] if k in PER_POSITION else v) for k, v in extra.items() if k != 'cf'}
        ys = tuple(y[:, :n] for y in Y)
        null = torch.zeros_like(text)
        with torch.no_grad():
            net(*xs, **dict(ex, text=null), targets=ys)
            base = {k: v.float() for k, v in self.logits.items()}  # the 450M without the caption
            tgt = dict(zip(self.net.head_names, ys))
            kw = dict(program=ex.get('program')) if self.net.config.n_programs else {}
            with_c = self.net.cascade(self.net.encode(*xs, text=text.to(torch.bfloat16), **kw), tgt)
            without = self.net.cascade(self.net.encode(*xs, text=null.to(torch.bfloat16), **kw), tgt)
        mask = (rows[:, None] & (ys[0] != -1)).float()               # (B, n)
        n_pos = mask.sum().clamp(min=1)
        kl, shift = 0.0, 0.0
        for name in self._model.head_names:
            delta = self.alpha * (with_c[name].float() - without[name].float())
            q = F.softmax(base[name] + delta, dim=-1)
            logp = F.log_softmax(own[name][:, :n].float(), dim=-1)
            kl = kl + ((q * (q.clamp(min=1e-12).log() - logp)).sum(-1) * mask).sum() / n_pos
            shift = shift + (delta.abs().mean(-1) * mask).sum() / n_pos
        parts['proxy_kl'] = kl.detach()
        parts['proxy_shift'] = shift.detach()
        parts['proxy_rows'] = rows.float().mean()
        return parts, loss + self.weight * kl
