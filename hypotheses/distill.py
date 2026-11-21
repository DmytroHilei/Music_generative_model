"""Token-level distillation from an ensemble of members (scaling plan 2026-11-20, items 3-4, E-E5 distill4): the loss
is alpha x the plain CE + (1 - alpha) x the sum over the five heads (program, pitch, duration, velocity, delta_time) of
KL(teacher || student), the teacher being the mean of the members' softmaxes at temperature T. Every head sees the
true earlier attributes (teacher forcing, as in training), masked targets (-1) are left out of each head's mean.
Members: run dirs (model_bf16.pt) of models with the student's heads, frozen in bf16, loaded under fork_rng (the
training RNG stream stays that of the plain run). With compile, the student's training forward (encode + cascade) and
each member are compiled once: K + 1 graphs for training, plus train.py's evaluation graph.
Expected (§8.6): with an ensemble gain g4 >= 0.015 the student on the repeated quarter keeps at least half of it
(retention (control - student) / (control - L_4) >= 0.5, student <= control - 0.012). Logged parts: kd_kl (the summed
KL), kd_teacher_ce (the teachers' mixture CE on the targets, 4 note heads).
    python tools/sweeps/aug_sweep.py --arms distill4   # --hypothesis=distill:members=<dir>,<dir>,...:alpha=0.0
"""

import torch
import torch.nn.functional as F

from hypotheses import Hypothesis as Base

NOTE_HEADS = ('pitch', 'velocity', 'duration', 'delta_time')


class Hypothesis(Base):
    name = 'distill'
    members = ''         # comma-separated run dirs
    alpha = 0.0          # weight of the plain CE
    T = 1.0              # softmax temperature of teachers and student in the KL

    def __init__(self, **args):
        super().__init__(**args)
        self.dirs = [d for d in str(self.members).split(',') if d]
        assert self.dirs and 0 <= self.alpha <= 1 and self.T > 0, (self.members, self.alpha, self.T)
        self.teachers, self.fns, self.student_fn, self.net = [], [], None, None

    def model(self, model):
        from musicar.model.checkpoint import load_model
        with torch.random.fork_rng(devices=[]):  # building a GPT draws its init: keep the training RNG stream
            self.teachers = [load_model(d, dtype=torch.bfloat16)[0].requires_grad_(False) for d in self.dirs]
        for t in self.teachers:
            assert t.head_names == model.head_names, f'distill: member heads {t.head_names} vs {model.head_names}'
        self.net = model
        print(f"distill: {len(self.teachers)} members ({', '.join(self.dirs)}), alpha {self.alpha:g}, T {self.T:g}",
              flush=True)

    @staticmethod
    def logits_fn(model):
        def fn(streams, extra, targets):
            return model.cascade(model.encode(*streams, **extra), dict(zip(model.head_names, targets)))
        return fn

    def setup(self, net, device):
        compiled = hasattr(net, '_orig_mod')
        self.teachers = [t.to(device) for t in self.teachers]
        wrap = torch.compile if compiled else (lambda f: f)
        self.fns = [wrap(self.logits_fn(t)) for t in self.teachers]
        self.student_fn = wrap(self.logits_fn(self.net))

    def forward(self, net, X, Y, train_kw, iter_num):
        assert not train_kw, f'distill: training loss options {train_kw} not supported'
        if self.student_fn is None:
            self.setup(net, X[0][0].device)
        streams, extra = X
        with torch.no_grad():
            teacher = [fn(streams, extra, Y) for fn in self.fns]
        logits = self.student_fn(streams, extra, Y)
        return self.loss(logits, teacher, Y)

    def loss(self, logits, teacher, Y):
        """(parts, loss) from the student's and the members' logits {head: (B, T, V)}."""
        names = self.net.head_names
        ls = self.net.config.label_smoothing
        parts, ce_loss, kl_loss, teacher_ce = {}, 0.0, 0.0, 0.0
        for name, t in zip(names, Y):
            lg = logits[name].float()
            valid = t != -1
            n = valid.sum().clamp(min=1)
            flat, tf = lg.reshape(-1, lg.size(-1)), t.reshape(-1)
            ce_loss = ce_loss + F.cross_entropy(flat, tf, ignore_index=-1, label_smoothing=ls, reduction='sum') / n
            p = torch.stack([torch.softmax(x[name].float() / self.T, -1) for x in teacher]).mean(0)
            logq = torch.log_softmax(lg / self.T, -1)
            kl = (torch.xlogy(p, p) - p * logq).sum(-1)
            kl_loss = kl_loss + (kl * valid).sum() / n * self.T ** 2
            with torch.no_grad():
                parts[name] = F.cross_entropy(flat, tf, ignore_index=-1, reduction='sum') / n
                if name in NOTE_HEADS:
                    pt = p.gather(-1, t.clamp(min=0)[..., None])[..., 0] if self.T == 1 else None
                    if pt is not None:
                        teacher_ce = teacher_ce - (torch.log(pt.clamp(min=1e-30)) * valid).sum() / n
        parts['kd_kl'] = kl_loss.detach()
        if self.T == 1:
            parts['kd_teacher_ce'] = teacher_ce
        loss = kl_loss if self.alpha == 0 else self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        return parts, loss
