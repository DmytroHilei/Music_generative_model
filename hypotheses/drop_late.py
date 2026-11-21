"""Late dropout: no dropout for the first `start` iterations, dropout p from then on (Liu et al. 2023, "Dropout reduces
underfitting": late dropout against overfitting; Xue et al. 2023: dropout against repeated data). Every nn.Dropout of
the model (embedding, attention output, MLP) and the attention's own dropout_p switch from 0 to p once, at the first
batch with iter_num >= start; under torch.compile that is one recompile of the training graph. Items 1-2 of the
2026-11-20 scaling plan (D-R2): on the repeated-data run (42M, a quarter of the replay files, ~3.9 passes) dropout
from 26% of the run on, when the data starts to repeat (start = 1266 of 4,883 iterations). Expected: late-run val CE
below drop10 (dropout 0.1 from step 0) and below control by more than T (0.012); then XL's dropout starts ~26% into
the run. Null if within seed noise of drop10.
    python tools/sweeps/aug_sweep.py --arms drop10_late    # --hypothesis=drop_late:p=0.1:start=1266
"""

from torch import nn

from hypotheses import Hypothesis as Base


class Hypothesis(Base):
    name = 'drop_late'
    p = 0.1              # the dropout from `start` on
    start = 0            # the first iteration with dropout

    def __init__(self, **args):
        super().__init__(**args)
        assert 0 <= self.p < 1 and self.start >= 0, (self.p, self.start)
        self.on = False
        self.drops, self.attns, self.net = [], [], None

    def model(self, model):
        """Keep the dropout modules and the attention modules (a float 'dropout' used as dropout_p); all at 0. The
        model's train() is wrapped so eval mode always sees 0: the compiled eval graph (guarded on p) stays valid and
        the switch costs one recompile, of the training graph only."""
        self.drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        self.attns = [m for m in model.modules() if not isinstance(m, nn.Dropout)
                      and isinstance(getattr(m, 'dropout', None), float)]
        assert self.drops, 'drop_late: the model has no nn.Dropout'
        self.set(0.0)
        self.net, plain_train = model, model.train

        def train(mode=True):
            out = plain_train(mode)
            self.set(float(self.p) if mode and self.on else 0.0)
            return out
        model.train = train  # eval() calls train(False); a compiled wrapper's train() reaches it through children()

    def set(self, p):
        for m in self.drops:
            m.p = p
        for m in self.attns:
            m.dropout = p

    def batch(self, X, Y, iter_num):
        if not self.on and iter_num >= self.start:
            self.on = True
            self.set(float(self.p) if self.net.training else 0.0)
            print(f"drop_late: dropout {self.p:g} from iteration {iter_num} on "
                  f"({len(self.drops)} modules, {len(self.attns)} attention)", flush=True)
        return X, Y
