"""Several hypotheses in one arm: parts=name/key=value/key=value;name/key=value. The dataset and batch hooks run in
the order given (each on the previous one's output), every part's model and optimizer hooks are called, and at most
one part may replace the training forward. train.py splits the --hypothesis spec on ':', so the parts use '/' between
the name and its arguments and ';' between parts (no '/', ';' or ':' inside a value). Items 1-2 of the 2026-11-20
scaling plan: BEST = REG + dyn + family when REG is itself a hypothesis (drop_late, wd_split).
Expected: the stacked arm at or below its best part (the gains of independent augmentations add up partly);
otherwise BEST := its best part.
    python tools/sweeps/aug_sweep.py --arm-json logs/reg_arms.json --arms BEST
    # --hypothesis=stack:parts=aug_dyn/shift=4;aug_family/rows=0.5
"""

import importlib
from ast import literal_eval

from hypotheses import Hypothesis as Base


def parse(part):
    """'name/key=value/key=value' -> (name, {key: value}); values as in hypotheses.load."""
    name, *kv = part.split('/')
    args = {}
    for item in kv:
        k, v = item.split('=', 1)
        try:
            args[k] = literal_eval(v)
        except (SyntaxError, ValueError):
            args[k] = v
    return name, args


class Hypothesis(Base):
    name = 'stack'
    parts = ''           # name/key=value;name/key=value

    def __init__(self, **args):
        super().__init__(**args)
        specs = [parse(p) for p in str(self.parts).split(';') if p.strip()]
        assert specs and all(n != 'stack' for n, _ in specs), self.parts
        self.hyps = [importlib.import_module(f'hypotheses.{n}').Hypothesis(**a) for n, a in specs]
        own = [h for h in self.hyps if type(h).forward is not Base.forward]
        assert len(own) <= 1, f'stack: more than one part replaces the forward ({", ".join(h.name for h in own)})'
        self.own_forward = own[0] if own else None

    def dataset(self, dataset):
        for h in self.hyps:
            dataset = h.dataset(dataset)
        return dataset

    def model(self, model):
        for h in self.hyps:
            h.model(model)

    def optimizer(self, optimizer):
        for h in self.hyps:
            h.optimizer(optimizer)

    def batch(self, X, Y, iter_num):
        for h in self.hyps:
            X, Y = h.batch(X, Y, iter_num)
        return X, Y

    def forward(self, net, X, Y, train_kw, iter_num):
        if self.own_forward is not None:
            return self.own_forward.forward(net, X, Y, train_kw, iter_num)
        return super().forward(net, X, Y, train_kw, iter_num)

    def __str__(self):
        return 'stack: ' + ' + '.join(map(str, self.hyps))
