"""Hypotheses: experimental training changes kept out of the main code until one earns its place.

train.py --hypothesis=<name>[:key=value:key=value] imports hypotheses/<name>.py and calls its Hypothesis at the fixed
points below; nothing else in train.py, the model or the loader changes for it. A hypothesis that works is then
written into the model / loader / train.py as a regular setting (with its tests) and its file deleted; one that fails
is deleted with its result in the notes (hypotheses/README.md).
"""

import importlib
from ast import literal_eval


class Hypothesis:
    """The hooks train.py calls, in this order; a hypothesis overrides the ones its idea needs. Its arguments are
    class attributes (the defaults), set from the spec string by __init__.
    dataset: the training NoteWindowDataset, before the DataLoader; returns the dataset to use.
    model: the model with its weights loaded, before .to(device), fp8 and the optimizer. Modules a hypothesis needs
      stay on the hypothesis object (the checkpoints stay those of the plain model; no resume with them).
    optimizer: the built optimizer; a hypothesis adds a param group for its own modules ('lr_mult' as the others).
    batch: a prepared training batch on the device, ((streams, extra), targets) -> the same layout; validation
      batches don't pass here.
    forward: the training forward of one micro-batch -> (parts, loss); the default is the plain one. net is the
      (compiled) model; extra forwards go through it too (one more compiled graph per new shape)."""
    name = ''

    def __init__(self, **args):
        for k, v in args.items():
            if not hasattr(self, k) or k in ('name',) or callable(getattr(self, k)):
                raise TypeError(f"{self.name}: unknown argument {k!r}")
            setattr(self, k, v)
        self.args = args

    def dataset(self, dataset):
        return dataset

    def model(self, model):
        pass

    def optimizer(self, optimizer):
        pass

    def batch(self, X, Y, iter_num):
        return X, Y

    def forward(self, net, X, Y, train_kw, iter_num):
        return net(*X[0], **X[1], targets=Y, **train_kw)

    def __str__(self):
        return self.name + ''.join(f' {k}={v}' for k, v in self.args.items())


def load(spec):
    """'name:key=value:key=value' -> the Hypothesis of hypotheses/name.py; values are Python literals when they parse,
    else strings (so --hypothesis=open_corrupt:p=0.5:streams=pitch,program reaches p=0.5, streams='pitch,program')."""
    name, *kv = str(spec).split(':')
    args = {}
    for item in kv:
        k, v = item.split('=', 1)
        try:
            args[k] = literal_eval(v)
        except (SyntaxError, ValueError):
            args[k] = v
    return importlib.import_module(f'hypotheses.{name}').Hypothesis(**args)
