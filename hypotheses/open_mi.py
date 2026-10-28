"""Caption prediction from the opening (an InfoNCE / mutual-information term). A training-only head (n_embd ->
text_dim) on the mean final state of the first `notes` notes after a BOS of each captioned window must pick that
window's caption (its mean LaBSE token state) out of the batch's captions, both ways, at temperature tau. The opening
states can only carry the caption by attending to the prefix slots, so this term pays attention to the slots directly,
where the plain CE gives ~no gradient once the trunk has turned the slots' keys away (probe of 2026-10-25); once the
attention flows, the CE can use what it reads. The "predict the condition back" regularizer of InfoGAN and of the
posterior-collapse literature. The head lives on the hypothesis (not in the checkpoint), lr x lr_mult, no weight decay.
    python train.py ... --hypothesis=open_mi:weight=0.1:tau=0.07
With compile: the forward hook on ln_f is traced by torch.compile; set --compile=False if that arm fails to compile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypotheses import Hypothesis as Base
from musicar.model.gpt import GPT


class Hypothesis(Base):
    name = 'open_mi'
    weight = 0.1
    tau = 0.07
    notes = 128
    lr_mult = 10.0

    def __init__(self, **args):
        super().__init__(**args)
        self.head = None
        self.states = None
        self._model = None

    def model(self, model):
        assert model.config.text_dim and model.config.text_prefix, 'open_mi needs a T2 text prefix model'
        with torch.random.fork_rng(devices=[]):   # a fixed init that leaves the run's RNG (data order) alone
            torch.manual_seed(0)
            self.head = nn.Linear(model.config.n_embd, model.config.text_dim)
        self._model = model
        model.transformer.ln_f.register_forward_hook(self._capture)   # the notes' final states, (B, T, C)

    def _capture(self, module, inputs, output):
        self.states = output

    def optimizer(self, optimizer):
        self.head.to(next(self._model.parameters()).device)
        optimizer.add_param_group({'params': list(self.head.parameters()), 'weight_decay': 0.0,
                                   'lr_mult': self.lr_mult})

    def forward(self, net, X, Y, train_kw, iter_num):
        parts, loss = net(*X[0], **X[1], targets=Y, **train_kw)
        text = X[1].get('text')
        if text is None:
            return parts, loss
        h, pitch = self.states, X[0][0]
        B = pitch.size(0)
        has = text.reshape(B, -1).abs().sum(1) > 0
        w = (GPT.opening_mask(pitch, self.notes) & (Y[0] != -1) & has[:, None]).to(h.dtype)   # (B, T)
        n = w.sum(1)
        pooled = (h * w[..., None]).sum(1) / n.clamp(min=1)[:, None]
        z = F.normalize(self.head(pooled).float(), dim=-1)                                      # (B, D)
        tokens = text if text.dim() == 3 else text[:, None]
        tmask = (tokens.abs().sum(-1, keepdim=True) > 0).to(tokens.dtype)
        c = F.normalize(((tokens * tmask).sum(1) / tmask.sum(1).clamp(min=1)).float(), dim=-1)  # (B, D)
        use = has & (n > 0)
        logits = (z @ c.t() / self.tau).masked_fill(~(use[:, None] & use[None, :]), torch.finfo(torch.float32).min)
        labels = torch.arange(B, device=pitch.device)
        ce = F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.t(), labels, reduction='none')
        n_use = use.sum().clamp(min=1)
        nce = (ce * use).sum() / (2 * n_use)
        parts['mi_nce'] = nce.detach()
        parts['mi_acc'] = (((logits.argmax(1) == labels) & use).sum() / n_use).detach()
        return parts, loss + self.weight * nce
