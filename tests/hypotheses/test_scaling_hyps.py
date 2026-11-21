"""drop_late, wd_split and stack (hypotheses/, scaling plan items 1-2) on CPU models:
python tests/hypotheses/test_scaling_hyps.py  (or pytest tests/)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hypotheses import load  # noqa: E402
from musicar.data.loader import DRUM_PROGRAM  # noqa: E402
from musicar.model.gpt import GPT, MusicConfig  # noqa: E402
from musicar.training.optim import build_optimizer, set_lr  # noqa: E402
from test_hypotheses import band, batch  # noqa: E402

# config/pretrain/scale_ladder.py at 42M, as train.py builds it (special tokens: pitch_size 130)
LADDER_42M = dict(n_layer=12, n_head=8, n_embd=512, block_size=2048, pitch_size=130, n_programs=129, pos_emb='rope',
                  norm='rmsnorm', mlp='swiglu', qk_norm=True, cond_inst=True, n_density=16, dropout=0.0)


def small(**kw):
    torch.manual_seed(0)
    return GPT(MusicConfig(**{**LADDER_42M, 'n_layer': 2, 'n_embd': 64, 'n_head': 2, 'block_size': 64, **kw}))


def test_drop_late_switches_once():
    m = small(dropout=0.2)
    hyp = load('drop_late:p=0.1:start=20')
    hyp.model(m)
    drops = [x for x in m.modules() if isinstance(x, torch.nn.Dropout)]
    attns = [x for x in m.modules() if type(x).__name__ == 'CausalSelfAttention']
    assert drops and attns and set(hyp.attns) == set(attns)
    assert all(x.p == 0.0 for x in drops) and all(x.dropout == 0.0 for x in attns)  # off until start
    X, Y = batch()
    assert hyp.batch(X, Y, 19) == (X, Y) and all(x.p == 0.0 for x in drops)
    assert hyp.batch(X, Y, 20) == (X, Y)
    assert all(x.p == 0.1 for x in drops) and all(x.dropout == 0.1 for x in attns)
    m.eval()  # eval mode sees no dropout (the compiled eval graph keeps p == 0)
    assert all(x.p == 0.0 for x in drops) and all(x.dropout == 0.0 for x in attns)
    m.train()
    assert all(x.p == 0.1 for x in drops) and all(x.dropout == 0.1 for x in attns)
    net = torch.nn.Sequential(m)  # a wrapper's eval() reaches the model's train() through children()
    net.eval()
    assert all(x.p == 0.0 for x in drops)
    net.train()
    for x in drops:
        x.p = 0.5
    hyp.batch(X, Y, 21)  # switched once only
    assert all(x.p == 0.5 for x in drops)


def test_wd_split_groups_at_42m():
    m = GPT(MusicConfig(**LADDER_42M))
    opt = build_optimizer('muon', m, 0.1, 1e-3, (0.9, 0.95), 'cpu')
    hyp = load('wd_split:emb=0.0')
    hyp.model(m)
    hyp.optimizer(opt)
    adamw = opt.optimizers[1]
    sizes = [(sum(p.numel() for p in g['params']), g['weight_decay']) for g in adamw.param_groups]
    assert sizes[0] == (3_556_864, 0.1) and sizes[-1] == (292_096, 0.0), sizes
    assert opt.optimizers[0].param_groups[0]['weight_decay'] == 0.1
    set_lr(opt, 5e-4)
    assert adamw.param_groups[-1]['lr'] == 5e-4  # the schedule reaches the new group


def test_wd_split_muon_only_and_step():
    m = small()
    opt = build_optimizer('muon', m, 0.1, 1e-3, (0.9, 0.95), 'cpu')
    n_groups = len(opt.param_groups)
    hyp = load('wd_split:muon=3.0')
    hyp.model(m)
    hyp.optimizer(opt)
    assert len(opt.param_groups) == n_groups and opt.optimizers[0].param_groups[0]['weight_decay'] == 3.0
    assert opt.optimizers[1].param_groups[0]['weight_decay'] == 0.1
    hyp = load('wd_split:emb=0.0:adam=0.3')
    opt = build_optimizer('muon', m, 0.1, 1e-3, (0.9, 0.95), 'cpu')
    hyp.model(m)
    hyp.optimizer(opt)
    assert [g['weight_decay'] for g in opt.optimizers[1].param_groups][::len(opt.optimizers[1].param_groups) - 1] \
        == [0.3, 0.0]
    X, Y = batch(T=16)
    X = (X[0], {'program': X[1]['program']})
    _, loss = m(*X[0], **X[1], targets=Y)
    loss.backward()
    before = m.transformer.program.weight.detach().clone()
    opt.step()  # one step runs with the new group
    assert not torch.equal(before, m.transformer.program.weight)


def test_stack_equals_parts_in_sequence():
    X, Y = band()
    prog = X[1]['program']
    inst = torch.zeros(3, DRUM_PROGRAM + 1)
    inst.scatter_(1, prog[:, 1:], 1.0)
    X = (X[0], dict(X[1], cond=(inst, torch.arange(3))))
    stacked = load('stack:parts=aug_dyn/shift=4/captioned=True;aug_family/rows=1.0')
    assert [h.name for h in stacked.hyps] == ['aug_dyn', 'aug_family'] and stacked.hyps[0].captioned is True
    assert str(stacked).startswith('stack: aug_dyn shift=4')
    dyn, fam = load('aug_dyn:shift=4:captioned=True'), load('aug_family:rows=1.0')
    for seed in range(5):
        torch.manual_seed(seed)
        (xs, ex), ys = stacked.batch(X, Y, 7)
        torch.manual_seed(seed)
        (xs2, ex2), ys2 = fam.batch(*dyn.batch(X, Y, 7), 7)
        assert all(torch.equal(a, b) for a, b in zip(xs, xs2)) and all(torch.equal(a, b) for a, b in zip(ys, ys2))
        assert torch.equal(ex['program'], ex2['program']) and torch.equal(ex['cond'][0], ex2['cond'][0])


def test_stack_hooks_and_forward_rule():
    m = small(dropout=0.1)
    opt = build_optimizer('muon', m, 0.1, 1e-3, (0.9, 0.95), 'cpu')
    hyp = load('stack:parts=drop_late/p=0.1/start=3;wd_split/muon=3.0')
    hyp.model(m)
    hyp.optimizer(opt)
    assert opt.optimizers[0].param_groups[0]['weight_decay'] == 3.0 and hyp.hyps[0].drops[0].p == 0.0
    X, Y = batch()
    hyp.batch(X, Y, 3)
    assert hyp.hyps[0].drops[0].p == 0.1 and hyp.own_forward is None
    for bad in ('stack:parts=open_cf;open_mi', 'stack:parts=stack'):
        try:
            load(bad)
            raise RuntimeError(bad)
        except AssertionError:
            pass


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
