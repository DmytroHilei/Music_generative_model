"""Optimizer, LR schedule, freezing and config checks (CPU): python tests/training/test_optim.py  (or pytest tests/)."""

import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import ROOT, batch, run, tiny  # noqa: E402
from musicar.training.optim import (CombinedOptimizer, build_optimizer, cooldown_start, freeze_params, lr_at,  # noqa: E402
                                    set_lr, zeropower_via_newtonschulz5)


def test_lr_schedules():
    kw = dict(learning_rate=1.0, min_lr=0.1, warmup_iters=10, lr_decay_iters=110, cooldown_frac=0.2)
    assert lr_at(0, 'cosine', **kw) == 1 / 11 and lr_at(9, 'cosine', **kw) == 10 / 11
    assert lr_at(10, 'cosine', **kw) == 1.0 and math.isclose(lr_at(60, 'cosine', **kw), 0.55)
    assert lr_at(110, 'cosine', **kw) == 0.1 and lr_at(500, 'cosine', **kw) == 0.1
    assert cooldown_start(110, 0.2) == 88
    assert lr_at(87, 'wsd', **kw) == 1.0 and lr_at(88, 'wsd', **kw) == 1.0
    assert math.isclose(lr_at(110, 'wsd', **kw), 0.1) and math.isclose(lr_at(500, 'wsd', **kw), 0.1)
    assert math.isclose(lr_at(99, 'wsd', **kw), 0.1 + (1 - math.sqrt(0.5)) * 0.9)


def test_adamw_groups():
    m = tiny(n_styles=4)
    opt = build_optimizer('adamw', m, 0.1, 1e-3, (0.9, 0.95), 'cpu')
    decay, nodecay = opt.param_groups
    names = {id(p): n for n, p in m.named_parameters()}
    assert all(p.dim() >= 2 and 'cond_emb' not in names[id(p)] for p in decay['params'])
    assert any('cond_emb' in names[id(p)] for p in nodecay['params']) and nodecay['weight_decay'] == 0.0
    assert len(decay['params']) + len(nodecay['params']) == len(list(m.parameters()))


def test_muon_groups_and_step():
    m = tiny(n_styles=4)
    opt = build_optimizer('muon', m, 0.1, 1e-3, (0.9, 0.95), 'cpu', style_lr_mult=20.0)
    assert isinstance(opt, CombinedOptimizer)
    muon, adamw = opt.optimizers
    block_mats = [p for p in m.transformer.h.parameters() if p.dim() == 2]
    assert len(muon.param_groups[0]['params']) == len(block_mats)
    assert [g.get('lr_mult', 1.0) for g in adamw.param_groups] == [1.0, 1.0, 20.0]
    streams, _, targets = batch()
    before = [p.detach().clone() for p in m.parameters()]
    _, loss = m.train()(*streams, targets=targets)
    loss.backward()
    opt.step()
    opt.zero_grad()
    assert all(not torch.equal(a, b) for a, b in zip(before, m.parameters()) if a.dim() == 2)
    state = opt.state_dict()
    opt.load_state_dict(state)
    assert all(p.grad is None for p in m.parameters())


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    for shape in ((32, 32), (16, 48), (48, 16)):
        s = torch.linalg.svdvals(zeropower_via_newtonschulz5(torch.randn(shape)).float())
        assert s.min() > 0.5 and s.max() < 1.3, (shape, s.min(), s.max())


def test_set_lr_uses_lr_mult():
    m = tiny(n_styles=4)
    opt = build_optimizer('muon', m, 0.1, 1e-3, (0.9, 0.95), 'cpu', style_lr_mult=20.0)
    set_lr(opt, 0.5)
    assert [g['lr'] for g in opt.param_groups] == [0.5, 0.5, 0.5, 10.0]


def test_freeze_params():
    m = tiny(n_styles=4)
    freeze_params(m, 'transformer.h.')
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith('transformer.h.'))
    assert m.transformer.style.weight.requires_grad
    opt = build_optimizer('adamw', m, 0.1, 1e-3, (0.9, 0.95), 'cpu')
    assert sum(len(g['params']) for g in opt.param_groups) == sum(p.requires_grad for p in m.parameters())
    try:
        freeze_params(tiny(), 'nothing.')
        raise AssertionError('a prefix that matches nothing must fail')
    except AssertionError as e:
        assert 'matches no parameter' in str(e), e


def test_configurator():
    configurator = (ROOT / 'musicar' / 'training' / 'configurator.py').read_text()
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / 'c.py'
        cfg.write_text("batch_size = 4\nout_dir = 'x'\n")
        g = dict(batch_size=1, learning_rate=1e-3, out_dir='a', compile=True, init_from=None)
        argv = sys.argv
        try:
            sys.argv = ['train.py', str(cfg), '--learning_rate=2', '--compile=False', '--init_from=resume',
                        '--out_dir=checkpoints/y']
            exec(configurator, g)
            assert (g['batch_size'], g['learning_rate'], g['compile']) == (4, 2, False)
            assert (g['init_from'], g['out_dir']) == ('resume', 'checkpoints/y')
            for bad in ('--nope=1', '--batch_size=1.5'):
                sys.argv = ['train.py', bad]
                try:
                    exec(configurator, g)
                    raise RuntimeError(bad)
                except (ValueError, AssertionError):
                    pass
        finally:
            sys.argv = argv


if __name__ == '__main__':
    run(globals())
