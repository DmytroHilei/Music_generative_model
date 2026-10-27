"""Hypotheses (hypotheses/): the loader and the hooks of each one, on tiny models (CPU):
python tests/hypotheses/test_hypotheses.py  (or pytest tests/)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hypotheses import load  # noqa: E402
from musicar.data.loader import BOS_PITCH  # noqa: E402
from musicar.model.gpt import GPT, MusicConfig  # noqa: E402
from musicar.training.optim import build_optimizer  # noqa: E402

TINY = dict(n_layer=2, n_head=2, n_embd=32, block_size=64, dropout=0.0, pitch_size=130, n_programs=129,
            pos_emb='rope', text_dim=8, text_prefix=4)


def tiny(**kw):
    torch.manual_seed(0)
    m = GPT(MusicConfig(**{**TINY, **kw}))
    torch.nn.init.normal_(m.transformer['text_prefix'].gate)  # a prefix that matters (the gate starts at 0.01)
    return m


def batch(B=3, T=16, seed=1):
    """A T2 batch: row 0 starts with a BOS and has a caption, row 1 starts with a BOS without one, row 2 has its BOS
    at note 5 and a caption. Captions = token states (B, 6, 8), rows 4-5 padding."""
    g = torch.Generator().manual_seed(seed)
    streams = [torch.randint(0, n, (B, T), generator=g) for n in (128, 32, 512, 512)]
    program = torch.randint(0, 129, (B, T), generator=g)
    targets = [torch.randint(0, n, (B, T), generator=g) for n in (130, 32, 512, 512, 129)]
    streams[0][0, 0] = streams[0][1, 0] = streams[0][2, 5] = BOS_PITCH
    text = torch.randn(B, 6, 8, generator=g)
    text[1] = 0
    text[:, 4:] = 0
    return (tuple(streams), dict(program=program, text=text)), tuple(targets)


def test_load_spec():
    hyp = load('open_cf')
    assert hyp.name == 'open_cf' and hyp.eta == 0.05 and str(hyp) == 'open_cf'
    hyp = load('open_corrupt:p=0.5:streams=pitch:rows=0.5')
    assert (hyp.p, hyp.streams, hyp.rows) == (0.5, ('pitch',), 0.5) and 'p=0.5' in str(hyp)
    for bad in ('open_cf:nope=1', 'open_cf:name=x', 'open_cf:forward=1'):
        try:
            load(bad)
            raise RuntimeError(bad)
        except TypeError:
            pass
    try:
        load('no_such_hypothesis')
        raise RuntimeError('loaded')
    except ModuleNotFoundError:
        pass
    # through train.py's configurator the spec stays a string
    g = dict(hypothesis='')
    argv, sys.argv = sys.argv, ['train.py', '--hypothesis=open_cf:eta=0.1']
    try:
        exec(Path('musicar/training/configurator.py').read_text(), g)
    finally:
        sys.argv = argv
    assert g['hypothesis'] == 'open_cf:eta=0.1' and load(g['hypothesis']).eta == 0.1


def test_open_corrupt():
    hyp = load('open_corrupt:p=1.0:notes=4:streams=pitch,program')
    X, Y = batch()
    torch.manual_seed(0)
    (xs, ex), Y2 = hyp.batch(X, Y, 0)
    assert Y2 is Y
    pitch0, prog0 = X[0][0], X[1]['program']
    # the opening = the 3 real notes after each BOS of a captioned row (row 0: notes 1-3, row 2: notes 6-8); the
    # BOS, the notes outside the opening and the row without a caption stay
    want = torch.zeros_like(pitch0, dtype=torch.bool)
    want[0, 1:4] = want[2, 6:9] = True
    for new, old in ((xs[0], pitch0), (ex['program'], prog0)):
        assert torch.equal(new[~want], old[~want]) and (new[want] != old[want]).float().mean() > 0.5
    assert (xs[0][want] < BOS_PITCH).all(), 'donors are real notes'
    assert xs[0][0, 0] == BOS_PITCH and xs[0][2, 5] == BOS_PITCH
    for i in (1, 2, 3):
        assert torch.equal(xs[i], X[0][i]), 'streams not named stay'
    # every row with captioned=False; nothing with p=0; a share of the rows with rows < 1
    (xs_all, _), _ = load('open_corrupt:p=1.0:notes=4:streams=pitch:captioned=False').batch(X, Y, 0)
    assert (xs_all[0][1, 1:4] != pitch0[1, 1:4]).any()
    assert load('open_corrupt:p=0').batch(X, Y, 0)[0] is X
    torch.manual_seed(3)
    touched = [(load('open_corrupt:p=1.0:notes=4:streams=pitch:rows=0.5').batch(X, Y, 0)[0][0][0] != pitch0).any()
               for _ in range(20)]
    assert any(touched) and not all(touched)


def test_open_mi():
    m = tiny()
    hyp = load('open_mi:weight=0.5:notes=4:lr_mult=10')
    hyp.model(m)
    opt = build_optimizer('adamw', m, 0.1, 1e-3, (0.9, 0.95), 'cpu', text_lr_mult=10.0)
    n_groups = len(opt.param_groups)
    hyp.optimizer(opt)
    assert len(opt.param_groups) == n_groups + 1 and opt.param_groups[-1]['lr_mult'] == 10
    assert all(id(p) not in {id(q) for g in opt.param_groups[:-1] for q in g['params']}
               for p in opt.param_groups[-1]['params'])
    X, Y = batch()
    m.train()
    parts0, loss0 = m(*X[0], **X[1], targets=Y)
    parts, loss = hyp.forward(m, X, Y, {}, 0)
    assert parts['mi_nce'] > 0 and 0 <= parts['mi_acc'] <= 1
    assert torch.allclose(loss, loss0 + 0.5 * parts['mi_nce'])
    assert torch.allclose(parts['pitch'], parts0['pitch'])
    loss.backward()
    assert hyp.head.weight.grad.abs().sum() > 0 and m.transformer['text_prefix'].gate.grad.abs().sum() > 0
    # without captions the term is 0 and the loss the plain one
    X2 = (X[0], dict(X[1], text=torch.zeros_like(X[1]['text'])))
    parts2, loss2 = hyp.forward(m, X2, Y, {}, 0)
    assert parts2['mi_nce'] == 0 and torch.allclose(loss2, m(*X2[0], **X2[1], targets=Y)[1])


def test_open_cf():
    m = tiny()
    hyp = load('open_cf:weight=2.0:eta=0.05:notes=4')
    X, Y = batch()
    parts0, loss0 = m(*X[0], **X[1], targets=Y)
    parts, loss = hyp.forward(m, X, Y, {}, 0)
    assert abs(parts['ocf_rows'].item() - 1 / 3) < 1e-6, 'row 0 alone starts with a BOS and has a caption'
    # by hand: the batch cropped to 4 notes, the targets of row 0 only, the own vs the rolled captions
    rows = torch.tensor([True, False, False])[:, None]
    xs = tuple(s[:, :4] for s in X[0])
    ys = tuple(torch.where(rows, y[:, :4], torch.full_like(y[:, :4], -1)) for y in Y)
    ex = dict(program=X[1]['program'][:, :4], text=X[1]['text'])
    _, own = m(*xs, **ex, targets=ys)
    _, other = m(*xs, **dict(ex, text=X[1]['text'].roll(1, 0)), targets=ys)
    hinge = torch.relu(0.05 - (other - own))
    assert torch.allclose(parts['ocf_margin'], other - own) and torch.allclose(loss, loss0 + 2.0 * hinge, atol=1e-6)
    assert torch.allclose(parts['pitch'], parts0['pitch'])
    # causal: the cropped own-caption CE is the full forward's CE over those targets
    ys_full = tuple(torch.where(rows & (torch.arange(16) < 4)[None], y, torch.full_like(y, -1)) for y in Y)
    assert torch.allclose(own, m(*X[0], **X[1], targets=ys_full)[1], atol=1e-5)
    loss.backward()
    assert m.transformer['text_prefix'].gate.grad.abs().sum() > 0


def test_open_proxy():
    """open_proxy: KL(q || p) on the opening, q = softmax(student without caption + alpha * teacher's caption shift),
    by hand; rows without a BOS first or without a caption count nothing; the teacher gets no gradient."""
    import tempfile
    m = tiny()
    teacher = tiny(n_layer=1)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'model_bf16.pt'
        from musicar.model.checkpoint import ARCH_KEYS
        args = {k: getattr(teacher.config, k) for k in ARCH_KEYS if hasattr(teacher.config, k)}
        torch.save({'model_args': args, 'model': teacher.state_dict(), 'iter_num': 7}, path)
        hyp = load(f'open_proxy:teacher={path}:alpha=0.5:weight=2.0:notes=4')
        hyp.model(m)
    assert not any(p.requires_grad for p in hyp.net.parameters())
    X, Y = batch()
    parts0, loss0 = m(*X[0], **X[1], targets=Y)
    parts, loss = hyp.forward(m, X, Y, {}, 0)
    assert abs(parts['proxy_rows'].item() - 1 / 3) < 1e-6 and parts['proxy_kl'] >= 0 and parts['proxy_shift'] > 0
    assert torch.allclose(loss, loss0 + 2.0 * parts['proxy_kl'])
    # by hand, row 0, positions 0..3
    xs = tuple(s[:, :4] for s in X[0])
    ys = tuple(y[:, :4] for y in Y)
    tgt = dict(zip(m.head_names, ys))
    text, null = X[1]['text'], torch.zeros_like(X[1]['text'])
    prog = X[1]['program'][:, :4]
    with torch.no_grad():
        own = m.cascade(m.encode(*xs, program=prog, text=text), tgt)
        base = m.cascade(m.encode(*xs, program=prog, text=null), tgt)
        t = hyp.net.float()
        tc = t.cascade(t.encode(*xs, program=prog, text=text), tgt)
        tn = t.cascade(t.encode(*xs, program=prog, text=null), tgt)
    kl = 0.0
    for name in m.head_names:
        q = torch.softmax(base[name][0].float() + 0.5 * (tc[name][0].float() - tn[name][0].float()), -1)
        logp = torch.log_softmax(own[name][0].float(), -1)
        kl = kl + (q * (q.log() - logp)).sum(-1).mean()
    assert torch.allclose(parts['proxy_kl'], kl, atol=2e-3), (parts['proxy_kl'], kl)   # bf16 teacher vs float
    loss.backward()
    assert m.transformer['text_prefix'].gate.grad.abs().sum() > 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
