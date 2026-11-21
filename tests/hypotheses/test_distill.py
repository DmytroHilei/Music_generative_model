"""distill (hypotheses/distill.py) on CPU tiny models: python tests/hypotheses/test_distill.py  (or pytest tests/)."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]

from hypotheses import load  # noqa: E402
from musicar.model.gpt import GPT, MusicConfig  # noqa: E402
from test_hypotheses import batch  # noqa: E402

ARGS = dict(n_layer=1, n_head=2, n_embd=32, block_size=64, pitch_size=130, n_programs=129, pos_emb='rope',
            norm='rmsnorm', mlp='swiglu')


def model(seed):
    torch.manual_seed(seed)
    return GPT(MusicConfig(**ARGS, dropout=0.0))


def members(*seeds):
    d = Path(tempfile.mkdtemp())
    for s in seeds:
        (d / f'm{s}').mkdir()
        torch.save(dict(model_args=ARGS, model=model(s).state_dict()), d / f'm{s}' / 'model_bf16.pt')
    return ','.join(str(d / f'm{s}') for s in seeds)


def xy():
    X, Y = batch()
    return (X[0], {'program': X[1]['program']}), Y


def hyp(spec, student):
    h = load(spec)
    state = torch.get_rng_state()
    h.model(student)
    assert torch.equal(state, torch.get_rng_state())  # loading the members leaves the CPU RNG alone
    return h


def test_alpha_one_is_the_plain_loss_and_members_frozen():
    student = model(0)
    h = hyp(f'distill:members={members(1, 2)}:alpha=1.0', student)
    assert len(h.teachers) == 2 and not any(p.requires_grad for t in h.teachers for p in t.parameters())
    assert all(p.dtype == torch.bfloat16 for t in h.teachers for p in t.parameters())
    X, Y = xy()
    parts, loss = h.forward(student, X, Y, {}, 0)
    ref_parts, ref = student(*X[0], **X[1], targets=Y)
    assert torch.allclose(loss, ref, atol=1e-5) and abs(parts['pitch'] - ref_parts['pitch']) < 1e-5
    assert float(parts['kd_kl']) > 0 and 'kd_teacher_ce' in parts


def test_identical_member_gives_no_kl_and_gradients_flow():
    student = model(0)
    h = hyp(f'distill:members={members(0)}', student)  # the student's own weights (in bf16)
    X, Y = xy()
    parts, loss = h.forward(student, X, Y, {}, 0)
    assert float(parts['kd_kl']) < 1e-3, parts['kd_kl']
    loss.backward()
    assert student.transformer.h[0].mlp.c_fc.weight.grad.abs().sum() > 0


def test_hand_kl_and_masking():
    student = model(0)
    h = hyp(f'distill:members={members(1)}', student)
    names = student.head_names
    g = torch.Generator().manual_seed(3)
    sizes = dict(pitch=130, velocity=32, duration=512, delta_time=512, program=129)
    logits = {n: torch.randn(1, 3, sizes[n], generator=g) for n in names}
    teach = [{n: torch.randn(1, 3, sizes[n], generator=g) for n in names} for _ in range(2)]
    Y = [torch.tensor([[1, 2, -1]]) for _ in names]
    parts, loss = h.loss(logits, teach, Y)
    want = 0.0
    for n in names:
        p = (teach[0][n].softmax(-1) + teach[1][n].softmax(-1)) / 2
        kl = (p * (p.log() - logits[n].log_softmax(-1))).sum(-1)[0, :2].mean()
        want += kl
    assert torch.allclose(loss, want, atol=1e-5)
    logits2 = {n: v.clone() for n, v in logits.items()}
    for v in logits2.values():
        v[0, 2] += 5.0  # a masked position: no effect
    assert torch.allclose(h.loss(logits2, teach, Y)[1], loss, atol=1e-6)


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
