"""MHA -> GQA conversion, layer by layer (musicar/model/gqa.py; CPU, tiny models):
python tests/model/test_gqa.py  (or pytest tests/)."""

import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import batch, run, tiny  # noqa: E402
from musicar.model.checkpoint import load_expanded  # noqa: E402
from musicar.model.gpt import CausalSelfAttention, rope_tables  # noqa: E402
from musicar.model.gqa import InputMoments, apply_pca_init, gqa_layers, pca_merge, teacher_weights  # noqa: E402

C, H, KV = 32, 4, 2
HS, HALF = C // H, C // H // 2


def attn_pair(config, n_kv):
    """A teacher (full) attention and an empty GQA one with n_kv heads."""
    torch.manual_seed(0)
    teacher = CausalSelfAttention(dataclasses.replace(config, n_kv_head=0)).eval()
    student = CausalSelfAttention(dataclasses.replace(config, n_kv_head=n_kv)).eval()
    return teacher, student


def rel_err(student, teacher, x, rope):
    with torch.no_grad():
        y, y_t = student(x, rope=rope), teacher(x, rope=rope)
    return ((y - y_t).pow(2).mean() / y_t.pow(2).mean()).item()


def merged(teacher, student, cov):
    w_attn, w_proj = pca_merge(teacher.c_attn.weight, teacher.c_proj.weight, cov, H, KV)
    with torch.no_grad():
        student.c_attn.weight.copy_(w_attn)
        student.c_proj.weight.copy_(w_proj)
    return student


def inputs(n=4, T=24, seed=0, mix=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, C, generator=g)
    return x @ mix if mix is not None else x


def test_pca_merge_exact_when_the_group_shares_its_keys_and_values():
    """Heads whose keys differ by a complex factor per rotation pair and whose values differ by an invertible map
    merge without error: the GQA layer reproduces the full attention on any input (RoPE included)."""
    config = tiny(n_head=H, n_embd=C, pos_emb='rope').config
    teacher, student = attn_pair(config, KV)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        w = teacher.c_attn.weight
        k = w[C:2 * C].view(H, HS, C)
        v = w[2 * C:].view(H, HS, C)
        for grp in range(KV):
            a, b = 2 * grp, 2 * grp + 1
            for f in range(HALF):
                c = torch.randn(2, generator=g)
                ra, ia = k[a, f].clone(), k[a, f + HALF].clone()
                k[b, f], k[b, f + HALF] = c[0] * ra - c[1] * ia, c[1] * ra + c[0] * ia
            v[b] = (torch.eye(HS) + 0.3 * torch.randn(HS, HS, generator=g)) @ v[a]
    x = inputs()
    cov = (x.reshape(-1, C).T @ x.reshape(-1, C)).double() / x.shape[0] / x.shape[1]
    merged(teacher, student, cov)
    rope = rope_tables(torch.arange(24), HS, 10000.0)
    assert rel_err(student, teacher, inputs(seed=1), rope) < 1e-9


def test_pca_merge_beats_mean_pooling():
    """On random weights and correlated inputs the data-aware merge leaves a smaller output error than the mean-pool
    (the first query rows stay those of the teacher up to the per-pair factor)."""
    config = tiny(n_head=H, n_embd=C, pos_emb='rope').config
    teacher, student = attn_pair(config, KV)
    mix = torch.randn(C, C, generator=torch.Generator().manual_seed(5)) * torch.linspace(1.5, 0.05, C)[:, None]
    x = inputs(n=16, mix=mix)
    cov = (x.reshape(-1, C).T @ x.reshape(-1, C)).double() / (16 * 24)
    rope = rope_tables(torch.arange(24), HS, 10000.0)
    mean = CausalSelfAttention(dataclasses.replace(config, n_kv_head=KV)).eval()
    sd = {k: v.clone() for k, v in teacher.state_dict().items()}
    from musicar.model.checkpoint import pool_kv_heads
    sd['c_attn.weight'] = pool_kv_heads(sd['c_attn.weight'], C, HS, KV)
    mean.load_state_dict(sd)
    test_x = inputs(n=8, seed=9, mix=mix)
    e_mean, e_pca = rel_err(mean, teacher, test_x, rope), rel_err(merged(teacher, student, cov), teacher, test_x, rope)
    assert e_pca < 0.8 * e_mean, (e_pca, e_mean)


def qk_gains(module, seed=4):
    """Non-trivial QK-norm gains (a trained model's are not all ones)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, CausalSelfAttention) and m.q_norm is not None:
                m.q_norm.weight.copy_(0.5 + torch.rand(HS, generator=g))
                m.k_norm.weight.copy_(0.5 + torch.rand(HS, generator=g))
    return module


def gqa_pair(**kw):
    """A full-attention tiny model and its GQA version loaded the way train.py's finetune does (mean-pool)."""
    base = qk_gains(tiny(n_head=H, n_embd=C, pos_emb='rope', n_layer=2, **kw))
    conv = tiny(n_head=H, n_embd=C, pos_emb='rope', n_layer=2, n_kv_head=KV, **kw)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    teachers = teacher_weights(state)
    load_expanded(conv, state)
    return base, conv, teachers


def test_teacher_stream_and_transfer_loss():
    """With the teachers attached the trunk computes the full-attention model's states in training, the loss is the
    summed transfer MSE and only the attention layers get gradients; in eval (and after detaching) the model runs its
    own GQA layers."""
    base, conv, teachers = gqa_pair()
    assert set(teachers) == {0, 1} and set(gqa_layers(conv)) == {0, 1}
    streams, _, targets = batch(T=24)
    with torch.no_grad():
        h_base, h_own = base.encode(*streams), conv.encode(*streams)
    assert not torch.allclose(h_base, h_own, atol=1e-3)
    keep = conv.attach_attn_teachers(teachers)
    assert keep == ['transformer.h.0.attn.', 'transformer.h.1.attn.']
    conv.train()
    with torch.no_grad():
        assert torch.allclose(conv.encode(*streams), h_base, atol=1e-5)
    for n, p in conv.named_parameters():
        p.requires_grad_(n.startswith(tuple(keep)))
    parts, loss = conv(*streams, targets=targets)
    assert torch.equal(loss, parts['transfer']) and loss > 0
    loss.backward()
    assert conv.transformer.h[1].attn.c_attn.weight.grad is not None
    assert all(p.grad is None for n, p in conv.named_parameters() if not n.startswith(tuple(keep)))
    conv.eval()
    with torch.no_grad():
        assert torch.allclose(conv.encode(*streams), h_own, atol=1e-5)
        parts, _ = conv(*streams, targets=targets)
    assert 'transfer' not in parts
    conv.detach_attn_teachers()
    conv.train()
    with torch.no_grad():
        assert torch.allclose(conv.encode(*streams), h_own, atol=1e-5)


def test_pca_merge_with_qk_norm_beats_mean_pooling():
    """QK-norm attention (non-trivial gains): the merge in the normalized space, back through the gains, leaves a
    smaller output error than the mean-pool."""
    config = tiny(n_head=H, n_embd=C, pos_emb='rope', qk_norm=True).config
    teacher, student = attn_pair(config, KV)
    qk_gains(teacher)
    mix = torch.randn(C, C, generator=torch.Generator().manual_seed(5)) * torch.linspace(1.5, 0.05, C)[:, None]
    x = inputs(n=16, mix=mix)
    cov = (x.reshape(-1, C).T @ x.reshape(-1, C)).double() / (16 * 24)
    rope = rope_tables(torch.arange(24), HS, 10000.0)
    from musicar.model.checkpoint import pool_kv_heads
    sd = {k: v.clone() for k, v in teacher.state_dict().items()}
    sd['c_attn.weight'] = pool_kv_heads(sd['c_attn.weight'], C, HS, KV)
    mean = CausalSelfAttention(dataclasses.replace(config, n_kv_head=KV)).eval()
    mean.load_state_dict(sd)
    w_attn, w_proj = pca_merge(teacher.c_attn.weight, teacher.c_proj.weight, cov, H, KV,
                               q_gain=teacher.q_norm.weight, k_gain=teacher.k_norm.weight)
    with torch.no_grad():
        student.load_state_dict(sd)
        student.c_attn.weight.copy_(w_attn)
        student.c_proj.weight.copy_(w_proj)
    test_x = inputs(n=8, seed=9, mix=mix)
    e_mean, e_pca = rel_err(mean, teacher, test_x, rope), rel_err(student, teacher, test_x, rope)
    assert e_pca < 0.8 * e_mean, (e_pca, e_mean)


def test_qk_norm_teacher_stream_is_the_full_model():
    """QK-norm model: the teachers carry the checkpoint's q_norm / k_norm, so in training the converted model
    computes the full-attention model's states exactly."""
    base, conv, teachers = gqa_pair(qk_norm=True)
    assert all(set(norms) == {'q_norm.weight', 'k_norm.weight'} for _, _, norms in teachers.values())
    streams, _, _ = batch(T=24)
    with torch.no_grad():
        h_base = base.encode(*streams)
    conv.attach_attn_teachers(teachers)
    conv.train()
    with torch.no_grad():
        assert torch.allclose(conv.encode(*streams), h_base, atol=1e-5)


def test_pca_init_from_moments_lowers_the_transfer_loss():
    """train.py's gqa_init='pca': moments through the teacher stream, then apply_pca_init; the transfer loss of the
    merged layers is below the mean-pool's (with and without QK-norm)."""
    for kw in ({}, dict(qk_norm=True)):
        pca_init_vs_mean(**kw)


def pca_init_vs_mean(**kw):
    _, conv, teachers = gqa_pair(**kw)
    conv.attach_attn_teachers(teachers)
    conv.train()
    streams, _, targets = batch(T=24, seed=7)
    with torch.no_grad():
        mean_loss = conv(*streams, targets=targets)[1].item()
        moments = InputMoments(gqa_layers(conv), bias=conv.config.bias)
        for seed in range(4):
            s, _, t = batch(T=24, seed=10 + seed)
            conv(*s, targets=t)
        moments.remove()
        assert moments.n == 4 * 3 * 24
        assert apply_pca_init(conv, moments.moments()) == [0, 1]
        pca_loss = conv(*streams, targets=targets)[1].item()
    assert pca_loss < mean_loss, (pca_loss, mean_loss)


if __name__ == '__main__':
    run(globals())
