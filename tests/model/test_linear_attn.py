"""Gated DeltaNet linear attention and the attention -> linear conversion (CPU, tiny models):
python tests/model/test_linear_attn.py  (or pytest tests/)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import batch, run, tiny  # noqa: E402
from musicar.model.checkpoint import ARCH_KEYS, load_expanded, merge_model_args  # noqa: E402
from musicar.model.gpt import KVCache  # noqa: E402
from musicar.model.linear_attn import (GatedDeltaNet, chunk_gated_delta_rule, linear_layers,  # noqa: E402
                                       recurrent_gated_delta_rule)


def inputs(B=2, H=3, T=45, dk=8, dv=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, dk, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, dk, generator=g), dim=-1)
    v = torch.randn(B, H, T, dv, generator=g)
    gate = -torch.rand(B, H, T, generator=g) * 0.5
    beta = torch.rand(B, H, T, generator=g)
    return q, k, v, gate, beta


def test_linear_layers():
    assert linear_layers('', 14) == set()
    assert linear_layers('hybrid4', 14) == {0, 1, 2, 4, 5, 6, 8, 9, 10, 12}
    assert linear_layers('hybrid4', 28) == {i for i in range(27) if (i + 1) % 4}
    assert linear_layers('0, 3', 4) == {0, 3}


def test_chunk_matches_recurrent():
    q, k, v, g, beta = inputs()
    state0 = torch.randn(2, 3, 8, 6) * 0.1
    o_ref, s_ref = recurrent_gated_delta_rule(q, k, v, g, beta, state=state0)
    for chunk in (8, 16, 64):  # T = 45: padded last chunk; 64 = one chunk
        o, s = chunk_gated_delta_rule(q, k, v, g, beta, chunk=chunk, state=state0)
        assert torch.allclose(o, o_ref, atol=1e-5), (chunk, (o - o_ref).abs().max())
        assert torch.allclose(s, s_ref, atol=1e-5), (chunk, (s - s_ref).abs().max())


def test_chunk_gradients():
    q, k, v, g, beta = (t.double().requires_grad_() for t in inputs(T=20))

    def grads(o):
        return torch.autograd.grad(o.pow(2).sum(), (q, k, v, g, beta))
    ref = grads(recurrent_gated_delta_rule(q, k, v, g, beta)[0])
    for a, b in zip(grads(chunk_gated_delta_rule(q, k, v, g, beta, chunk=8)[0]), ref):
        assert torch.allclose(a.float(), b.float(), atol=1e-4)


def test_delta_rule_recall():
    """beta = 1, no decay, orthonormal keys: the state returns each written value exactly."""
    B, H, T, d = 1, 1, 6, 8
    k = torch.eye(d)[:T][None, None]
    v = torch.randn(B, H, T, d)
    o, S = chunk_gated_delta_rule(k, k, v, torch.zeros(B, H, T), torch.ones(B, H, T), chunk=4)
    assert torch.allclose(o, v, atol=1e-5)
    # writing a new value for the first key replaces the old one (delta rule), it is not added to it
    v2 = torch.randn(B, H, 1, d)
    o2, _ = chunk_gated_delta_rule(k[:, :, :1], k[:, :, :1], v2, torch.zeros(B, H, 1), torch.ones(B, H, 1), state=S)
    assert torch.allclose(o2, v2, atol=1e-5)


def test_layer_is_causal():
    m = tiny(linear_attn='hybrid4')
    layer = m.transformer.h[0].attn
    assert isinstance(layer, GatedDeltaNet) and not isinstance(m.transformer.h[1].attn, GatedDeltaNet)
    x = torch.randn(2, 20, 32)
    y = layer(x)
    x2 = x.clone()
    x2[:, 12:] = torch.randn(2, 8, 32)
    assert torch.allclose(layer(x2)[:, :12], y[:, :12], atol=1e-5)
    assert not torch.allclose(layer(x2)[:, 12:], y[:, 12:])


def test_model_forward_backward():
    m = tiny(linear_attn='0,1', block_size=128).train()
    streams, _, targets = batch(T=100)  # > one chunk
    parts, loss = m(*streams, targets=targets)
    loss.backward()
    layer = m.transformer.h[0].attn
    for p in (layer.c_attn.weight, layer.conv.weight, layer.ab_proj.weight, layer.A_log, layer.dt_bias,
              layer.out_gate.weight, layer.o_norm.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_conversion_and_transfer():
    """An attention model's weights load into the hybrid (c_attn / c_proj carry over, the rest is new); with the
    teachers attached the trunk computes exactly the attention model's states in training, the loss is the transfer
    MSE and only the linear layers get gradients; in eval the model runs its own linear layers."""
    base = tiny(pos_emb='rope', n_layer=3)
    saved = {k: getattr(base.config, k) for k in ARCH_KEYS}
    args = merge_model_args(saved, {**saved, 'linear_attn': 'hybrid4'}, True, 64)
    assert args['linear_attn'] == 'hybrid4'
    assert merge_model_args(saved, {**saved, 'linear_attn': 'hybrid4'}, False, 64)['linear_attn'] == ''  # a resume keeps the checkpoint's layers
    conv = tiny(pos_emb='rope', n_layer=3, linear_attn='hybrid4')
    new = load_expanded(conv, {k: v.clone() for k, v in base.state_dict().items()})
    assert all(k.startswith(('transformer.h.0.attn.', 'transformer.h.1.attn.')) for k in new), new
    assert not any(k.endswith(('c_attn.weight', 'c_proj.weight')) for k in new)
    assert torch.equal(conv.transformer.h[0].attn.c_attn.weight, base.transformer.h[0].attn.c_attn.weight)

    streams, _, targets = batch(T=24)
    with torch.no_grad():
        h_base = base.encode(*streams)
        h_lin = conv.encode(*streams)
    assert not torch.allclose(h_base, h_lin, atol=1e-3)
    keep = conv.attach_attn_teachers()
    assert keep == ['transformer.h.0.attn.', 'transformer.h.1.attn.']
    conv.train()
    with torch.no_grad():
        assert torch.allclose(conv.encode(*streams), h_base, atol=1e-5)
    for n, p in conv.named_parameters():
        p.requires_grad_(n.startswith(tuple(keep)))
    parts, loss = conv(*streams, targets=targets)
    assert torch.equal(loss, parts['transfer']) and loss > 0
    loss.backward()
    assert conv.transformer.h[0].attn.ab_proj.weight.grad is not None
    assert not any('teacher' in n for n, _ in conv.named_parameters())
    assert not any('teacher' in k for k in conv.state_dict())
    conv.eval()
    with torch.no_grad():
        assert torch.allclose(conv.encode(*streams), h_lin, atol=1e-5)


def test_transfer_reduces_mse():
    """A few AdamW steps on the transfer loss bring the linear layer's output closer to the attention's."""
    base = tiny(pos_emb='rope', n_layer=2)
    conv = tiny(pos_emb='rope', n_layer=2, linear_attn='0')
    load_expanded(conv, {k: v.clone() for k, v in base.state_dict().items()})
    keep = conv.attach_attn_teachers()
    params = [p for n, p in conv.named_parameters() if n.startswith(tuple(keep))]
    opt = torch.optim.AdamW(params, lr=3e-3)
    conv.train()
    losses = []
    for step in range(60):
        streams, _, targets = batch(T=32, seed=step)
        _, loss = conv(*streams, targets=targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) < 0.7 * sum(losses[:10]), (losses[:3], losses[-3:])


def test_layer_cache_prefill_step_chunk():
    """Generation state of one layer: prefill + one-note steps + an appended chunk = the full sequence; a prefill and
    a step at slot 0 start from zero (the junk of a CUDA-graph warm-up does not leak in)."""
    layer = tiny(linear_attn='0', block_size=64).transformer.h[0].attn
    x = torch.randn(2, 30, 32)
    with torch.no_grad():
        full = layer(x)
        cache = KVCache(1, 2, 2, 64, 16, 'cpu', torch.float32)
        cache.pos.fill_(5)
        layer(x[:, :7], cache)                       # junk state
        parts = [layer(x[:, :20], cache)]            # prefill resets
        for t in range(20, 25):
            cache.pos.fill_(t)
            parts.append(layer(x[:, t:t + 1], cache))
        cache.chunk_start = 25
        parts.append(layer(x[:, 25:], cache))
        cache.chunk_start = None
        assert torch.allclose(torch.cat(parts, 1), full, atol=1e-4), (torch.cat(parts, 1) - full).abs().max()
        cache.pos.fill_(0)
        assert torch.allclose(layer(x[:, :1], cache), full[:, :1], atol=1e-5)   # step at slot 0 starts fresh


def test_model_cached_decode_matches_forward():
    m = tiny(linear_attn='hybrid4', block_size=64, n_layer=4)
    streams, _, _ = batch(B=2, T=24)
    with torch.no_grad():
        full = m.encode(*streams)
        cache = KVCache(4, 2, m.config.n_head, 64, m.config.n_embd // m.config.n_head, 'cpu', torch.float32)
        hs = [m.encode(*(s[:, :16] for s in streams), cache=cache)]
        for t in range(16, 24):
            cache.pos.fill_(t)
            hs.append(m.encode(*(s[:, t:t + 1] for s in streams), cache=cache))
    assert torch.allclose(torch.cat(hs, 1), full, atol=1e-4), (torch.cat(hs, 1) - full).abs().max()


def test_cuda_graph_decode_matches_eager():
    """generate() with the one-note step captured in a CUDA graph gives the eager notes (cache refills included)."""
    if not torch.cuda.is_available():
        print('  (skipped: needs CUDA)')
        return
    from musicar.generation.sampling import generate
    m = tiny(linear_attn='hybrid2', n_layer=4, pos_emb='rope', block_size=48).cuda()
    start = [torch.full((2, 1), 60, device='cuda')] + [torch.zeros(2, 1, dtype=torch.long, device='cuda')] * 3
    out = []
    for graph in (False, True):
        torch.manual_seed(1)
        out.append(generate(m, *start, max_new_tokens=120, top_k=1, program=None, cuda_graph=graph))
    assert all(torch.equal(a, b) for a, b in zip(*out))


def test_linear_lr_mult_groups():
    from musicar.training.optim import build_adamw, set_lr
    m = tiny(linear_attn='0', n_layer=2)
    opt = build_adamw(m, 0.1, 1e-4, (0.9, 0.95), 'cpu', linear_lr_mult=10.0)
    set_lr(opt, 1e-4)
    lin = {id(p) for p in m.transformer.h[0].attn.parameters()}
    for g in opt.param_groups:
        if g['params']:
            want = 1e-3 if id(g['params'][0]) in lin else 1e-4
            assert all((id(p) in lin) == (want == 1e-3) for p in g['params']) and abs(g['lr'] - want) < 1e-12
    assert sum(p.numel() for g in opt.param_groups for p in g['params']) == \
        sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_fla_kernel_matches():
    """The fused kernel (MUSICAR_GDN_KERNEL=fla) gives the PyTorch path's outputs and gradients (CUDA + fla-core only)."""
    import importlib.util
    import musicar.model.linear_attn as la
    if not torch.cuda.is_available() or importlib.util.find_spec('fla') is None:
        print('  (skipped: needs CUDA and fla-core)')
        return
    torch.manual_seed(0)
    layer = tiny(linear_attn='0', n_embd=128, n_head=2).transformer.h[0].attn.cuda()
    x = torch.randn(2, 150, 128, device='cuda')
    outs = []
    for kernel in ('torch', 'fla'):
        la.KERNEL = kernel
        layer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = layer.mix(x)
        y.float().pow(2).sum().backward()
        outs.append((y.float(), layer.c_attn.weight.grad.clone(), layer.A_log.grad.clone()))
    la.KERNEL = 'torch'
    for a, b in zip(*outs):
        assert (a - b).norm() / b.norm() < 0.03, (a - b).norm() / b.norm()


if __name__ == '__main__':
    run(globals())
