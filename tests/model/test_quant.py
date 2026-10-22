"""Fake quantization (CPU, tiny models): python tests/model/test_quant.py  (or pytest tests/)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import batch, tiny  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.model.quant import fake_quantize, parse, quantize_dynamic, quantize_symmetric  # noqa: E402


def test_parse():
    assert parse('w4g32-a8-kv8') == dict(w=4, group=32, a=8, kv=8)
    assert parse('w8') == dict(w=8, group=0, a=0, kv=0)
    assert parse('none') == parse('') == dict(w=0, group=0, a=0, kv=0)
    try:
        parse('w4x')
    except ValueError:
        pass
    else:
        raise AssertionError('bad spec accepted')


def test_levels():
    w = torch.randn(6, 64)
    q = quantize_symmetric(w, 4, 32)
    for g in q.reshape(6, 2, 32):  # per group: at most 16 distinct values on a uniform grid through 0
        for row in g:
            assert len(row.unique()) <= 16
    scale = w.reshape(6, 2, 32).abs().amax(-1, keepdim=True) / 7
    assert torch.allclose(q.reshape(6, 2, 32) / scale, (q.reshape(6, 2, 32) / scale).round(), atol=1e-4)
    assert (q - w).abs().max() <= scale.max() / 2 + 1e-6
    x = torch.randn(3, 5, 40)
    xq = quantize_dynamic(x)
    assert all(len(r.unique()) <= 256 for r in xq.reshape(-1, 40))
    assert (xq - x).abs().max() < (x.amax(-1) - x.amin(-1)).max() / 255


def test_fake_quantize_model():
    streams, _, _ = batch(B=2, T=12)
    for kw in (dict(), dict(pos_emb='rope', norm='rmsnorm', mlp='swiglu', qk_norm=True)):
        ref = tiny(**kw)
        with torch.no_grad():
            h0 = ref.encode(*streams)
        for spec, tol in (('w8-a8-kv8', 0.15), ('w4g16-a8-kv8', 1.0)):
            m = tiny(**kw)
            print(fake_quantize(m, spec))
            with torch.no_grad():
                h = m.encode(*streams)
            err = (h - h0).abs().mean() / h0.abs().mean()
            assert 0 < err < tol, (kw, spec, float(err))
        m = tiny(**kw)
        assert fake_quantize(m, 'none') == 'none'
        with torch.no_grad():
            assert torch.equal(m.encode(*streams), h0)
        # generation with the KV hook and the cache: greedy steps = full forwards of the quantized model
        m = tiny(**kw)
        fake_quantize(m, 'w8-a8-kv8')
        out = generate(m, *streams, max_new_tokens=10, top_k=1, cuda_graph=False)
        assert out[0].shape == (2, 22)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
