"""Model checks (CPU, tiny models): python tests/model/test_gpt.py  (or pytest tests/)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import batch, run, tiny  # noqa: E402
from musicar.model.gpt import KVCache, apply_rope, rope_tables  # noqa: E402

VARIANTS = [dict(), dict(pos_emb='rope'), dict(pos_emb='rope', norm='rmsnorm', mlp='swiglu', qk_norm=True)]


def test_forward_parts_and_loss():
    for kw in VARIANTS:
        m = tiny(**kw)
        streams, _, targets = batch()
        parts, loss = m(*streams, targets=targets)
        assert list(parts) == ['pitch', 'velocity', 'duration', 'delta_time'], list(parts)
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
        loss.backward()
        assert all(p.grad is not None for p in m.parameters())


def test_programs_head():
    m = tiny(n_programs=129)
    streams, program, targets = batch(n_programs=129)
    parts, _ = m(*streams, program=program, targets=targets)
    assert m.head_names[-1] == 'program' and 'program' in parts
    assert m.cascade.order == ('delta_time', 'program', 'pitch', 'duration', 'velocity')


def test_masked_targets_give_zero_not_nan():
    m = tiny()
    streams, _, targets = batch()
    targets[2] = torch.full_like(targets[2], -1)  # e.g. the duration head of a drums-only window
    parts, loss = m(*streams, targets=targets)
    assert parts['duration'] == 0 and torch.isfinite(loss)


def test_label_smoothing_only_in_loss():
    streams, _, targets = batch()
    a, b = tiny(label_smoothing=0.0), tiny(label_smoothing=0.3)
    with torch.no_grad():
        pa, la = a(*streams, targets=targets)
        pb, lb = b(*streams, targets=targets)
    assert all(torch.allclose(pa[k], pb[k]) for k in pa), 'parts are the true CE'
    assert torch.allclose(la, sum(pa.values())) and not torch.allclose(la, lb)


def test_causal():
    for kw in VARIANTS:
        m = tiny(**kw)
        streams, _, _ = batch(T=12)
        changed = [s.clone() for s in streams]
        changed[0][:, 8:] = (changed[0][:, 8:] + 7) % 128
        with torch.no_grad():
            h, h2 = m.encode(*streams), m.encode(*changed)
        assert torch.allclose(h[:, :8], h2[:, :8], atol=1e-6) and not torch.allclose(h[:, 8:], h2[:, 8:])


def test_kv_cache_matches_full_forward():
    for kw in VARIANTS:
        m = tiny(**kw)
        streams, _, _ = batch(B=2, T=20)
        with torch.no_grad():
            full = m.encode(*streams)
            for n0 in (1, 5):
                cache = KVCache(2, 2, 2, 20, 16, 'cpu', torch.float32)
                h = m.encode(*(s[:, :n0] for s in streams), cache=cache)
                assert torch.allclose(h, full[:, :n0], atol=1e-5), (kw, n0)
                for i in range(n0, 20):
                    cache.pos.fill_(i)
                    h = m.encode(*(s[:, i:i + 1] for s in streams), cache=cache)
                    assert torch.allclose(h[:, 0], full[:, i], atol=1e-5), (kw, n0, i)


def test_rope_is_relative():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(pq, pk):
        cq, sq = rope_tables(torch.tensor([pq]), 16, 10000.0)
        ck, sk = rope_tables(torch.tensor([pk]), 16, 10000.0)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum()
    assert torch.allclose(score(3, 1), score(40, 38), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)
    cos, sin = rope_tables(torch.tensor([0]), 16, 10000.0)
    assert torch.allclose(apply_rope(q, cos, sin), q), 'position 0 = no rotation'


def test_zero_init_conditioning():
    """A fresh style / instrument-set / density input changes nothing until it is trained."""
    streams, program, _ = batch(n_programs=129)
    plain = tiny(n_programs=129)
    m = tiny(n_programs=129, n_styles=5, cond_inst=True, n_density=16)
    m.load_state_dict(plain.state_dict(), strict=False)
    cond = (torch.ones(3, 129), torch.tensor([1, 5, 15]))
    with torch.no_grad():
        h0 = plain.encode(*streams, program=program)
        h1 = m.encode(*streams, program=program, style=torch.tensor([1, 2, 4]), cond=cond)
    assert torch.allclose(h0, h1, atol=1e-6)


def test_style_changes_output_once_trained():
    m = tiny(n_styles=5)
    torch.nn.init.normal_(m.transformer['style'].weight)
    streams, _, _ = batch()
    with torch.no_grad():
        h0 = m.encode(*streams, style=torch.zeros(3, dtype=torch.long))
        h1 = m.encode(*streams, style=torch.tensor([1, 2, 3]))
    assert not torch.allclose(h0, h1)


def test_crop_block_size():
    m = tiny()
    m.crop_block_size(32)
    assert m.config.block_size == 32 and m.transformer.wpe.weight.shape[0] == 32
    streams, _, _ = batch(T=32)
    assert m.encode(*streams).shape == (3, 32, 32)


def test_activation_checkpointing_same_grads():
    streams, _, targets = batch(T=16)
    grads = []
    for act_ckpt, save in ((0, ''), (-1, ''), (-1, 'attn,mm')):
        m = tiny(pos_emb='rope').train()
        m.act_ckpt, m.act_ckpt_save = act_ckpt, save
        _, loss = m(*streams, targets=targets)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6) and torch.allclose(grads[0], grads[2], atol=1e-6)


if __name__ == '__main__':
    run(globals())
