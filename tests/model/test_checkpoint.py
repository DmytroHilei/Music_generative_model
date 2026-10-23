"""Checkpoint checks (CPU, tiny models): python tests/model/test_checkpoint.py  (or pytest tests/)."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import TINY, batch, run, tiny  # noqa: E402
from musicar.model.checkpoint import (LEGACY_DEFAULTS, load_expanded, load_model, merge_model_args,  # noqa: E402
                                      resolve_checkpoint, save_atomic, strip_compile_prefix)

WANTED = {**{k: v for k, v in TINY.items() if k != 'dropout'}, 'bias': False, 'pitch_size': 130, 'velocity_size': 32,
          'duration_size': 512, 'delta_time_size': 512, **LEGACY_DEFAULTS}


def test_grow_pitch_vocab_keeps_old_rows():
    old, new = tiny(), tiny(pitch_size=130)
    grown = load_expanded(new, old.state_dict())
    assert 'transformer.music_embeddings.pitch.weight' in grown and 'cascade.heads.pitch.out.weight' in grown
    w_old, w_new = old.transformer.music_embeddings.pitch.weight, new.transformer.music_embeddings.pitch.weight
    assert torch.equal(w_new[:128], w_old)
    streams, _, targets = batch()
    with torch.no_grad():
        p_old, _ = old(*streams, targets=targets)
        p_new, _ = new(*streams, targets=targets)
    for k in ('velocity', 'duration', 'delta_time'):  # only the pitch head's vocabulary changed
        assert torch.allclose(p_old[k], p_new[k], atol=1e-5), k


def test_grow_to_programs_is_function_preserving():
    old, new = tiny(), tiny(n_programs=129)
    load_expanded(new, old.state_dict())
    assert torch.all(new.cascade.heads['program'].out.weight == 0)
    streams, program, targets = batch(n_programs=129)
    with torch.no_grad():
        p_old, _ = old(*streams, targets=targets[:4])
        p_new, _ = new(*streams, program=program, targets=targets)
    for k in p_old:
        assert torch.allclose(p_old[k], p_new[k], atol=1e-5), k
    assert torch.allclose(p_new['program'], torch.log(torch.tensor(129.0)), atol=1e-4), 'uniform instrument guess'


def test_learned_positions_stretch_and_rope_drop():
    old = tiny()
    long = tiny(block_size=128)
    changed = load_expanded(long, old.state_dict())
    assert any('stretched 64 -> 128' in c for c in changed), changed
    w_old, w_new = old.transformer.wpe.weight, long.transformer.wpe.weight
    assert torch.allclose(w_new[0], w_old[0]) and torch.allclose(w_new[-1], w_old[-1])
    rope = tiny(pos_emb='rope')
    assert 'transformer.wpe.weight (dropped)' in load_expanded(rope, old.state_dict())


def test_gqa_conversion_mean_pools_kv_heads():
    """A finetune into fewer KV heads mean-pools each group: an MHA checkpoint whose heads are equal within groups
    converts exactly; any other keeps the query rows and the averaged K / V rows."""
    g = tiny(n_head=4, n_kv_head=2, pos_emb='rope')
    mha_state = {}
    for k, v in g.state_dict().items():
        if k.endswith('attn.c_attn.weight'):
            q, kk, vv = v.split([32, 16, 16])
            rep = lambda x: x.view(2, 1, 8, 32).expand(2, 2, 8, 32).reshape(32, 32)  # noqa: E731
            v = torch.cat([q, rep(kk), rep(vv)])
        mha_state[k] = v.clone()
    new = tiny(n_head=4, n_kv_head=2, pos_emb='rope', n_layer=2)
    changed = load_expanded(new, mha_state)
    assert sum('mean-pooled to 2 heads' in c for c in changed) == 2, changed
    assert all(torch.allclose(a, b) for a, b in zip(new.state_dict().values(), g.state_dict().values()))
    mha = tiny(n_head=4, pos_emb='rope')
    w = mha.transformer.h[0].attn.c_attn.weight
    new = tiny(n_head=4, n_kv_head=1, pos_emb='rope')
    load_expanded(new, mha.state_dict())
    w1 = new.transformer.h[0].attn.c_attn.weight
    assert torch.equal(w1[:32], w[:32])
    assert torch.allclose(w1[32:40], w[32:64].view(4, 8, 32).mean(0)) and torch.allclose(w1[40:], w[64:].view(4, 8, 32).mean(0))


def test_merge_model_args_gqa():
    saved = {k: v for k, v in WANTED.items() if k not in LEGACY_DEFAULTS}
    assert merge_model_args(saved, {**WANTED, 'n_kv_head': 1}, True, 64)['n_kv_head'] == 1, 'finetune: pool'
    assert merge_model_args(saved, {**WANTED, 'n_kv_head': 1}, False, 64)['n_kv_head'] == 0, 'resume: the checkpoint'
    assert merge_model_args({**saved, 'n_kv_head': 1}, {**WANTED, 'n_kv_head': 0}, True, 64)['n_kv_head'] == 1, \
        'KV heads are never grown back'


def test_merge_model_args_adds_remain():
    saved = {k: v for k, v in WANTED.items() if k not in LEGACY_DEFAULTS}
    assert merge_model_args(saved, {**WANTED, 'n_remain': 33}, True, 64)['n_remain'] == 33, 'finetune: added'
    assert merge_model_args(saved, {**WANTED, 'n_remain': 33}, False, 64)['n_remain'] == 0, 'resume: the checkpoint'


def test_merge_model_args():
    saved = {k: v for k, v in WANTED.items() if k not in LEGACY_DEFAULTS}
    saved['pitch_size'] = 128
    resume = merge_model_args(saved, {**WANTED, 'n_styles': 10}, False, 256)
    assert resume['pitch_size'] == 128 and resume['n_styles'] == 0 and resume['block_size'] == 64
    assert resume['pos_emb'] == 'learned', 'a checkpoint without pos_emb used learned positions'
    ft = merge_model_args(saved, {**WANTED, 'n_styles': 10}, True, 256)
    assert ft['pitch_size'] == 130 and ft['n_styles'] == 10 and ft['block_size'] == 256
    assert merge_model_args(saved, WANTED, True, 32)['block_size'] == 64, 'learned positions never shrink'
    rope = merge_model_args(saved, {**WANTED, 'pos_emb': 'rope'}, True, 32)
    assert rope['pos_emb'] == 'rope' and rope['block_size'] == 32
    assert merge_model_args({**saved, 'n_embd': 64}, WANTED, True, 64)['n_embd'] == 64, 'the shape is the checkpoint'


def test_save_resolve_load():
    m = tiny(pos_emb='rope', n_programs=129)
    with tempfile.TemporaryDirectory() as d:
        state = {'_orig_mod.' + k: v for k, v in m.state_dict().items()}  # saved from a compiled model
        args = {k: getattr(m.config, k) for k in WANTED}
        save_atomic({'model': state, 'model_args': args, 'iter_num': 7}, f'{d}/run/ckpt.pt')
        assert not Path(f'{d}/run/ckpt.pt.tmp').exists()
        assert resolve_checkpoint(f'{d}/run') == Path(f'{d}/run/ckpt.pt')
        save_atomic({'model': m.state_dict(), 'model_args': args}, f'{d}/run/best.pt')
        assert resolve_checkpoint(f'{d}/run').name == 'best.pt', 'best.pt before ckpt.pt'
        loaded, it = load_model(f'{d}/run/ckpt.pt')
        assert it == 7 and not loaded.training and loaded.config.dropout == 0.0
        assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), m.state_dict().values()))
        try:
            resolve_checkpoint(d)
            raise AssertionError('a dir without checkpoints must raise')
        except FileNotFoundError:
            pass
    assert list(strip_compile_prefix({'_orig_mod.a': 1, 'b': 2})) == ['b', 'a']


if __name__ == '__main__':
    run(globals())
