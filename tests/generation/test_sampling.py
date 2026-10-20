"""Sampling checks (CPU, tiny untrained models): python tests/generation/test_sampling.py  (or pytest tests/)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import batch, run, tiny  # noqa: E402
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM, EOS_PITCH  # noqa: E402
from musicar.generation.sampling import generate, prompt_inputs, with_bos  # noqa: E402
from musicar.model.gpt import STREAM_ORDER  # noqa: E402

KW = dict(cuda_graph=False)


def greedy_reference(m, streams, n_new):
    """Greedy decoding with a full forward per note (no cache): the ground truth for top_k=1."""
    out = [s.clone() for s in streams]
    for _ in range(n_new):
        with torch.no_grad():
            h = m.encode(*out)[:, [-1]]
            nxt = m.cascade.sample(h, lambda logits, name: logits[:, -1].argmax(-1, keepdim=True))
        out = [torch.cat([o, nxt[name]], 1) for o, name in zip(out, STREAM_ORDER)]
    return out


def test_greedy_matches_full_forward():
    for kw in (dict(), dict(pos_emb='rope', norm='rmsnorm', mlp='swiglu', qk_norm=True)):
        m = tiny(**kw)
        streams, _, _ = batch(B=2, T=5)
        got = generate(m, *streams, max_new_tokens=30, top_k=1, **KW)
        want = greedy_reference(m, streams, 30)
        assert all(torch.equal(a, b) for a, b in zip(got, want)), kw


def test_past_block_size_refills():
    m = tiny(block_size=32)
    streams, _, _ = batch(B=2, T=4)
    out = generate(m, *streams, max_new_tokens=80, slide=8, anchor=2, **KW)
    assert all(o.shape == (2, 84) for o in out)
    assert all(torch.equal(o[:, :4], s) for o, s in zip(out, streams)), 'the prompt is kept'


def test_special_tokens():
    m = tiny(pitch_size=130)
    with torch.no_grad():  # make BOS and EOS the likeliest pitches
        m.cascade.heads['pitch'].out.weight[BOS_PITCH:] = 0
        m.cascade.heads['pitch'].out.bias = None
        m.cascade.heads['pitch'].out.weight[BOS_PITCH:] += 3 * m.cascade.heads['pitch'].out.weight[:BOS_PITCH].std()
    start = [torch.full((4, 1), BOS_PITCH)] + [torch.zeros(4, 1, dtype=torch.long)] * 3
    torch.manual_seed(0)
    pitch = generate(m, *start, max_new_tokens=40, min_new=10, temperature=1.0, **KW)[0][:, 1:]
    assert not (pitch == BOS_PITCH).any(), 'BOS is never sampled'
    assert not (pitch[:, :10] == EOS_PITCH).any(), 'no EOS before min_new'


def test_programs():
    m = tiny(n_programs=129)
    streams, program, _ = batch(B=2, T=4, n_programs=129)
    out = generate(m, *streams, max_new_tokens=20, program=33, programs=program, return_programs=True, **KW)
    assert torch.all(out[4][:, 4:] == 33) and torch.equal(out[4][:, :4], program)
    allowed = [0, 33, DRUM_PROGRAM]
    out = generate(m, *streams, max_new_tokens=40, program=None, programs=program, allowed_programs=allowed,
                   return_programs=True, temperature=1.0, **KW)
    assert set(out[4][:, 4:].unique().tolist()) <= set(allowed)


def test_drum_durations_zero():
    # drum notes are trained with duration 0 as input: sampled drums and prompt drums both get it
    m = tiny(n_programs=129)
    streams, program, _ = batch(B=2, T=4, n_programs=129)
    program[:, 1] = DRUM_PROGRAM
    streams[2][:, 1] = 7
    torch.manual_seed(0)
    out = generate(m, *streams, max_new_tokens=60, program=None, programs=program, allowed_programs=[0, DRUM_PROGRAM],
                   return_programs=True, temperature=1.0, **KW)
    drums = out[4] == DRUM_PROGRAM
    assert drums[:, 4:].any() and (~drums[:, 4:]).any()
    assert torch.all(out[2][drums] == 0) and torch.all(out[2][:, 1] == 0)
    assert (out[2][:, 4:][~drums[:, 4:]] != 0).any(), 'other instruments keep their sampled durations'


def test_dt_bias_and_density_target():
    m = tiny()
    streams, _, _ = batch(B=4, T=4)
    torch.manual_seed(0)
    slow = generate(m, *streams, max_new_tokens=60, dt_bias=3.0, **KW)[3][:, 4:].float().mean()
    torch.manual_seed(0)
    fast = generate(m, *streams, max_new_tokens=60, dt_bias=-1.0, **KW)[3][:, 4:].float().mean()
    assert slow > 2 * fast, (slow, fast)
    torch.manual_seed(0)
    out = generate(m, *streams, max_new_tokens=200, target_nps=[2.0, 2.0, 50.0, 50.0], **KW)[3][:, -64:]
    nps = 64 / (out.sum(1).float() * 0.02).clamp(min=1e-3)
    assert nps[:2].max() < nps[2:].min(), nps


def test_style_guidance():
    m = tiny(n_styles=4)
    torch.nn.init.normal_(m.transformer['style'].weight)
    streams, _, _ = batch(B=2, T=4)
    torch.manual_seed(0)
    a = generate(m, *streams, max_new_tokens=20, style=2, **KW)
    torch.manual_seed(0)
    b = generate(m, *streams, max_new_tokens=20, style=2, cfg_scale=1.0, **KW)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), 'cfg_scale 1 = plain conditional sampling'
    g = generate(m, *streams, max_new_tokens=20, style=2, cfg_scale=3.0, top_p=0.9, **KW)
    assert g[0].shape == (2, 24)


def test_prompt_helpers():
    streams = tuple(torch.ones(2, 3, dtype=torch.long) for _ in range(4))
    s, p = with_bos(streams, torch.full((2, 3), 33))
    assert s[0][:, 0].tolist() == [BOS_PITCH] * 2 and all(t[:, 0].sum() == 0 for t in s[1:]) and p[:, 0].sum() == 0
    tokens = torch.ones(2, 3, 4, dtype=torch.long)
    s, p, cond = prompt_inputs(tiny(), tokens, None)
    assert s[0].shape == (2, 3) and cond is None, 'no BOS for a model without special tokens'
    s, p, cond = prompt_inputs(tiny(pitch_size=130, n_programs=129, cond_inst=True), tokens,
                               torch.full((2, 3), 33), band=True)
    assert s[0].shape == (2, 4) and cond[0][:, 33].tolist() == [1.0, 1.0] and cond[1].sum() == 0


if __name__ == '__main__':
    run(globals())
