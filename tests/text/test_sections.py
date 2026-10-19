"""Section captions (CPU, tiny models, synthetic stores): python tests/text/test_sections.py  (or pytest tests/)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.data.loader import NoteWindowDataset, collate_windows  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.model.checkpoint import merge_model_args  # noqa: E402
from musicar.model.gpt import KVCache, section_layout, section_query_index  # noqa: E402
from musicar.text.captions import section_attributes, section_caption  # noqa: E402
from test_text import TINY, batch, make_store, tiny  # noqa: E402

D = 8  # text_dim of the tiny models


def section_model(mode, **kw):
    m = tiny(text_dim=D, text_prefix=4, section_len=8, section_slots=2, section_mode=mode, **kw)
    torch.nn.init.normal_(m.transformer['section_prefix'].gate)
    torch.nn.init.normal_(m.transformer['section_prefix'].null)
    for xa in m.transformer['section_xattn'].values() if mode == 'xattn' else ():
        torch.nn.init.constant_(xa.gate, 0.5)
    return m


def test_section_layout_by_hand():
    # 4 notes, prefix 1, 2 slots per caption; captions before note 0 and note 2, one unused (pos > t)
    dest, read = section_layout(torch.tensor([[0, 2, 5]]), 4, 1, 2)
    # sequence: P S0 S0 n0 n1 S1 S1 n2 n3 (unused U U at 9, 10)
    assert dest.tolist() == [[0, 3, 4, 7, 8, 1, 2, 5, 6, 9, 10]]
    assert read.tolist() == [[3, 6, 7, 8]]  # n0, the last S1 vector (before n2), n2, n3
    # a caption right after the last note: note 3's prediction reads it
    dest, read = section_layout(torch.tensor([[0, 4]]), 4, 0, 1)
    assert read.tolist() == [[1, 2, 3, 5]]
    q = section_query_index(torch.tensor([[0, 2, 9]]), 1 + 4, 1)
    assert q.tolist() == [[0, 0, 1, 1, 1]]  # prefix, n0 (predicts n1), n1 (predicts n2: caption 1), ...


def sections_input(B=3, E=3, T=16):
    g = torch.Generator().manual_seed(3)
    states = torch.randn(B, E, 5, D, generator=g)
    states[1, 1] = 0  # a null caption
    pos = torch.tensor([[0, 8, T + 1], [0, 3, 11], [0, 16, T + 1]])
    return states, pos


def test_section_forward_both_modes():
    streams, program, targets = batch()
    states, pos = sections_input()
    text = torch.randn(3, 5, D)
    for mode in ('prefix', 'xattn'):
        m = section_model(mode)
        parts, loss = m(*streams, program=program, targets=targets, text=text, sections=(states, pos), late_from=8)
        assert 'late_pitch' in parts and torch.isfinite(loss)
        loss.backward()
        assert m.transformer['section_prefix'].queries.grad.abs().sum() > 0, mode
        with torch.no_grad():
            vecs = m.section_vectors(states)
            h = m.encode(*streams, program=program, text=text, sections=(vecs, pos))
            other = states.clone()
            other[0, 1] = torch.randn(5, D)  # caption of row 0's second section (from note 8 on)
            h2 = m.encode(*streams, program=program, text=text, sections=(m.section_vectors(other), pos))
        assert h.shape == (3, 16, 32)
        # causal: predictions before the caption arrives (outputs of notes 0..6 predict notes 1..7) don't change
        assert torch.allclose(h[0, :7], h2[0, :7], atol=1e-5), mode
        assert not torch.allclose(h[0, 7:], h2[0, 7:]), mode
        assert torch.allclose(h[1:], h2[1:], atol=1e-5)


def test_xattn_zero_gate_is_function_preserving():
    m = tiny(text_dim=D, text_prefix=4, section_len=8, section_slots=2, section_mode='xattn')
    streams, program, _ = batch()
    states, pos = sections_input()
    text = torch.randn(3, 5, D)
    with torch.no_grad():
        a = m.encode(*streams, program=program, text=text, sections=(m.section_vectors(states), pos))
        b = m.encode(*streams, program=program, text=text)
    assert torch.allclose(a, b, atol=1e-6)


def test_section_cache_matches_forward():
    """Prefill + one-note steps (+ appended section vectors for 'prefix') = the teacher-forced forward."""
    T, S = 20, 8
    streams, program, _ = batch(B=2, T=T)
    text = torch.randn(2, 5, D)
    states = torch.randn(2, 3, 5, D)
    pos = torch.tensor([[0, 8, 16]] * 2)  # section starts, lead 0
    for mode in ('prefix', 'xattn'):
        m = section_model(mode)
        with torch.no_grad():
            vecs = m.section_vectors(states)
            full = m.encode(*streams, program=program, text=text, sections=(vecs, pos))
            for n0 in (1, 5, 8):
                cache = KVCache(2, 2, 2, 4 + T + 3 * 2, 16, 'cpu', torch.float32)
                n_ev = int((pos[0] <= n0).sum())
                h = m.encode(*(t[:, :n0] for t in streams), program=program[:, :n0], cache=cache, text=text,
                             step=False, sections=(vecs[:, :n_ev], pos[:, :n_ev]))
                assert torch.allclose(h, full[:, :n0], atol=1e-5), (mode, n0)
                extra = 2 * n_ev if mode == 'prefix' else 0
                for i in range(n0, T - 1):
                    cache.pos.fill_(4 + i + extra)
                    ev = int((pos[0] <= i + 1).sum()) - 1  # the caption of the note being predicted
                    h = m.encode(*(t[:, i:i + 1] for t in streams), program=program[:, i:i + 1], cache=cache,
                                 sections=(vecs[:, ev:ev + 1], None))
                    if mode == 'prefix' and (i + 1) in pos[0].tolist():
                        h = m.encode_chunk(vecs[:, ev], cache, 4 + i + 1 + extra)
                        extra += 2
                    assert torch.allclose(h[:, 0], full[:, i], atol=1e-4), (mode, n0, i)


def test_generate_with_sections():
    start = [torch.full((2, 1), 128)] + [torch.zeros(2, 1, dtype=torch.long)] * 3
    kw = dict(program=None, cuda_graph=False, return_programs=True)
    for mode in ('prefix', 'xattn'):
        for lead in (0, 4):
            m = section_model(mode, section_lead=lead)
            torch.manual_seed(5)
            plain = generate(m, *start, max_new_tokens=30, **kw)
            torch.manual_seed(5)
            zero = generate(m, *start, max_new_tokens=30, sections=torch.zeros(2, 5, D), **kw)
            assert all(torch.equal(a, b) for a, b in zip(plain, zero)), (mode, lead, 'zero = null captions')
            secs = torch.randn(3, 5, D)
            torch.manual_seed(5)
            out = generate(m, *start, max_new_tokens=150, text=torch.randn(5, D), sections=secs, cfg_scale=2.0,
                           slide=16, **kw)  # past block_size 64: refills with the captions in force
            assert out[0].shape == (2, 151) and (out[0][:, 1:] < 130).all()
            torch.manual_seed(5)
            changed = generate(m, *start, max_new_tokens=30, sections=torch.cat([secs[:1], -secs[1:]]), **kw)
            torch.manual_seed(5)
            same = generate(m, *start, max_new_tokens=30, sections=secs, **kw)
            first = 8 - lead  # the second caption comes before note 8 - lead (index in the output incl. BOS)
            assert torch.equal(changed[0][:, :first], same[0][:, :first]), (mode, lead)


def test_section_caption_from_notes():
    rng = np.random.default_rng(0)
    n = 64
    tok = np.stack([rng.integers(60, 72, n), np.full(n, 28), np.full(n, 10), np.full(n, 3)], 1)
    piano = section_attributes(tok, np.zeros(n, np.int64))
    assert piano['solo'] and piano['instruments'] == ['piano'] and piano['pace'] == 'high' and \
        piano['dynamics'] == 'high', piano
    drums = section_attributes(tok, np.full(n, 128))
    assert drums.get('drums') and 'instruments' not in drums
    band = section_attributes(tok, np.array([0, 33, 128, 25] * 16))
    assert set(band['instruments']) >= {'piano', 'drums'} and 'no_drums' not in band and 'solo' not in band
    import random
    caps = [section_caption(tok, np.zeros(n, np.int64), random.Random(i)) for i in range(20)]
    assert all(caps) and section_caption(tok[:0], np.zeros(0, np.int64)) == ''


def test_dataset_section_events():
    with tempfile.TemporaryDirectory() as d:
        root = make_store(Path(d), notes=80)  # 80 notes + BOS + EOS = 82 rows per file
        kw = dict(root_dir=d, block_size=32, special_tokens=True, programs=True, captions=str(root / 'labels'),
                  cache_dir=str(root / 'cache'), section_len=16)
        src = f"store:{root / 'cache' / 'toy'}"
        val = NoteWindowDataset(src, split='train', eval_stride=32, **kw)
        assert val.section_events == 4
        x, _ = val[1]  # file 0, window at row 32: sections 2 (in force), 3 at window index 16, 4 at 32
        caps, pos = x[-2], x[-1]
        assert pos.tolist() == [0, 16, 32, 33] and all(caps[:3]) and caps[3] == ''
        assert x[-2] == val[1][0][-2], 'validation section captions are fixed'
        lead = NoteWindowDataset(src, split='train', eval_stride=32, section_lead=4, **kw)
        assert lead[1][0][-1].tolist() == [0, 12, 28, 33]  # sections 3 / 4 arrive 4 rows early
        none = NoteWindowDataset(src, split='train', eval_stride=32, section_caption_mode='none', **kw)
        assert none[1][0][-2] == [''] * 4
        shuf = NoteWindowDataset(src, split='train', eval_stride=32, section_caption_mode='shuffled', **kw)
        assert shuf[1][0][-2] != caps
        train = NoteWindowDataset(src, split='train', augment=True, section_dropout=0.0, **kw)
        (streams, extra), y = collate_windows([train[i] for i in range(4)],
                                              extra_names=('program', 'text', 'sec_text', 'sec_pos'))
        assert len(extra['sec_text']) == 4 and all(len(c) == 4 for c in extra['sec_text'])
        assert extra['sec_pos'].shape == (4, 4) and (extra['sec_pos'][:, 0] == 0).all()


def test_merge_model_args_adds_sections():
    saved = {k: v for k, v in TINY.items() if k != 'dropout'}
    saved.update(text_dim=D, text_prefix=4, section_len=0, section_lead=0)  # a run after this change, no sections
    wanted = {**saved, 'section_len': 256, 'section_lead': 128, 'section_slots': 4, 'section_mode': 'xattn',
              'section_xattn_every': 2, 'text_center': False, 'n_styles': 0, 'cond_inst': False, 'n_density': 0,
              'bias': False, 'rope_base': 1e4, 'norm': 'layernorm', 'mlp': 'gelu', 'qk_norm': False,
              'velocity_size': 32, 'duration_size': 512, 'delta_time_size': 512}
    args = merge_model_args(saved, wanted, True, 64)
    assert (args['section_len'], args['section_lead'], args['section_mode']) == (256, 128, 'xattn')
    assert merge_model_args(saved, wanted, False, 64)['section_len'] == 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
