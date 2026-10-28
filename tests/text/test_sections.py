"""Section captions (CPU, tiny models, synthetic stores): python tests/text/test_sections.py  (or pytest tests/)."""

import hashlib
import random
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
from musicar.model.gpt import STREAM_ORDER, KVCache, section_layout, section_query_index  # noqa: E402
from musicar.text.captions import (hold_caption, section_attributes, section_caption, section_change,  # noqa: E402
                                   section_changed)
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


def test_pad_multiple_is_exact():
    """Padding the trunk's sequence at the end (fp8: positions divisible by 16) changes no output, and gradients are
    the same; the window here is 4 prefix + 16 notes + 3 x 2 section slots = 26 positions."""
    streams, program, targets = batch()
    states, pos = sections_input()
    text = torch.randn(3, 5, D)
    m = section_model('prefix')
    outs = []
    for pad in (0, 16):
        m.pad_multiple = pad
        m.zero_grad()
        parts, loss = m(*streams, program=program, targets=targets, text=text, sections=(states, pos))
        loss.backward()
        outs.append((loss.detach(), m.transformer.h[0].attn.c_attn.weight.grad.clone()))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6)
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-6)


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


def greedy_sections(m, text, secs, n_new, slide):
    """Greedy decoding with a full forward per note on generate()'s window (refilled to L - slide notes when full),
    the section captions laid out like NoteWindowDataset._sections: the one in force at the window start, then those
    arriving inside it."""
    L, S, lead = m.config.block_size, m.config.section_len, m.config.section_lead
    R = m.config.section_repeat or S
    B = text.size(0)
    out = [torch.full((B, 1), 128)] + [torch.zeros(B, 1, dtype=torch.long)] * 3
    prog = torch.zeros(B, 1, dtype=torch.long)
    vecs = m.section_vectors(secs)
    at = [max(0, k * S - lead) for k in range(len(secs))]
    arrive = [(p + j * R, k) for k, p in enumerate(at) for j in range(-(-S // R))
              if j == 0 or k + 1 >= len(at) or p + j * R < at[k + 1]]
    a = 0
    for _ in range(n_new):
        n = out[0].size(1)
        if n - a > L:
            a = n - (L - slide)
        events = [(0, max((p, k) for p, k in arrive if p <= a)[1])] + [(p - a, k) for p, k in arrive if a < p <= n]
        sec = (torch.stack([vecs[k] for _, k in events], 1), torch.tensor([p for p, _ in events]).expand(B, -1))
        h = m.encode(*(o[:, a:n] for o in out), program=prog[:, a:n], text=text, sections=sec)[:, [-1]]
        picked = {}

        def pick(logits, name):
            logits = logits[:, -1].clone()
            if name == 'pitch':
                logits[:, 128] = -float('Inf')  # BOS is never sampled
            nxt = logits.argmax(-1, keepdim=True)
            if name == 'duration':
                nxt = torch.where(picked['program'] == 128, 0, nxt)
            picked[name] = nxt
            return nxt
        nxt = m.cascade.sample(h, pick)
        out = [torch.cat([o, nxt[name]], 1) for o, name in zip(out, STREAM_ORDER)]
        prog = torch.cat([prog, nxt['program']], 1)
    return out


def test_refills_match_training_layout():
    """Past block_size (refills, with section starts on and off the refill points), generate() feeds the captions a
    training window at the same start gets."""
    B, n_new = 2, 100
    text = torch.randn(B, 5, D)
    secs = torch.randn(n_new // 8 + 2, B, 5, D)
    start = [torch.full((B, 1), 128)] + [torch.zeros(B, 1, dtype=torch.long)] * 3
    for mode in ('prefix', 'xattn'):
        for lead, repeat in ((0, 0), (4, 0), (0, 3), (4, 3)):
            m = section_model(mode, section_lead=lead, section_repeat=repeat)
            for slide in (8, 12):
                got = generate(m, *start, max_new_tokens=n_new, top_k=1, text=text, sections=secs, slide=slide,
                               program=None, cuda_graph=False)
                with torch.no_grad():
                    want = greedy_sections(m, text, secs, n_new, slide)
                assert all(torch.equal(g, w) for g, w in zip(got, want)), (mode, lead, repeat, slide)


def test_guidance_negatives():
    """text_neg / sections_neg equal to the conditional captions: guidance changes nothing; a different negative
    does."""
    start = [torch.full((2, 1), 128)] + [torch.zeros(2, 1, dtype=torch.long)] * 3
    kw = dict(max_new_tokens=40, top_k=1, program=None, cuda_graph=False)
    m = section_model('prefix')
    text, secs = torch.randn(5, D), torch.randn(6, 5, D)
    plain = generate(m, *start, text=text, sections=secs, **kw)
    same = generate(m, *start, text=text, sections=secs, cfg_scale=3.0, text_neg=text, sections_neg=secs, **kw)
    assert all(torch.equal(a, b) for a, b in zip(plain, same))
    prev = torch.cat([secs[:2], torch.zeros(4, 5, D)])  # contrast only the first two sections
    other = generate(m, *start, text=text, sections=secs, cfg_scale=3.0, text_neg=text, sections_neg=prev, **kw)
    assert not all(torch.equal(a, b) for a, b in zip(plain, other))


def test_double_guidance_and_boost():
    """cfg_section: equal scales = ordinary guidance, both 1 = no guidance; cfg_boost changes only notes near a
    section start."""
    start = [torch.full((2, 1), 128)] + [torch.zeros(2, 1, dtype=torch.long)] * 3
    kw = dict(max_new_tokens=40, top_k=1, program=None, cuda_graph=False)
    m = section_model('prefix')
    text, secs = torch.randn(5, D), torch.randn(6, 5, D)
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))  # noqa: E731
    plain = generate(m, *start, text=text, sections=secs, **kw)
    guided = generate(m, *start, text=text, sections=secs, cfg_scale=2.0, **kw)
    assert same(generate(m, *start, text=text, sections=secs, cfg_scale=2.0, cfg_section=2.0, **kw), guided)
    assert same(generate(m, *start, text=text, sections=secs, cfg_scale=1.0, cfg_section=1.0, **kw), plain)
    double = generate(m, *start, text=text, sections=secs, cfg_scale=1.0, cfg_section=4.0, **kw)
    assert not same(double, plain)
    boost = generate(m, *start, text=text, sections=secs, cfg_scale=2.0, cfg_boost=3.0, cfg_boost_notes=2, **kw)
    assert not same(boost, guided)
    assert torch.equal(boost[0][:, :1], guided[0][:, :1])  # the prompt (BOS)
    assert same(generate(m, *start, text=text, sections=secs, cfg_scale=2.0, cfg_boost=0.0, **kw), guided)


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


def test_section_attributes_v2():
    """v2: section-level thresholds, 'rhythm' + 'texture' instead of 'pace'; texture changes count and get named."""
    import random
    n = 64
    single = np.stack([np.arange(n) % 12 + 60, np.full(n, 28), np.full(n, 10), np.full(n, 3)], 1)
    chords = single.copy()
    chords[:, 3] = np.where(np.arange(n) % 8 == 0, 12, 0)  # 8 notes per onset, 1.5 onsets/s
    piano = np.zeros(n, np.int64)
    a, c = section_attributes(single, piano, 'v2'), section_attributes(chords, piano, 'v2')
    assert 'pace' not in a and a['rhythm'] == 'high' and a['texture'] == 'low' and a['dynamics'] == 'high', a
    assert c['rhythm'] == 'low' and c['texture'] == 'high', c
    assert 'texture' not in section_attributes(single, piano) and 'pace' in section_attributes(single, piano)  # v1
    assert section_change(a, c) == 2 and section_changed(a, c) == {'rhythm': 'low', 'texture': 'high'}
    caps = [section_caption(chords, piano, random.Random(i), prev=a, version='v2').lower() for i in range(30)]
    assert all(any(w in cap for w in ('chordal', 'dense', 'акордов', 'щільн')) for cap in caps), caps
    busy = [section_caption(single, piano, random.Random(i), lang='en', version='v2') for i in range(60)]
    assert any('busy' in cap for cap in busy) and not any('dense' in cap for cap in busy), busy


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


def test_section_change():
    a = {'dynamics': 'low', 'instruments': ['piano', 'violin'], 'no_drums': True}
    assert section_change(a, a) == 0 and section_change(a, {}) == 0
    assert section_change(a, {**a, 'dynamics': 'high', 'register': 'high'}) == 2
    assert section_change(a, {**a, 'instruments': ['piano', 'cello']}) == 1  # violin out, cello in
    assert section_change(a, {**a, 'instruments': ['piano']}) == 0  # one instrument: below the threshold
    assert section_change(a, {'dynamics': 'low', 'instruments': ['piano', 'violin', 'drums']}) == 1  # with drums


def caption_sections(n=200, seed=0):
    """Random sections (tokens, programs) of 20-300 notes: piano, a band, drums only, a large band."""
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = int(g.integers(20, 300))
        tok = np.stack([g.integers(30, 100, m), g.integers(int(g.integers(0, 20)), int(g.integers(20, 32)), m),
                        g.integers(1, 60, m), g.integers(0, int(g.integers(2, 40)), m)], 1)
        progs = [[0], [0, 33, 128], [128], [40, 41, 42, 0, 33, 25, 128]][i % 4]
        out.append((tok.astype(np.int64), g.choice(progs, m).astype(np.int64)))
    return out


def test_plain_section_captions_unchanged():
    # the plain captions (and their draws) as before the change-aware option (2026-10-20): a run resumed across the
    # change (long_A256) keeps its training and fixed validation captions
    caps = [section_caption(tok, prog, random.Random(i * 10 + s), uk_share=0.3)
            for i, (tok, prog) in enumerate(caption_sections()) for s in (0, 1)]
    assert hashlib.sha1('\n'.join(caps).encode()).hexdigest() == 'bc2b2b097256ef7031b931e5bb4612af7e24aa19'


def test_section_changed():
    soft = {'dynamics': 'low', 'pace': 'high', 'instruments': ['piano'], 'solo': True}
    assert section_changed(soft, soft) == {} and section_changed(soft, {}) == {}
    loud = {'dynamics': 'high', 'instruments': ['piano', 'violin', 'drums']}
    assert section_changed(soft, loud) == {'dynamics': 'high', 'instruments': ['drums', 'violin']}
    assert section_changed(loud, soft) == {'dynamics': 'low', 'pace': 'high', 'solo': True}  # drums stopped
    band = {'instruments': ['piano', 'bass'], 'no_drums': True}
    assert section_changed(loud, band) == {'no_drums': True, 'instruments': ['bass']}  # middle dynamics: no word
    assert section_changed(band, {'drums': True}) == {'drums': True}


def test_change_captions_name_the_change():
    from musicar.text import vocab as V
    n = 256
    tok = np.stack([np.full(n, 60), np.full(n, 10), np.full(n, 10), np.full(n, 10)], 1)  # soft, low notes
    piano = np.zeros(n, np.int64)
    prev = section_attributes(tok, piano)
    loud = tok.copy()
    loud[:, 0], loud[:, 1] = 90, 31  # loud, high register
    band = np.array([0, 40] * (n // 2))  # piano + violin
    assert section_changed(prev, section_attributes(loud, band)).keys() >= {'dynamics', 'register', 'instruments'}
    violin = V.INSTRUMENT['violin'][0] + [f[0][:6] for f in V.INSTRUMENT['violin'][1]]
    for i in range(60):
        for lang in ('en', 'uk'):
            c = section_caption(loud, band, random.Random(i), lang=lang, prev=prev).lower()
            assert any(w in c for w in ('loud', 'powerful', 'гучн', 'потужн')), c
            assert any(w in c for w in ('high register', 'high-pitched', 'bright', 'регістр')), c
            assert any(w in c for w in violin), c
    # the first section of a file (prev None) and an unchanged section read like plain captions
    assert section_caption(loud, band, random.Random(0)) == section_caption(loud, band, random.Random(0), prev=None)


def test_dataset_change_windows():
    with tempfile.TemporaryDirectory() as d:
        root = make_store(Path(d), notes=80)
        store = root / 'cache' / 'toy_train'
        tokens = np.fromfile(store / 'tokens.u16', np.uint16).reshape(-1, 4)
        tokens[:, :2], tokens[:, 3] = (60, 10), 10  # even notes: no changes
        tokens[80 + 47:160, :2] = (100, 31)  # file 1 from row 48 (section 3 of 16 rows): high register, loud
        tokens.tofile(store / 'tokens.u16')
        kw = dict(root_dir=d, block_size=32, special_tokens=True, programs=True, captions=str(root / 'labels'),
                  cache_dir=str(root / 'cache'), section_len=16, augment=True, section_dropout=0.0)
        src = f"store:{root / 'cache' / 'toy'}"
        ds = NoteWindowDataset(src, split='train', change_frac=1.0, change_min=2, section_lead=8, **kw)
        assert ds.changes.tolist() == [[1, 48]] and (store / 'section_changes_16_1.npy').exists()
        for seed in range(20):
            random.seed(seed)
            x, _ = ds[0]
            random.seed(seed)
            random.random()  # the change_frac draw
            file_idx, v, _ = ds._change_window(None)
            assert file_idx == 1 and 48 - 28 <= v <= 48 - 12  # boundary at window index lead + 4 .. 32 - 4
            assert 40 - v in x[-1].tolist()  # its caption arrives inside the window, 8 rows early
        np.save(store / 'section_changes_16_1.npy', np.zeros(36, np.int8))  # the cache is read, not recomputed
        try:
            NoteWindowDataset(src, split='train', change_frac=1.0, change_min=2, **kw)
            raise RuntimeError('expected no section changes')
        except AssertionError as e:
            assert 'no section changes' in str(e)


def test_dataset_section_style():
    with tempfile.TemporaryDirectory() as d:
        root = make_store(Path(d), notes=80)
        store = root / 'cache' / 'toy_train'
        tokens = np.fromfile(store / 'tokens.u16', np.uint16).reshape(-1, 4)
        tokens[:, :2], tokens[:, 3] = (60, 10), 10
        tokens[80 + 47:160, :2] = (100, 31)  # file 1 from row 48 (section 3): high register, loud
        tokens.tofile(store / 'tokens.u16')
        kw = dict(root_dir=d, block_size=32, special_tokens=True, programs=True, captions=str(root / 'labels'),
                  cache_dir=str(root / 'cache'), section_len=16, eval_stride=32)
        src = f"store:{root / 'cache' / 'toy'}"
        plain = NoteWindowDataset(src, split='train', **kw)
        change = NoteWindowDataset(src, split='train', section_style='change', **kw)
        assert plain._section_caption(1, 0) == change._section_caption(1, 0), 'first section: plain'
        assert plain._section_caption(1, 2) == change._section_caption(1, 2), 'nothing changed: plain'
        c = change._section_caption(1, 3).lower()
        assert any(w in c for w in ('loud', 'powerful', 'гучн', 'потужн')), c
        assert any(w in c for w in ('high register', 'high-pitched', 'bright', 'регістр')), c
        v2 = NoteWindowDataset(src, split='train', section_style='change', section_attrs='v2', **kw)
        assert any(w in v2._section_caption(1, 3).lower() for w in ('loud', 'powerful', 'гучн', 'потужн'))
        v2._find_changes(2)
        assert list(store.glob('section_changes_*_v2.npy')), 'v2 change scores: own cache file'


def test_dataset_previous_and_repeat():
    with tempfile.TemporaryDirectory() as d:
        root = make_store(Path(d), notes=80)
        kw = dict(root_dir=d, block_size=32, special_tokens=True, programs=True, captions=str(root / 'labels'),
                  cache_dir=str(root / 'cache'), section_len=16, eval_stride=32, section_style='change')
        src = f"store:{root / 'cache' / 'toy'}"
        true = NoteWindowDataset(src, split='train', **kw)
        prev = NoteWindowDataset(src, split='train', section_caption_mode='previous', **kw)
        # file 0, window at row 32: sections 2, 3, 4 -> previous gives the captions of 1, 2, 3
        assert prev[1][0][-2][1:3] == true[1][0][-2][:2] and prev[1][0][-2][0] == true[0][0][-2][1]
        rep = NoteWindowDataset(src, split='train', section_repeat=6, **kw)
        assert rep.section_events == 1 + 3 * 3
        caps, pos = rep[1][0][-2:]  # sections 2 (rows 32, 38, 44), 3 (48, 54, 60), 4 (64)
        assert pos.tolist() == [0, 6, 12, 16, 22, 28, 32, 33, 33, 33]
        assert caps[:3] == [true[1][0][-2][0]] * 3 and caps[3:6] == [true[1][0][-2][1]] * 3
        assert caps[6] == true[1][0][-2][2] and caps[7:] == [''] * 3


def test_dataset_splice_and_counterfactual():
    with tempfile.TemporaryDirectory() as d:
        root = make_store(Path(d), n_files=6, notes=80)
        kw = dict(root_dir=d, block_size=32, special_tokens=True, programs=True, captions=str(root / 'labels'),
                  cache_dir=str(root / 'cache'), section_len=16, augment=False, section_dropout=0.0,
                  section_style='change', caption_dropout=0.0)
        src = f"store:{root / 'cache' / 'toy'}"
        ds = NoteWindowDataset(src, split='train', splice_frac=1.0, counterfactual=True, **kw)
        plain = NoteWindowDataset(src, split='train', **kw)
        n_cf = 0
        calls, own = [], ds._section_caption

        def record(f, k, prev_notes=None):
            # (file, section, the first file's section whose notes come before it in a splice)
            match = [j for j in range(5) if prev_notes is not None and
                     np.array_equal(prev_notes[0], plain._section_notes(f1, j)[0])] if prev_notes is not None else []
            calls.append((f, k, match[0] if match else None))
            return own(f, k, prev_notes)
        ds._section_caption = record
        for seed in range(30):
            random.seed(seed)
            random.random()
            f1 = ds._splice_window(None)[0]
            calls.clear()
            random.seed(seed)
            x, y = ds[0]
            random.seed(seed)
            random.random()  # the splice_frac draw
            f1, v1, b, f2, v2 = ds._splice_window(None)
            assert f1 != f2 and (v1 + b) % 16 == 0 and (v2 + b) % 16 == 0 and 4 <= b <= 28
            r1, _ = plain._read(f1, v1)
            r2, _ = plain._read(f2, v2)
            assert torch.equal(x[0], torch.cat([r1[:b], r2[b:]])[:-1, 0]), seed
            assert torch.equal(y[0], torch.cat([r1[:b], r2[b:]])[1:, 0]), seed
            caps, pos, cf_text, cf_event, cf_start = x[-5:]
            assert x[-6] == '', 'no file caption in a spliced window'
            assert b in pos.tolist()
            e = pos.tolist().index(b)
            k1, k2 = (v1 + b) // 16, (v2 + b) // 16
            assert (f2, k2, k1 - 1) in calls, 'the first spliced section: change-aware against the notes before b'
            assert all(k < k1 for f, k, _ in calls if f == f1 and _ is None), 'no first-file section after b'
            if int(cf_start) >= 0:
                n_cf += 1
                assert int(cf_start) == b and int(cf_event) == e and cf_text and cf_text != caps[e]
        assert n_cf >= 20
        (streams, extra), _ = collate_windows([ds[i] for i in range(3)], extra_names=(
            'program', 'text', 'sec_text', 'sec_pos', 'cf_text', 'cf_event', 'cf_start'))
        assert extra['cf_start'].shape == (3,) and len(extra['cf_text']) == 3


def test_counterfactual_loss():
    m = section_model('prefix')
    m.add_section_cls()
    B, T, E, L = 3, 24, 4, 5
    streams, program, tgt = batch(B, T)
    torch.manual_seed(2)
    text = torch.randn(B, L, D)
    states = torch.randn(B, E, L, D)
    pos = torch.tensor([[0, 8, 16, 25]] * B)
    sections = (states, pos)
    base_parts, base = m(*streams, program=program, targets=tgt, text=text, sections=sections)
    rows = torch.tensor([0, 2])
    cf = dict(rows=rows, states=states[rows].clone(), start=torch.tensor([8, -1]), notes=6, eta=0.05, weight=1.0,
              cls_weight=0.5)
    parts, loss = m(*streams, program=program, targets=tgt, text=text, sections=sections, cf=cf)
    # identical captions: margin 0, hinge = eta, used = 1 of 2 rows
    assert abs(float(parts['cf_margin'])) < 1e-5 and abs(float(parts['cf_hinge']) - 0.05) < 1e-5
    assert float(parts['cf_used']) == 0.5 and float(parts['cls_acc']) == 0.5  # same states: same logit for both
    cf['states'][0, 1] = torch.randn(L, D)  # the changed section (event 1, before note 8) gets another caption
    parts, loss = m(*streams, program=program, targets=tgt, text=text, sections=sections, cf=cf)
    assert abs(float(parts['cf_margin'])) > 1e-6
    loss.backward()
    assert m.transformer['section_prefix'].gate.grad.abs().sum() > 0 and m.aux_section_cls.weight.grad.abs().sum() > 0
    assert torch.allclose(base, loss - 1.0 * parts['cf_hinge'] - 0.5 * parts['cls_bce'], atol=1e-5)


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



def test_hold():
    """generate's hold: a section caption returned at note hold_at is what every later section gets (the same notes as
    passing it up front, greedy, with guidance and past block_size), and the returned instruments are the only ones
    sampled after it."""
    start = [torch.full((2, 1), 128)] + [torch.zeros(2, 1, dtype=torch.long)] * 3
    for mode in ('prefix', 'xattn'):
        m = section_model(mode, n_programs=12)
        X = torch.randn(2, 5, D)
        kw = dict(max_new_tokens=100, top_k=1, text=torch.randn(5, D), cfg_scale=2.0, slide=16, program=None,
                  cuda_graph=False, return_programs=True)
        seen = []

        def hold(notes):
            seen.append(notes[0].shape)
            return X, None
        got = generate(m, *start, hold=hold, **kw)
        want = generate(m, *start, sections=torch.stack([torch.zeros(2, 5, D), X]), **kw)
        assert seen == [(2, 8)], seen  # once, at the first section's end (section_len 8, BOS included)
        assert all(torch.equal(g, w) for g, w in zip(got, want)), mode
        torch.manual_seed(3)
        out = generate(m, *start, hold=lambda notes: (None, [[2, 5], None]), **{**kw, 'top_k': None})
        assert set(out[4][0, 8:].tolist()) <= {2, 5} and len(set(out[4][1, 8:].tolist())) > 2, mode


def test_hold_caption():
    rng = np.random.default_rng(0)
    tok = np.stack([rng.integers(40, 80, 256), rng.integers(24, 30, 256), rng.integers(5, 20, 256),
                    rng.choice([0, 0, 5, 10], 256)], 1)
    prog = rng.choice([0, 33, 128, 25, 48], 256, p=[.4, .2, .25, .1, .05])
    cap = hold_caption(tok, prog)
    assert cap == hold_caption(tok, prog) and 'piano' in cap and 'drums' in cap, cap  # the same words every time
    assert 'string' not in cap  # 5 instruments: the 3 with the most notes
    assert hold_caption(tok[:0], prog[:0]) == ''
    fast = tok.copy()
    fast[:, 3] = rng.choice([0, 0, 1, 2], 256)  # ~50 notes/s: no 'busy' / 'dense' held
    assert 'busy' in hold_caption(fast, prog, calm_above=1e9), hold_caption(fast, prog, calm_above=1e9)
    assert not {'busy', 'dense'} & set(hold_caption(fast, prog).lower().split()), hold_caption(fast, prog)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
