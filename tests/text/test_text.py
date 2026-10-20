"""Text control checks (CPU, tiny models, synthetic stores): python tests/text/test_text.py  (or pytest tests/)."""

import json
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musicar.text.captions import StoreCaptions, instrument_name  # noqa: E402
from musicar.model.checkpoint import load_expanded, merge_model_args  # noqa: E402
from musicar.data.loader import NoteWindowDataset, collate_windows  # noqa: E402
from musicar.model.gpt import GPT, MusicConfig  # noqa: E402
from musicar.training.optim import build_optimizer, freeze_params  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402

TINY = dict(n_layer=2, n_head=2, n_embd=32, block_size=64, dropout=0.0, pitch_size=130, n_programs=129,
            pos_emb='rope')


def tiny(**kw):
    torch.manual_seed(0)
    return GPT(MusicConfig(**{**TINY, **kw})).eval()


def batch(B=3, T=16):
    g = torch.Generator().manual_seed(1)
    streams = [torch.randint(0, n, (B, T), generator=g) for n in (128, 32, 512, 512)]
    program = torch.randint(0, 129, (B, T), generator=g)
    targets = [torch.randint(0, n, (B, T), generator=g) for n in (130, 32, 512, 512, 129)]
    return streams, program, targets


def test_zero_init_is_function_preserving():
    old = tiny()
    new = tiny(text_dim=8)
    grown = load_expanded(new, old.state_dict())
    assert grown == ['transformer.text_proj.weight'], grown
    streams, program, targets = batch()
    text = torch.randn(3, 8)
    with torch.no_grad():
        _, l_old = old(*streams, program=program, targets=targets)
        _, l_new = new(*streams, program=program, targets=targets, text=text)
    assert torch.allclose(l_old, l_new), (l_old, l_new)


def test_text_changes_output_once_trained():
    m = tiny(text_dim=8)
    torch.nn.init.normal_(m.transformer['text_proj'].weight)
    streams, program, _ = batch()
    with torch.no_grad():
        h0 = m.encode(*streams, program=program, text=torch.zeros(3, 8))
        h1 = m.encode(*streams, program=program, text=torch.randn(3, 8))
        hn = m.encode(*streams, program=program)
    assert torch.allclose(h0, hn) and not torch.allclose(h0, h1)


def test_text_center():
    old = tiny()
    m = tiny(text_dim=8, text_center=True)
    grown = load_expanded(m, old.state_dict())
    assert sorted(grown) == ['text_mean', 'transformer.text_proj.weight'], grown
    m.text_mean.copy_(torch.full((8,), 0.3))
    torch.nn.init.normal_(m.transformer['text_proj'].weight)
    streams, program, _ = batch()
    t = torch.randn(3, 8)
    with torch.no_grad():
        hn = m.encode(*streams, program=program)
        h0 = m.encode(*streams, program=program, text=torch.zeros(3, 8))
        h1 = m.encode(*streams, program=program, text=t)
        h2 = m.encode(*streams, program=program, text=2 * (t - 0.3) + 0.3)  # same direction from the mean
    assert torch.allclose(h0, hn), 'the zero vector must stay no caption'
    assert not torch.allclose(h0, h1) and torch.allclose(h1, h2, atol=1e-5)


def prefix_model():
    m = tiny(text_dim=8, text_prefix=4)
    torch.nn.init.normal_(m.transformer['text_prefix'].gate)
    torch.nn.init.normal_(m.transformer['text_prefix'].null)
    return m


def test_text_prefix_forward():
    m = prefix_model()
    assert 'text_proj' not in m.transformer
    streams, program, targets = batch()
    states = torch.randn(3, 5, 8)
    states[0, 3:] = 0  # padding
    states[2] = 0      # no caption
    with torch.no_grad():
        h = m.encode(*streams, program=program, text=states)
        hn = m.encode(*streams, program=program)
        h_null = m.encode(*streams, program=program, text=torch.zeros(3, 5, 8))
        h_pad = m.encode(*streams, program=program, text=torch.cat([states, torch.zeros(3, 2, 8)], 1))
    assert h.shape == (3, 16, 32)
    assert torch.allclose(h[2], hn[2], atol=1e-5) and torch.allclose(hn, h_null, atol=1e-5), 'zero = null prefix'
    assert not torch.allclose(h[0], hn[0])
    assert torch.allclose(h, h_pad, atol=1e-5), 'padding rows must not matter'
    parts, loss = m(*streams, program=program, targets=targets, text=states)
    loss.backward()
    assert m.transformer['text_prefix'].queries.grad.abs().sum() > 0


def test_text_prefix_cache_matches_forward():
    from musicar.model.gpt import KVCache
    m = prefix_model()
    streams, program, _ = batch(B=2, T=10)
    states = torch.randn(2, 5, 8)
    with torch.no_grad():
        full = m.encode(*streams, program=program, text=states)
        for n0 in (1, 4):  # a 1-note prefill (BOS only) must not be taken for a one-note step
            cache = KVCache(2, 2, 2, 4 + 10, 16, 'cpu', torch.float32)
            h = m.encode(*(t[:, :n0] for t in streams), program=program[:, :n0], cache=cache, text=states,
                         step=False)
            assert torch.allclose(h, full[:, :n0], atol=1e-5), n0
            for i in range(n0, 10):
                cache.pos.fill_(4 + i)
                h = m.encode(*(t[:, i:i + 1] for t in streams), program=program[:, i:i + 1], cache=cache)
                assert torch.allclose(h[:, 0], full[:, i], atol=1e-4), (n0, i)


def test_text_prefix_generate():
    m = prefix_model()
    start = [torch.full((2, 1), 128)] + [torch.zeros(2, 1, dtype=torch.long)] * 3
    kw = dict(max_new_tokens=20, program=None, cuda_graph=False, return_programs=True)
    torch.manual_seed(5)
    plain = generate(m, *start, **kw)
    torch.manual_seed(5)
    zero = generate(m, *start, text=torch.zeros(5, 8), **kw)
    assert all(torch.equal(a, b) for a, b in zip(plain, zero)), 'a zero caption must equal no caption'
    torch.manual_seed(5)
    guided = generate(m, *start, text=torch.randn(5, 8), cfg_scale=2.0, slide=16, **kw)
    assert guided[0].shape == (2, 21)
    torch.manual_seed(5)  # past block_size: the cache is refilled (prefix rebuilt)
    long = generate(m, *start, text=torch.randn(5, 8), **{**kw, 'max_new_tokens': 100})
    assert long[0].shape == (2, 101)


def test_merge_model_args_adds_text():
    saved = {k: v for k, v in TINY.items() if k != 'dropout'}
    wanted = {**saved, 'text_dim': 768, 'text_center': False, 'text_prefix': 0, 'n_styles': 0, 'rope_base': 10000.0, 'norm': 'layernorm', 'mlp': 'gelu',
              'qk_norm': False, 'cond_inst': False, 'n_density': 0, 'bias': False, 'velocity_size': 32,
              'duration_size': 512, 'delta_time_size': 512}
    assert merge_model_args(saved, wanted, True, 64)['text_dim'] == 768
    assert merge_model_args(saved, wanted, False, 64)['text_dim'] == 0  # resume keeps the checkpoint's
    assert merge_model_args(saved, {**wanted, 'text_center': True}, True, 64)['text_center']


def test_freeze_trunk_and_text_group():
    m = tiny(text_dim=8)
    freeze_params(m, '*', except_prefixes='transformer.text_proj')
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable == ['transformer.text_proj.weight'], trainable
    opt = build_optimizer('adamw', m, 0.1, 1e-3, (0.9, 0.95), 'cpu', text_lr_mult=10.0)
    assert [g.get('lr_mult', 1.0) for g in opt.param_groups] == [10.0]


def test_generate_with_text_and_guidance():
    m = tiny(text_dim=8)
    torch.nn.init.normal_(m.transformer['text_proj'].weight)
    start = [torch.full((2, 1), 128)] + [torch.zeros(2, 1, dtype=torch.long)] * 3
    kw = dict(max_new_tokens=20, program=None, cuda_graph=False, return_programs=True)
    torch.manual_seed(5)
    plain = generate(m, *start, **kw)
    torch.manual_seed(5)
    zero = generate(m, *start, text=torch.zeros(8), **kw)
    assert all(torch.equal(a, b) for a, b in zip(plain, zero)), 'a zero caption must equal no caption'
    torch.manual_seed(5)
    guided = generate(m, *start, text=torch.randn(8), cfg_scale=2.0, **kw)
    assert guided[0].shape == (2, 21)


def make_store(root, n_files=6, notes=80):
    """A tiny multi-instrument store <root>/cache/toy_train + label tables for files 0 and 3."""
    store = root / 'cache' / 'toy_train'
    store.mkdir(parents=True)
    rng = np.random.default_rng(0)
    tokens = np.stack([rng.integers(40, 80, n_files * notes), rng.integers(0, 32, n_files * notes),
                       rng.integers(1, 50, n_files * notes), rng.integers(0, 20, n_files * notes)], 1)
    tokens.astype(np.uint16).tofile(store / 'tokens.u16')
    np.full(n_files * notes, 33, np.uint8).tofile(store / 'programs.u8')
    np.save(store / 'offsets.npy', np.arange(0, (n_files + 1) * notes, notes, dtype=np.int64))
    (store / 'meta.json').write_text(json.dumps(dict(n_files=n_files, n_notes=n_files * notes, programs=True)))
    labels = root / 'labels'
    labels.mkdir()
    rows = [dict(file=0, md5='a', midicaps=dict(caption='A sad jazz piece.', genre=['jazz'], mood=['sad'],
                                                key='D minor', tempo_word='Slow', instrument_summary=['Piano'])),
            dict(file=3, md5='b', xmidi=[['happy', 'pop']])]
    (labels / 'toy_train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    bits = np.zeros((n_files, 136), np.uint8)
    bits[:, 33] = 1
    feats = dict(programs=np.packbits(bits, axis=1, bitorder='little'), seconds=np.full(n_files, 60.0, np.float32),
                 ops=np.linspace(1, 10, n_files).astype(np.float32), vel_mean=np.full(n_files, 21.0, np.float32),
                 pitch_mean=np.full(n_files, 60.0, np.float32), key=np.full(n_files, 14, np.int16),
                 key_conf=np.full(n_files, 0.2, np.float32), drums=np.zeros(n_files, np.float32))
    np.savez(labels / 'toy_train.features.npz', **feats)
    return root


def test_captions_and_dataset():
    with tempfile.TemporaryDirectory() as d:
        root = make_store(Path(d))
        c = StoreCaptions('toy_train', root / 'labels')
        assert c.labeled(0) and c.labeled(3) and not c.labeled(1)
        rng = random.Random(0)
        caps = [c.caption(i, rng, 'en') for i in range(6) for _ in range(5)]
        assert all(caps), caps
        assert any('bass' in x for x in caps[5:]), caps  # program 33 from the features
        uk = [c.caption(i, rng, 'uk') for i in range(6) for _ in range(20)]
        assert all(uk) and not any(re.search('[A-Za-z]', x) for x in uk), [x for x in uk if re.search('[A-Za-z]', x)]
        assert any('бас' in x for x in uk), uk
        assert instrument_name(80) == 'synth lead' and instrument_name(89) == 'synth pad'
        kw = dict(root_dir=d, split='train', block_size=32, special_tokens=True, programs=True,
                  captions=str(root / 'labels'), cache_dir=str(root / 'cache'))
        src = f"store:{root / 'cache' / 'toy'}"
        ds = NoteWindowDataset(src + '#labeled', augment=True, caption_dropout=0.0, **kw)
        assert sorted(ds.file_ids.tolist()) == [0, 3]
        (streams, extra), y = collate_windows([ds[i] for i in range(4)], extra_names=('program', 'text'))
        assert isinstance(extra['text'], list) and all(isinstance(t, str) and t for t in extra['text'])
        assert NoteWindowDataset(src + '#unlabeled', **kw).file_ids.tolist() == [1, 2, 4, 5]
        val = {mode: NoteWindowDataset(src, eval_stride=32, caption_mode=mode, **kw)
               for mode in ('true', 'shuffled', 'none')}
        true = [val['true'][i][0][-1] for i in range(len(val['true']))]
        assert true == [val['true'][i][0][-1] for i in range(len(val['true']))], 'val captions must be fixed'
        shuffled = [val['shuffled'][i][0][-1] for i in range(len(val['shuffled']))]
        assert shuffled != true
        assert all(val['none'][i][0][-1] == '' for i in range(len(val['none'])))


def test_caption_vocab_and_uk_rendering():
    from musicar.text import vocab as V
    from musicar.text.captions import render_uk
    words = set()

    def walk(o):
        if isinstance(o, str):
            words.update(re.findall(r"[\w'-]+", o))
        elif isinstance(o, dict):
            [walk(x) for x in o.values()]
        elif isinstance(o, (list, tuple)):
            [walk(x) for x in o]
    for name in dir(V):
        walk(getattr(V, name))
    mixed = {w for w in words if re.search('[а-яіїєґА-ЯІЇЄҐ]', w) and re.search('[a-zA-Z]', w)}
    assert not mixed, mixed
    rng = random.Random(0)
    for _ in range(200):  # feminine nouns get feminine adjectives, masculine nouns masculine ones
        text = render_uk({'mood': ['x_sad'], 'genre': ['jazz']}, rng).lower().rstrip('.')
        if text.startswith(('сумний, ', 'меланхолійний, ')):
            continue  # tags
        for noun in ('композиці', 'мелоді', 'пісн', "п'єс", 'музик'):
            if noun in text:
                assert re.search('(сумн|меланхолійн)(а|у) джазов(а|у)', text), text
        for noun in ('твір', 'трек'):
            if noun in text:
                assert re.search('(сумний|меланхолійний) джазовий', text), text


def test_fix_uk_captions():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_prep' / 'labels'))
    from fix_uk_captions import fix
    assert fix('Мелодійний поп-пісня')[0] == 'Мелодійна поп-пісня'
    assert fix('Щасливий і мотиваційний поп-пісня')[0] == 'Щаслива і мотиваційна поп-пісня'
    assert fix('цей різдвяний пісня')[0] == 'ця різдвяна пісня'
    assert fix('акустичний фортепіано')[0] == 'акустичне фортепіано'
    assert fix('у модератному темпі, з модерним темпом')[0] == 'у помірному темпі, з помірним темпом'
    assert fix('кінematографічним')[0] == 'кінематографічним'
    assert fix('захоплюючу мелодію')[0] == 'захопливу мелодію'
    assert fix('Класичний твір, який')[0] == 'Класичний твір, який'


def test_no_bare_captions():
    """Kept attributes that say nothing alone ('solo' without its instrument, a length word next to 'music') must not
    leave a bare noun ('Трек', 'a piece')."""
    with tempfile.TemporaryDirectory() as d:
        bits = np.zeros((2, 136), np.uint8)
        bits[0, 0] = 1  # file 0: solo piano and nothing else to say; file 1: no instruments, long
        feats = dict(programs=np.packbits(bits, axis=1, bitorder='little'), seconds=np.array([60, 300], np.float32),
                     ops=np.full(2, 5.0, np.float32), vel_mean=np.full(2, 21.0, np.float32),
                     pitch_mean=np.full(2, 60.0, np.float32), key=np.full(2, -1, np.int16),
                     key_conf=np.zeros(2, np.float32), drums=np.zeros(2, np.float32))
        np.savez(Path(d) / 'toy.features.npz', **feats)
        c = StoreCaptions('toy', d)
        rng = random.Random(0)
        for lang, piano, long in (('en', 'piano', 'long|lengthy'), ('uk', 'фортепіано|піаніно', 'довг')):
            for i, word in ((0, piano), (1, long)):
                caps = [c.caption(i, rng, lang) for _ in range(300)]
                assert all(re.search(word, x.lower()) for x in caps), [x for x in caps if not re.search(word, x.lower())]


def test_gigamidi_styles():
    from musicar.text import vocab as V
    from musicar.text.captions import gigamidi_ids
    for key, ids in [*V.GIGAMIDI_STYLE.items(), *V.GIGAMIDI_ARTIST.items()]:
        assert ids and all((i in V.GENRE) != (i in V.MOOD) for i in ids), key
    assert not set(V.GIGAMIDI_STYLE) & V.GIGAMIDI_SKIP and (V.ERAS | V.DECADES) <= set(V.GENRE)
    for g, e in V.GENRE.items():
        assert e['en'] and e['uk_tag'] and (e.get('uk') or e.get('prefix') or e.get('uk_post')), g
    assert gigamidi_ids({'styles': ['classical', 'baroque']}) == {'genre': ['baroque']}
    assert gigamidi_ids({'styles': ['rock', 'hits of the 1970s', 'christmas festive']}) == \
        {'genre': ['70s', 'rock'], 'use': ['christmas']}
    assert gigamidi_ids({'styles': ['love songs'], 'artist': 'Video Games'}) == {'mood': ['love'], 'use': ['game']}
    assert gigamidi_ids({'styles': ['midi karaoke'], 'artist': 'Queen'}) == {}
    with tempfile.TemporaryDirectory() as d:
        rows = [dict(file=0, md5='a', gigamidi=dict(styles=['classical', 'baroque'], artist='Bach, Johann Sebastian')),
                dict(file=1, md5='b', gigamidi=dict(artist='Queen', title='Bohemian Rhapsody')),
                dict(file=2, md5='c', gigamidi=dict(styles=['hits of the 1980s', 'pop', 'rock']))]
        (Path(d) / 'toy.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        c = StoreCaptions('toy', d)
        assert c.labeled(0) and not c.labeled(1) and c.labeled(2)
        rng = random.Random(0)
        bach = [c.caption(0, rng, 'en') for _ in range(100)]
        assert all('baroque' in x.lower() for x in bach) and not any('classical' in x for x in bach), bach
        pop = [c.caption(2, rng, 'en') for _ in range(200)]
        assert any("an '80s" in x for x in pop), pop
        assert not any(re.search(r"\ba '80s|'80s and", x, re.I) for x in pop), pop
        uk = [c.caption(i, rng, 'uk') for i in (0, 2) for _ in range(100)]
        assert not any(re.search('[A-Za-z]', x) for x in uk), uk
        assert any('барок' in x for x in uk) and any('80-х' in x for x in uk), uk
        assert c.caption(1, rng, 'en') == ''


def test_gigamidi_meta_styles():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_prep' / 'labels'))
    from gigamidi_meta import _clean, styles_of
    row = {'music_styles_curated': float('nan'), 'music_style_scraped': 'italian%2cfrench%2cspanish,midi karaoke',
           'music_style_audio_text_Discogs': 'HEAVY_METAL', 'music_style_audio_text_Lastfm': 'HEAVY_METAL'}
    assert styles_of(row) == ['italian,french,spanish', 'midi karaoke', 'heavy metal']
    assert _clean('Pink\nFloyd ') == 'Pink Floyd' and _clean('  ') is None and _clean(float('nan')) is None


def test_text_encoder():
    try:
        from musicar.text.encoder import TextEncoder
        enc = TextEncoder()
    except Exception as e:  # LaBSE not downloaded
        print(f'  skipped: {e}')
        return
    v = enc.encode(['a sad slow piano ballad', 'сумна повільна фортепіанна балада', 'a fast happy drum beat', ''])
    assert v.shape == (4, 768) and torch.all(v[3] == 0)
    assert torch.allclose(v[:3].norm(dim=1), torch.ones(3), atol=1e-4)
    assert v[0] @ v[1] > v[0] @ v[2], 'the Ukrainian translation must be nearer than another prompt'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
