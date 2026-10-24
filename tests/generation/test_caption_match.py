"""musicar/generation/caption_match.py checks (CPU): python tests/generation/test_caption_match.py
(or tests/run.py)."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run  # noqa: E402
from musicar.generation.caption_match import (FAMILY, FAMILY_OF_ID, FileAttributes, allowed_programs,  # noqa: E402
                                              match, note_attributes, parse_caption)
from musicar.text import vocab as V  # noqa: E402
from musicar.text.captions import NOTE_NAMES, render_en, render_uk  # noqa: E402


def labels(w):
    return {label for label, _ in w['instruments']}


def test_parse_examples():
    w = parse_caption('A slow romantic rock ballad in E minor with piano, strings, electric guitar, drums, bass and '
                      'choir')
    assert labels(w) == {'piano', 'strings', 'electric guitar', 'bass', 'choir'} and w['drums']
    assert w['key'] == (4, 'minor') and w['pace'] == 'low' and not w['no_drums']
    u = parse_caption('Повільна романтична рок-балада в мі мінорі з фортепіано, струнними, електрогітарою, ударними, '
                      'басом і хором')
    assert labels(u) == labels(w) and u['drums'] and u['key'] == (4, 'minor') and u['pace'] == 'low'
    w = parse_caption('a sad solo piano piece in a minor key')  # 'a minor key' = the mode, not A minor
    assert labels(w) == {'piano'} and w['solo'] and w['key'] == (None, 'minor')
    w = parse_caption('Зіграй тиху класичну пісню Генделя для фортепіано соло у мінорі')
    assert labels(w) == {'piano'} and w['solo'] and w['key'] == (None, 'minor') and w['dynamics'] == 'low'
    w = parse_caption('a fast drum and bass track without drums')  # the genre names no instruments
    assert labels(w) == set() and w['no_drums'] and not w['drums'] and w['pace'] == 'high'
    w = parse_caption("a deep, sparse piece for double bass, Eb major")
    assert labels(w) == {'double bass'} and w['key'] == (3, 'major') and w['register'] == 'low' and w['pace'] == 'low'
    w = parse_caption('Енергійна музика у тональності ля-бемоль мажор зі слеп-басом')
    assert labels(w) == {'slap bass'} and w['key'] == (8, 'major') and w['energy'] == 'energetic'
    w = parse_caption('a soft and loud song with guitar')  # contradicting levels are not checked
    assert labels(w) == {'guitar'} and w['dynamics'] is None
    assert parse_caption('') == parse_caption('a piece by Chopin')  # nothing checkable
    sad = frozenset({'sad', 'x_sad', 'p_sad'})  # 'melancholic' names all three label sources' sad
    for text in ('A melancholic pop rock song with drums', 'Меланхолійна поп-рок пісня з ударними'):
        w = parse_caption(text)
        assert w['genres'] == [frozenset({'poprock'})] and w['moods'] == [sad] and w['drums'], text
    w = parse_caption('a soft deep jazz piece, Зіграй у стилі саундтреку')  # 'soft' / 'deep' are levels too
    assert w['dynamics'] == 'low' and w['register'] == 'low' and frozenset({'deep'}) in w['moods']
    assert {frozenset({'jazz'}), frozenset({'soundtrack'})} <= set(w['genres'])


def test_instrument_limits():
    """'only' / 'solo' (one instrument) / 'without' become hard limits; a plain list or a guitar solo in a band
    don't."""
    piano, guitar, bass = {0, 1}, set(range(24, 32)), set(range(32, 40))
    for text, want in (('solo piano', piano), ('piano only', piano), ('Only piano and drums', piano | {128}),
                       ('just drums', {128}), ('лише фортепіано', piano), ('фортепіано соло', piano),
                       ('тільки фортепіано і бас', piano | bass), ('solo piano without drums', piano)):
        assert set(allowed_programs(parse_caption(text))) == want, text
    for text in ('a rock song with a guitar solo, bass and drums', 'a sad slow piano ballad', 'piano and bass'):
        assert allowed_programs(parse_caption(text)) is None, text
    for text, gone in (('a song without guitar', guitar), ('пісня без гітари та ударних', guitar | {128}),
                       ('a song without guitar, bass or drums', guitar | bass | {128}),
                       ('a piano piece with no drums and no bass', bass | {128})):
        w = parse_caption(text)
        assert set(range(129)) - set(allowed_programs(w)) == gone, text
        assert not {label for label, _ in w['instruments']} & {'guitar', 'bass'}, 'banned ones are not wanted'
    assert allowed_programs(parse_caption('solo piano'), base=[0, 33]) == [0]
    a = dict(present={0, 33}, drums=0.0, ops=5.0, vel_mean=21.0, pitch_mean=60.0, key=0, key_conf=0.2)
    assert match(parse_caption('piano only'), a)[1]['only'] == 0.0 and match(parse_caption('no bass'), a)[0] == 0.0


def test_parse_round_trip():
    """Captions rendered by the training caption builder (EN + UK) parse back to their instruments (as GM programs),
    drums, key, 'no drums', genre and mood, and never to a level they don't name (unless a mood word says it)."""
    ids = [i for i in V.INSTRUMENT if i != 'drums']
    for seed in range(400):
        rng = random.Random(seed)
        inst = rng.sample(ids, rng.randint(1, 3)) + (['drums'] if rng.random() < 0.4 else [])
        a = dict(instruments=inst)
        if 'drums' not in inst and rng.random() < 0.4:
            a['no_drums'] = True
        if len(inst) == 1 and rng.random() < 0.5:
            a['solo'] = True
            a.pop('no_drums', None)
        tonic, mode = rng.choice(NOTE_NAMES), rng.choice(['major', 'minor'])
        a['key'] = (tonic, mode)
        for name in ('pace', 'dynamics', 'register'):
            if rng.random() < 0.5:
                a[name] = rng.choice(['low', 'high'])
        if rng.random() < 0.6:
            a['genre'] = [rng.choice(sorted(V.GENRE))]
        if rng.random() < 0.6:
            a['mood'] = [rng.choice(sorted(m for m in V.MOOD if V.MOOD[m].get('kind') != 'use'))]
        want = [FAMILY[FAMILY_OF_ID[i]] if i in FAMILY_OF_ID else {p for p in range(129) if V.GM_TO_ID[p] == i}
                for i in inst if i != 'drums']
        pc = NOTE_NAMES.index(tonic)
        for text in (render_en(dict(a), random.Random(seed)), render_uk(dict(a), random.Random(seed))):
            w = parse_caption(text)
            got = [progs for _, progs in w['instruments']]  # a word of two ids ('saxophone') gives one label
            assert all(any(e & g for g in got) for e in want) and all(any(g & e for e in want) for g in got), \
                (text, labels(w), inst)
            assert w['drums'] == ('drums' in inst), text
            assert w['no_drums'] == bool(a.get('no_drums')), text
            assert w['key'] in ((pc, mode), (None, mode)), (text, w['key'])
            for name in ('pace', 'dynamics', 'register'):
                assert w[name] in (None, a.get(name)) or a.get('mood'), (text, name, w[name])  # moods say 'soft' too
            for kind, named in (('genres', a.get('genre', [])), ('moods', a.get('mood', []))):
                assert all(any(i in got for got in w[kind]) for i in named), (text, kind, w[kind], named)


def band(n=600, seed=0, drums=True, nps=10.0):
    """(tokens (n, 4), programs (n,)): guitar 27, bass 33 and optionally drums in turn, ~nps notes/s."""
    rng = np.random.default_rng(seed)
    progs = np.array([27, 33, 128] if drums else [27, 33])[np.arange(n) % (3 if drums else 2)]
    dt = np.full(n, max(1, round(1 / nps / 0.02)))
    tok = np.stack([rng.integers(52, 72, n), np.full(n, 22), np.full(n, 20), dt], 1)
    return tok, progs


def test_match_notes():
    a = note_attributes(*band())
    assert a['present'] == {27, 33, 128} and a['runaway'] == 0.0
    score, checks = match(parse_caption('a rock song with drums, electric guitar and bass'), a)
    assert score == 1.0 and checks == {'electric guitar': 1.0, 'bass': 1.0, 'drums': 1.0}
    score, checks = match(parse_caption('solo piano without drums'), a)
    assert checks == {'piano': 0.0, 'no drums': 0.0, 'solo': 0.0, 'only': 0.0} and score == 0.0
    assert match(parse_caption('a guitar piece without drums'), note_attributes(*band(drums=False)))[0] == 1.0
    assert note_attributes(*band(nps=60.0))['runaway'] == 1.0, 'every 10-s window over 40 notes/s'
    assert match(parse_caption('a piece by Chopin'), a) == (1.0, {})
    assert note_attributes(np.zeros((0, 4)), np.zeros(0))['n'] == 0


def test_match_levels_and_key():
    tok, prog = band(drums=False)
    tok[:, 0] = np.where(np.arange(len(tok)) % 3 == 0, 57, np.where(np.arange(len(tok)) % 3 == 1, 60, 64))  # A minor
    a = note_attributes(tok, prog)
    assert a['key'] == 21 and a['key_conf'] > 0
    assert match(parse_caption('a piece in A minor'), a)[1]['key'] == 1.0
    assert match(parse_caption('a piece in C minor'), a)[1]['key'] == 0.5, 'mode right, tonic wrong'
    assert match(parse_caption('a piece in a major key'), a)[1]['key'] == 0.0
    slow = note_attributes(*band(drums=False, nps=1.0))
    assert match(parse_caption('a slow piece'), slow)[1]['pace'] == 1.0
    assert match(parse_caption('a fast piece'), slow)[1]['pace'] == 0.0
    assert match(parse_caption('a calm piece'), slow)[1]['energy'] == 1.0


def test_genre_mood_labels():
    a = dict(present={27, 33, 128}, drums=0.3, ops=5.0, vel_mean=21.0, pitch_mean=60.0, key=21, key_conf=0.2,
             genres={'pop'}, moods={'happy'})
    _, checks = match(parse_caption('a melancholic pop rock song'), a)
    assert checks == {'poprock': 0.5, 'p_sad/sad/x_sad': 0.0}, 'a related genre, another mood'
    _, checks = match(parse_caption('a happy pop song'), a)
    assert checks == {'pop': 1.0, 'happy/x_happy': 1.0}, checks
    _, checks = match(parse_caption('a sad jazz piece'), {**a, 'genres': set(), 'moods': set()})
    assert checks == {'jazz': 0.5, 'p_sad/sad/x_sad': 0.5}, 'no labels: undecided'
    assert match(parse_caption('a sad jazz piece'), note_attributes(*band()))[1] == {}, 'notes have no labels'
    both = {**a, 'genres': {'pop', 'rock'}}
    assert match(parse_caption('a pop rock song'), both)[1] == {'poprock': 1.0}, 'all parts of a compound genre'
    score, checks = match(parse_caption('a melancholic song with drums, electric guitar and bass'), a)
    assert len(checks) == 4 and score == 0.5, 'instruments 1 and mood 0 weigh the same, however many instruments'


def test_file_attributes():
    bits = np.zeros((2, 136), np.uint8)
    bits[0, [27, 33, 128]] = 1
    bits[1, 0] = 1
    f = dict(programs=np.packbits(bits, axis=1, bitorder='little'), drums=np.array([0.3, 0.0]),
             ops=np.array([5.0, 2.0]), vel_mean=np.array([21.0, 18.0]), pitch_mean=np.array([60.0, 60.0]),
             key=np.array([21, 0]), key_conf=np.array([0.2, 0.2]))
    attrs = FileAttributes(f)
    w = parse_caption('solo piano, soft, in C major')
    assert match(w, attrs(1))[0] == 1.0 and match(w, attrs(0))[1]['piano'] == 0.0
    assert match(parse_caption('drums, electric guitar and bass in A minor'), attrs(0))[0] == 1.0


if __name__ == '__main__':
    run(globals())
