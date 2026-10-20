"""
Captions per file of a token store, for text control (context/claude/text_control.md, build step 5).

Per file it reads data/labels/<store>.*:
  .jsonl         MidiCaps fields + full caption, XMIDI (emotion, genre) labels, GigaMIDI v2 metadata (styles, artist,
                 title; styles -> genre / mood ids through vocab.GIGAMIDI_STYLE)  (tier 1, data_prep/labels/build_labels.py)
  .features.npz  key, onsets/s, register, velocity, length, instruments, drum share    (every file, song_features.py)
  .pseudo.npz    confident predicted mood group / genre (pseudo_labels.py), used only for files >= 20 s that are
                 not drum-only (out-of-distribution files get confident nonsense) and only without true labels
and data/captions/<store>.uk.fixed.jsonl: Ukrainian translations of the MidiCaps captions (data_prep/labels/fix_uk_captions.py).

A caption is drawn per window, in Ukrainian with probability uk_share, else in English. A MidiCaps file gives its full
caption (the English original or its Ukrainian translation) 40% of the time; otherwise the file's attributes are
collected as language-neutral ids, a random subset is kept (mood / genre p = 0.8, instruments 0.7, the rest 0.5, at
least one; clashing words such as 'calm' + 'loud' lose one side), and rendered with the hand-written vocabulary of
musicar/text/vocab.py: a noun phrase (adjectives in a fixed order, at most 3, Ukrainian ones agreed with the noun's
gender), the same after a request ('Play ...', 'Зіграй ...' with the accusative), led by the instruments, or tags.

    python -m musicar.text.captions gigamidi_replay_train --n 40 --lang uk     # samples for review
"""

import json
import random
from pathlib import Path

import numpy as np

from musicar.text import vocab as V

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
MIDICAPS_FULL = 0.4
UK_SHARE = 0.3
P_ATTR = dict(mood=0.8, genre=0.8, drums=1.0, ballad=1.0, instruments=0.7, use=0.6, solo=1.0,
              no_drums=0.4)  # other attributes: 0.5
NOT_ALONE = {'ballad', 'solo'}  # don't count as the one kept attribute: 'ballad' only sets the noun, 'solo' needs
                                # the instrument
FULL_LIST = 0.8   # files with <= 3 instruments: all of them named
KEEP_DRUMS = 0.7  # more than 3: drums kept in the named subset
MAX_ADJ = 3
PSEUDO_MIN_SECONDS = 20.0
DRUM_ONLY = 0.9
# feature quantiles of the local GigaMIDI stores (2026-10-17): onsets/s 20 / 80%, mean pitch of pitched files 20 / 80%,
# velocity bin 20 / 80%, length; key names only above the median key margin
OPS_Q = (3.24, 7.75)
PITCH_Q = (49.0, 68.1)
VEL_Q = (19.3, 23.8)
SECONDS_Q = (20.0, 180.0)
KEY_CONF_MEDIAN = 0.072
UK_VOWELS = set('аеєиіїоуюя')
# section captions: what can change inside a piece, read from the section's own notes (the file caption keeps mood,
# genre, key, length); per attribute the chance of naming it
SECTION_P = dict(instruments=0.7, pace=0.6, dynamics=0.6, register=0.5, drums=1.0, solo=1.0, no_drums=0.5)
DT = 0.02  # seconds per time bin
DRUM_PROGRAM = 128


def instrument_name(program):
    """GM program (128 = drums) -> its common English name ('Lead 1 (square)' -> 'synth lead')."""
    return V.INSTRUMENT[V.GM_TO_ID[int(program)]][0][0]


def _level(x, q):
    return 'low' if x < q[0] else 'high' if x > q[1] else None


def _join(items, last=' and '):
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + last + items[-1]


def _article(word):
    w = word.lstrip("'").lower()
    vowel = w[:1] in 'aeiou' and not w.startswith(('eu', 'uni', 'use')) or w.startswith(('8', '11', '18'))
    return 'an' if vowel else 'a'


def _uk_acc(phrase):
    """Feminine nominative -> accusative on the last word: сумна -> сумну, літня -> літню."""
    head, _, last = phrase.rpartition(' ')
    last = last[:-1] + 'у' if last.endswith('а') else last[:-1] + 'ю' if last.endswith('я') else last
    return f'{head} {last}' if head else last


def _uk_z(word):
    """з / зі before an instrumental: зі скрипкою, з фортепіано."""
    w = word.lower()
    return 'зі' if w[:1] in 'зсцчшщж' and len(w) > 1 and w[1] not in UK_VOWELS else 'з'


def _dedupe(words):
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def gigamidi_ids(gm):
    """GigaMIDI metadata of a file ({'styles': [...], 'artist': ...}) -> {'genre' / 'mood' / 'use': ids}. An era
    (Baroque, ...) replaces 'classical'; eras and decades come first ("a Baroque piece", "a '70s rock song")."""
    ids = [i for s in gm.get('styles', ()) for i in V.GIGAMIDI_STYLE.get(s, ())]
    ids = _dedupe(ids + V.GIGAMIDI_ARTIST.get(str(gm.get('artist') or '').lower(), []))
    if set(ids) & V.ERAS:
        ids = [i for i in ids if i != 'classical']
    out = {}
    for i in sorted(ids, key=lambda i: i not in V.ERAS | V.DECADES):
        kind = 'genre' if i in V.GENRE else V.MOOD[i].get('kind', 'mood')
        out.setdefault(kind, []).append(i)
    return out


class StoreCaptions:
    """Captions of the files of one token store (file index = position in the store). Missing label files -> every
    caption is ''. labeled(i): the file has MidiCaps or XMIDI labels or a mapped GigaMIDI style (tier 1)."""

    def __init__(self, store_name, labels_dir='data/labels', uk_share=UK_SHARE, captions_dir='data/captions'):
        d = Path(labels_dir)
        self.name = store_name
        self.uk_share = uk_share
        self.labels = {}
        if (d / f'{store_name}.jsonl').exists():
            for line in (d / f'{store_name}.jsonl').read_text().splitlines():
                row = json.loads(line)
                self.labels[row['file']] = row
        self.gm_ids = {i: ids for i, row in self.labels.items()
                       if 'gigamidi' in row and (ids := gigamidi_ids(row['gigamidi']))}
        self.uk_full = {}
        uk = Path(captions_dir) / f'{store_name}.uk.fixed.jsonl'
        if uk.exists():
            for line in uk.read_text().splitlines():
                row = json.loads(line)
                self.uk_full[row['file']] = row['uk']
        self.features = dict(np.load(d / f'{store_name}.features.npz')) \
            if (d / f'{store_name}.features.npz').exists() else None
        self.pseudo, self.pseudo_classes = None, {}
        if (d / f'{store_name}.pseudo.npz').exists():
            self.pseudo = dict(np.load(d / f'{store_name}.pseudo.npz'))
            meta = json.loads((d / f'{store_name}.pseudo.json').read_text())
            self.pseudo_classes = {t: v['classes'] for t, v in meta['tasks'].items()}
        self.instruments = None
        if self.features is not None:
            bits = np.unpackbits(self.features['programs'], axis=1, bitorder='little')[:, :129].astype(bool)
            self.instruments = bits

    def labeled(self, i):
        row = self.labels.get(int(i))
        return bool(row and ('midicaps' in row or 'xmidi' in row or int(i) in self.gm_ids))

    def caption(self, i, rng=random, lang=None):
        """A caption for file i, drawn with rng (random.Random or the random module); lang 'en' / 'uk' / None (drawn:
        'uk' with probability uk_share)."""
        i = int(i)
        lang = lang or ('uk' if rng.random() < self.uk_share else 'en')
        row = self.labels.get(i, {})
        mc = row.get('midicaps')
        full = (mc or {}).get('caption') if lang == 'en' else self.uk_full.get(i)
        if full and rng.random() < MIDICAPS_FULL:
            return full
        attrs = self._attributes(i, row, rng)
        if not attrs:
            return ''
        keep = {k: v for k, v in attrs.items() if rng.random() < P_ATTR.get(k, 0.5)}
        if not set(keep) - NOT_ALONE:
            k = rng.choice([k for k in attrs if k not in NOT_ALONE] or list(attrs))
            keep[k] = attrs[k]
        _resolve_clashes(keep)
        return render_en(keep, rng) if lang == 'en' else render_uk(keep, rng)

    # ---------------------------------------------------------------- attributes (language-neutral ids)

    def _attributes(self, i, row, rng):
        a = {}
        f = self.features
        drum_only = f is not None and f['drums'][i] >= DRUM_ONLY
        mc, xm = row.get('midicaps'), row.get('xmidi')
        if mc:
            tags = [m for m in (mc.get('mood') or []) if m in V.MOOD or m in ('slow', 'ballad')]
            if 'ballad' in tags:
                a['ballad'] = True
            if 'slow' in tags and not mc.get('tempo_word'):
                a['tempo'] = ('slow', None, None)
            feel = [m for m in tags if m in V.MOOD and V.MOOD[m].get('kind') != 'use']
            use = [m for m in tags if m in V.MOOD and V.MOOD[m].get('kind') == 'use']
            if feel:
                a['mood'] = rng.sample(feel, min(len(feel), rng.choice([1, 1, 2])))
            if use:
                a['use'] = [rng.choice(use)]
            genres = [g for g in (mc.get('genre') or []) if g in V.GENRE]
            if genres:
                a['genre'] = genres[:rng.choice([1, 1, 2])]
            if mc.get('tempo_word') in V.TEMPO_WORD:
                cls, italian = V.TEMPO_WORD[mc['tempo_word']]
                a['tempo'] = (cls, italian, mc.get('tempo'))
            key = str(mc.get('key') or '').split()
            if len(key) == 2 and key[0] in V.KEY_UK and key[1] in V.MODE_UK:
                a['key'] = tuple(key)
            if mc.get('time_signature') in V.TIME_SIGNATURES:
                a['time_sig'] = mc['time_signature']
            names = _dedupe(V.MIDICAPS_INSTRUMENT.get(n.lower(), n.lower()) for n in mc.get('instrument_summary') or [])
            names = [n for n in names if n in V.INSTRUMENT]
            if names:
                a['instruments'] = rng.sample(names, min(len(names), rng.randint(1, 3)))
        gm = self.gm_ids.get(i)
        if gm:  # site / Musicmap genres replace MidiCaps' tagger genres; moods only where MidiCaps has none
            if gm.get('genre'):
                g = gm['genre']
                a['genre'] = [g[j] for j in sorted(rng.sample(range(len(g)), min(len(g), rng.choice([1, 1, 2]))))]
            for kind in ('mood', 'use'):
                if gm.get(kind) and kind not in a:
                    a[kind] = [rng.choice(gm[kind])]
        if xm:
            emotion, genre = rng.choice(xm)
            a['mood'] = [f'x_{emotion}']
            if genre in V.GENRE:
                a['genre'] = [genre]
        if f is None:
            return a
        if drum_only:
            a['drums'] = True
            a.pop('instruments', None)
        else:  # the feature set lists every instrument with >= 2% of the notes, so leaving one out means something
            ids = _dedupe(V.GM_TO_ID[p] for p in np.flatnonzero(self.instruments[i]))
            if ids:
                if len(ids) <= 3 and rng.random() < FULL_LIST:
                    a['instruments'] = rng.sample(ids, len(ids))
                else:
                    pick = rng.sample(ids, min(len(ids), rng.randint(1, 3)))
                    if 'drums' in ids and 'drums' not in pick and rng.random() < KEEP_DRUMS:
                        pick[-1] = 'drums'
                    a['instruments'] = pick
                if len(ids) == 1 and ids[0] != 'drums':
                    a['solo'] = True
                elif 'drums' not in ids:
                    a['no_drums'] = True
        pseudo_ok = (not mc and not xm and not gm and self.pseudo is not None and not drum_only
                     and f['seconds'][i] >= PSEUDO_MIN_SECONDS)
        if pseudo_ok:
            if self.pseudo['xmidi_mood_keep'][i]:
                a['mood'] = ['p_' + self.pseudo_classes['xmidi_mood'][int(self.pseudo['xmidi_mood_pred'][i])]]
            if self.pseudo['midicaps_genre_keep'][i]:
                g = self.pseudo_classes['midicaps_genre'][int(self.pseudo['midicaps_genre_pred'][i])]
                if g in V.GENRE:
                    a['genre'] = [g]
        for name, value, q in (('pace', f['ops'][i], OPS_Q), ('dynamics', f['vel_mean'][i], VEL_Q),
                               ('length', f['seconds'][i], SECONDS_Q)):
            level = _level(value, q)
            if level and not (name == 'pace' and 'tempo' in a):  # MidiCaps' own tempo word already says it
                a[name] = level
        if not drum_only:
            level = _level(f['pitch_mean'][i], PITCH_Q)
            if level:
                a['register'] = level
            key = int(f['key'][i])
            if 'key' not in a and key >= 0 and f['key_conf'][i] >= KEY_CONF_MEDIAN:
                a['key'] = (NOTE_NAMES[key % 12], 'minor' if key >= 12 else 'major')
        return a


def section_attributes(tok, prog):
    """Language-neutral attribute ids of a stretch of notes from the notes themselves (tok (n, 4) pitch, velocity,
    duration, delta_time bins; prog (n,) GM programs, 128 = drums), with the file caption's thresholds: pace from
    onsets/s, dynamics from the mean velocity bin, register from the mean pitch of pitched notes, the instruments with
    >= 2% of the notes (at least 4), drums-only / solo / without drums. {} for no notes."""
    n = len(tok)
    if n == 0:
        return {}
    pitch, vel, dt = tok[:, 0], tok[:, 1], tok[:, 3]
    a = {}
    drums = prog == DRUM_PROGRAM
    if drums.mean() >= DRUM_ONLY:
        a['drums'] = True
    else:
        counts = np.bincount(prog, minlength=129)
        present = np.flatnonzero((counts >= 4) & (counts >= 0.02 * n))
        ids = _dedupe(V.GM_TO_ID[int(p)] for p in present)
        if ids:
            a['instruments'] = ids
            if len(ids) == 1 and ids[0] != 'drums':
                a['solo'] = True
            elif 'drums' not in ids:
                a['no_drums'] = True
        level = _level(float(pitch[~drums].mean()), PITCH_Q)
        if level:
            a['register'] = level
    seconds = max(float(dt[1:].sum()) * DT, 0.5)
    for name, value, q in (('pace', float((dt[1:] > 0).sum()) / seconds, OPS_Q), ('dynamics', float(vel.mean()), VEL_Q)):
        level = _level(value, q)
        if level:
            a[name] = level
    return a


def section_change(a, b):
    """How much a section differs from the one before it (section_attributes dicts a, b): one point each for another
    dynamics / pace / register level, another drums state (drums only / with drums / without), and instrument sets
    that differ by >= 2 instruments (one instrument near the 2% threshold flips often). 0 when b has no notes."""
    if not b:
        return 0

    def drums(x):
        return 'only' if x.get('drums') else 'without' if x.get('no_drums') or x.get('solo') else 'with'
    score = sum(a.get(k) != b.get(k) for k in ('dynamics', 'pace', 'register')) + (drums(a) != drums(b))
    return score + (len(set(a.get('instruments', ())) ^ set(b.get('instruments', ()))) >= 2)


def section_caption(tok, prog, rng=random, uk_share=UK_SHARE, lang=None):
    """A caption of a section from its own notes (section_attributes): a random subset (SECTION_P, at least one;
    instrument lists of <= 3 named whole with FULL_LIST, else 1-3 of them) rendered like the file captions. '' when
    nothing describes it."""
    attrs = section_attributes(tok, prog)
    if not attrs:
        return ''
    lang = lang or ('uk' if rng.random() < uk_share else 'en')
    ids = attrs.get('instruments')
    if ids and not (len(ids) <= 3 and rng.random() < FULL_LIST):
        attrs['instruments'] = rng.sample(ids, min(len(ids), rng.randint(1, 3)))
    elif ids:
        attrs['instruments'] = rng.sample(ids, len(ids))
    keep = {k: v for k, v in attrs.items() if rng.random() < SECTION_P.get(k, 0.5)}
    if not keep:
        k = rng.choice(list(attrs))
        keep[k] = attrs[k]
    if keep.get('solo') and 'instruments' not in keep:
        keep['instruments'] = attrs['instruments']
    return render_en(keep, rng) if lang == 'en' else render_uk(keep, rng)


def _resolve_clashes(a):
    """Drop the loudness / density / pace word that contradicts the mood or the tempo ('calm' + 'loud', 'lazy' +
    'busy', 'slow' + 'busy')."""
    moods = set(a.get('mood', [])) | set(a.get('use', []))
    tempo = a.get('tempo', (None,))[0]
    calm = bool(moods & V.CALM_MOODS) or tempo == 'slow'
    energetic = bool(moods & V.ENERGETIC_MOODS) or tempo in ('fast', 'very fast')
    if calm and a.get('dynamics') == 'high' or energetic and a.get('dynamics') == 'low':
        a.pop('dynamics')
    if calm and a.get('pace') == 'high' or energetic and a.get('pace') == 'low':
        a.pop('pace')


# ---------------------------------------------------------------- English

def render_en(a, rng):
    if a.get('drums'):
        noun = rng.choice(V.DRUM_NOUNS_EN)
    elif a.get('ballad'):
        noun = 'ballad'
    else:  # no 'music' next to a length word: 'short music' reads oddly
        noun = 'music' if rng.random() < 0.15 and 'length' not in a else rng.choice(V.NOUNS_EN)
    adjs, post = [], []
    feel = []  # mood words that may move into 'that feels ...'
    if 'length' in a:
        adjs.append(rng.choice(V.LENGTH[a['length']]['en']))
    for m in a.get('mood', []) + a.get('use', []):
        e = V.MOOD[m]
        if e.get('en') and not (e.get('en_post') and rng.random() < 0.4):
            word = rng.choice(e['en'])
            adjs.append(word)
            if e.get('kind') != 'use':
                feel.append(word)
        elif e.get('en_post'):
            post.append(rng.choice(e['en_post']))
    for name, table in (('dynamics', V.DYNAMICS), ('pace', V.PACE)):
        if name in a:
            adjs.append(rng.choice(table[a[name]]['en']))
    if 'tempo' in a:
        cls, italian, bpm = a['tempo']
        if bpm and rng.random() < 0.3:
            post.append(f'at {int(bpm)} BPM')
        elif italian and rng.random() < 0.4:
            post.append(f'at {_article(italian)} {italian} tempo')
        elif rng.random() < 0.6:
            adjs.append(rng.choice(V.TEMPO[cls]['en']))
        else:
            post.append(V.TEMPO[cls]['en_post'][0])
    if 'register' in a:
        if rng.random() < 0.4:
            adjs.append(rng.choice(V.REGISTER[a['register']]['en']))
        else:
            post.append(V.REGISTER[a['register']]['en_post'][0])
    adjs = _dedupe(adjs)[:MAX_ADJ]
    feel = [w for w in feel if w in adjs]
    genres = _dedupe(rng.choice(V.GENRE[g]['en']) for g in a.get('genre', []))
    era = set(a.get('genre', [])) & (V.ERAS | V.DECADES)  # "a '70s rock song", not "a '70s and rock song"
    genre = ' '.join(genres) if len(genres) < 2 or rng.random() < 0.5 or era else ' and '.join(genres)
    instruments = [rng.choice(V.INSTRUMENT[i][0]) for i in a.get('instruments', [])]
    solo = a.get('solo') and len(instruments) == 1
    if a.get('no_drums'):
        post.append(rng.choice(['without drums', 'with no drums']))
    if 'time_sig' in a:
        post.append(f"in {a['time_sig']} time")
    rng.shuffle(post)
    if 'key' in a:  # last: ', D minor' reads as an afterthought
        tonic, mode = a['key']
        key = f'{tonic} {mode}'
        post.append(rng.choice([f'in a {mode} key'] if rng.random() < 0.25 else
                               [f'in {key}', f'in the key of {key}', f', {key}']))

    tags = adjs + genres + ([f'solo {instruments[0]}'] if solo else instruments) \
        + (['no drums'] if a.get('no_drums') else []) + ([f'{a["key"][0]} {a["key"][1]}'] if 'key' in a else [])
    if 'tempo' in a and a['tempo'][1] and not any(t in V.TEMPO[a['tempo'][0]]['en'] for t in tags):
        tags.append(a['tempo'][1])
    form = rng.random()
    if form < 0.2 and len(_dedupe(tags)) > 1:  # tags: "sad, jazz, piano, D minor"
        return _finish_en(', '.join(_dedupe(tags)), rng)
    relative = ''
    if feel and rng.random() < 0.15:  # "a jazz piece that feels sad"
        word = feel[-1]
        adjs.remove(word)
        relative = rng.choice([f' that feels {word}', f' with a {word} mood', f' with a {word} feel'])
    head = ' '.join(w for w in adjs + [genre, noun] if w)
    np_ = head if noun == 'music' else f'{_article(head)} {head}'
    if solo and rng.random() < 0.5:  # "a sad solo piano piece"
        head = ' '.join(w for w in adjs + [genre, 'solo', instruments[0], noun] if w)
        np_ = f'{_article(head)} {head}'
        instruments = []
    elif solo:
        instruments = [f'solo {instruments[0]}']
    if instruments and form < 0.35:  # "piano and bass playing a sad jazz piece"
        text = f'{_join(instruments)} playing {np_}{relative}' + ''.join(' ' + p if p[0] != ',' else p for p in post)
        return _finish_en(text, rng)
    if instruments:
        prep = rng.choice(['with', 'featuring', 'for'])
        np_ += f' {prep} {_join(instruments)}'
    text = np_ + relative + ''.join(' ' + p if p[0] != ',' else p for p in post)
    if rng.random() < 0.35:
        return _finish_en(f'{rng.choice(V.REQUEST_EN)} {text}', rng, capital=False)
    return _finish_en(text, rng)


def _finish_en(text, rng, capital=None):
    if capital is None:
        capital = rng.random() < 0.5
    if capital:
        text = text[0].upper() + text[1:]
    return text + ('.' if rng.random() < 0.2 else '')


# ---------------------------------------------------------------- Ukrainian

def render_uk(a, rng):
    if a.get('drums'):
        nom, acc, gender = rng.choice(V.DRUM_NOUNS_UK)
    elif a.get('ballad'):
        nom, acc, gender = 'балада', 'баладу', 'f'
    else:  # no 'музика' next to a length word
        nom, acc, gender = V.MUSIC_UK if rng.random() < 0.15 and 'length' not in a else rng.choice(V.NOUNS_UK)
    fem = gender == 'f'
    adjs, post, tags = [], [], []  # adjs: (m, f) pairs

    if 'length' in a:
        adjs.append(rng.choice(V.LENGTH[a['length']]['uk']))
    for m in a.get('mood', []) + a.get('use', []):
        e = V.MOOD[m]
        if e.get('uk') and not (e.get('uk_post') and rng.random() < 0.4):
            adjs.append(rng.choice(e['uk']))
        elif e.get('uk_post'):
            post.append(rng.choice(e['uk_post']))
    for name, table in (('dynamics', V.DYNAMICS), ('pace', V.PACE)):
        if name in a:
            adjs.append(rng.choice(table[a[name]]['uk']))
    if 'tempo' in a:
        cls, italian, bpm = a['tempo']
        t = V.TEMPO[cls]
        if bpm and rng.random() < 0.3:
            post.append(f'у темпі {int(bpm)} ударів на хвилину')
        elif italian and rng.random() < 0.4:
            post.append(f'у темпі {V.ITALIAN_UK[italian]}')
            tags.append(V.ITALIAN_UK[italian])
        elif t['uk'] and rng.random() < 0.6:
            adjs.append(rng.choice(t['uk']))
        else:
            post.append(rng.choice(t['uk_post']))
    if 'register' in a:
        post.append(V.REGISTER[a['register']]['uk_post'][0])
    seen = set()
    adjs = [p for p in adjs if not (p[0] in seen or seen.add(p[0]))][:MAX_ADJ]
    tags = [p[0] for p in adjs] + tags

    prefix = ''
    if a.get('genre'):
        g = V.GENRE[rng.choice(a['genre'])]
        tags.append(g['uk_tag'])
        if g.get('uk'):
            adjs.append(rng.choice(g['uk']))  # the genre adjective stands next to the noun
        if g.get('prefix'):
            if a.get('drums'):
                post.append(f"у стилі {g['uk_tag']}")
            else:
                prefix = g['prefix'] + '-'
        if g.get('uk_post'):
            post.append(rng.choice(g['uk_post']))
    instruments = [rng.choice(V.INSTRUMENT[i][1]) for i in a.get('instruments', [])]
    solo = a.get('solo') and len(instruments) == 1
    tags += [f'{instruments[0][0]} соло'] if solo else [ins[0] for ins in instruments]
    if a.get('no_drums'):
        tags.append('без ударних')
        post.append(rng.choice(['без ударних', 'без барабанів']))
    if 'time_sig' in a:
        post.append(f"у розмірі {a['time_sig']}")
    rng.shuffle(post)
    if 'key' in a:  # last, before ', де грає ...'
        tonic, mode = a['key']
        tonic_uk, (mode_nom, mode_loc) = V.KEY_UK[tonic], V.MODE_UK[mode]
        tags.append(f'{tonic_uk} {mode_nom}')
        post.append(rng.choice([f'у {mode_loc}'] if rng.random() < 0.25 else
                               [f'у тональності {tonic_uk} {mode_nom}', f'у {tonic_uk} {mode_loc}',
                                f', {tonic_uk} {mode_nom}']))

    form = rng.random()
    if form < 0.2 and len(_dedupe(tags)) > 1:  # tags: "сумний, джаз, фортепіано, ре мінор"
        return _finish_uk(', '.join(_dedupe(tags)), rng)
    request = rng.random() < 0.35
    words = [(_uk_acc(f) if request else f) if fem else m for m, f in adjs]
    head = ' '.join(words + [prefix + (acc if request else nom)])
    if solo:
        head += f' для {instruments[0][1]} соло'
    elif instruments:
        style = rng.random()
        if style < 0.4:
            head += ' для ' + _join([ins[1] for ins in instruments], rng.choice([' і ', ' та ']))
        elif style < 0.75:
            ins = _join([x[2] for x in instruments], rng.choice([' і ', ' та ']))
            head += f' {_uk_z(ins)} {ins}'
        else:
            plural = len(instruments) > 1 or instruments[0][3]
            names = _join([x[0] for x in instruments], rng.choice([' і ', ' та ']))
            post.append(f", де {'грають' if plural else 'грає'} {names}")
    text = head + ''.join(' ' + p if p[0] != ',' else p for p in post)
    if request:
        return _finish_uk(f'{rng.choice(V.REQUEST_UK)} {text}', rng, capital=False)
    return _finish_uk(text, rng)


def _finish_uk(text, rng, capital=None):
    if capital is None:
        capital = rng.random() < 0.6
    if capital:
        text = text[0].upper() + text[1:]
    return text + ('.' if rng.random() < 0.2 else '')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='print sampled captions of a store for review')
    p.add_argument('store')
    p.add_argument('--n', type=int, default=40)
    p.add_argument('--lang', default=None, help="'en' / 'uk' (default: drawn)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-full', action='store_true', help='templates only (no MidiCaps full captions)')
    p.add_argument('--labels-dir', default='data/labels')
    p.add_argument('--files', default='all', help="'all' / 'gigamidi': only files with a mapped GigaMIDI style")
    args = p.parse_args()
    if args.no_full:
        MIDICAPS_FULL = 0.0
    c = StoreCaptions(args.store, args.labels_dir)
    rng = random.Random(args.seed)
    n_files = len(c.features['seconds']) if c.features is not None else max(c.labels) + 1
    pool = sorted(c.gm_ids) if args.files == 'gigamidi' else range(n_files)
    for _ in range(args.n):
        i = rng.choice(pool)
        text = c.caption(i, rng, args.lang) or "''"
        print(f"{text}    [{', '.join(c.labels[i]['gigamidi'].get('styles', []))}]" if args.files == 'gigamidi' else text)
