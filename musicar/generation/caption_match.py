"""
What a caption asks for that can be checked on notes, and how well notes (or a store file) match it: best-of-N in
generate.py and the attribute part of caption retrieval (musicar/generation/retrieval.py).

parse_caption(text) reads, in English or Ukrainian, the word forms of the caption vocabulary (musicar/text/vocab.py,
the words the training captions are rendered from): named instruments (the generic 'guitar' / 'bass' / 'strings' /
'гітара' / 'бас' / 'струнні' accept any instrument of their kind), drums or no drums, solo, key or mode, pace (sparse /
busy, slow / fast), dynamics (soft / loud), register (low / high), calm / energetic moods, and genres and moods
as label ids. Genres and moods can't be measured on notes: they are only checked against a store file's labels
(retrieval); artists are left to the model.

match(wants, attrs) scores 0-1: the mean over the asked-for checks, judged with the caption builder's thresholds
(musicar/text/captions.py: an instrument counts with >= 2% of the notes and >= 8 notes, the pace / dynamics /
register quantiles, the key-confidence median), so notes are judged by the measures their training captions came
from. note_attributes() also gives the runaway share: 10-second windows over 40 notes/s (eval/compare/text_follow.py).
"""

import re

import numpy as np

from musicar.data.features import DT, FIELDS, file_features
from musicar.text import vocab as V
from musicar.text.captions import KEY_CONF_MEDIAN, NOTE_NAMES, OPS_Q, PITCH_Q, VEL_Q, _level

DRUMS = 128
PRESENT = 0.02          # share of the notes from which an instrument counts (and >= PRESENT_NOTES notes)
PRESENT_NOTES = 8
RUNAWAY_NPS = 40.0      # notes/s of a runaway window
RUNAWAY_WINDOW = 10.0   # seconds
LEVELS = ('pace', 'dynamics', 'register')
# generic words: any instrument of the kind (the vocabulary ids 'electric bass' / 'strings' are named 'bass' /
# 'strings' too, so they get the whole family)
FAMILY = {'guitar': frozenset(range(24, 32)), 'bass': frozenset(range(32, 40)),
          'strings': frozenset({40, 41, 42, 44, 45, 48, 49, 50, 51})}
FAMILY_OF_ID = {'electric bass': 'bass', 'strings': 'strings'}
GUITAR_WORDS = ('guitar', 'guitars', 'гітара', 'гітари', 'гітарою', 'гітарі', 'гітар', 'гітару')
NO_DRUMS = r'without drums|with no drums|no drums|без ударних|без барабанів'
ONLY = r'(?<![\w-])(only|just|exclusively|nothing but|лише|тільки|виключно|винятково)(?![\w-])'
WITHOUT = r'(?<![\w-])(?:without|with no|no|без)\s+(?:any |a |an |the )?'
NOTE_PC = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
EN_KEY = re.compile(r'(?<![\w#])([a-g])(#|b)?\s+(major|minor)\b(?!\s+key)')
EN_MODE = re.compile(r'\b(?:in\s+)?an?\s+(major|minor)\s+key\b')
UK_MODE = re.compile(r'(?<![\w-])у\s+(мажорі|мінорі)(?![\w-])|(?<![\w-])(мажорн|мінорн)\w*')


def _norm(text):
    """Lower case, one kind of apostrophe, single spaces, and Ukrainian 'в' read as 'у' (the vocabulary's phrases
    use 'у': 'у повільному темпі', 'у мінорі')."""
    text = re.sub(r"[’ʼ`´]", "'", str(text).lower())
    text = re.sub(r'(?<![\w-])в(?![\w-])', 'у', text)
    return ' ' + re.sub(r'\s+', ' ', text) + ' '


def _stem(m, f):
    """Common start of a Ukrainian adjective's masculine and feminine forms ('тихий', 'тиха' -> 'тих'), matched with
    any ending ('тиху', 'тихе')."""
    n = 0
    while n < min(len(m), len(f)) and m[n] == f[n]:
        n += 1
    return m[:n]


def _word(s):
    return r'(?<![\w-])' + re.escape(s) + r'(?![\w-])'


def _uk_adj(m, f):
    return r'(?<![\w-])' + re.escape(_stem(m, f)) + r"[\w']{0,4}(?![\w-])"


def _label_patterns(table):
    """[(regex, frozenset of ids)] for the genre or mood vocabulary: English names and phrases, Ukrainian adjectives
    (any ending), genre prefixes ('поп-рок' in 'поп-рок пісня'), tags and phrases. A word of several ids ('sad': the
    MidiCaps, XMIDI and pseudo-label moods; 'folk': folk and traditional) accepts any of them."""
    owners = {}
    for k, e in table.items():
        words = [w.lower() for w in e.get('en', []) + e.get('en_post', []) + e.get('uk_post', [])]
        words += [e['uk_tag'].lower()] if e.get('uk_tag') else []
        for w in words:
            owners.setdefault(('w', w), set()).add(k)
        if e.get('prefix'):  # 'гранж-твір', 'поп-рок пісня'
            owners.setdefault(('p', e['prefix'].lower()), set()).add(k)
        for pair in e.get('uk') or []:
            owners.setdefault(('a', pair), set()).add(k)
    pattern = dict(w=_word, p=lambda w: r'(?<![\w-])' + re.escape(w) + r'(?![\w])', a=lambda pair: _uk_adj(*pair))
    out = [(pattern[kind](w), len(w if kind != 'a' else w[0]), frozenset(ids)) for (kind, w), ids in owners.items()]
    out.sort(key=lambda x: -x[1])
    return [(re.compile(p), ids) for p, _, ids in out]


def _genre_parts():
    """Genre id -> the genre ids its English name is made of ('poprock' -> {'pop', 'rock'}), for partial matches."""
    by_name = {w.lower(): g for g, e in V.GENRE.items() for w in e.get('en', [])}
    return {g: {by_name[w] for w in e.get('en', [''])[0].lower().split() if w in by_name and by_name[w] != g}
            for g, e in V.GENRE.items()}


def _instrument_patterns():
    """[(regex, label, GM programs)], longest surface first, so 'double bass' is read before 'bass'. A word of
    several ids ('saxophone': alto and tenor; 'дзвіночки': glockenspiel and bells) accepts any of them."""
    owners = {}
    for iid, (en, uk) in V.INSTRUMENT.items():
        if iid == 'drums':
            continue
        fam = FAMILY_OF_ID.get(iid)
        progs = FAMILY[fam] if fam else frozenset(p for p in range(129) if V.GM_TO_ID[p] == iid)
        for s in set(en) | {form for forms in uk for form in forms[:3]}:
            owners.setdefault(s.lower(), {})[fam or iid] = progs
    for w in GUITAR_WORDS:
        owners[w] = {'guitar': FAMILY['guitar']}
    out = [(s, '/'.join(sorted(ids)), frozenset().union(*ids.values())) for s, ids in owners.items()]
    out.sort(key=lambda x: -len(x[0]))
    return out


def _level_patterns():
    """[(regex, attribute, value)]: pace / dynamics / register / energy words in English and Ukrainian."""
    out = []

    def add(table, attr, value_of=lambda k: k):
        for k, e in table.items():
            for w in e.get('en', []) + e.get('en_post', []):
                out.append((_word(w.lower()), attr, value_of(k)))
            for pair in e.get('uk') or []:
                out.append((_uk_adj(*pair), attr, value_of(k)))
            for w in e.get('uk_post', []):
                out.append((_word(w.lower()), attr, value_of(k)))

    add(V.PACE, 'pace')
    add(V.RHYTHM, 'pace')
    add(V.DYNAMICS, 'dynamics')
    add(V.REGISTER, 'register')
    add(V.TEMPO, 'pace', lambda k: {'slow': 'low', 'moderate': 'mid'}.get(k, 'high'))
    for word, (cls, italian) in V.TEMPO_WORD.items():
        level = {'slow': 'low', 'moderate': 'mid'}.get(cls, 'high')
        for w in filter(None, (italian, V.ITALIAN_UK.get(italian or ''))):
            out.append((_word(w), 'pace', level))
    for ids, energy in ((V.CALM_MOODS, 'calm'), (V.ENERGETIC_MOODS, 'energetic')):
        add({k: V.MOOD[k] for k in ids if k in V.MOOD}, 'energy', lambda k, e=energy: e)
    out.sort(key=lambda x: -len(x[0]))
    return [(re.compile(p), attr, value) for p, attr, value in out]


_INSTRUMENT_WORDS = _instrument_patterns()
_INSTRUMENTS = [(re.compile(_word(s)), label, progs) for s, label, progs in _INSTRUMENT_WORDS]
_DRUM_WORDS = [w.lower() for w in V.INSTRUMENT['drums'][0]] + [f for fs in V.INSTRUMENT['drums'][1] for f in fs[:3]]
_BAN_WORDS = sorted(_INSTRUMENT_WORDS + [(w, 'drums', frozenset({DRUMS})) for w in _DRUM_WORDS],
                    key=lambda x: -len(x[0]))
_BAN_ALT = '|'.join(re.escape(s) for s, _, _ in _BAN_WORDS)
_BAN_OF = {s: (label, progs) for s, label, progs in _BAN_WORDS}
# 'without X', 'without X and Y', 'без X, Y та Z': every instrument of the list
_AND = r'\s*(?:,|and|or|nor|та|і|й|чи|ні)\s+(?:any |a |an |the )?'
_WITHOUT = re.compile(WITHOUT + rf'({_BAN_ALT})((?:{_AND}(?:{_BAN_ALT}))*)' + r'(?![\w-])')
_BAN_ITEM = re.compile(rf'(?<![\w-])({_BAN_ALT})(?![\w-])')
_LEVELS = _level_patterns()
_GENRES = _label_patterns(V.GENRE)
_MOODS = _label_patterns(V.MOOD)
GENRE_PARTS = _genre_parts()
_DRUM_NOUNS = sorted([w.lower() for w in V.DRUM_NOUNS_EN] + [f for n in V.DRUM_NOUNS_UK for f in n[:2]]
                     + [w.lower() for w in V.INSTRUMENT['drums'][0]]
                     + [f for forms in V.INSTRUMENT['drums'][1] for f in forms[:3]], key=len, reverse=True)
_UK_TONIC = sorted(V.KEY_UK.items(), key=lambda kv: -len(kv[1]))
UK_KEY = re.compile(r'(?<![\w-])(' + '|'.join(re.escape(u) for _, u in _UK_TONIC) + r')\s+(мажор|мінор)\w*')
_UK_PC = {u: (NOTE_PC[t[0].lower()] + (1 if t[1:] == '#' else -1 if t[1:] == 'b' else 0)) % 12 for t, u in _UK_TONIC}


def parse_caption(text):
    """The checkable wants of a caption: dict(instruments=[(label, GM programs)], drums, no_drums, solo, key=(pitch
    class or None, 'major' / 'minor') or None, pace / dynamics / register = 'low' / 'mid' / 'high' or None,
    energy = 'calm' / 'energetic' or None, genres / moods = [frozenset of label ids], one per word, only = the GM
    programs the caption allows exclusively or None, banned = [(label, GM programs)] it excludes).
    only: 'only' / 'just' / 'лише' / 'тільки' ... with named instruments ('piano only', 'лише фортепіано і бас'), or
    'solo' with one named instrument and no drums ('solo piano', 'фортепіано соло'); a plain list stays soft (captions
    name a subset of the instruments), as does 'a song with a guitar solo, bass and drums'. banned: 'without X',
    'no X', 'без X' (drums: no_drums)."""
    t = _norm(text)
    w = dict(instruments=[], drums=False, no_drums=False, solo=False, key=None, energy=None, genres=[], moods=[],
             only=None, banned=[], **{k: None for k in LEVELS})
    labels = t  # genres and moods: read from their own copy (their words may also be level words: 'soft', 'deep')
    for kind, patterns in (('genres', _GENRES), ('moods', _MOODS)):
        for rx, ids in patterns:
            while (m := rx.search(labels)):
                labels = labels[:m.start()] + ' | ' + labels[m.end():]
                if ids not in w[kind]:
                    w[kind].append(ids)

    def cut(m):
        nonlocal t
        t = t[:m.start()] + ' | ' + t[m.end():]

    def take(pattern):
        found = []
        while (m := re.search(pattern, t)):
            found.append(m)
            cut(m)
        return found

    t = re.sub(_word('drum and bass'), ' | ', t)  # a genre, not drums + bass
    w['no_drums'] = bool(take(NO_DRUMS))
    if (m := EN_MODE.search(t)):
        w['key'] = (None, m.group(1))
        cut(m)
    if (m := UK_KEY.search(t)):
        w['key'] = (_UK_PC[m.group(1)], 'major' if m.group(2) == 'мажор' else 'minor')
        cut(m)
    if (m := EN_KEY.search(t)):
        pc = (NOTE_PC[m.group(1)] + {'#': 1, 'b': -1}.get(m.group(2), 0)) % 12
        w['key'] = (pc, m.group(3))
        cut(m)
    if (m := UK_MODE.search(t)):
        if w['key'] is None:
            word = m.group(1) or m.group(2)
            w['key'] = (None, 'major' if word.startswith('мажор') else 'minor')
        cut(m)
    w['solo'] = bool(re.search(r'(?<![\w-])(solo|соло)(?![\w-])', t))
    strict = bool(re.search(ONLY, t))
    while (m := _WITHOUT.search(t)):
        cut(m)
        for word in [m.group(1)] + _BAN_ITEM.findall(m.group(2)):
            label, progs = _BAN_OF[word]
            if label == 'drums':
                w['no_drums'] = True
            elif all(label != b for b, _ in w['banned']):
                w['banned'].append((label, progs))
    for s in _DRUM_NOUNS:
        if take(_word(s)):
            w['drums'] = not w['no_drums']
    seen = set()
    for rx, label, progs in _INSTRUMENTS:
        while (m := rx.search(t)):
            cut(m)
            if label not in seen:
                seen.add(label)
                w['instruments'].append((label, progs))
    named = frozenset().union(*(progs for _, progs in w['instruments'])) | ({DRUMS} if w['drums'] else frozenset())
    if named and (strict or (w['solo'] and len(w['instruments']) == 1 and not w['drums'])):
        w['only'] = named
    for rx, attr, value in _LEVELS:
        while (m := rx.search(t)):
            cut(m)
            if w[attr] not in (None, value):  # 'soft' and 'loud' both: no check
                value = 'clash'
            w[attr] = value
    for attr in LEVELS + ('energy',):
        if w[attr] == 'clash':
            w[attr] = None
    return w


def describe(wants):
    """A short line of what is checked."""
    parts = [', '.join(label for label, _ in wants['instruments'])] if wants['instruments'] else []
    parts += [k.replace('_', ' ') for k in ('drums', 'no_drums', 'solo') if wants[k]]
    if wants['only'] is not None:
        parts.append('only these instruments')
    parts += [f'not {label}' for label, _ in wants['banned']]
    if wants['key']:
        pc, mode = wants['key']
        parts.append(('' if pc is None else NOTE_NAMES[pc] + ' ') + mode)
    parts += [f'{k} {wants[k]}' for k in LEVELS + ('energy',) if wants[k]]
    parts += [f"{k[:-1]} {'/'.join(sorted(ids))}" for k in ('genres', 'moods') for ids in wants[k]]
    return '; '.join(parts) or 'nothing checkable'


def note_attributes(tok, prog):
    """Attributes of a note sequence (tok (n, 4) pitch, velocity, duration, delta time; prog (n,) GM program,
    128 = drums) for match(), plus nps, seconds and the runaway share."""
    tok, prog = np.asarray(tok, dtype=np.int64), np.asarray(prog, dtype=np.int64)
    n = len(prog)
    if n == 0:
        return dict(present=set(), drums=0.0, ops=0.0, vel_mean=0.0, pitch_mean=0.0, key=-1, key_conf=0.0, nps=0.0,
                    seconds=0.0, runaway=0.0, n=0)
    f = dict(zip(FIELDS, file_features(tok, prog)[0]))
    counts = np.bincount(prog, minlength=129)
    present = set(np.flatnonzero((counts >= PRESENT_NOTES) & (counts >= PRESENT * n)).tolist())
    onset = np.cumsum(tok[:, 3]) * DT
    counts_w = np.bincount((onset // RUNAWAY_WINDOW).astype(np.int64))
    span = np.full(len(counts_w), RUNAWAY_WINDOW)
    span[-1] = onset[-1] - (len(counts_w) - 1) * RUNAWAY_WINDOW + DT  # the last window: up to the last onset
    keep = span >= RUNAWAY_WINDOW / 2
    keep[0] = True  # a short piece still has its one window
    rate = counts_w[keep] / span[keep]
    return dict(present=present, drums=float(f['drums']), ops=float(f['ops']), vel_mean=float(f['vel_mean']),
                pitch_mean=float(f['pitch_mean']), key=int(f['key']), key_conf=float(f['key_conf']),
                nps=float(f['nps']), seconds=float(f['seconds']), runaway=float((rate > RUNAWAY_NPS).mean()), n=n)


class FileAttributes:
    """Attributes of store files for match(), from their feature table (data/labels/<store>.features.npz) and, with
    caps (the store's StoreCaptions), their genre and mood labels."""

    def __init__(self, features, caps=None):
        self.f, self.caps = features, caps
        self.bits = np.unpackbits(features['programs'], axis=1, bitorder='little')[:, :129].astype(bool)

    def __call__(self, i):
        f = self.f
        a = dict(present=set(np.flatnonzero(self.bits[i]).tolist()), drums=float(f['drums'][i]),
                 ops=float(f['ops'][i]), vel_mean=float(f['vel_mean'][i]), pitch_mean=float(f['pitch_mean'][i]),
                 key=int(f['key'][i]), key_conf=float(f['key_conf'][i]))
        if self.caps is not None:
            a['genres'], a['moods'] = self.caps.genre_mood(i)
        return a


def match(wants, a, categories=False):
    """(score 0-1, {check: 0-1}), with categories=True also {category: {check: 0-1}} (instruments, key, levels,
    genre, mood: what the caption asked for). Score = the mean over the asked-for categories (instruments incl. drums
    / solo, key, levels incl. energy, genre, mood) of the mean of their checks, so a long instrument list doesn't
    outweigh the mood; 1.0 when nothing is checkable. Levels: 1 on the asked level, 0.5 one step off (the middle), 0 on the opposite
    one; key: half for the mode, half for the tonic, halved when the estimate is unsure (confidence under the caption
    builder's median). Genres and moods only when attrs has labels (FileAttributes with caps): 1 a match (a compound
    genre: all its parts, 'pop' + 'rock' for 'pop rock'), 0.5 a related genre or no labels of the kind, 0 another."""
    checks, cats, named = {}, {}, {}

    def put(cat, name, value):
        checks[name] = float(value)
        cats.setdefault(cat, []).append(float(value))
        named.setdefault(cat, {})[name] = float(value)

    for label, progs in wants['instruments']:
        put('instruments', label, bool(progs & a['present']))
    if wants['drums']:
        put('instruments', 'drums', DRUMS in a['present'])
    if wants['no_drums']:
        put('instruments', 'no drums', DRUMS not in a['present'])
    pitched = a['present'] - {DRUMS}
    if wants['solo']:
        put('instruments', 'solo', len(pitched) == 1 and DRUMS not in a['present'])
    if wants['only'] is not None:
        put('instruments', 'only', a['present'] <= wants['only'])
    for label, progs in wants['banned']:
        put('instruments', f'not {label}', not progs & a['present'])
    if wants['key']:
        pc, mode = wants['key']
        v = 0.0
        if a['key'] >= 0:
            mode_ok = ('minor' if a['key'] >= 12 else 'major') == mode
            v = float(mode_ok) if pc is None else 0.5 * mode_ok + 0.5 * (mode_ok and a['key'] % 12 == pc)
            v = v if a['key_conf'] >= KEY_CONF_MEDIAN else v / 2
        put('key', 'key', v)
    for name, value, q in (('pace', a['ops'], OPS_Q), ('dynamics', a['vel_mean'], VEL_Q),
                           ('register', a['pitch_mean'], PITCH_Q)):
        if wants[name] is None or (name == 'register' and not pitched):
            continue
        got = _level(value, q) or 'mid'
        put('levels', name, 1.0 if got == wants[name] else 0.5 if 'mid' in (got, wants[name]) else 0.0)
    if wants['energy']:
        bad = 'high' if wants['energy'] == 'calm' else 'low'
        off = (_level(a['ops'], OPS_Q) == bad) + (_level(a['vel_mean'], VEL_Q) == bad)
        put('levels', 'energy', 1.0 - 0.5 * off)
    if 'genres' in a:
        have = a['genres'] | set().union(*(GENRE_PARTS[g] for g in a['genres']))
        for ids in wants['genres']:
            full = ids & a['genres'] or any(GENRE_PARTS[g] and GENRE_PARTS[g] <= a['genres'] for g in ids)
            near = set().union(*(GENRE_PARTS[g] | {g} for g in ids)) & have
            put('genre', '/'.join(sorted(ids)), 1.0 if full else 0.5 if near or not a['genres'] else 0.0)
        for ids in wants['moods']:
            put('mood', '/'.join(sorted(ids)), 1.0 if ids & a['moods'] else 0.5 if not a['moods'] else 0.0)
    score = float(np.mean([np.mean(v) for v in cats.values()])) if cats else 1.0
    return (score, checks, named) if categories else (score, checks)


def allowed_programs(wants, base=None):
    """The GM programs (0-128) a caption leaves to the instrument head as a hard mask: its 'only' set, minus what it
    bans ('without X', 'no drums'), within `base` (e.g. --instruments) when given; None when it constrains nothing."""
    if wants is None:
        return None
    banned = frozenset().union(*(progs for _, progs in wants['banned'])) | ({DRUMS} if wants['no_drums'] else set())
    if wants['only'] is None and not banned:
        return base
    allowed = set(wants['only'] if wants['only'] is not None else range(129)) - banned
    if base is not None:
        allowed &= set(base)
    return sorted(allowed)
