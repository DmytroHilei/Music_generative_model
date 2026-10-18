"""
Caption text written by the local Qwen3 (Ollama), once, offline (text-control plan 15h, step 1.6, Qwen part).

    python data_prep/labels/caption_bank.py paraphrase                 # English template paraphrases
    python data_prep/labels/caption_bank.py uk-labels                  # Ukrainian captions per mood / emotion x genre
    python data_prep/labels/caption_bank.py translate --stores data/cache/gigamidi_replay_train,...   # MidiCaps -> UK

paraphrase: every base template in TEMPLATES gets --n rewrites (several calls at rising temperature); placeholders
go to Qwen as [MOOD]-style tokens and come back as {mood}; families that already have --n rewrites in the output
file are skipped. A rewrite is kept only with exactly the base template's placeholders, no other braces, at most 2.5x the base length, no
non-Latin letters, and not a duplicate. -> data/captions/templates_en.json {family: {slots, base, paraphrases}}.

uk-labels: for each combination of a mood / emotion word and a genre (XMIDI emotion x XMIDI genre, 4 mood groups x
XMIDI + MidiCaps genres, and each word alone), --n short Ukrainian descriptions. -> data/captions/uk_labels.json
{"emotion=sad|genre=jazz": [...]}.

translate: the MidiCaps caption of every labeled file of each store (data/labels/<store>.jsonl) into Ukrainian.
Appends to data/captions/<store>.uk.jsonl ({"file", "uk"}), skips files already there, so it resumes after a stop.

Ukrainian checks (both): Cyrillic >= 60% of the letters, none of ы э ъ ё (Russian), no CJK, and for translations
every number of the English caption present (tempo, time signature) and 0.6-2.2x its length in characters.
The prompt carries a hand-written glossary (only the entries found in the caption), the Ukrainian name of the
caption's key (which the answer must contain) and one worked example (EXAMPLE_EN / EXAMPLE_UK).
A failed answer is asked again (new seed, up to --retries); still failing -> left out and counted.
Requests run --workers at a time (Ollama serves them in parallel up to its OLLAMA_NUM_PARALLEL).
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from musicar.jobstatus import JobStatus

OUT = Path('data/captions')

# Slot values the caption sampler fills in (examples go into the prompt so the rewrites stay grammatical)
SLOTS = {
    'mood': 'a mood adjective, e.g. "sad", "calm", "energetic", "tense", "romantic", "dark", "happy"',
    'genre': 'a genre word used before a noun, e.g. "jazz", "rock", "classical", "pop", "electronic", "country"',
    'tempo': 'a tempo adjective, e.g. "slow", "moderate", "fast", "very fast"',
    'bpm': 'a number of beats per minute, e.g. "92", "128"',
    'key': 'a key, e.g. "C minor", "F# major"',
    'mode': '"major" or "minor"',
    'instruments': 'a list of instruments, e.g. "piano", "piano and strings", "drums, bass and electric guitar"',
    'pace': 'how busy the notes are, e.g. "sparse", "steady", "busy"',
    'register': '"low", "middle" or "high"',
    'length': 'a length adjective, e.g. "short", "medium-length", "long"',
    'time_sig': 'a time signature, e.g. "3/4", "6/8"',
}
TEMPLATES = {
    'mood': 'a {mood} piece of music',
    'genre': 'a {genre} song',
    'mood_genre': 'a {mood} {genre} piece',
    'instruments': 'music for {instruments}',
    'genre_instruments': 'a {genre} piece played on {instruments}',
    'mood_genre_instruments': 'a {mood} {genre} piece with {instruments}',
    'tempo_mood': 'a {tempo}, {mood} piece',
    'tempo_genre_mode': 'a {tempo} {genre} piece in a {mode} key',
    'genre_key': 'a {genre} piece in {key}',
    'key_bpm_instruments': 'a piece in {key} at {bpm} BPM, played by {instruments}',
    'pace_register': 'a {pace} piece in a {register} register',
    'pace_instruments': 'a {pace} arrangement for {instruments}',
    'length_genre': 'a {length} {genre} piece',
    'genre_time_sig': 'a {genre} piece in {time_sig} time',
    'drums_tempo': 'a {tempo} drum pattern',
    'drums_bpm': 'a drum loop at {bpm} BPM',
    'full': 'a {length}, {tempo} {mood} {genre} piece in {key} with {instruments}',
}
XMIDI_EMOTIONS = ('exciting', 'warm', 'happy', 'romantic', 'funny', 'sad', 'angry', 'lazy', 'quiet', 'fear',
                  'magnificent')
XMIDI_GENRES = ('rock', 'pop', 'country', 'jazz', 'classical', 'traditional')
MOODS = ('energetic', 'calm', 'sad', 'tense')
MIDICAPS_GENRES = ('ambient', 'classical', 'country', 'easylistening', 'electronic', 'experimental', 'folk',
                   'instrumentalpop', 'jazz', 'latin', 'pop', 'popfolk', 'reggae', 'rock', 'soundtrack', 'world')
GENRE_WORDS = {'easylistening': 'easy listening', 'instrumentalpop': 'instrumental pop', 'popfolk': 'pop folk',
               'traditional': 'traditional folk'}
# Ukrainian words given to Qwen (only the entries found in the text), so terms and key names come out right
TONICS = {'C': 'до', 'C#': 'до-дієз', 'Db': 'ре-бемоль', 'D': 'ре', 'D#': 'ре-дієз', 'Eb': 'мі-бемоль', 'E': 'мі',
          'F': 'фа', 'F#': 'фа-дієз', 'Gb': 'соль-бемоль', 'G': 'соль', 'G#': 'соль-дієз', 'Ab': 'ля-бемоль', 'A': 'ля',
          'A#': 'ля-дієз', 'Bb': 'сі-бемоль', 'B': 'сі'}
GLOSSARY = {
    'key': 'тональність (in the key of X = у тональності X)', 'time signature': 'розмір (у розмірі 4/4)',
    'tempo': 'темп', 'chord progression': 'послідовність акордів', 'bpm': 'ударів на хвилину',
    'melodic': 'мелодійний', 'film': 'кінематографічний / для фільму', 'soundtrack': 'саундтрек',
    'relaxing': 'розслаблювальний', 'meditative': 'медитативний', 'corporate': 'корпоративний',
    'motivational': 'мотиваційний', 'epic': 'епічний', 'space': 'космічний', 'inspiring': 'надихаючий',
    'dream': 'мрійливий', 'uplifting': 'піднесений', 'upbeat': 'бадьорий', 'christmas': 'різдвяний',
    'ambient': 'ембієнт', 'easy listening': 'легка музика', 'easylistening': 'легка музика',
    'instrumental pop': 'інструментальний поп', 'instrumentalpop': 'інструментальний поп',
    'largo': 'ларго', 'grave': 'граве', 'adagio': 'адажіо', 'andante': 'анданте', 'moderato': 'модерато',
    'allegro': 'алегро', 'vivace': 'віваче', 'presto': 'престо', 'prestissimo': 'престіссімо',
    'piano': 'фортепіано', 'drums': 'ударні', 'string ensemble': 'струнний ансамбль',
    'acoustic guitar': 'акустична гітара', 'electric bass': 'бас-гітара', 'acoustic bass': 'контрабас',
    'electric guitar': 'електрогітара', 'clean electric guitar': 'електрогітара з чистим звуком',
    'fretless bass': 'безладова бас-гітара', 'synth pad': 'синтезаторний пед',
    'distortion guitar': 'гітара з дисторшном', 'overdriven guitar': 'гітара з овердрайвом',
    'synth strings': 'синтезаторні струнні', 'brass section': 'мідна духова секція', 'trumpet': 'труба',
    'flute': 'флейта', 'choir aahs': 'хор', 'choir': 'хор', 'trombone': 'тромбон', 'synth lead': 'лід-синтезатор',
    'clarinet': 'кларнет', 'violin': 'скрипка', 'alto saxophone': 'альт-саксофон', 'synth bass': 'синтезаторний бас',
    'rock organ': 'рок-орган', 'vibraphone': 'вібрафон', 'tenor saxophone': 'тенор-саксофон',
    'synth effects': 'синтезаторні ефекти', 'synth brass': 'синтезаторні мідні', 'french horn': 'валторна',
    'voice oohs': 'вокальний хор', 'accordion': 'акордеон', 'harp': 'арфа', 'synth voice': 'синтезований голос',
    'electric piano': 'електропіаніно', 'harpsichord': 'клавесин', 'oboe': 'гобой', 'cello': 'віолончель',
    'hammond organ': 'орган Гаммонда', 'percussive organ': 'перкусійний орган', 'slap bass': 'слеп-бас',
    'harmonica': 'губна гармоніка', 'piccolo': 'флейта-пікколо', 'clavinet': 'клавінет', 'tuba': 'туба',
    'church organ': 'церковний орган', 'pizzicato strings': 'струнні піцикато', 'bassoon': 'фагот',
    'viola': 'альт', 'banjo': 'банджо', 'soprano saxophone': 'сопрано-саксофон', 'ocarina': 'окарина',
    'marimba': 'марімба', 'pan flute': 'флейта Пана', 'tango accordion': 'бандонеон',
    'tremolo strings': 'струнні тремоло', 'baritone saxophone': 'баритон-саксофон', 'music box': 'музична скринька',
    'contrabass': 'контрабас', 'recorder': 'блок-флейта', 'celesta': 'челеста', 'timpani': 'литаври',
    'glockenspiel': 'дзвіночки', 'muted trumpet': 'труба із сурдиною', 'tubular bells': 'оркестрові дзвони',
    'english horn': 'англійський ріжок', 'xylophone': 'ксилофон', 'sitar': 'ситар', 'steel drums': 'стил-драм',
}
LABEL_WORDS = {
    'exciting': 'захопливий', 'warm': 'теплий', 'happy': 'радісний', 'romantic': 'романтичний', 'funny': 'кумедний',
    'sad': 'сумний', 'angry': 'гнівний', 'lazy': 'неквапливий, лінивий', 'quiet': 'тихий', 'fear': 'моторошний',
    'magnificent': 'величний', 'energetic': 'енергійний', 'calm': 'спокійний', 'tense': 'напружений',
    'rock': 'рок', 'pop': 'поп', 'country': 'кантрі', 'jazz': 'джаз', 'classical': 'класична музика',
    'traditional folk': 'народна музика', 'ambient': 'ембієнт', 'easy listening': 'легка музика',
    'electronic': 'електронна музика', 'experimental': 'експериментальна музика', 'folk': 'фолк',
    'instrumental pop': 'інструментальний поп', 'latin': 'латиноамериканська музика', 'pop folk': 'поп-фолк',
    'reggae': 'реґі', 'soundtrack': 'саундтрек', 'world': 'етнічна музика',
}
# one worked example for the translation prompt (a gigamidi_replay_train caption, translated by hand)
EXAMPLE_EN = ('A classical piece that could be part of a film soundtrack, this short composition is characterized by '
              'its slow tempo and relaxing, meditative mood. The string ensemble takes the lead, accompanied by the '
              'majestic tones of the French horn and the subtle support of the drums. Set in the key of Bb major and '
              'following a 4/4 time signature, it features a chord progression of F, Bb, C, F, and Bb, creating an '
              'emotionally epic atmosphere.')
EXAMPLE_UK = ('Класичний твір, який міг би стати частиною саундтреку до фільму: ця коротка композиція вирізняється '
              'повільним темпом і розслаблювальним, медитативним настроєм. Провідну роль грає струнний ансамбль, його '
              'супроводжують величні звуки валторни та делікатна підтримка ударних. Композиція написана в тональності '
              'сі-бемоль мажор, у розмірі 4/4, з послідовністю акордів F, Bb, C, F, Bb, що створює емоційно-епічну '
              'атмосферу.')
SCHEMA = {'type': 'object', 'properties': {'captions': {'type': 'array', 'items': {'type': 'string'}}},
          'required': ['captions']}
SCHEMA_UK = {'type': 'object', 'properties': {'uk': {'type': 'string'}}, 'required': ['uk']}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('task', choices=('paraphrase', 'uk-labels', 'translate'))
    p.add_argument('--model', default='qwen3:8b')
    p.add_argument('--host', default='http://localhost:11434')
    p.add_argument('--n', type=int, default=0, help='rewrites per template (paraphrase, default 24) or per '
                   'combination (uk-labels, default 6)')
    p.add_argument('--stores', default='data/cache/gigamidi_replay_train,data/cache/gigamidi_heldout_validation')
    p.add_argument('--labels-dir', default='data/labels')
    p.add_argument('--max-files', type=int, default=0, help='translate: per store, 0 = all')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--cpu', action='store_true', help='run the model on the CPU (num_gpu 0), for tests')
    p.add_argument('--out-dir', default=str(OUT))
    return p.parse_args()


class Qwen:
    def __init__(self, args):
        self.url, self.model, self.cpu = f'{args.host}/api/chat', args.model, args.cpu
        self.tokens, self.lock = 0, threading.Lock()

    def __call__(self, system, user, schema, temperature, seed, num_predict=1024, shots=()):
        options = dict(temperature=temperature, top_p=0.95, seed=seed, num_predict=num_predict)
        if self.cpu:
            options['num_gpu'] = 0
        r = requests.post(self.url, timeout=600, json=dict(
            model=self.model, think=False, stream=False, format=schema, keep_alive='10m', options=options,
            messages=[{'role': 'system', 'content': system}, *shots, {'role': 'user', 'content': user}]))
        r.raise_for_status()
        out = r.json()
        with self.lock:
            self.tokens += out.get('eval_count', 0)
        try:
            return json.loads(out['message']['content'])
        except json.JSONDecodeError:
            return None


def placeholders(s):
    return sorted(re.findall(r'\{(\w+)\}', s))


ADJ_SLOTS = ('mood', 'tempo', 'length', 'pace', 'register')


def adjective_as_noun(s):
    """'in {key} of {length}', 'in {tempo}, {length}, and {mood}': an adjective slot standing where a noun belongs.
    After the slot, further slots joined by ',' / 'and' are skipped; then the end of the text or a preposition means
    no noun follows ('a {tempo}, {mood} piece' is fine). A linking verb before it is fine ('feels {mood}')."""
    for m in re.finditer(rf'\{{(?:{"|".join(ADJ_SLOTS)})\}}', s):
        rest = re.sub(r'^(?:\s*(?:,|and\b)\s*\{\w+\})*', '', s[m.end():]).strip()
        rest = re.sub(r'^,?\s*and\b', '', rest).strip() if not rest.startswith(('and {', ', {')) else rest
        if (not rest or re.match(r'(?:[,.;]|(?:and|with|of|in|at|that|for)\b)', rest)) and \
                not re.search(r"(feels|feel|is|'s|sounds|seems|are|feeling)\s*$", s[:m.start()]):
            return True
    return False


def ok_paraphrase(s, base):
    if adjective_as_noun(s):
        return False
    if placeholders(s) != placeholders(base) or re.sub(r'\{\w+\}', '', s).count('{') or '}' in re.sub(r'\{\w+\}', '', s):
        return False
    if len(s.split()) > 2.5 * len(base.split()) + 3 or re.search(r'[^\x00-\x7F’—–]', s):
        return False
    return True


def uk_key(key):
    """'Bb major' -> 'сі-бемоль мажор'; None for a missing / odd key."""
    parts = (key or '').split()
    if len(parts) != 2 or parts[0] not in TONICS or parts[1] not in ('major', 'minor'):
        return None
    return f"{TONICS[parts[0]]} {'мажор' if parts[1] == 'major' else 'мінор'}"


def glossary_for(text):
    low = text.lower()
    return {k: v for k, v in GLOSSARY.items() if re.search(rf'\b{re.escape(k)}\b', low)}


def ok_uk(s, en=None, key=None):
    letters = re.findall(r'[^\W\d_]', s)
    if not letters or sum('Ѐ' <= c <= 'ӿ' for c in letters) < 0.6 * len(letters):
        return False
    if re.search(r'[ыэъёЫЭЪЁ]', s) or re.search(r'[぀-ヿ㐀-鿿가-힯]', s):
        return False
    if en is not None:
        if not 0.6 <= len(s) / max(len(en), 1) <= 2.2:
            return False
        if any(n not in s for n in re.findall(r'\d+(?:/\d+)?', en)):
            return False
    if key and key not in s.lower():
        return False
    return True


def to_tokens(s):
    """'{mood}' -> '[MOOD]': Qwen keeps bracketed capitals, but fills in '{mood}' with an example word."""
    return re.sub(r'\{(\w+)\}', lambda m: f'[{m.group(1).upper()}]', s)


def from_tokens(s):
    return re.sub(r'\[([A-Z_]+)\]', lambda m: f'{{{m.group(1).lower()}}}', s)


def run_paraphrase(args, qwen, out_dir):
    n = args.n or 24
    system = ('You write short English descriptions of pieces of music for a text-to-music model. '
              'Answer with JSON only.')
    path = out_dir / 'templates_en.json'
    bank = json.loads(path.read_text()) if path.exists() else {}
    for v in bank.values():  # checks may have been tightened since the file was written
        v['paraphrases'] = [c for c in v['paraphrases'] if ok_paraphrase(c, v['base'])]
    todo = [f for f in TEMPLATES if len(bank.get(f, {}).get('paraphrases', [])) < n]
    job = JobStatus('caption_paraphrase', len(todo), args.model)
    for k, family in enumerate(tqdm(todo, desc='templates')):
        base = TEMPLATES[family]
        slot_help = '\n'.join(f'- {to_tokens("{" + s + "}")}: {SLOTS[s].split(", e.g.")[0]}'
                               for s in dict.fromkeys(placeholders(base)))
        seen, kept = {base.lower()}, []
        for attempt in range(args.retries + 2):
            if len(kept) >= n:
                break
            user = (f'Template: "{to_tokens(base)}"\nPlaceholders in square brackets are filled in later by a '
                    f'program. Copy them literally, in capitals with the brackets, each exactly once; never replace '
                    f'them with a word:\n{slot_help}\n\nWrite {n} different rewrites of the template with the same '
                    f'meaning, the way a person would ask for this music: vary the wording, word order and length '
                    f'(3 to 20 words), some as requests ("give me ...", "I want ..."), some as plain descriptions. '
                    f'No extra facts (no other instruments, moods or tempos). Return {{"captions": [...]}}.')
            ans = qwen(system, user, SCHEMA, temperature=0.7 + 0.15 * attempt, seed=attempt)
            for c in (ans or {}).get('captions', []):
                c = from_tokens(c.strip().strip('"').rstrip('.'))
                if c.lower() not in seen and ok_paraphrase(c, base) and '[' not in c:
                    seen.add(c.lower())
                    kept.append(c)
        bank[family] = dict(slots=placeholders(base), base=base, paraphrases=kept[:n])
        job.update(k + 1, detail=f'{family}: {len(kept[:n])}')
    bank = {f: bank[f] for f in TEMPLATES if f in bank}
    path.write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    job.finish(f'{path}')
    print('\n'.join(f'{f:24s} {len(v["paraphrases"]):3d}  e.g. {v["paraphrases"][:2]}' for f, v in bank.items()))
    print(f'wrote {path}, {qwen.tokens} generated tokens')


def label_combos():
    g = lambda x: GENRE_WORDS.get(x, x)  # noqa: E731
    combos = {}
    for e in XMIDI_EMOTIONS:
        combos[f'emotion={e}'] = f'mood: {e}'
        for genre in XMIDI_GENRES:
            combos[f'emotion={e}|genre={genre}'] = f'mood: {e}; genre: {g(genre)}'
    for m in MOODS:
        combos[f'mood={m}'] = f'mood: {m}'
        for genre in sorted(set(XMIDI_GENRES) | set(MIDICAPS_GENRES)):
            combos[f'mood={m}|genre={genre}'] = f'mood: {m}; genre: {g(genre)}'
    for genre in sorted(set(XMIDI_GENRES) | set(MIDICAPS_GENRES)):
        combos[f'genre={genre}'] = f'genre: {g(genre)}'
    return combos


def run_uk_labels(args, qwen, out_dir):
    n = args.n or 6
    system = ('Ти пишеш короткі описи музичних творів українською мовою для моделі, що генерує музику з тексту. '
              'Пиши природною сучасною українською, без русизмів і без суржику. Відповідай лише JSON.')
    combos = label_combos()
    path = out_dir / 'uk_labels.json'
    bank = json.loads(path.read_text()) if path.exists() else {}
    todo = [k for k in combos if len(bank.get(k, [])) < n]
    bad = 0

    def one(key):
        kept, seen = [], set()
        for attempt in range(args.retries + 1):
            words = '; '.join(f'{w} = {LABEL_WORDS[w]}' for w in LABEL_WORDS if re.search(rf'\b{w}\b', combos[key]))
            user = (f'Attributes of the music (in English): {combos[key]}.\nUkrainian words for them (inflect as '
                    f'needed, synonyms are fine): {words}.\nWrite {n} different short Ukrainian '
                    f'descriptions of such a piece (3 to 12 words), the way a person would ask for it: some as '
                    f'requests ("зіграй ...", "хочу ..."), some as plain descriptions ("сумна джазова композиція"). '
                    f'Use correct gender and case agreement. Only these attributes, no other facts. '
                    f'Return {{"captions": [...]}}.')
            ans = qwen(system, user, SCHEMA, temperature=0.7 + 0.1 * attempt, seed=attempt)
            for s in (ans or {}).get('captions', []):
                s = s.strip().strip('"').rstrip('.')
                if s.lower() not in seen and ok_uk(s):
                    seen.add(s.lower())
                    kept.append(s)
            if len(kept) >= n:
                break
        return key, kept[:n]

    job = JobStatus('caption_uk_labels', len(todo), args.model)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(tqdm(as_completed([pool.submit(one, k) for k in todo]), total=len(todo))):
            key, kept = fut.result()
            bank[key] = kept
            bad += len(kept) < n
            job.update(i + 1)
            if i % 20 == 0:
                path.write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    path.write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    job.finish(f'{path}')
    print(f'wrote {path}: {len(bank)} combinations, {bad} with fewer than {n}, {qwen.tokens} generated tokens')


def run_translate(args, qwen, out_dir):
    system = ('You translate English descriptions of music into natural, modern Ukrainian (no Russian words, no '
              'surzhyk). Keep every number as it is (tempo, time signature like 4/4). Keys in Ukrainian notation '
              '(e.g. "Bb major" -> "сі-бемоль мажор", "F# minor" -> "фа-дієз мінор"); chord progressions keep '
              'their chord symbols (F, Bb, Am). Instrument and genre names as Ukrainian musicians say them. '
              'Answer with JSON only: {"uk": "..."}.')
    for store in args.stores.split(','):
        name = Path(store).name
        rows = [json.loads(line) for line in open(Path(args.labels_dir) / f'{name}.jsonl', encoding='utf-8')]
        rows = [r for r in rows if r.get('midicaps', {}).get('caption')]
        rows = rows[:args.max_files] if args.max_files else rows
        path = out_dir / f'{name}.uk.jsonl'
        done = {json.loads(line)['file'] for line in open(path, encoding='utf-8')} if path.exists() else set()
        todo = [r for r in rows if r['file'] not in done]
        print(f'{name}: {len(rows)} MidiCaps captions, {len(done)} done, {len(todo)} to translate')
        lock, failed = threading.Lock(), 0
        job = JobStatus('caption_translate', len(todo), f'{name}, {args.model}')

        shots = ({'role': 'user', 'content': EXAMPLE_EN},
                 {'role': 'assistant', 'content': json.dumps({'uk': EXAMPLE_UK}, ensure_ascii=False)})

        def one(r):
            en = r['midicaps']['caption']
            key = uk_key(r['midicaps'].get('key')) if re.search(r'\b[A-G][#b]? (major|minor)\b', en) else None
            terms = glossary_for(en)
            if key:
                terms[r['midicaps']['key']] = key
            user = en + ('\n\nGlossary for this text (use these words, inflected as needed): '
                         + '; '.join(f'{k} = {v}' for k, v in terms.items()) if terms else '')
            for attempt in range(args.retries + 1):
                ans = qwen(system, user, SCHEMA_UK, temperature=0.3 + 0.15 * attempt, seed=attempt,
                           num_predict=768, shots=shots)
                uk = ((ans or {}).get('uk') or '').strip()
                if ok_uk(uk, en, key):
                    return r['file'], uk
            return r['file'], None

        with open(path, 'a', encoding='utf-8') as f, ThreadPoolExecutor(args.workers) as pool:
            futs = [pool.submit(one, r) for r in todo]
            for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc=name)):
                file, uk = fut.result()
                with lock:
                    if uk is None:
                        failed += 1
                    else:
                        f.write(json.dumps(dict(file=file, uk=uk), ensure_ascii=False) + '\n')
                        f.flush()
                job.update(i + 1, detail=f'{name}, {failed} failed, {qwen.tokens} tokens')
        job.finish(f'{path}: {len(todo) - failed} new, {failed} failed')
        print(f'wrote {path}: {len(todo) - failed} new, {failed} failed checks, {qwen.tokens} generated tokens')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qwen = Qwen(args)
    {'paraphrase': run_paraphrase, 'uk-labels': run_uk_labels, 'translate': run_translate}[args.task](args, qwen, out_dir)


if __name__ == '__main__':
    main()
