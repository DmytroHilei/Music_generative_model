"""
Fix the Ukrainian MidiCaps translations (data_prep/caption_bank.py, Qwen3 8B) with rules, drop the ones that stay
broken (context/claude/text_control.md, caption review 2026-10-18).

    python data_prep/fix_uk_captions.py                      # every data/captions/*.uk.jsonl
    python data_prep/fix_uk_captions.py --samples 30         # + before / after pairs for review

Rules, in order: Latin look-alike letters inside Cyrillic words -> Cyrillic (кінematографічним); word, stem and phrase
replacements for systematic errors (Russianisms, calques, garbled 'cinematic', 'moderato' as an adjective,
transliterated keys); a masculine adjective (or two, joined by і / та / ,) right before a feminine noun -> feminine,
before фортепіано / піаніно -> neuter. A caption is dropped when it still has a word with Latin and Cyrillic letters,
a blocklisted fragment (garbled words, chord names in Cyrillic), or 2+ words that neither the uk_UA Hunspell
dictionary (data/dicts/uk_UA.aff / .dic from github.com/LibreOffice/dictionaries, uk_UA/; read with spylls) nor
the allowlist of music jargon knows.
Writes data/captions/<store>.uk.fixed.jsonl (file, uk) and prints the counts.
"""

import argparse
import json
import random
import re
from pathlib import Path

CYR = 'А-Яа-яІіЇїЄєҐґ'
WORD = re.compile(rf"[{CYR}A-Za-z'’]+(?:-[{CYR}A-Za-z'’]+)*")
LOOKALIKE = str.maketrans('aceopxiykmtbhABCEHKMOPTX', 'асеорхіукмтвнАВСЕНКМОРТХ')

# stem replacements (prefix of a word -> prefix; the ending is kept)
STEMS = [
    ('модератив', 'помірн'), ('модератн', 'помірн'), ('Модератив', 'Помірн'), ('Модератн', 'Помірн'),
    ('умерен', 'помірн'), ('Умерен', 'Помірн'),
    ('перегріт', 'перевантажен'), ('захоплююч', 'захоплив'), ('Захоплююч', 'Захоплив'),
    ('надихаюч', 'надихальн'), ('Надихаюч', 'Надихальн'), ('мотивуюч', 'мотиваційн'), ('Мотивуюч', 'Мотиваційн'),
    ('підходящ', 'придатн'), ('діюч', 'динамічн'), ('затишков', 'затишн'), ('фортепіанов', 'фортепіанн'),
    ('кіноромантичн', 'кінематографічн'), ('кінорепертуарн', 'кінематографічн'), ('кіноробоч', 'кінематографічн'),
    ('кінорамн', 'кінематографічн'), ('кіноробн', 'кінематографічн'), ('кінореперативн', 'кінематографічн'),
    ('кіноритмічн', 'кінематографічн'), ('Кіноромантичн', 'Кінематографічн'), ('марімб', 'маримб'),
    ('гонкі-тонк', 'хонкі-тонк'), ('сплітуют', 'сплітают'), ('кіноробськ', 'кінематографічн'),
]
PHRASES = [(re.compile(r'\b([Мм])одерн(\w*) темп'),  # 'modern tempo' = moderate
            lambda m: ('П' if m.group(1) == 'М' else 'п') + 'омірн' + m.group(2) + ' темп')]
WORDS = {'твіру': 'твору', 'твірі': 'творі', 'вдохновення': 'натхнення', 'престіссімо': 'престисимо',
         'зроблячи': 'роблячи', 'фюзіон': "ф'южн", 'фьюжн': "ф'южн", 'бі-мінор': 'сі мінор',
         'еб-мінор': 'мі-бемоль мінор', 'бі-мажор': 'сі мажор', 'еб-мажор': 'мі-бемоль мажор',
         'аб-мажор': 'ля-бемоль мажор', 'аб-мінор': 'ля-бемоль мінор'}
BLOCK = re.compile(r'\bнотк\b|\b(тапет|ооо|аах|участию|лейд|дуе|редов|іскривлен|іскажен|висловлююч)\w*|\bвівач(?!е\b)|'
                   r'(?<![\w-])ї(?![\w-])|\b(ам|ем|дм|бм|гм|см|фм|еб|ред|д)\b', re.I)
ALLOW = re.compile(r'^(синтезатор\w*|безладов\w*|пед\w*|пад\w*|овердрайв\w*|дисторшн\w*|слеп-бас\w*|клавінет\w*|'
                   r'маримб\w*|калімб\w*|кото|шакухачі|сякухаті|саундтреков\w*|хонкі-тонк\w*|гаммонд\w*|фідл\w*|'
                   r'каліоп\w*|лід\w*|стил-драм\w*|дулцімер\w*|ф\'южн|ембієнт\w*|техно|хаус|чилаут|інді|ситар\w*|'
                   r'челест\w*|банджо|кантрі|реггі|свінг\w*|сямісен\w*|шехна\w*|гобо\w*|вібрафон\w*|'
                   r'пульсуюч\w*|престисимо|віваче|модерато|анданте|ларго|граве|адажіо|алегро|престо)$', re.I)
FEM_NOUNS = r'(?:пісня|композиція|балада|мелодія|музика|п\'єса|п’єса|тема|арфа|гітара|партія|мелодика|робота|річ)'
GENDER = re.compile(rf"((?:[Цц]ей )?(?:[{CYR}]+(?:ий|ій)(?:,? (?:і|та|й) |, | ))+)((?:[{CYR}]+-)*{FEM_NOUNS})\b")
NEUTER = re.compile(rf"((?:[{CYR}]+(?:ий|ій)(?:,? (?:і|та|й) |, | ))+)(фортепіано|піаніно)\b")


def _adj(word, gender):
    if word.lower() in ('свій', 'мій', 'твій', 'чий'):
        return word
    if gender == 'f':
        return word[:-2] + ('я' if word.endswith('ій') else 'а')
    return word[:-2] + ('є' if word.endswith('ій') else 'е')


def _agree(m, gender):
    adjs = re.sub(rf'[{CYR}]+(?:ий|ій)\b', lambda w: _adj(w.group(0), gender), m.group(1))
    if gender == 'f':
        adjs = re.sub(r'\b([Цц])ей ', r'\1я ', adjs)
    return adjs + m.group(2)


def fix(text):
    """-> (fixed text, list of applied rule names)."""
    applied = []

    def lookalike(m):
        w = m.group(0)
        if re.search(f'[{CYR}]', w) and re.search('[A-Za-z]', w):
            applied.append('script')
            return w.translate(LOOKALIKE)
        return w
    text = WORD.sub(lookalike, text)

    def words(m):
        w = m.group(0)
        if w in WORDS:
            applied.append('word')
            return WORDS[w]
        for old, new in STEMS:
            if w.startswith(old):
                applied.append('stem')
                return new + w[len(old):]
        return w
    text = WORD.sub(words, text)
    for pattern, repl in PHRASES:
        new = pattern.sub(repl, text)
        if new != text:
            applied.append('phrase')
            text = new
    for pattern, gender in ((GENDER, 'f'), (NEUTER, 'n')):
        new = pattern.sub(lambda m: _agree(m, gender), text)
        if new != text:
            applied.append('gender')
            text = new
    return text, applied


def problems(text, known):
    """Reasons to drop a fixed caption ([] = keep)."""
    out = []
    if any(re.search(f'[{CYR}]', w) and re.search('[A-Za-z]', w) for w in WORD.findall(text)):
        out.append('mixed script')
    if BLOCK.search(text):
        out.append(f'blocked: {BLOCK.search(text).group(0)}')
    unknown = [w for w in re.findall(rf"[{CYR}'’]+(?:-[{CYR}'’]+)*", text)
               if len(w) > 1 and not ALLOW.match(w) and not known(w)]
    if len(unknown) >= 2:
        out.append(f'unknown words: {", ".join(unknown[:4])}')
    return out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dir', default='data/captions')
    p.add_argument('--dict', default='data/dicts/uk_UA')
    p.add_argument('--samples', type=int, default=0, help='print this many before / after pairs (changed captions)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    from spylls.hunspell import Dictionary
    d = Dictionary.from_files(args.dict)
    cache = {}

    def known(w):
        if w not in cache:
            parts = [x for x in w.split('-') if x]
            cache[w] = d.lookup(w) or d.lookup(w.lower()) or all(d.lookup(x) or d.lookup(x.lower()) or ALLOW.match(x)
                                                                  for x in parts)
        return cache[w]

    changed, dropped = [], []
    for src in sorted(Path(args.dir).glob('*.uk.jsonl')):
        rows = [json.loads(line) for line in src.read_text().splitlines()]
        out, counts = [], {'kept': 0, 'changed': 0, 'dropped': 0}
        rule_counts = {}
        for r in rows:
            text, applied = fix(r['uk'])
            for a in set(applied):
                rule_counts[a] = rule_counts.get(a, 0) + 1
            bad = problems(text, known)
            if bad:
                counts['dropped'] += 1
                dropped.append((r['uk'], bad))
                continue
            counts['kept'] += 1
            if applied:
                counts['changed'] += 1
                changed.append((r['uk'], text))
            out.append({'file': r['file'], 'uk': text})
        dst = src.with_name(src.name.replace('.uk.jsonl', '.uk.fixed.jsonl'))
        dst.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
        print(f'{src.name}: {len(rows)} -> {counts} | rules applied (captions): {rule_counts} -> {dst}')
    if args.samples:
        rng = random.Random(args.seed)
        print('\n## changed')
        for before, after in rng.sample(changed, min(args.samples, len(changed))):
            print(f'- {before}\n+ {after}\n')
        print('## dropped')
        for before, why in rng.sample(dropped, min(args.samples // 2, len(dropped))):
            print(f'x {before[:200]}\n  ({"; ".join(why)})\n')


if __name__ == '__main__':
    main()
