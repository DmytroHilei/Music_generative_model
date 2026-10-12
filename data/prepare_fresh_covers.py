"""
New piano covers of validation-split songs: a covers val extension and a fresh covers test.

Candidates are the songs whose title hash puts them in validation (ukrainian_reduction_v4.csv val rows, the same split
as the multi-track and covers CSVs), that have no cover yet, with a Cyrillic title and not by a non-Ukrainian-language
artist (seed_compare's list). None of them is in any train set. Language: a title with Russian-only letters (ы э ъ ё)
or Russian word forms (RUSSIAN_HINT) is dropped; a title without Ukrainian-only letters (і ї є ґ, apostrophe) is kept
only when the artist's titles in the reduction/cover CSVs have Ukrainian-only letters more than twice as often as
Russian-only ones. Some Russian titles still pass: split prints the chosen songs, and lines 'artist|title' in
data/finetune/fresh/exclude.txt drop songs.

    python data/prepare_fresh_covers.py candidates   # -> data/finetune/fresh/{candidates.csv, artists.txt}
    .venv-audio/bin/python data/fetch_songs.py --mode covers --only-known-songs --ukrainian-only --per-song 1 \\
        --per-artist 1000 --songs-csv data/finetune/fresh/candidates.csv --artists data/finetune/fresh/artists.txt \\
        --output data/covers_fresh ...
    .venv-audio/bin/python data/transcribe_covers.py --input data/covers_fresh --output data/covers_fresh
    python data/prepare_fresh_covers.py split --val-songs 8
        # -> data/finetune/ukrainian_covers_v5.csv (v4 + val-songs new val covers)
        #    data/finetune/ukrainian_covers_fresh_test.csv (the other new covers, split=validation)

artists.txt is ordered by the artist's number of covers in covers v4 (artists with many covers first), so a run cut
short by --max-minutes has searched the likeliest songs. The val/test assignment orders the found songs by a salted
crc32 of the title, so it doesn't depend on download order.
"""

import argparse
import csv
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
from prepare_finetune import norm_title  # noqa: E402

FRESH = ROOT / 'data/finetune/fresh'
COVERS_ROOT = ROOT / 'data/covers_fresh'
NON_UA_ARTISTS = {'Мішель Андраде', 'Пошлая Молли', 'Луна', 'Витас', 'Ундервуд', 'Pianoбой', 'Ані Лорак'}  # = seed_compare
UKRAINIAN = re.compile('[а-яіїєґ]', re.I)
RUSSIAN_ONLY = re.compile('[ыэъё]', re.I)
UKRAINIAN_ONLY = re.compile("[іїєґʼ']", re.I)
RUSSIAN_HINT = re.compile(r'\w(ая|яя|ое|ее|ые|ие)\b|\b(и|что|тебя|меня|люблю|любви|время|года|под|тот|уходит)\b', re.I)


def key(artist, title):
    return artist, norm_title(str(title))


def candidates(args):
    red = pd.read_csv(args.reductions)
    cov = pd.read_csv(args.covers)
    covered = {key(a, t) for a, t in zip(cov.artist, cov.title)}
    titles = pd.concat([red, cov]).groupby('artist').title.agg(list)
    ukrainian_artist = {a: sum(bool(UKRAINIAN_ONLY.search(str(x))) for x in ts) >
                        2 * sum(bool(RUSSIAN_ONLY.search(str(x))) for x in ts) for a, ts in titles.items()}
    rows, seen = [], set()
    for a, t in zip(red.artist[red.split == 'validation'], red.title[red.split == 'validation']):
        k = key(a, t)
        if k in covered or k in seen or a in NON_UA_ARTISTS:
            continue
        if not UKRAINIAN.search(str(t)) or RUSSIAN_ONLY.search(str(t)) or RUSSIAN_HINT.search(str(t)):
            continue
        if not UKRAINIAN_ONLY.search(str(t)) and not ukrainian_artist.get(a, False):
            continue
        assert zlib.crc32(k[1].encode()) % 100 < 10, f'{k} is not a validation title'
        seen.add(k)
        rows.append({'artist': a, 'song': str(t).strip()})
    FRESH.mkdir(parents=True, exist_ok=True)
    with open(FRESH / 'candidates.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['artist', 'song'])
        w.writeheader()
        w.writerows(rows)

    # aliases from the existing artist lists (first name = folder name), most-covered artists first
    aliases = {}
    for txt in sorted((ROOT / 'data').glob('artists*.txt')):
        for line in txt.read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.startswith('#'):
                aliases.setdefault(line.split('|')[0].strip(), line.strip())
    n_cov = Counter(cov.artist[cov.split == 'train'])
    n_cand = Counter(r['artist'] for r in rows)
    order = sorted(n_cand, key=lambda a: (-n_cov[a], -n_cand[a], a))
    (FRESH / 'artists.txt').write_text('# fresh covers: artists of the candidate songs, most covers in v4 first\n' +
                                       '\n'.join(aliases.get(a, a) for a in order) + '\n', encoding='utf-8')
    print(f'{len(rows)} candidate songs of {len(order)} artists -> {FRESH}/candidates.csv, artists.txt '
          f'({sum(n_cov[a] > 0 for a in order)} artists already have covers in v4)')


def split(args):
    cand = pd.read_csv(FRESH / 'candidates.csv')
    # fetch_songs writes the song as its normalize() of the candidate title: compare both via norm_title
    by_key = {key(a, s): s for a, s in zip(cand.artist, cand.song)}
    with open(COVERS_ROOT / 'songs.csv', newline='', encoding='utf-8') as f:
        song_of = {r['id']: (r['artist'], r['song']) for r in csv.DictReader(f)}
    excluded = set()
    if (FRESH / 'exclude.txt').exists():
        excluded = {key(*line.split('|', 1)) for line in (FRESH / 'exclude.txt').read_text(encoding='utf-8').splitlines()
                    if '|' in line and not line.startswith('#')}
    found = {}
    for midi in sorted(COVERS_ROOT.glob('*/midi/*.mid')):
        m = re.search(r'\[([\w-]+)\]$', midi.stem)
        if not m or m.group(1) not in song_of:
            continue
        artist, fetched = song_of[m.group(1)]
        title = by_key.get(key(artist, fetched))
        if title is None:
            print(f'  not a candidate, skipped: {midi.name}')
            continue
        if key(artist, title) in excluded:
            continue
        found.setdefault(key(artist, title), (artist, title, midi))  # one cover per song
    order = sorted(found, key=lambda k: zlib.crc32(f'fresh:{k[0]}:{k[1]}'.encode()))
    val_keys, test_keys = order[:args.val_songs], order[args.val_songs:]
    row = lambda k: {'split': 'validation', 'midi_filename': str(found[k][2]), 'artist': found[k][0],
                     'title': found[k][1], 'style': 'cover'}
    v4 = pd.read_csv(args.covers)
    v5 = pd.concat([v4, pd.DataFrame([row(k) for k in val_keys])], ignore_index=True)
    v5.to_csv(args.out_val, index=False)
    pd.DataFrame([row(k) for k in test_keys], columns=v4.columns).to_csv(args.out_test, index=False)
    for name, keys in (('validation +', val_keys), ('fresh test', test_keys)):
        print(f'{name} ({len(keys)}): ' + '; '.join(f'{found[k][0]}|{found[k][1]}' for k in keys))
    print(f'{len(found)} new songs with a cover: {len(val_keys)} -> {args.out_val} (validation, {len(v5)} rows), '
          f'{len(test_keys)} -> {args.out_test}')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('step', choices=['candidates', 'split'])
    ap.add_argument('--reductions', default=str(ROOT / 'data/finetune/ukrainian_reduction_v4.csv'))
    ap.add_argument('--covers', default=str(ROOT / 'data/finetune/ukrainian_covers_v4.csv'))
    ap.add_argument('--val-songs', type=int, default=8, help='split: new songs added to the covers val, rest = test')
    ap.add_argument('--out-val', default=str(ROOT / 'data/finetune/ukrainian_covers_v5.csv'))
    ap.add_argument('--out-test', default=str(ROOT / 'data/finetune/ukrainian_covers_fresh_test.csv'))
    args = ap.parse_args()
    candidates(args) if args.step == 'candidates' else split(args)


if __name__ == '__main__':
    main()
