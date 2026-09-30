"""
Builds the fine-tune CSV from the piano reductions in data/finetune/<artist>/midi/ (skryabin_test excluded).

    python data/prepare_finetune.py            # -> data/finetune/ukrainian.csv

- Song title: from data/audio/songs.csv via the YouTube id in the file name ("... [id].mid"), else from the file
  name (skryabin_local: "skryabin-kolorova-(meloua.com).mid"). Titles are transliterated (Cyrillic -> Latin) and
  reduced to letters/digits so the same song from two sources matches.
- Dedupe: one file per (artist, title). skryabin_local counts as artist Скрябін.
- Split by song title (crc32 hash, --val-percent), so no song is in both train and validation.
Columns: split, midi_filename (absolute), artist, title.
"""

import argparse
import csv
import re
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSLIT = dict(zip('абвгґдеєжзиіїйклмнопрстуфхцчшщьюяэыъё',
                    ['a', 'b', 'v', 'h', 'g', 'd', 'e', 'ie', 'zh', 'z', 'y', 'i', 'i', 'i', 'k', 'l', 'm', 'n', 'o',
                     'p', 'r', 's', 't', 'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'iu', 'ia', 'e', 'y', '', 'e']))
ARTIST_ALIASES = {'skryabin_local': 'Скрябін'}


def norm_title(text):
    text = text.lower().replace("'", '').replace('’', '')
    text = ''.join(TRANSLIT.get(c, c) for c in text)
    # common transliteration variants collapse to one spelling
    text = text.replace('kh', 'h').replace('y', 'i').replace('j', 'i').replace('g', 'h')
    return re.sub(r'[^a-z0-9]+', '', text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-percent', type=int, default=10)
    ap.add_argument('--out', default=str(ROOT / 'data/finetune/ukrainian.csv'))
    ap.add_argument('--style', default='', help="write a 'style' column with this label for every song (e.g. "
                    "'reduction' = one domain label); without it the loader uses the artist as the style")
    args = ap.parse_args()

    by_id = {}
    songs_csv = ROOT / 'data/audio/songs.csv'
    if songs_csv.exists():
        with open(songs_csv, newline='', encoding='utf-8') as f:
            by_id = {row['id']: row['song'] for row in csv.DictReader(f)}

    rows, seen, dupes = [], {}, []
    for midi in sorted((ROOT / 'data/finetune').glob('*/midi/*.mid')):
        folder = midi.parent.parent.name
        if folder == 'skryabin_test':
            continue
        artist = ARTIST_ALIASES.get(folder, folder)
        m = re.search(r'\[([\w-]{11})\]', midi.stem)
        if m and m.group(1) in by_id:
            title = by_id[m.group(1)]
        else:
            title = re.sub(r'\(meloua\.com\)', '', midi.stem)
            title = re.sub(r'^(iryna-bilyk-)?skryabin-', '', title).replace('-', ' ') if 'meloua' in midi.stem \
                else re.split(r'\s[-–—]\s', title, maxsplit=1)[-1]  # "Скрябін - Title" / "Скрябін, X - Title"
        key = (artist, norm_title(title))
        if key in seen:
            dupes.append((midi.name, seen[key]))
            continue
        seen[key] = midi.name
        split = 'validation' if zlib.crc32(key[1].encode()) % 100 < args.val_percent else 'train'
        rows.append(dict(split=split, midi_filename=str(midi), artist=artist, title=title.strip(),
                         **({'style': args.style} if args.style else {})))

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'midi_filename', 'artist', 'title'] + (['style'] if args.style else []))
        w.writeheader()
        w.writerows(rows)
    n_val = sum(r['split'] == 'validation' for r in rows)
    print(f"{len(rows)} songs ({len(rows) - n_val} train / {n_val} validation), {len(dupes)} duplicates dropped -> {args.out}")
    for a, b in dupes:
        print(f"  dup: {a}  ==  {b}")


if __name__ == '__main__':
    main()
