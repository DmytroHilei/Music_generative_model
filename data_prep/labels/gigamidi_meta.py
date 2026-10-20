"""
GigaMIDI v2 metadata -> data/gigamidi/gigamidi_meta.parquet, one row per file with an artist, title or style:

    python data_prep/labels/gigamidi_meta.py

Source: Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv (1.25 GB) of the Hugging Face dataset Metacreation/GigaMIDI
(v2.0.0, CC BY-NC 4.0), keyed by the md5 of the file bytes like data/gigamidi/gigamidi.csv.
Columns: md5, artist, title, styles, curated, spotify, mbid, match_score. styles = the tokens of music_styles_curated
(Musicmap), music_style_scraped (site categories) and the three audio-matched columns (Discogs / Last.fm / Tagtraum
classes): lower case, '_' -> ' ', scraped values split on ',' / ';' before URL-decoding (so 'musical%2cfilm%2ctv' stays
one category). build_labels.py joins the rows to the stores; musicar/text/vocab.py GIGAMIDI_STYLE maps the tokens.

Coverage (2026-10-20): 2,136,218 files, 2.47B notes. Artist / title 207,980 files (0.54B notes, 23,846 artists, 702
with >= 50 files); any style 204,509 (0.43B notes; curated 71,791, scraped 130,877, audio-matched 9,897). Not already
MidiCaps / XMIDI (268,203 files, 1.00B notes): 192,311 styled files, 0.37B notes. Split of the styled files: train
163,602 / validation 20,288 / test 20,619.
"""

import argparse
import ast
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import pandas as pd

STYLE_COLUMNS = ('music_styles_curated', 'music_style_scraped', 'music_style_audio_text_Discogs',
                 'music_style_audio_text_Lastfm', 'music_style_audio_text_Tagtraum')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv', default='data/gigamidi/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv')
    p.add_argument('--out', default='data/gigamidi/gigamidi_meta.parquet')
    return p.parse_args()


def styles_of(row):
    """Style tokens of one CSV row (a mapping column -> str / NaN), in column order, without repeats."""
    out = []
    for column in STYLE_COLUMNS:
        value = row.get(column)
        if not isinstance(value, str):
            continue
        for part in re.split('[,;]', value):
            token = urllib.parse.unquote(part).replace('_', ' ').strip().lower()
            if token and token not in out:
                out.append(token)
    return out


def _clean(x):
    """Whitespace runs (some artists contain line breaks: 'Pink\\nFloyd') -> one space; '' / NaN -> None."""
    if not isinstance(x, str):
        return None
    return ' '.join(x.split()) or None


def _mbids(x):
    return [str(m) for m in ast.literal_eval(x)] if isinstance(x, str) and x.startswith('[') else None


def main():
    args = parse_args()
    columns = ['md5', 'artist', 'title', *STYLE_COLUMNS, 'audio_text_matches_sid', 'audio_text_matches_mbid',
               'audio_text_matches_score']
    d = pd.read_csv(args.csv, usecols=columns, dtype=str)
    n_files = len(d)
    d = d[d[list(STYLE_COLUMNS)].notna().any(axis=1) | d.artist.notna() | d.title.notna()]
    out = pd.DataFrame({
        'md5': d.md5.values,
        'artist': [_clean(x) for x in d.artist],
        'title': [_clean(x) for x in d.title],
        'styles': [styles_of(r) for r in d[list(STYLE_COLUMNS)].to_dict('records')],
        'curated': [_clean(x) for x in d.music_styles_curated],
        'spotify': [_clean(x) for x in d.audio_text_matches_sid],
        'mbid': [_mbids(x) for x in d.audio_text_matches_mbid],
        'match_score': pd.to_numeric(d.audio_text_matches_score, errors='coerce').values,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    styled = out.styles.map(len) > 0
    print(f'{n_files:,} files -> {len(out):,} rows ({styled.sum():,} with styles, {out.artist.notna().sum():,} '
          f'with an artist) -> {args.out}')

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from musicar.text import vocab as V
    counts = Counter(t for styles in out.styles for t in styles)
    unknown = {t: n for t, n in counts.most_common() if t not in V.GIGAMIDI_STYLE and t not in V.GIGAMIDI_SKIP}
    mapped = out.styles.map(lambda s: any(t in V.GIGAMIDI_STYLE for t in s))
    print(f'{len(counts)} style tokens; files with a mapped style: {mapped.sum():,}; skipped tokens: '
          f'{sum(n for t, n in counts.items() if t in V.GIGAMIDI_SKIP):,}; unknown: {unknown}')


if __name__ == '__main__':
    main()
