"""
Label side table of the Aria stores from their metadata CSV (data_prep/datasets/prepare_aria.py writes one row per
store file, in store order per split):

    python data_prep/labels/aria_labels.py                                   # aria_train, aria_validation
    python data_prep/labels/aria_labels.py --csv data/aria/aria_poprock.csv --name aria_poprock

Writes data/labels/<name>_<split>.jsonl, one line per file with a genre or composer:
    {"file": index in the store, "aria": {"genre": ..., "composer": ...}}
musicar/text/captions.py turns the genre into a genre id and the composer (lower-case surname) into an artist through
musicar/text/artists.py. Aria (2026-10-20): 224,322 of 367,318 train files have a genre, 117,564 a composer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv', default='data/aria/aria.csv')
    p.add_argument('--name', default='aria', help='store prefix: <cache-dir>/<name>_<split>')
    p.add_argument('--cache-dir', default='data/cache')
    p.add_argument('--out-dir', default='data/labels')
    return p.parse_args()


def aria_rows(meta, offsets):
    """Label rows of one split: meta = its CSV rows in store order (genre, composer, n_notes; '' = missing), offsets =
    the store's offsets.npy (checked against n_notes)."""
    assert len(meta) == len(offsets) - 1, (len(meta), len(offsets) - 1)
    assert (np.diff(offsets) == meta.n_notes.astype(np.int64).values).all(), 'CSV rows are not in store order'
    rows = []
    for i, (genre, composer) in enumerate(zip(meta.genre, meta.composer)):
        label = {k: v.strip() for k, v in (('genre', genre), ('composer', composer)) if v.strip()}
        if label:
            rows.append({'file': i, 'aria': label})
    return rows


def main():
    args = parse_args()
    meta = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation'):
        store = Path(args.cache_dir) / f'{args.name}_{split}'
        rows = aria_rows(meta[meta.split == split], np.load(store / 'offsets.npy'))
        with open(out_dir / f'{store.name}.jsonl', 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        print(f"{store.name}: {len(rows):,} labeled files ({sum('genre' in r['aria'] for r in rows):,} genre, "
              f"{sum('composer' in r['aria'] for r in rows):,} composer) -> {out_dir / f'{store.name}.jsonl'}")


if __name__ == '__main__':
    main()
