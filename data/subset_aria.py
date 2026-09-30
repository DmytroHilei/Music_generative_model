"""
Cuts a genre subset out of the tokenized Aria store (the source archive isn't needed):

    python data/subset_aria.py --genres pop,rock --name aria_poprock

-> data/cache/<name>_{train,validation} (tokens.u16, offsets.npy, meta.json) and data/aria/<name>.csv (the matching
rows of aria.csv, same order as the store, so style labels line up).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--genres', required=True, help='comma-separated, e.g. pop,rock')
ap.add_argument('--name', required=True)
ap.add_argument('--source', default='aria')
args = ap.parse_args()
genres = set(args.genres.split(','))

meta = pd.read_csv(f'data/aria/{args.source}.csv', dtype=str, keep_default_na=False)
out_rows = []
for split in ('train', 'validation'):
    src = Path(f'data/cache/{args.source}_{split}')
    offsets = np.load(src / 'offsets.npy')
    tokens = np.memmap(src / 'tokens.u16', dtype=np.uint16, mode='r', shape=(int(offsets[-1]), 4))
    rows = meta[meta['split'] == split].reset_index(drop=True)
    assert len(rows) == len(offsets) - 1, f'{split}: metadata and store out of sync'
    keep = rows.index[rows['genre'].isin(genres)].to_numpy()
    dst = Path(f'data/cache/{args.name}_{split}')
    dst.mkdir(parents=True, exist_ok=True)
    new_offsets = [0]
    with open(dst / 'tokens.u16', 'wb') as f:
        for i in keep:
            chunk = np.asarray(tokens[offsets[i]:offsets[i + 1]])
            f.write(chunk.tobytes())
            new_offsets.append(new_offsets[-1] + len(chunk))
    np.save(dst / 'offsets.npy', np.array(new_offsets, dtype=np.int64))
    (dst / 'meta.json').write_text(json.dumps(dict(n_files=len(keep), n_notes=new_offsets[-1], n_empty_or_failed=0)))
    out_rows.append(rows.loc[keep])
    print(f'{split}: {len(keep):,} files, {new_offsets[-1]:,} notes -> {dst}')
pd.concat(out_rows).to_csv(f'data/aria/{args.name}.csv', index=False)
print(f'metadata: data/aria/{args.name}.csv')
