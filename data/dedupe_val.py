"""
Write a copy of a GigaMIDI evaluation store without the files whose notes also occur in its train split.

    python data/dedupe_val.py                     # data/cache/gigamidi_validation -> data/cache/gigamidi_clean_validation
    python data/dedupe_val.py --split test --kind exact

Uses the per-file hashes that data/prepare_discover.py caches in data/gigamidi/<giga>_hashes.npz (built from the
stores on its first run). --kind loose (default) drops a file when its sorted (onset, pitch) pairs match any train
file (same notes, any instrument/velocity/duration), which includes every exact match. --split test also drops the
files that occur in validation (checkpoints are picked on validation). Duplicates inside the evaluation split itself
are kept once. Train with val_csv_path='store:data/cache/<giga>_clean'.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SPLIT_IDS = {'train': 0, 'validation': 1, 'test': 2}  # order of giga_index() in prepare_discover.py


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--giga', default='gigamidi')
    p.add_argument('--split', default='validation', choices=['validation', 'test'])
    p.add_argument('--kind', default='loose', choices=['exact', 'loose'])
    p.add_argument('--cache-dir', default='data/cache')
    args = p.parse_args()

    z = np.load(Path('data/gigamidi') / f'{args.giga}_hashes.npz')
    h, split = z[args.kind], z['split']
    seen = h[split == SPLIT_IDS['train']]
    if args.split == 'test':
        seen = np.concatenate([seen, h[split == SPLIT_IDS['validation']]])
    ev = h[split == SPLIT_IDS[args.split]]
    leaked = np.isin(ev, seen)
    _, first = np.unique(ev, return_index=True)
    once = np.zeros(len(ev), bool)
    once[first] = True
    keep = ~leaked & once

    src = Path(args.cache_dir) / f'{args.giga}_{args.split}'
    dst = Path(args.cache_dir) / f'{args.giga}_clean_{args.split}'
    meta = json.loads((src / 'meta.json').read_text())
    offsets = np.load(src / 'offsets.npy')
    assert len(offsets) - 1 == len(ev), f'{src}: {len(offsets) - 1} files but {len(ev)} hashes'
    tokens = np.memmap(src / 'tokens.u16', dtype=np.uint16, mode='r', shape=(meta['n_notes'], 4))
    programs = np.memmap(src / 'programs.u8', dtype=np.uint8, mode='r', shape=(meta['n_notes'],))

    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'meta.json').unlink(missing_ok=True)  # incomplete until written last
    new_offsets = [0]
    with open(dst / 'tokens.u16', 'wb') as ft, open(dst / 'programs.u8', 'wb') as fp:
        for i in np.flatnonzero(keep):
            a, b = offsets[i], offsets[i + 1]
            ft.write(tokens[a:b].tobytes())
            fp.write(programs[a:b].tobytes())
            new_offsets.append(new_offsets[-1] + int(b - a))
    np.save(dst / 'offsets.npy', np.array(new_offsets, dtype=np.int64))
    out = dict(n_files=int(keep.sum()), n_notes=new_offsets[-1], programs=True, n_empty_or_failed=0,
               source=str(src), dropped_in_train=int(leaked.sum()),  # test: in train or validation
               dropped_repeats=int((~leaked & ~once).sum()), kind=args.kind)
    (dst / 'meta.json').write_text(json.dumps(out))
    print(f"{src}: {len(ev):,} files, {meta['n_notes']:,} notes -> {dst}: {out['n_files']:,} files, "
          f"{out['n_notes']:,} notes ({out['dropped_in_train']:,} also in train"
          f"{' or validation' if args.split == 'test' else ''}, {out['dropped_repeats']:,} repeats)")


if __name__ == '__main__':
    main()
