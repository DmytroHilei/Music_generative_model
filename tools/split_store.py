"""
Split a token store into two by a hash of each file's tokens, e.g. the clean GigaMIDI test split (never seen in
pretraining) into a replay store for fine-tunes and a smaller held-out test (2026-10-12):

    python tools/split_store.py --store data/cache/gigamidi_clean_test --held-frac 0.1 \\
        --out-rest data/cache/gigamidi_replay_train --out-held data/cache/gigamidi_heldout_validation

Train with csv_path='...,store:data/cache/gigamidi_replay' and evaluate on 'store:data/cache/gigamidi_heldout'
(the loader appends _<split>). The hash is crc32 of the file's token bytes, so the split doesn't depend on file order
and a rerun gives the same files. Copies tokens.u16 (+ programs.u8) and writes offsets.npy, meta.json last.
"""

import argparse
import contextlib
import json
import zlib
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--store', required=True)
    ap.add_argument('--held-frac', type=float, default=0.1)
    ap.add_argument('--out-rest', required=True)
    ap.add_argument('--out-held', required=True)
    args = ap.parse_args()

    src = Path(args.store)
    offsets = np.load(src / 'offsets.npy')
    n = int(offsets[-1])
    tokens = np.memmap(src / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    has_programs = (src / 'programs.u8').exists()
    programs = np.memmap(src / 'programs.u8', dtype=np.uint8, mode='r', shape=(n,)) if has_programs else None

    held = np.array([zlib.crc32(tokens[a:b].tobytes()) % 10_000 < args.held_frac * 10_000
                     for a, b in zip(offsets[:-1], offsets[1:])])
    for out, mask, part in ((Path(args.out_held), held, 'held'), (Path(args.out_rest), ~held, 'rest')):
        out.mkdir(parents=True, exist_ok=True)
        (out / 'meta.json').unlink(missing_ok=True)  # incomplete until the end
        new_offsets = [0]
        with open(out / 'tokens.u16', 'wb') as ft, \
                open(out / 'programs.u8', 'wb') if has_programs else contextlib.nullcontext() as fp:
            for i in np.flatnonzero(mask):
                a, b = offsets[i], offsets[i + 1]
                ft.write(tokens[a:b].tobytes())
                if has_programs:
                    fp.write(programs[a:b].tobytes())
                new_offsets.append(new_offsets[-1] + int(b - a))
        np.save(out / 'offsets.npy', np.array(new_offsets, dtype=np.int64))
        meta = dict(n_files=len(new_offsets) - 1, n_notes=new_offsets[-1], **({'programs': True} if has_programs else {}),
                    n_empty_or_failed=0, source=str(src), part=part, held_frac=args.held_frac,
                    hash='crc32(tokens) % 10000')
        (out / 'meta.json').write_text(json.dumps(meta))
        print(f"{out}: {meta['n_files']:,} files, {meta['n_notes']:,} notes")


if __name__ == '__main__':
    main()
