"""
Aria-MIDI tier membership: which files of the full tier are also in the pruned / deduped / unique tiers.

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/aria_tiers.py \
        --pruned raw/aria/aria-midi-v1-pruned-ext.tar.gz --deduped raw/aria/aria-midi-v1-deduped-ext.tar.gz \
        --unique raw/aria/aria-midi-v1-unique-ext.tar.gz

Each tar.gz is streamed once ('r|gz'), reading member names only (no file contents). The stems (<file_id>_<segment>,
the .mid name without extension, as in prepare_aria.py) of each tier go to <census>/aria_tiers_<tier>.txt as soon as
that tar is listed (so a crash does not redo it; delete the .txt to force), then <census>/aria_tiers.csv is written:
stem, in_pruned, in_deduped, in_unique (0/1), one row per stem in any listed tier. Once it exists the tier tars can be
deleted. Rebuild the aria table afterwards (global_index.py --rebuild aria; it also rebuilds when the csv is newer).
"""

import argparse
import csv
import tarfile
from pathlib import Path

from tables import TIERS


def list_stems(tar_path):
    stems = []
    with tarfile.open(tar_path, 'r|gz') as tf:
        for m in tf:
            if m.isfile() and m.name.endswith('.mid'):
                stems.append(Path(m.name).stem)
    return stems


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for t in TIERS:
        ap.add_argument(f'--{t}', default=None, help=f'path of the {t} tier .tar.gz')
    ap.add_argument('--census-dir', default='census')
    args = ap.parse_args(argv)
    census = Path(args.census_dir)
    census.mkdir(parents=True, exist_ok=True)
    members = {}
    for t in TIERS:
        txt = census / f'aria_tiers_{t}.txt'
        if not txt.exists():
            tar = getattr(args, t)
            if tar is None:
                print(f'{t}: no tar given and no {txt}: column will be all 0')
                continue
            stems = list_stems(tar)
            tmp = txt.with_suffix('.tmp')
            tmp.write_text(''.join(s + '\n' for s in stems))
            tmp.replace(txt)
            print(f'{t}: {len(stems):,} files listed from {tar}')
        members[t] = set(txt.read_text().split())  # ~1M short strings at most: fine in RAM
    allstems = sorted(set().union(*members.values())) if members else []
    out = census / 'aria_tiers.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stem'] + [f'in_{t}' for t in TIERS])
        for s in allstems:
            w.writerow([s] + [int(s in members.get(t, ())) for t in TIERS])
    print(f'{out}: {len(allstems):,} stems; ' + ', '.join(f'{t} {len(members.get(t, ())):,}' for t in TIERS))


if __name__ == '__main__':
    main()
