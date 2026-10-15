"""
Label side table for text conditioning: which files of a token store have MidiCaps captions / XMIDI labels.

    python data_prep/build_labels.py --stores data/cache/gigamidi_replay_train,data/cache/gigamidi_heldout_validation

A store file is matched to its GigaMIDI md5 through the exact note hash (tokens + programs, `note_hashes` of
prepare_discover.py) cached in data/gigamidi/gigamidi_hashes.npz, so it works for any store cut from GigaMIDI (the
full splits, gigamidi_clean_*, the split_store.py parts). MidiCaps (data/midicaps/train.json) is keyed by the Lakh
md5 = the GigaMIDI file name; XMIDI files (data/xmidi/XMIDI.zip) are matched by the md5 of their bytes.

Writes data/labels/<store name>.jsonl, one line per labeled file:
    {"file": index in the store, "md5": ..., "midicaps": {caption, genre, mood, key, time_signature, tempo,
     tempo_word, duration_word, chord_summary, instrument_summary, test_set}, "xmidi": [[emotion, genre], ...]}
(xmidi is a list: 1,699 XMIDI md5s carry two or more different labels). Unlabeled files are left out.
"""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from prepare_discover import note_hashes

MIDICAPS_FIELDS = ('caption', 'genre', 'mood', 'key', 'time_signature', 'tempo', 'tempo_word', 'duration_word',
                   'chord_summary', 'instrument_summary', 'test_set')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stores', required=True, help='comma-separated store directories')
    p.add_argument('--hashes', default='data/gigamidi/gigamidi_hashes.npz')
    p.add_argument('--midicaps', default='data/midicaps/train.json')
    p.add_argument('--xmidi', default='data/xmidi/XMIDI.zip')
    p.add_argument('--out-dir', default='data/labels')
    return p.parse_args()


def load_midicaps(path):
    caps = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            caps[Path(row['location']).stem] = {k: row[k] for k in MIDICAPS_FIELDS}
    return caps


def load_xmidi(path):
    labels = {}
    with zipfile.ZipFile(path) as z:
        for info in tqdm(z.infolist(), desc='xmidi md5'):
            if info.is_dir():
                continue
            m = re.match(r'XMIDI_([^_]+)_([^_]+)_', Path(info.filename).name)
            md5 = hashlib.md5(z.read(info)).hexdigest()
            pair = [m.group(1), m.group(2)]
            if pair not in labels.setdefault(md5, []):
                labels[md5].append(pair)
    return labels


def main():
    args = parse_args()
    z = np.load(args.hashes)
    md5_of = dict(zip(z['exact'].tolist(), (m.hex() for m in z['md5'])))
    caps = load_midicaps(args.midicaps)
    xmidi = load_xmidi(args.xmidi)
    print(f'{len(md5_of):,} GigaMIDI hashes, {len(caps):,} MidiCaps captions, {len(xmidi):,} XMIDI md5s')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for store in args.stores.split(','):
        store = Path(store)
        offsets = np.load(store / 'offsets.npy')
        n = int(offsets[-1])
        tokens = np.memmap(store / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
        programs = np.memmap(store / 'programs.u8', dtype=np.uint8, mode='r', shape=(n,))
        counts = dict(files=len(offsets) - 1, md5=0, midicaps=0, xmidi=0, both=0, labeled_notes=0)
        with open(out_dir / f'{store.name}.jsonl', 'w', encoding='utf-8') as f:
            for i, (a, b) in enumerate(tqdm(zip(offsets[:-1], offsets[1:]), total=counts['files'], desc=store.name)):
                md5 = md5_of.get(note_hashes(tokens[a:b], programs[a:b])[0])
                if md5 is None:
                    continue
                counts['md5'] += 1
                row = {'file': i, 'md5': md5}
                if md5 in caps:
                    row['midicaps'] = caps[md5]
                if md5 in xmidi:
                    row['xmidi'] = xmidi[md5]
                if len(row) == 2:
                    continue
                counts['midicaps'] += 'midicaps' in row
                counts['xmidi'] += 'xmidi' in row
                counts['both'] += 'midicaps' in row and 'xmidi' in row
                counts['labeled_notes'] += int(b - a)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(store.name, json.dumps(counts))


if __name__ == '__main__':
    main()
