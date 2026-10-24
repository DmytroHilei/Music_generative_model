"""
How much a piece repeats itself, chord by chord (musicar/generation/repetition.py): a chord = the pitched notes of
one onset (drums left out), as a set, with the time since the previous chord. For every chord, the longest earlier
passage that the passage ending at it copies exactly, and where that passage is; both with chords as sets of
(instrument, pitch) — exact copies — and of pitch classes — similar copies (another voicing or instrument).

    python eval/probes/repetition.py samples/retrieve_govoryly/*.mid --skip 128
    python eval/probes/repetition.py samples/retrieve_govoryly/en_1.mid --skip 128 --notes en_1_notes.csv

--skip: notes at the start not judged (a borrowed prompt; they still count as earlier passages). --notes: every
judged pitched note with its chord's copy length and the time of the passage it copies. Reference, 30 random real
GigaMIDI band songs (notes 128-1,500, 2026-10-24): share of the pitched notes inside exact copies of >= 16 chords
0-74% (median ~13%), longest copies 0-159 chords.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musicar.data.loader import tokenize_tracks  # noqa: E402
from musicar.generation.repetition import chords, copies  # noqa: E402


def covered_share(length, notes, later, min_len):
    covered = np.zeros(len(length), bool)
    for i in np.flatnonzero(length >= min_len):
        covered[i - length[i] + 1:i + 1] = True
    return float(notes[later & covered].sum() / max(1, notes[later].sum()))


def analyze(path, skip=0, notes_csv=None):
    res = tokenize_tracks(path)
    if res is None:
        raise SystemExit(f'{path}: no notes')
    tok, prog = res[0].astype(np.int64), res[1].astype(np.int64)
    out = {}
    for key in ('notes', 'pitch_class'):
        keys, notes, onsets, first = chords(tok[:, 0], tok[:, 3], prog, key)
        length, source = copies(keys)
        later = first >= skip
        if not later.any():
            raise SystemExit(f'{path}: no pitched notes after --skip {skip}')
        out[key] = dict(share16=covered_share(length, notes, later, 16),
                        share32=covered_share(length, notes, later, 32), longest=int(length[later].max()),
                        chords=int(later.sum()))
        if key == 'notes' and notes_csv:
            onset_of = np.cumsum(tok[:, 3])
            chord_of = np.searchsorted(onsets, onset_of)  # each note's chord (pitched notes only below)
            with open(notes_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['note', 'seconds', 'program', 'pitch', 'chord', 'copy_length_chords',
                            'copied_from_seconds'])
                for i in range(skip, len(tok)):
                    c = chord_of[i]
                    if prog[i] == 128 or c >= len(onsets) or onsets[c] != onset_of[i]:
                        continue  # drums
                    src = onsets[source[c] - length[c] + 1] * 0.02 if source[c] >= 0 else None
                    w.writerow([i, f'{onset_of[i] * 0.02:.2f}', int(prog[i]), int(tok[i, 0]), int(c), int(length[c]),
                                '' if src is None else f'{src:.2f}'])
    return out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('midi', nargs='+')
    p.add_argument('--skip', type=int, default=0, help='first notes not judged (a borrowed prompt)')
    p.add_argument('--notes', default='', help='one file: write every judged pitched note to this CSV')
    args = p.parse_args()
    if args.notes and len(args.midi) != 1:
        raise SystemExit('--notes takes one MIDI file')
    print(f"{'':40s} {'exact copies (chords)':>30s}   {'similar (pitch classes)':>30s}")
    print(f"{'file':<40s} {'chords':>6s} {'>=16':>6s} {'>=32':>6s} {'longest':>8s}   {'>=16':>6s} {'>=32':>6s} "
          f"{'longest':>8s}")
    rows = []
    for path in args.midi:
        r = analyze(path, args.skip, args.notes or None)
        rows.append(r)
        e, s = r['notes'], r['pitch_class']
        print(f"{Path(path).name[:40]:<40s} {e['chords']:6d} {e['share16']:6.0%} {e['share32']:6.0%} "
              f"{e['longest']:8d}   {s['share16']:6.0%} {s['share32']:6.0%} {s['longest']:8d}")
    if len(rows) > 1:
        m = lambda k, f: np.mean([r[k][f] for r in rows])  # noqa: E731
        md = lambda k: np.median([r[k]['longest'] for r in rows])  # noqa: E731
        print(f"{'mean (longest: median)':<40s} {'':6s} {m('notes', 'share16'):6.0%} {m('notes', 'share32'):6.0%} "
              f"{md('notes'):8.0f}   {m('pitch_class', 'share16'):6.0%} {m('pitch_class', 'share32'):6.0%} "
              f"{md('pitch_class'):8.0f}")


if __name__ == '__main__':
    main()
