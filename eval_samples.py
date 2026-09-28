"""
Objective sanity metrics for generated MIDI: compare simple statistics of generated files with real files.

    python eval_samples.py --generated samples/*.mid --reference-csv data/combined.csv --split validation

For each feature we compute a histogram over all notes of each set and report the overlapping area (OA, 0..1,
1 = identical distributions, cf. Yang & Lerch 2020) plus the mean of both sets. This doesn't measure "musicality",
but it catches obvious failures: wrong density, stuck pitches, no rhythm structure, runaway gaps.
"""

import argparse
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

# name -> histogram bin edges
FEATURES = {
    'pitch': np.arange(0, 129),
    'pitch_class': np.arange(0, 13),
    'velocity': np.arange(0, 129, 4),
    'duration_s': np.concatenate([[0], np.geomspace(0.01, 10, 40)]),
    'ioi_s': np.concatenate([[0, 0.005], np.geomspace(0.01, 10, 40)]),  # onset gap to previous note
    'polyphony': np.arange(0, 17),           # notes sounding at each onset
    'interval': np.arange(-24, 26),          # pitch step between consecutive onsets (clipped)
}


def note_features(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = sorted((n for inst in midi.instruments if not inst.is_drum for n in inst.notes),
                   key=lambda n: (n.start, n.pitch))
    if len(notes) < 2:
        return None
    starts = np.array([n.start for n in notes])
    ends = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes])
    # polyphony at each onset: notes that started before and are still sounding
    order = np.argsort(ends)
    poly = np.searchsorted(starts, starts, side='right') - np.searchsorted(ends[order], starts, side='right')
    total_time = ends.max() - starts.min()
    return {
        'pitch': pitch,
        'pitch_class': pitch % 12,
        'velocity': np.array([n.velocity for n in notes]),
        'duration_s': ends - starts,
        'ioi_s': np.diff(starts),
        'polyphony': np.clip(poly, 0, 16),
        'interval': np.clip(np.diff(pitch), -24, 24),
        '_notes_per_s': len(notes) / max(total_time, 1e-6),
    }


def collect(paths):
    feats = [f for f in map(note_features, paths) if f is not None]
    merged = {k: np.concatenate([f[k] for f in feats]) for k in FEATURES}
    merged['_notes_per_s'] = np.array([f['_notes_per_s'] for f in feats])
    return merged, len(feats)


def overlap_area(a, b, bins):
    ha, _ = np.histogram(np.clip(a, bins[0], bins[-1]), bins=bins)
    hb, _ = np.histogram(np.clip(b, bins[0], bins[-1]), bins=bins)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return np.minimum(ha, hb).sum()


def pitch_class_entropy(pc):
    h = np.bincount(pc, minlength=12) / len(pc)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generated', nargs='+', required=True, help='generated MIDI files (globs ok)')
    parser.add_argument('--reference', nargs='*', default=[], help='reference MIDI files (globs ok)')
    parser.add_argument('--reference-csv', default=None, help='or: dataset CSV to take reference files from')
    parser.add_argument('--root-dir', default='.')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--max-reference', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    gen = [p for g in args.generated for p in glob.glob(g)]
    ref = [p for g in args.reference for p in glob.glob(g)]
    if args.reference_csv:
        df = pd.read_csv(args.reference_csv)
        ref += [str(Path(args.root_dir) / f) for f in df[df['split'] == args.split]['midi_filename']]
    ref = [p for p in ref if Path(p).exists()]
    random.Random(args.seed).shuffle(ref)
    ref = ref[:args.max_reference]
    assert gen and ref, f"need generated ({len(gen)}) and reference ({len(ref)}) files"

    g, ng = collect(gen)
    r, nr = collect(ref)
    print(f"generated: {ng} files, {len(g['pitch']):,} notes | reference: {nr} files, {len(r['pitch']):,} notes\n")
    print(f"{'feature':<14}{'OA':>6}{'gen mean':>12}{'ref mean':>12}")
    oas = []
    for name, bins in FEATURES.items():
        oa = overlap_area(g[name], r[name], bins)
        oas.append(oa)
        print(f"{name:<14}{oa:>6.3f}{g[name].mean():>12.3f}{r[name].mean():>12.3f}")
    print(f"{'notes/s':<14}{'':>6}{g['_notes_per_s'].mean():>12.3f}{r['_notes_per_s'].mean():>12.3f}")
    print(f"{'pc entropy':<14}{'':>6}{pitch_class_entropy(g['pitch_class']):>12.3f}"
          f"{pitch_class_entropy(r['pitch_class']):>12.3f}")
    print(f"\nmean OA: {np.mean(oas):.3f}")


if __name__ == '__main__':
    main()
