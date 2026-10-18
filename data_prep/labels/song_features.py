"""
Per-song features of every file in a token store, for the caption builder (text-control plan 15h, step 1.3).

    python data_prep/labels/song_features.py --stores data/cache/gigamidi_replay_train,data/cache/gigamidi_heldout_validation

Writes data/labels/<store>.features.npz, one row per store file (labeled or not):
    seconds, n_notes, nps (notes/s), ops (onsets/s), notes_per_onset, pitch_mean, pitch_std, pitch_lo, pitch_hi
    (5th / 95th percentile), vel_mean, vel_std (velocity bins 0-31), key (0-11 major on that tonic, 12-23 minor,
    -1 = none), key_conf (correlation margin over the second-best key), tempo (BPM, 0 = none), tempo_conf,
    drums (share of drum notes), programs (n, 17) packed bits: GM programs 0-127 + drums (128) with >= 2% of the
    notes and >= 8 notes.
Pitch statistics, key and tempo leave out drum notes. Key: Krumhansl-Schmuckler over the whole song
(musicar/eval/structure_eval.py). Tempo: autocorrelation of the onset signal (20 ms frames, onsets weighted by
note count x velocity) over the first 120 s, lags 0.3-1.5 s (40-200 BPM), with a log-normal prior around 110 BPM
against octave errors (librosa-style); conf = peak height relative to lag 0.

--check compares key and tempo with MidiCaps (data/labels/<store>.jsonl from build_labels.py) on the labeled files.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from musicar.data.loader import DRUM_PROGRAM
from musicar.eval.structure_eval import PROFILES

DT = 0.02
FIELDS = ('seconds', 'n_notes', 'nps', 'ops', 'notes_per_onset', 'pitch_mean', 'pitch_std', 'pitch_lo', 'pitch_hi',
          'vel_mean', 'vel_std', 'key', 'key_conf', 'tempo', 'tempo_conf', 'drums')
PITCH_CLASSES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
_store = {}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stores', required=True, help='comma-separated store directories')
    p.add_argument('--out-dir', default='data/labels')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--check', action='store_true', help='compare key / tempo with MidiCaps on the labeled files')
    return p.parse_args()


def key_estimate(pitch, dur):
    """(key index, margin of the best correlation over the second best); -1 without pitched notes."""
    h = np.bincount(pitch % 12, weights=dur + 1e-3, minlength=12)
    if h.sum() == 0 or h.std() == 0:
        return -1, 0.0
    hc = (h - h.mean()) / h.std()
    pc = (PROFILES - PROFILES.mean(1, keepdims=True)) / PROFILES.std(1, keepdims=True)
    r = pc @ hc / 12
    order = np.argsort(r)
    return int(order[-1]), float(r[order[-1]] - r[order[-2]])


def tempo_estimate(onset_frames, weights, max_seconds=120.0):
    """(BPM, confidence) from the autocorrelation of the onset signal; (0, 0) for too few onsets."""
    keep = onset_frames < max_seconds / DT
    f, w = onset_frames[keep], weights[keep]
    if len(f) < 16:
        return 0.0, 0.0
    sig = np.bincount(f, weights=w).astype(np.float64)
    sig -= sig.mean()
    n = len(sig)
    spec = np.fft.rfft(sig, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return 0.0, 0.0
    lags = np.arange(int(0.3 / DT), min(int(1.5 / DT), n - 1) + 1)
    if len(lags) == 0:
        return 0.0, 0.0
    bpm = 60.0 / (lags * DT)
    prior = np.exp(-0.5 * (np.log2(bpm / 110.0) / 1.0) ** 2)
    score = ac[lags] / ac[0]
    i = int(np.argmax(score.clip(min=0) * prior))
    return float(bpm[i]), float(score[i])


def file_features(tok, prog):
    pitch, vel, dur, dt = (tok[:, j].astype(np.int64) for j in range(4))
    onset = np.cumsum(dt)
    seconds = max(onset[-1] * DT + dur[-1] * DT, 1e-3)
    n = len(tok)
    n_onsets = max(int((dt > 0).sum()) + 1, 1)
    drums = prog == DRUM_PROGRAM
    pitched = ~drums
    p = pitch[pitched]
    if len(p):
        lo, hi = np.percentile(p, [5, 95])
        key, kconf = key_estimate(p, dur[pitched].astype(np.float64))
    else:
        lo = hi = 0
        key, kconf = -1, 0.0
    frames, inv = np.unique(onset, return_inverse=True)
    weights = np.bincount(inv, weights=vel + 1)
    tempo, tconf = tempo_estimate(frames, weights)
    counts = np.bincount(prog, minlength=129)
    present = (counts >= 8) & (counts >= 0.02 * n)
    feats = [seconds, n, n / seconds, n_onsets / seconds, n / n_onsets, p.mean() if len(p) else 0,
             p.std() if len(p) else 0, lo, hi, vel.mean(), vel.std(), key, kconf, tempo, tconf, drums.mean()]
    return np.array(feats, dtype=np.float64), np.packbits(present[:129], bitorder='little')


def _init(store):
    offsets = np.load(Path(store) / 'offsets.npy')
    n = int(offsets[-1])
    _store['offsets'] = offsets
    _store['tokens'] = np.memmap(Path(store) / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    path = Path(store) / 'programs.u8'
    _store['programs'] = np.memmap(path, dtype=np.uint8, mode='r', shape=(n,)) if path.exists() else None


def _work(rng):
    o, rows, bits = _store['offsets'], [], []
    for i in range(*rng):
        a, b = int(o[i]), int(o[i + 1])
        prog = (_store['programs'][a:b].astype(np.int64) if _store['programs'] is not None
                else np.zeros(b - a, np.int64))
        f, pb = file_features(np.asarray(_store['tokens'][a:b]), prog)
        rows.append(f)
        bits.append(pb)
    return rng[0], np.stack(rows), np.stack(bits)


def check(out, store, labels_dir):
    path = Path(labels_dir) / f'{Path(store).name}.jsonl'
    if not path.exists():
        print(f'{path}: no label table, skipping the check')
        return
    z = np.load(out)
    key_hit = mode_hit = rel = n_key = 0
    tempo_err, octave_ok = [], 0
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        m = r.get('midicaps')
        if not m:
            continue
        i = r['file']
        if m.get('key'):
            tonic, mode = m['key'].rsplit(' ', 1)
            ref = PITCH_CLASSES.index(tonic.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#')
                                      .replace('Ab', 'G#').replace('Bb', 'A#')) + (12 if mode == 'minor' else 0)
            k = int(z['key'][i])
            n_key += 1
            key_hit += k == ref
            mode_hit += (k >= 12) == (ref >= 12)
            rel += k >= 0 and ((k < 12 and ref == 12 + (k + 9) % 12) or (k >= 12 and ref == (k - 12 + 3) % 12))
        if m.get('tempo') and z['tempo'][i] > 0:
            ratio = z['tempo'][i] / m['tempo']
            tempo_err.append(abs(np.log2(ratio)))
            octave_ok += min(abs(np.log2(ratio) - o) for o in (-1, 0, 1)) < np.log2(1.08)
    e = np.array(tempo_err)
    print(f'{Path(store).name} vs MidiCaps: key exact {key_hit / n_key:.1%}, mode {mode_hit / n_key:.1%}, '
          f'relative major/minor confusions {rel / n_key:.1%} ({n_key} files); tempo within 8% '
          f'{(e < np.log2(1.08)).mean():.1%}, within 8% allowing x2 / x0.5 {octave_ok / len(e):.1%} ({len(e)} files)')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for store in args.stores.split(','):
        n = len(np.load(Path(store) / 'offsets.npy')) - 1
        ranges = [(a, min(a + 500, n)) for a in range(0, n, 500)]
        feats = np.zeros((n, len(FIELDS)))
        bits = np.zeros((n, 17), np.uint8)
        with Pool(args.workers, initializer=_init, initargs=(store,)) as pool:
            for a, f, b in tqdm(pool.imap_unordered(_work, ranges), total=len(ranges), desc=Path(store).name):
                feats[a:a + len(f)], bits[a:a + len(f)] = f, b
        arrays = {name: feats[:, j].astype(np.int16 if name in ('key',) else np.float32)
                  for j, name in enumerate(FIELDS)}
        arrays['n_notes'] = feats[:, FIELDS.index('n_notes')].astype(np.int32)
        out = out_dir / f'{Path(store).name}.features.npz'
        np.savez(out, programs=bits, **arrays)
        print(f'{out}: {n:,} files, median {np.median(arrays["seconds"]):.0f} s, {np.median(arrays["nps"]):.1f} '
              f'notes/s, tempo found {(arrays["tempo"] > 0).mean():.0%}, key found {(arrays["key"] >= 0).mean():.0%}')
        if args.check:
            check(out, store, args.out_dir)


if __name__ == '__main__':
    main()
