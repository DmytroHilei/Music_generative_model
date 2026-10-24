"""Per-song note features (the caption tables' measures, data_prep/labels/song_features.py writes them for every store
file): length, density, pitch statistics, velocity, key (Krumhansl-Schmuckler, musicar/eval/structure_eval.py),
tempo (onset autocorrelation) and the instruments with >= 2% of the notes. Also used on generated notes
(musicar/generation/caption_match.py), so both are measured the same way."""

import numpy as np

from musicar.data.loader import DRUM_PROGRAM
from musicar.eval.structure_eval import PROFILES

DT = 0.02
FIELDS = ('seconds', 'n_notes', 'nps', 'ops', 'notes_per_onset', 'pitch_mean', 'pitch_std', 'pitch_lo', 'pitch_hi',
          'vel_mean', 'vel_std', 'key', 'key_conf', 'tempo', 'tempo_conf', 'drums')
PITCH_CLASSES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


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
