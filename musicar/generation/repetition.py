"""
Verbatim repetition of pitched passages: a sampling penalty against it and its measure.

The unit is a chord: the pitched notes (drums left out: their patterns repeat by design) that share an onset, as a
set of (instrument, pitch) — key 'notes' — or of pitch classes — key 'pitch_class', where a repeat with another
voicing or instrument counts too — plus the time since the previous chord. Sets, because the order of notes within
an onset varies (8.5% of same-onset neighbours in real data are out of pitch order after the 20-ms quantization, 16%
in sampled continuations), so a note-by-note match misses replayed loops. A passage copies an earlier one when their
chords are equal one for one. Sampled continuations of band songs replay their own earlier passages for minutes
(2026-10-24: one copy of 315 chords = 83% of a piece; random real GigaMIDI band songs: longest copies 0-159 chords,
median ~20).

RepeatPenalty: the DRY sampler (p-e-w, 2024, "Don't Repeat Yourself") on chords. When the last m chords equal the m
chords before an earlier chord E, with m >= allowed, and the note being sampled could continue E (same timing, and
the notes already played at this onset are all in E), the pitches of E it would add lose multiplier *
base^(m - allowed) logits: continuing the copy gets ever less likely, any other pitch keeps its odds. State per row
on the device, no GPU <-> CPU sync.

copies(): for every chord, the longest earlier passage that the passage ending at it copies, and where (the measure,
eval/probes/repetition.py and generate.py --best-of).
"""

import numpy as np
import torch

from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM

PITCHES = 128
TIME_BINS = 512   # time since the previous chord, in delta-time bins, capped
KEYS = ('notes', 'pitch_class')
CHORD = 32        # notes of a chord kept for the penalty (more still count in its key)
_GAP_MUL = 0x9E3779B97F4A7C15 - (1 << 64)  # odd 64-bit multiplier of the gap in the chord hash
_WEIGHTS = np.random.default_rng(1234).integers(1, 1 << 62, size=PITCHES * 129, dtype=np.int64)


def _code(pitch, program, key):
    return pitch % 12 if key == 'pitch_class' else program * PITCHES + pitch


def chords(pitch, delta_time, programs, key='notes'):
    """The chords of one row (numpy arrays of a whole sequence): (keys (c,) hashes, notes (c,) pitched notes per
    chord, onsets (c,) in delta-time bins, first note index (c,)). A chord's key: its set of codes and its gap to the
    previous chord (the first: to the start)."""
    pitch, delta_time, programs = (np.asarray(x, dtype=np.int64) for x in (pitch, delta_time, programs))
    onset = np.cumsum(delta_time)
    keep = np.flatnonzero((programs != DRUM_PROGRAM) & (pitch < BOS_PITCH))
    groups = {}
    for i in keep:
        groups.setdefault(int(onset[i]), []).append(i)
    keys, notes, onsets, first, prev = [], [], [], [], 0
    for t, idx in groups.items():  # dict order = onset order (onsets never decrease)
        codes = {int(_code(pitch[i], programs[i], key)) for i in idx}
        h = (min(t - prev, TIME_BINS - 1) * _GAP_MUL + int(sum(int(_WEIGHTS[c]) for c in codes))) & ((1 << 64) - 1)
        keys.append(h)
        notes.append(len(idx))
        onsets.append(t)
        first.append(idx[0])
        prev = t
    return (np.array(keys, dtype=np.uint64), np.array(notes, np.int64), np.array(onsets, np.int64),
            np.array(first, np.int64))


def copies(keys):
    """(length, source) per item: the longest m such that keys[i-m+1..i] == keys[j-m+1..j] for some j < i, and that
    j (-1 when keys[i] is new)."""
    keys = np.asarray(keys)
    n = len(keys)
    length, source = np.zeros(n, np.int64), np.full(n, -1, np.int64)
    run = np.zeros(n, np.int64)  # run[j]: match length of the passages ending at the latest item and at j
    for i in range(1, n):
        eq = keys[:i] == keys[i]
        new = np.zeros(n, np.int64)
        new[1:i] = np.where(eq[1:], run[:i - 1] + 1, 0)
        new[0] = eq[0]
        run = new
        j = int(run[:i].argmax())
        if run[j]:
            length[i], source[i] = run[j], j
    return length, source


def copy_share(pitch, delta_time, programs, min_len=16, skip=0, key='notes'):
    """Share of the pitched notes after note index `skip` that lie in chords inside an exact copy of at least
    min_len chords (a chord ending a copy of m covers the m chords up to it)."""
    keys, notes, _, first = chords(pitch, delta_time, programs, key)
    if not len(keys):
        return 0.0
    length = copies(keys)[0]
    covered = np.zeros(len(keys), bool)
    for i in np.flatnonzero(length >= min_len):
        covered[i - length[i] + 1:i + 1] = True
    later = first >= skip
    return float(notes[later & covered].sum() / max(1, notes[later].sum()))


class RepeatPenalty:
    """DRY penalty on the pitch head of generate() for Bo rows of up to `total` notes (see the module docstring)."""

    def __init__(self, Bo, total, multiplier, base=1.15, allowed=16, device='cpu', key='notes', cap=50.0):
        assert key in KEYS, key
        self.mult, self.base, self.allowed, self.cap, self.key = float(multiplier), float(base), int(allowed), \
            float(cap), key
        z = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=device)  # noqa: E731
        self.hash, self.gap, self.run = z(Bo, total), z(Bo, total), z(Bo, total)  # closed chords
        self.codes = torch.full((Bo, total, CHORD), -1, dtype=torch.long, device=device)
        self.n = z(Bo)                                   # closed chords
        self.open = torch.full((Bo, CHORD), -1, dtype=torch.long, device=device)  # the chord being played
        self.open_n, self.open_sum, self.open_gap = z(Bo), z(Bo), z(Bo)
        self.onset, self.last = z(Bo), z(Bo)  # latest note's onset; onset of the open (or latest) chord
        self.dt, self.prog = z(Bo), z(Bo)     # the note being sampled: its delta time and instrument
        self.cols = torch.arange(total, device=device)
        self.w = torch.as_tensor(_WEIGHTS, device=device)
        self.rows = torch.arange(Bo, device=device)

    def _chord_hash(self):
        return self.open_gap * _GAP_MUL + self.open_sum

    def _closed_runs(self, close):
        """Runs as if the open chord were closed (rows in `close`), else the current ones."""
        h = self._chord_hash()
        eq = (self.hash == h[:, None]) & (self.cols[None] < self.n[:, None])
        prev = torch.cat([torch.zeros_like(self.run[:, :1]), self.run[:, :-1]], 1)  # run[j - 1]
        return torch.where(close[:, None], torch.where(eq, prev + 1, 0), self.run)

    def push(self, pitch, delta_time, program):
        """Append one note per row ((Bo,) tensors): drums and special tokens only move the clock."""
        self.onset += delta_time
        pitched = (program != DRUM_PROGRAM) & (pitch < BOS_PITCH)
        new = pitched & ((self.open_n == 0) | (self.onset != self.last))
        close = new & (self.open_n > 0)
        self.run = self._closed_runs(close)
        at = close[:, None] & (self.cols[None] == self.n[:, None])
        self.hash = torch.where(at, self._chord_hash()[:, None], self.hash)
        self.gap = torch.where(at, self.open_gap[:, None], self.gap)
        self.codes = torch.where(at[:, :, None], self.open[:, None, :], self.codes)
        self.n += close.long()
        self.open_gap = torch.where(new, (self.onset - self.last).clamp(max=TIME_BINS - 1), self.open_gap)
        self.open = torch.where(new[:, None], -1, self.open)
        self.open_n = torch.where(new, 0, self.open_n)
        self.open_sum = torch.where(new, 0, self.open_sum)
        self.last = torch.where(new, self.onset, self.last)
        code = _code(pitch, program, self.key).clamp(min=0, max=len(self.w) - 1)
        add = pitched & ~(self.open == code[:, None]).any(1)
        slot = add[:, None] & (torch.arange(CHORD, device=code.device)[None] == self.open_n[:, None])
        self.open = torch.where(slot, code[:, None], self.open)
        self.open_n += add.long()
        self.open_sum += torch.where(add, self.w[code], 0)

    def prime(self, pitch, delta_time, programs):
        """The prompt, (Bo, T) tensors, note by note."""
        for t in range(pitch.size(1)):
            self.push(pitch[:, t], delta_time[:, t], programs[:, t])

    def penalize(self, logits):
        """logits (Bo, V) of the pitch head for the note whose delta time (self.dt) and instrument (self.prog) are
        decided: lowered on the pitches that would continue a copy of >= allowed chords."""
        onset = self.onset + self.dt
        joins = (self.open_n > 0) & (onset == self.last)  # the note adds to the open chord
        # a note starting a chord closes the open one: the copy so far includes it, and it may be the continuation
        starts = ~joins & (self.open_n > 0)
        run = self._closed_runs(starts)
        n = self.n + starts.long()
        cols = self.cols[None]
        codes = torch.where((cols == self.n[:, None])[:, :, None] & starts[:, None, None], self.open[:, None, :],
                            self.codes)
        gap = torch.where((cols == self.n[:, None]) & starts[:, None], self.open_gap[:, None], self.gap)
        prev = torch.cat([torch.zeros_like(run[:, :1]), run[:, :-1]], 1)  # copy length before chord j
        want_gap = torch.where(joins, self.open_gap, (onset - self.last).clamp(max=TIME_BINS - 1))
        last = torch.where(joins, self.n, n)  # chords that can be continued: 1 .. last - 1 (closed ones)
        cand = (prev >= self.allowed) & (cols >= 1) & (cols < last[:, None]) & (gap == want_gap[:, None])
        played = torch.where(joins[:, None], self.open, -1)[:, None, None, :8]  # notes already at this onset
        contained = ((codes[:, :, :, None] == played) | (played < 0)).any(2).all(-1)  # (Bo, E): played within it
        cand &= contained & (self.prog != DRUM_PROGRAM)[:, None]
        value = torch.where(cand, self.mult * self.base ** (prev - self.allowed).clamp(max=200).float(), 0.0)
        value = value.clamp(max=self.cap)[:, :, None].expand_as(codes)
        fresh = (codes >= 0) & ~(codes[:, :, :, None] == played[:, :, 0, None, :]).any(-1)  # not played yet
        if self.key == 'pitch_class':
            pen = torch.zeros(logits.size(0), 12, device=logits.device).scatter_reduce(
                1, codes.clamp(min=0).flatten(1) % 12, torch.where(fresh, value, 0.0).flatten(1), 'amax')
            pen = pen[:, torch.arange(logits.size(1), device=logits.device) % 12]
            pen[:, PITCHES:] = 0  # BOS / EOS
        else:
            fresh &= codes // PITCHES == self.prog[:, None, None]
            pen = torch.zeros_like(logits).scatter_reduce(1, (codes.clamp(min=0) % PITCHES).flatten(1),
                                                          torch.where(fresh, value, 0.0).flatten(1), 'amax')
        return logits - pen
