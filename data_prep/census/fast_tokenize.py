"""
Fast, bit-identical drop-in replacements for musicar.data.loader.tokenize_tracks / tokenize_midi.

    fast_tokenize_tracks(data) == tokenize_tracks(data)   ((n, 4) uint16 tokens, (n,) uint8 programs) or None
    fast_tokenize_midi(data)   == tokenize_midi(data)     (n, 4) uint16 tokens or None

The originals parse with pretty_midi on top of mido (pure Python), ~75 files/s on this box. Here the whole chain is
ported to C (fast_tokenize.c, compiled on first import with gcc into data_prep/census/_build/ and loaded with ctypes):

  * mido 1.3.3 parsing, including every way it raises on a broken file: missing MThd/MTrk, truncated data (EOFError),
    a track whose events overrun its chunk size (mido keeps reading until EOF), running status without a previous
    status, undefined status bytes (0xf4 0xf5 0xf9 0xfd), data bytes > 127, wrong message length (running status on a
    1-byte system message), meta/sysex payloads over 1e6 bytes or with bytes > 127 (sysex), and meta decoders that
    raise: sequence_number of length 1, channel_prefix of length 0, set_tempo < 3 bytes, smpte_offset (frame-rate bits,
    minutes/seconds > 59, sub_frames > 99), time_signature < 4 bytes or a denominator 2**k that mido's
    math.log(2**k, 2) check rejects (table computed here in Python, so it matches this interpreter), key_signature
    < 2 bytes or outside -7..7 / mode 0-1. Sysex running status (the peek byte discarded) is reproduced, and so is a mido quirk:
    meta messages of unknown type (e.g. 0x08, 0x0a) are built without their delta time, so the delta is lost and
    every later event of the track moves earlier by it.
  * pretty_midi 0.2.11: num_tracks <= 0 or an empty track (max([]) raises), ticks_per_beat 0, tempo 0, any event at
    an absolute tick >= 1e7 (MAX_TICK), track-0 time signature with numerator 0; the tempo map from track 0 only (a tempo
    event at tick 0 replaces the map, repeats of the same tick scale are ignored); tick -> seconds with the same float
    operations as _update_tick_to_time (segment start + scale * ticks, no FMA); note pairing per (channel, pitch) FIFO
    with velocity-0 note-ons as note-offs, one note-off closing every open note from earlier ticks, the same-tick
    note-on kept only if something else was closed (otherwise dropped), spurious note-offs ignored, notes left open at
    the end of a track dropped; instruments keyed by (program at note-off, channel, track), drums = channel 9, in
    creation order.
  * loader: stable sort by (onset, program, pitch) / (onset, pitch) with instrument order then note order as the
    tiebreak (= Python's stable list.sort), quantize_time with Python round (half to even = C rint) on the same
    float division, velocity * bins // 128.

Fallback: the original pretty_midi tokenizer runs if the C library can't be built/loaded, or the C code returns
FALLBACK: negative (SMPTE-style) ticks_per_beat, which pretty_midi uses as is, so time runs backwards (Note() raises
when end < start, signatures at negative times raise); out of memory; output larger than the buffer. data_prep/census/test_fast_tokenize.py checks equality.

Single-threaded and stateless; safe to call from worker processes (each loads its own copy of the library).
"""

import ctypes
import hashlib
import io
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / 'fast_tokenize.c'
_BUILD = _HERE / '_build'

FALLBACK = -2
BROKEN = -1
STATS = {'fast': 0, 'fallback': 0}   # calls answered by C / by the original tokenizer (this process)

# time_signature denominators 2**k that mido rejects: its check is math.log(value, 2) == int(math.log(value, 2))
_BAD_DEN = bytes(int(math.log(2 ** k, 2) != int(math.log(2 ** k, 2))) for k in range(256))


def _build():
    src = _SRC.read_bytes()
    so = _BUILD / f'fast_tokenize_{hashlib.sha1(src).hexdigest()[:12]}.so'
    if not so.exists():
        _BUILD.mkdir(exist_ok=True)
        tmp = so.with_suffix(f'.{os.getpid()}.tmp')
        subprocess.run(['gcc', '-O2', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math', '-o', str(tmp),
                        str(_SRC), '-lm'], check=True, capture_output=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    f = lib.ft_tokenize
    f.restype = ctypes.c_int64
    f.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_double,
                  ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    return f


try:
    _ft = _build()
    BUILD_ERROR = None
except Exception as e:  # no compiler etc.: everything goes to the original tokenizer
    _ft = None
    BUILD_ERROR = repr(e)
    print(f'fast_tokenize: C build failed, using pretty_midi ({BUILD_ERROR})', file=sys.stderr)


def _as_bytes(midi):
    if isinstance(midi, bytes):
        return midi
    if isinstance(midi, (bytearray, memoryview)):
        return bytes(midi)
    if isinstance(midi, (str, Path)):
        return Path(midi).read_bytes()
    return midi.read()


def _run(data, mode, velocity_bins, time_resolution, max_duration_bin, max_delta_bin):
    """(n, tokens, programs) from C; n = FALLBACK when C can't answer."""
    if _ft is None:
        return FALLBACK, None, None
    cap = len(data) // 3 + 1          # a note needs a note-on of >= 3 bytes (delta + 2 data bytes)
    tokens = np.empty((cap, 4), np.uint16)
    programs = np.empty(cap, np.uint8)
    n = _ft(data, len(data), mode, _BAD_DEN, velocity_bins, time_resolution, max_duration_bin, max_delta_bin,
            tokens.ctypes.data, programs.ctypes.data, cap)
    return n, tokens, programs


def fast_tokenize_tracks(midi, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """Same result as musicar.data.loader.tokenize_tracks: ((n, 4) uint16 tokens, (n,) uint8 programs) or None."""
    data = _as_bytes(midi)
    n, tokens, programs = _run(data, 0, velocity_bins, time_resolution, max_duration_bin, max_delta_bin)
    if n == FALLBACK:
        STATS['fallback'] += 1
        from musicar.data.loader import tokenize_tracks
        return tokenize_tracks(io.BytesIO(data), velocity_bins, time_resolution, max_duration_bin, max_delta_bin)
    STATS['fast'] += 1
    if n <= 0:
        return None
    return tokens[:n].copy(), programs[:n].copy()


def fast_tokenize_midi(midi, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """Same result as musicar.data.loader.tokenize_midi: (n, 4) uint16 tokens (non-drum notes) or None."""
    data = _as_bytes(midi)
    n, tokens, _ = _run(data, 1, velocity_bins, time_resolution, max_duration_bin, max_delta_bin)
    if n == FALLBACK:
        STATS['fallback'] += 1
        from musicar.data.loader import tokenize_midi
        return tokenize_midi(io.BytesIO(data), velocity_bins, time_resolution, max_duration_bin, max_delta_bin)
    STATS['fast'] += 1
    if n <= 0:
        return None
    return tokens[:n].copy()
