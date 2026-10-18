"""Tokenization and MIDI writing checks: python tests/data/test_tokenize.py  (or pytest tests/)."""

import io
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run  # noqa: E402
from musicar.data.loader import (DRUM_PROGRAM, quantize_time, quantize_velocity, tokenize_midi,  # noqa: E402
                                 tokenize_tracks)
from musicar.data.midi_io import tokens_to_midi  # noqa: E402


def midi_bytes(tracks):
    """tracks: list of (program, is_drum, [(start, end, pitch, velocity)]) -> MIDI file bytes."""
    pm = pretty_midi.PrettyMIDI()
    for program, drum, notes in tracks:
        inst = pretty_midi.Instrument(program=program, is_drum=drum)
        inst.notes = [pretty_midi.Note(velocity=v, pitch=p, start=s, end=e) for s, e, p, v in notes]
        pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


PIANO = [(0.0, 0.5, 60, 100), (0.0, 0.5, 64, 90), (0.5, 1.0, 67, 64), (1.2, 1.3, 72, 20)]
BASS = [(0.25, 0.75, 36, 80)]
DRUMS = [(0.0, 0.05, 36, 127), (0.5, 0.55, 38, 100)]


def test_quantize():
    assert quantize_time(0.0, 0.02, 511) == 0 and quantize_time(0.031, 0.02, 511) == 2
    assert quantize_time(100.0, 0.02, 511) == 511 and quantize_time(-1.0, 0.02, 511) == 0
    assert quantize_velocity(0, 32) == 0 and quantize_velocity(127, 32) == 31 and quantize_velocity(64, 32) == 16


def test_tokenize_midi_piano():
    data = midi_bytes([(0, False, PIANO), (33, False, BASS), (0, True, DRUMS)])
    t = tokenize_midi(data)  # bytes, file-like and paths all work
    assert np.array_equal(t, tokenize_midi(io.BytesIO(data)))
    assert t.dtype == np.uint16 and t.shape == (5, 4), 'non-drum tracks merged, drums dropped'
    assert t[:, 0].tolist() == [60, 64, 36, 67, 72], 'sorted by onset, then pitch'
    assert t[:, 3].tolist() == [0, 0, 12, 12, 35], 'delta_time = onset gap in 20 ms bins, chords 0'
    assert t[:, 2].tolist() == [25, 25, 25, 25, 5] and t[0, 1] == 25
    assert tokenize_midi(b'not a midi') is None
    assert tokenize_midi(midi_bytes([(0, True, DRUMS)])) is None, 'drums only = empty'


def test_tokenize_tracks_multi():
    t, programs = tokenize_tracks(midi_bytes([(0, False, PIANO), (33, False, BASS), (0, True, DRUMS)]))
    assert len(t) == len(programs) == 7 and programs.dtype == np.uint8
    assert programs[:3].tolist() == [0, 0, DRUM_PROGRAM], 'onset 0: sorted by program, then pitch'
    assert sorted(set(programs.tolist())) == [0, 33, DRUM_PROGRAM]


def test_tokens_to_midi_round_trip():
    data = midi_bytes([(0, False, PIANO), (33, False, BASS), (0, True, DRUMS)])
    t, programs = tokenize_tracks(data)
    pm = tokens_to_midi(*t.T.astype(int), programs=programs.tolist())
    by_prog = {(i.program, i.is_drum): i for i in pm.instruments}
    assert set(by_prog) == {(0, False), (33, False), (0, True)}
    back, back_programs = tokenize_tracks(io.BytesIO(_write(pm)))
    assert np.array_equal(back[:, 0], t[:, 0]) and np.array_equal(back[:, 3], t[:, 3]), 'pitches and onsets survive'
    assert np.array_equal(back_programs, programs)
    assert all(abs(n.end - n.start - 0.1) < 1e-6 for n in by_prog[(0, True)].notes), 'fixed drum length'


def test_tokens_to_midi_polyphony_and_gaps():
    pm = tokens_to_midi([60, 64, 67, 72], [10, 30, 20, 5], [10, 10, 10, 10], [0, 0, 0, 500], max_polyphony=2,
                        max_delta=1.0)
    notes = pm.instruments[0].notes
    assert [n.pitch for n in notes] == [64, 67, 72], 'the 2 loudest of the chord'
    assert abs(notes[-1].start - 1.0) < 1e-6, 'gap capped at max_delta'


def _write(pm):
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


if __name__ == '__main__':
    run(globals())
