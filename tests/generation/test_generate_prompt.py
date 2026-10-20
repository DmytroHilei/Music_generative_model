"""generate.py prompt rows (CPU, no model): python tests/generation/test_generate_prompt.py  (or pytest tests/)."""

import sys
import tempfile
import types
from pathlib import Path

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run  # noqa: E402
import generate  # noqa: E402
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM  # noqa: E402


def prompt_file(path):
    pm = pretty_midi.PrettyMIDI()
    for program, drum, pitch in ((0, False, 60), (33, False, 36), (0, True, 42)):
        inst = pretty_midi.Instrument(program=program, is_drum=drum)
        inst.notes = [pretty_midi.Note(velocity=90, pitch=pitch, start=k * 0.25, end=k * 0.25 + 0.2) for k in range(8)]
        pm.instruments.append(inst)
    pm.write(str(path))


def test_midi_prompt_keeps_instruments():
    # a multi-instrument model reads a --prompt MIDI with every track and its instrument, drums included (as its
    # token stores); a piano model merges the pitched tracks into piano and drops the drums
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'prompt.mid'
        prompt_file(path)
        cfg = types.SimpleNamespace(n_programs=129, velocity_size=32, duration_size=512, delta_time_size=512,
                                    pitch_size=BOS_PITCH + 2)
        args = types.SimpleNamespace(prompt=str(path), prompt_start=0, prompt_notes=24)
        streams, labels, programs = generate.prompt_rows(cfg, args, None, 0, 2, 'cpu')
        assert streams[0].shape == (2, 25) and streams[0][0, 0] == BOS_PITCH and programs[0, 0] == 0
        assert programs[0, 1:4].tolist() == [0, 33, DRUM_PROGRAM]  # (onset, program, pitch) order
        assert streams[0][0, 1:4].tolist() == [60, 36, 42] and 'Drums' in labels[0]
        cfg.n_programs = 0
        streams, _, programs = generate.prompt_rows(cfg, args, None, 0, 2, 'cpu')
        assert streams[0].shape == (2, 17) and DRUM_PROGRAM not in programs.tolist()[0]


if __name__ == '__main__':
    run(globals())
