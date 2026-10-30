"""
Shared parts of the head-to-head adapters for other labs' models (eval/compare/head2head.py: each system writes
results/head2head/<task>/<system>/<prompt id>_<k>.mid, then `head2head.py score` reads every folder the same way).
The weights and the authors' code are in checkpoints/baselines/<model>/ (README there).

A continuation file is the prompt MIDI as exported (its notes untouched) plus the model's new notes only: the
models re-encode the prompt in their own token grids, so their full outputs would move the prompt notes.
"""

import io
import json
import time
from collections import Counter
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'head2head'
BASE = ROOT / 'checkpoints' / 'baselines'


def prompts(task, n=0):
    return json.loads((OUT / task / 'prompts.json').read_text())['prompts'][:n or None]


def out_dir(task, system, config):
    d = OUT / task / system
    d.mkdir(parents=True, exist_ok=True)
    (d / 'config.json').write_text(json.dumps(dict(system=system, task=task, **config), indent=1))
    return d


def to_pm(midi):
    """pretty_midi.PrettyMIDI of a path, MIDI bytes, a mido.MidiFile or a PrettyMIDI (returned as is)."""
    if isinstance(midi, pretty_midi.PrettyMIDI):
        return midi
    if isinstance(midi, (str, Path)):
        return pretty_midi.PrettyMIDI(str(midi))
    if isinstance(midi, (bytes, bytearray)):
        return pretty_midi.PrettyMIDI(io.BytesIO(midi))
    buf = io.BytesIO()
    midi.save(file=buf)
    buf.seek(0)
    return pretty_midi.PrettyMIDI(buf)


def note_rows(pm):
    """(program, is_drum, pitch, start, end, velocity) of every note; drums under program 0."""
    return [(0 if i.is_drum else i.program, i.is_drum, n.pitch, n.start, n.end, n.velocity)
            for i in pm.instruments for n in i.notes]


def last_onset(path):
    return max(n.start for i in to_pm(path).instruments for n in i.notes)


def write_continuation(prompt_path, generated, path, minus=None):
    """The prompt file + the notes of `generated` (a decoded MIDI) that are not in `minus` (the model's own decoding
    of the prompt, for models that return prompt + continuation as one sequence). Notes match without their end: a
    new note can shorten a prompt note of its pitch in the full decoding (Aria). Returns the new note count."""
    new = note_rows(to_pm(generated))
    if minus is not None:
        old = Counter((p, d, k, round(s, 4)) for p, d, k, s, e, _ in note_rows(to_pm(minus)))
        kept = []
        for row in new:
            key = (*row[:3], round(row[3], 4))
            if old[key] > 0:
                old[key] -= 1
            else:
                kept.append(row)
        new = kept
    pm = to_pm(prompt_path)
    tracks = {(0 if i.is_drum else i.program, i.is_drum): i for i in pm.instruments}
    for prog, drum, pitch, start, end, vel in new:
        if (prog, drum) not in tracks:
            tracks[prog, drum] = pretty_midi.Instrument(program=prog, is_drum=drum)
            pm.instruments.append(tracks[prog, drum])
        tracks[prog, drum].notes.append(pretty_midi.Note(velocity=int(vel), pitch=int(pitch), start=start,
                                                         end=max(end, start + 0.01)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return len(new)


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0, self.done = total, label, time.time(), 0

    def step(self, what):
        self.done += 1
        el = time.time() - self.t0
        print(f'[{time.strftime("%H:%M:%S")}] {self.label} {self.done}/{self.total} {what} '
              f'({el / self.done:.0f} s each)', flush=True)
