"""
docs/listen MIDI -> web/public/listen/<name>.json in the engine's note wire format, plus index.json for the page.

The page's mock engine replays these files as event streams, and the home page draws its sample cards from them.
Notes: [onset_s, program, pitch, duration_s, velocity], program 128 = drums, sorted by onset (as the engine sends).

Run from the repository root: .venv/bin/python web/scripts/listen_to_json.py
"""

import json
from pathlib import Path

import pretty_midi

SRC, DST = Path('docs/listen'), Path('web/public/listen')
SAMPLES = [  # (prompt, files, instruments in the result) as in docs/listen/README.md
    ('A slow, quiet solo piano piece in a minor key', ['slow_piano_1', 'slow_piano_2'], 'piano'),
    ('A fast, energetic rock song with drums, electric guitar and bass', ['fast_rock_1', 'fast_rock_2'],
     'guitars · bass · drums'),
    ('A calm acoustic guitar piece without drums, in a high register', ['guitar_nodrums_1', 'guitar_nodrums_2'],
     'guitar'),
    ('A lively jazz piece with piano, double bass and drums', ['jazz_band_1', 'jazz_band_2'],
     'piano · double bass · drums'),
    ('Швидка весела пісня з барабанами та електрогітарою', ['uk_fast_drums_1'], 'drums · bass · guitar'),
    ('Повільна сумна мелодія для фортепіано', ['uk_slow_piano_1', 'uk_slow_piano_2'], 'piano'),
]


def notes_of(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [[round(float(n.start), 3), 128 if ins.is_drum else int(ins.program), int(n.pitch),
              round(float(n.end - n.start), 3), int(n.velocity)] for ins in pm.instruments for n in ins.notes]
    return sorted(notes)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    index = []
    for prompt, files, instruments in SAMPLES:
        for name in files:
            notes = notes_of(SRC / f'{name}.mid')
            (DST / f'{name}.json').write_text(json.dumps(notes, separators=(',', ':')))
            index.append(dict(name=name, prompt=prompt, instruments=instruments, n_notes=len(notes),
                              duration_s=round(max(n[0] + n[3] for n in notes), 1)))
    (DST / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f'{len(index)} files -> {DST}')


if __name__ == '__main__':
    main()
