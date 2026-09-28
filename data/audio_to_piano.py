"""
Band recording (mp3) -> piano reduction MIDI.

Transcribing a full band mix with a piano transcriber gives garbage (vocals, guitars, drums all become
ghost notes). Instead:
    1. Demucs htdemucs_6s separates vocals / drums / bass / guitar / piano / other.
    2. basic-pitch transcribes each part separately:
         vocals                  -> melody  (monophonic: highest note at any time)
         bass                    -> bass    (monophonic: lowest note, kept in piano bass range)
         piano + guitar + other  -> harmony (short / quiet notes dropped, range limited, polyphony capped)
    3. Everything is merged into one piano track (the file the model trains on) plus a debug MIDI with
       one track per part, so each part can be listened to separately.
    4. Key and mode are estimated (Krumhansl-Schmuckler) and written to the CSV, to filter minor-key songs.

Run with the audio env (not the training env):
    .venv-audio/bin/python data/audio_to_piano.py --input Skryabin --output data/finetune/skryabin --limit 3
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

STEMS = ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other')
HARMONY_STEMS = ('piano', 'guitar', 'other')

# Krumhansl-Kessler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, help='folder with mp3/wav files')
    parser.add_argument('--output', required=True, help='output folder (midi/, debug/, stems/, songs.csv)')
    parser.add_argument('--limit', type=int, default=None, help='only process the first N files')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--harmony-max-poly', type=int, default=3, help='max harmony notes sounding together')
    parser.add_argument('--harmony-min-note-ms', type=float, default=100)
    parser.add_argument('--harmony-drop-quiet', type=float, default=0.3,
                        help='drop this fraction of the quietest harmony notes')
    parser.add_argument('--melody-legato-ms', type=float, default=350,
                        help='extend a melody note to the next one if the gap is shorter than this (sung phrases)')
    parser.add_argument('--bass-legato-ms', type=float, default=500)
    parser.add_argument('--merge-gap-ms', type=float, default=80,
                        help='same pitch re-triggered within this gap becomes one note (pads, strums)')
    parser.add_argument('--harmony-range', type=int, nargs=2, default=(48, 84), help='harmony pitch range (C3..C6)')
    parser.add_argument('--bass-range', type=int, nargs=2, default=(28, 55), help='bass pitch range (E1..G3)')
    parser.add_argument('--melody-range', type=int, nargs=2, default=(55, 96), help='melody pitch range')
    parser.add_argument('--min-note-ms', type=float, default=80)
    return parser.parse_args()


def separate(audio_path, stems_root, device):
    """Demucs 6-stem separation, cached: returns dict stem -> wav path."""
    out_dir = stems_root / 'htdemucs_6s' / audio_path.stem
    paths = {s: out_dir / f'{s}.wav' for s in STEMS}
    if not all(p.exists() for p in paths.values()):
        subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs_6s', '-d', device,
                        '-o', str(stems_root), str(audio_path)], check=True)
    return paths


def mix_stems(paths, out_path):
    audio, sr = None, None
    for p in paths:
        a, sr = sf.read(str(p), dtype='float32')
        audio = a if audio is None else audio + a
    sf.write(str(out_path), audio, sr)
    return out_path


def transcribe(wav_path, min_note_ms, fmin=None, fmax=None, onset_threshold=0.5, frame_threshold=0.3):
    """basic-pitch -> list of pretty_midi.Note."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi, _ = predict(str(wav_path), ICASSP_2022_MODEL_PATH,
                         onset_threshold=onset_threshold, frame_threshold=frame_threshold,
                         minimum_note_length=min_note_ms, minimum_frequency=fmin, maximum_frequency=fmax,
                         multiple_pitch_bends=False, melodia_trick=True)
    return [n for inst in midi.instruments for n in inst.notes]


def fold_into_range(notes, lo, hi):
    for n in notes:
        while n.pitch < lo:
            n.pitch += 12
        while n.pitch > hi:
            n.pitch -= 12
    return notes


def monophonic(notes, keep='highest'):
    """Skyline: when notes overlap keep the highest (melody) or lowest (bass), cut the other one."""
    notes = sorted(notes, key=lambda n: (n.start, -n.pitch if keep == 'highest' else n.pitch))
    out = []
    for n in notes:
        if out and n.start < out[-1].end:
            prev = out[-1]
            better = n.pitch > prev.pitch if keep == 'highest' else n.pitch < prev.pitch
            if not better:
                continue
            prev.end = n.start  # new note wins, shorten the previous one
            if prev.end - prev.start < 0.03:
                out.pop()
        out.append(n)
    return out


def cap_polyphony(notes, max_poly):
    """Drop the quietest notes whenever more than max_poly notes sound at once."""
    notes = sorted(notes, key=lambda n: n.start)
    kept = []
    for n in notes:
        sounding = [k for k in kept if k.end > n.start]
        if len(sounding) < max_poly:
            kept.append(n)
            continue
        weakest = min(sounding, key=lambda k: k.velocity)
        if n.velocity > weakest.velocity:
            weakest.end = n.start
            kept.append(n)
    return [n for n in kept if n.end - n.start > 0.03]


def merge_retriggers(notes, gap):
    """Join consecutive notes of the same pitch separated by < gap seconds (sustained pads split into repeats)."""
    by_pitch = {}
    for n in sorted(notes, key=lambda n: n.start):
        prev = by_pitch.get(n.pitch)
        if prev is not None and n.start - prev.end < gap:
            prev.end = max(prev.end, n.end)
            prev.velocity = max(prev.velocity, n.velocity)
            continue
        by_pitch[n.pitch] = n
        yield n


def legato(notes, max_gap):
    """Monophonic line: extend each note to the next onset when the gap is < max_gap seconds,
    so syllable fragments / plucked bass become held notes like a pianist would play them."""
    notes = sorted(notes, key=lambda n: n.start)
    for cur, nxt in zip(notes, notes[1:]):
        if nxt.start - cur.end < max_gap:
            cur.end = max(cur.end, nxt.start)
    return notes


def drop_quiet(notes, fraction):
    if not notes or fraction <= 0:
        return notes
    threshold = np.quantile([n.velocity for n in notes], fraction)
    return [n for n in notes if n.velocity > threshold]


def drop_duplicates(notes, others, tol=0.05):
    """Remove harmony notes that double a melody/bass note (same pitch, same onset)."""
    index = {(o.pitch, round(o.start / tol)) for o in others}
    return [n for n in notes if (n.pitch, round(n.start / tol)) not in index]


def estimate_key(notes):
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.end - n.start
    if hist.sum() == 0:
        return 'unknown', 'unknown', 0.0
    best = max(
        ((np.corrcoef(hist, np.roll(profile, tonic))[0, 1], tonic, mode)
         for mode, profile in (('major', MAJOR), ('minor', MINOR)) for tonic in range(12)),
        key=lambda x: x[0],
    )
    return NOTE_NAMES[best[1]], best[2], float(best[0])


def reduce_song(audio_path, out_root, args):
    stems_root = out_root / 'stems'
    stems = separate(audio_path, stems_root, args.device)
    harmony_wav = mix_stems([stems[s] for s in HARMONY_STEMS], stems[STEMS[0]].parent / 'harmony_mix.wav')

    melody = transcribe(stems['vocals'], args.min_note_ms, fmin=80, fmax=1100, onset_threshold=0.6)
    melody = monophonic(fold_into_range(melody, *args.melody_range), keep='highest')
    melody = legato(list(merge_retriggers(melody, args.merge_gap_ms / 1000)), args.melody_legato_ms / 1000)

    bass = transcribe(stems['bass'], args.min_note_ms, fmin=30, fmax=400)
    bass = monophonic(fold_into_range(bass, *args.bass_range), keep='lowest')
    bass = legato(list(merge_retriggers(bass, args.merge_gap_ms / 1000)), args.bass_legato_ms / 1000)

    harmony = transcribe(harmony_wav, args.harmony_min_note_ms, onset_threshold=0.6, frame_threshold=0.4)
    harmony = fold_into_range(harmony, *args.harmony_range)
    harmony = list(merge_retriggers(harmony, args.merge_gap_ms / 1000))
    harmony = drop_quiet(harmony, args.harmony_drop_quiet)
    harmony = drop_duplicates(harmony, melody + bass)
    harmony = cap_polyphony(harmony, args.harmony_max_poly)

    # melody a bit louder than the accompaniment, like a pianist would play it
    for n in melody:
        n.velocity = int(np.clip(n.velocity * 1.15 + 10, 1, 127))

    parts = {'melody': melody, 'bass': bass, 'harmony': harmony}
    piano = pretty_midi.PrettyMIDI()
    piano.instruments.append(pretty_midi.Instrument(program=0, name='piano'))
    piano.instruments[0].notes = sorted(melody + bass + harmony, key=lambda n: (n.start, n.pitch))

    debug = pretty_midi.PrettyMIDI()
    for name, notes in parts.items():
        inst = pretty_midi.Instrument(program=0, name=name)
        inst.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in notes]
        debug.instruments.append(inst)

    (out_root / 'midi').mkdir(parents=True, exist_ok=True)
    (out_root / 'debug').mkdir(parents=True, exist_ok=True)
    midi_path = out_root / 'midi' / f'{audio_path.stem}.mid'
    piano.write(str(midi_path))
    debug.write(str(out_root / 'debug' / f'{audio_path.stem}.parts.mid'))

    tonic, mode, conf = estimate_key(piano.instruments[0].notes)
    return {
        'midi_filename': str(midi_path.relative_to(out_root)),
        'source_audio': str(audio_path),
        'n_melody': len(melody), 'n_bass': len(bass), 'n_harmony': len(harmony),
        'key': f'{tonic} {mode}', 'mode': mode, 'key_confidence': round(conf, 3),
    }


def main():
    args = parse_args()
    in_dir, out_root = Path(args.input), Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in ('.mp3', '.wav', '.flac', '.m4a'))
    if args.limit:
        files = files[:args.limit]

    csv_path = out_root / 'songs.csv'
    done = set()
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            done = {row['source_audio'] for row in csv.DictReader(f)}

    fields = ['midi_filename', 'source_audio', 'n_melody', 'n_bass', 'n_harmony', 'key', 'mode', 'key_confidence']
    new_file = not csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for i, audio in enumerate(files):
            if str(audio) in done:
                print(f'[{i + 1}/{len(files)}] skip {audio.name}')
                continue
            print(f'[{i + 1}/{len(files)}] {audio.name}')
            try:
                row = reduce_song(audio, out_root, args)
            except Exception as e:  # one broken file shouldn't stop a batch of hundreds
                print(f'  FAILED: {e!r}')
                continue
            writer.writerow(row)
            f.flush()
            print(f"  melody {row['n_melody']}, bass {row['n_bass']}, harmony {row['n_harmony']}, key {row['key']}")


if __name__ == '__main__':
    main()
