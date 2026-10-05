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
    5. --multi: from the same stems, a multi-track MIDI (multi/<song>.mid, stats in multi.csv) with one GM instrument
       per stem: vocals -> melody line (--vocal-program), bass 33, guitar, piano 0, other (--other-program), drums
       via ADTOF (kick/snare/tom/hi-hat/cymbal). Stems quieter than --stem-gate-db vs the mix are left out (Demucs
       leakage, e.g. a "piano" stem in a guitar song).

Run with the audio env (not the training env):
    .venv-audio/bin/python data/audio_to_piano.py --input Skryabin --output data/finetune/skryabin --limit 3
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
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
    parser.add_argument('--min-age', type=float, default=0, help='skip files modified in the last N seconds (still '
                        'being written by a running download)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-file', default=None,
                        help='exit before the next song once this file exists (a worker stops between songs)')
    parser.add_argument('--delete-audio', action='store_true',
                        help='delete the source audio after its MIDI is written (re-downloadable via songs.csv; saves disk)')
    parser.add_argument('--keep-stems', action='store_true',
                        help='keep the Demucs wavs (~350 MB per song!); by default deleted after transcription')
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
    parser.add_argument('--multi', action='store_true', help='also write the multi-track MIDI (multi/, multi.csv)')
    parser.add_argument('--no-piano', action='store_true', help='skip the piano reduction (with --multi)')
    parser.add_argument('--vocal-program', type=int, default=53, help='GM program of the vocal melody (53 voice oohs)')
    parser.add_argument('--guitar-program', type=int, default=27, help='GM program of the guitar stem (27 clean electric)')
    parser.add_argument('--other-program', type=int, default=48, help='GM program of the "other" stem (48 strings)')
    parser.add_argument('--stem-gate-db', type=float, default=-30,
                        help='leave a stem out of the multi-track MIDI when its level vs the full mix is below this')
    parser.add_argument('--multi-max-poly', type=int, default=6, help='max notes sounding together per polyphonic stem')
    parser.add_argument('--multi-drop-quiet', type=float, default=0.2,
                        help='drop this fraction of the quietest notes of each polyphonic stem')
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


_adtof = None


def transcribe_drums(wav_path, device):
    """ADTOF Frame_RNN (kick 35, snare 38, tom 47, hi-hat 42, cymbal 49) -> list of pretty_midi.Note; the velocity
    comes from the peak activation (ADTOF itself writes every hit at 100)."""
    global _adtof
    import torch
    from adtof_pytorch import (LABELS_5, PeakPicker, calculate_n_bins, create_frame_rnn_model, get_default_weights_path,
                               load_audio_for_model, load_pytorch_weights)
    if _adtof is None:
        dev = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
        _adtof = load_pytorch_weights(create_frame_rnn_model(calculate_n_bins()), get_default_weights_path()).eval().to(dev)
    with torch.no_grad():
        act = _adtof(load_audio_for_model(str(wav_path)).to(next(_adtof.parameters()).device)).cpu().numpy()[0]
    peaks = PeakPicker(fps=100).pick(act)[0]  # pitch -> onset times; frames are 1/100 s
    return [pretty_midi.Note(int(np.clip(40 + 80 * act[min(round(t * 100), len(act) - 1), c], 1, 127)), pitch, t, t + 0.1)
            for c, pitch in enumerate(LABELS_5) for t in peaks[pitch]]


def stem_levels(paths):
    """Level of each stem in dB relative to the full mix (the sum of the stems)."""
    audio = {s: sf.read(str(p), dtype='float32')[0] for s, p in paths.items()}
    mix_power = np.mean(sum(audio.values()) ** 2) + 1e-12
    return {s: float(10 * np.log10(np.mean(a ** 2) / mix_power + 1e-12)) for s, a in audio.items()}


def copy_notes(notes):
    return [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in notes]


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


def reduce_song(audio_path, out_root, args, piano=True, multi=False):
    """Demucs once, then the piano reduction and/or the multi-track MIDI: returns (piano row, multi row), None
    for a part not asked for."""
    stems_root = out_root / 'stems'
    stems = separate(audio_path, stems_root, args.device)
    raw_vocals = transcribe(stems['vocals'], args.min_note_ms, fmin=80, fmax=1100, onset_threshold=0.6)
    raw_bass = transcribe(stems['bass'], args.min_note_ms, fmin=30, fmax=400)
    row = reduce_to_piano(audio_path, out_root, args, stems, copy_notes(raw_vocals), copy_notes(raw_bass)) \
        if piano else None
    multi_row = multi_track(audio_path, out_root, args, stems, raw_vocals, raw_bass) if multi else None
    if not args.keep_stems:
        shutil.rmtree(stems['vocals'].parent, ignore_errors=True)
    return row, multi_row


def reduce_to_piano(audio_path, out_root, args, stems, melody, bass):
    harmony_wav = mix_stems([stems[s] for s in HARMONY_STEMS], stems[STEMS[0]].parent / 'harmony_mix.wav')

    melody = monophonic(fold_into_range(melody, *args.melody_range), keep='highest')
    melody = legato(list(merge_retriggers(melody, args.merge_gap_ms / 1000)), args.melody_legato_ms / 1000)

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


def multi_track(audio_path, out_root, args, stems, vocals, bass):
    """One GM instrument per Demucs stem, at the transcribed pitches (no folding into a piano range)."""
    levels = stem_levels(stems)
    gap = args.merge_gap_ms / 1000
    parts = {}  # stem -> (program, is_drum, notes)
    vocals = monophonic(vocals, keep='highest')
    parts['vocals'] = (args.vocal_program, False, legato(list(merge_retriggers(vocals, gap)), args.melody_legato_ms / 1000))
    bass = monophonic(bass, keep='lowest')
    parts['bass'] = (33, False, legato(list(merge_retriggers(bass, gap)), args.bass_legato_ms / 1000))
    for stem, program in (('guitar', args.guitar_program), ('piano', 0), ('other', args.other_program)):
        if levels[stem] < args.stem_gate_db:
            continue
        notes = transcribe(stems[stem], args.min_note_ms)
        notes = drop_quiet(list(merge_retriggers(notes, gap)), args.multi_drop_quiet)
        parts[stem] = (program, False, cap_polyphony(notes, args.multi_max_poly))
    if levels['drums'] >= args.stem_gate_db:
        parts['drums'] = (0, True, transcribe_drums(stems['drums'], args.device))
    parts = {s: p for s, p in parts.items() if levels[s] >= args.stem_gate_db and p[2]}

    midi = pretty_midi.PrettyMIDI()
    for stem, (program, is_drum, notes) in parts.items():
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=stem)
        inst.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        midi.instruments.append(inst)
    (out_root / 'multi').mkdir(parents=True, exist_ok=True)
    midi_path = out_root / 'multi' / f'{audio_path.stem}.mid'
    midi.write(str(midi_path))
    return {'midi_filename': str(midi_path.relative_to(out_root)), 'source_audio': str(audio_path),
            **{f'n_{s}': len(parts[s][2]) if s in parts else 0 for s in STEMS},
            **{f'db_{s}': round(levels[s], 1) for s in STEMS}}


def main():
    args = parse_args()
    in_dir, out_root = Path(args.input), Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in ('.mp3', '.wav', '.flac', '.m4a')
                   and time.time() - p.stat().st_mtime >= args.min_age)
    if args.limit:
        files = files[:args.limit]

    fields = ['midi_filename', 'source_audio', 'n_melody', 'n_bass', 'n_harmony', 'key', 'mode', 'key_confidence']
    multi_fields = ['midi_filename', 'source_audio'] + [f'n_{s}' for s in STEMS] + [f'db_{s}' for s in STEMS]
    failed_path = out_root / 'failed.txt'  # songs that failed once are not retried (a worker loop would spin on them)
    failed = set(failed_path.read_text(encoding='utf-8').splitlines()) if failed_path.exists() else set()
    with ExitStack() as stack:
        piano_csv = open_csv(stack, out_root / 'songs.csv', fields) if not args.no_piano else None
        multi_csv = open_csv(stack, out_root / 'multi.csv', multi_fields) if args.multi else None
        for i, audio in enumerate(files):
            if args.stop_file and Path(args.stop_file).exists():
                print(f'stop file {args.stop_file} found, exiting before {audio.name}')
                break
            # a song is redone only for the outputs it is missing (e.g. multi-track for an already reduced song)
            piano, multi = (c is not None and str(audio) not in c[2] for c in (piano_csv, multi_csv))
            if (not piano and not multi) or str(audio) in failed:
                print(f'[{i + 1}/{len(files)}] skip {audio.name}')
                continue
            print(f'[{i + 1}/{len(files)}] {audio.name}')
            try:
                row, multi_row = reduce_song(audio, out_root, args, piano=piano, multi=multi)
            except Exception as e:  # one broken file shouldn't stop a batch of hundreds
                print(f'  FAILED: {e!r}')
                with open(failed_path, 'a', encoding='utf-8') as f:
                    f.write(f'{audio}\n')
                continue
            for c, r in ((piano_csv, row), (multi_csv, multi_row)):
                if r is not None:
                    c[1].writerow(r)
                    c[0].flush()
            if args.delete_audio:  # only songs processed in this run: skipped (already done) files are never touched
                audio.unlink()
            if row:
                print(f"  melody {row['n_melody']}, bass {row['n_bass']}, harmony {row['n_harmony']}, key {row['key']}")
            if multi_row:
                print('  multi: ' + ', '.join(f"{s} {multi_row[f'n_{s}']} ({multi_row[f'db_{s}']:+.0f} dB)" for s in STEMS))


def open_csv(stack, path, fields):
    """Append-mode CSV: (file, writer, source_audio values already in it)."""
    done = set()
    if path.exists():
        with open(path, newline='', encoding='utf-8') as f:
            done = {r['source_audio'] for r in csv.DictReader(f)}
    f = stack.enter_context(open(path, 'a', newline='', encoding='utf-8'))
    writer = csv.DictWriter(f, fieldnames=fields)
    if not done and f.tell() == 0:
        writer.writeheader()
    return f, writer, done


if __name__ == '__main__':
    main()
