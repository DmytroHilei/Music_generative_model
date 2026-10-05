"""
Solo piano covers (mp3 from data/fetch_songs.py --mode covers) -> MIDI with the Kong et al. piano transcription model
(piano_transcription_inference, high-resolution onsets/offsets + pedal; checkpoint in ~/piano_transcription_inference_data).

Unlike data/audio_to_piano.py there is no source separation or reduction: the audio already is a human piano
arrangement, so the MIDI is used as is. Note lengths are key releases (pedal kept as CC64 only), the same convention
as the MAESTRO tokens. Output mirrors the reductions: <output>/<artist>/midi/<audio stem>.mid, plus a stats line per
file in <output>/transcribed.csv (notes, seconds, notes/s, pitch range) for filtering.

    .venv-audio/bin/python data/transcribe_covers.py --input data/covers --output data/covers
"""

import argparse
import csv
import time
from pathlib import Path

import librosa
import pretty_midi
import torch
from piano_transcription_inference import PianoTranscription, sample_rate


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', default='data/covers', help='<input>/<artist>/*.mp3')
    ap.add_argument('--output', default='data/covers', help='-> <output>/<artist>/midi/*.mid')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--delete-audio', action='store_true', help='delete each mp3 once its MIDI is written')
    ap.add_argument('--min-age', type=float, default=0, help='skip mp3s modified in the last N seconds (still being '
                    'written by a running download)')
    ap.add_argument('--stop-file', default=None,
                    help='exit before the next mp3 once this file exists (a worker stops between songs)')
    return ap.parse_args()


def main():
    args = parse_args()
    files = sorted(p for p in Path(args.input).glob('*/*.mp3') if time.time() - p.stat().st_mtime >= args.min_age)
    print(f'{len(files)} mp3 files')
    model = PianoTranscription(device=torch.device(args.device), checkpoint_path=None)
    stats_path = Path(args.output) / 'transcribed.csv'
    new_stats = not stats_path.exists()
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_stats:
            w.writerow(['artist', 'file', 'notes', 'seconds', 'notes_per_sec', 'pitch_min', 'pitch_max'])
        for i, mp3 in enumerate(files, 1):
            if args.stop_file and Path(args.stop_file).exists():
                print(f'stop file {args.stop_file} found, exiting before {mp3.name}')
                break
            artist = mp3.parent.name
            out = Path(args.output) / artist / 'midi' / (mp3.stem + '.mid')
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                audio, _ = librosa.load(str(mp3), sr=sample_rate, mono=True)
                model.transcribe(audio, str(out))
                notes = [n for inst in pretty_midi.PrettyMIDI(str(out)).instruments for n in inst.notes]
            except Exception as e:
                print(f'[{i}/{len(files)}] FAILED {mp3.name}: {e!r}')
                out.unlink(missing_ok=True)
                continue
            secs = len(audio) / sample_rate
            pitches = [n.pitch for n in notes] or [0]
            w.writerow([artist, out.name, len(notes), round(secs, 1), round(len(notes) / max(secs, 1e-6), 2),
                        min(pitches), max(pitches)])
            f.flush()
            print(f'[{i}/{len(files)}] {artist} — {mp3.stem}: {len(notes)} notes, {secs:.0f} s, '
                  f'{len(notes) / max(secs, 1e-6):.1f} notes/s ({time.time() - t0:.0f} s)')
            if args.delete_audio:
                mp3.unlink()
    print('TRANSCRIPTION DONE')


if __name__ == '__main__':
    main()
