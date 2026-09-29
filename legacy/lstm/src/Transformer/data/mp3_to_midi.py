"""
Convert a folder of MP3s to MIDI using piano_transcription_inference,
then write a CSV compatible with MaestroDataset for fine-tuning.

Usage:
    python data/mp3_to_midi.py --input Skryabin --output Skryabin/midi

Install deps first:
    pip install piano_transcription_inference demucs
"""

import argparse
import csv
import random
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="Folder containing MP3 files")
    parser.add_argument("--output", type=str, required=True,
                        help="Folder to write MIDI files into")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of songs held out for validation")
    parser.add_argument("--use-demucs", action="store_true",
                        help="Separate audio with Demucs before transcribing "
                             "(slower but cleaner for multi-instrument recordings)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def separate_with_demucs(mp3_path: Path, tmp_dir: Path) -> Path:
    import subprocess
    subprocess.run(
        ["python", "-m", "demucs", "--two-stems=other",
         "-o", str(tmp_dir), str(mp3_path)],
        check=True,
    )
    # demucs writes to tmp_dir/htdemucs/<stem name>/other.wav
    stem_files = list(tmp_dir.rglob("other.wav"))
    if not stem_files:
        raise FileNotFoundError(f"Demucs output not found for {mp3_path}")
    return stem_files[0]


def transcribe(audio_path: Path, midi_path: Path, device: str) -> None:
    import librosa
    from piano_transcription_inference import PianoTranscription

    # bypass piano_transcription_inference.load_audio — incompatible with librosa >= 0.10
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    transcriptor = PianoTranscription(device=device, checkpoint_path=None)
    transcriptor.transcribe(audio, str(midi_path))


def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mp3_files = sorted(input_dir.glob("*.mp3")) + sorted(input_dir.glob("*.MP3"))
    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {input_dir}")
    print(f"Found {len(mp3_files)} MP3 files")

    tmp_dir = output_dir / "_demucs_tmp"

    midi_paths = []
    for mp3 in mp3_files:
        midi_path = output_dir / (mp3.stem + ".mid")
        if midi_path.exists():
            print(f"  [skip] {midi_path.name} already exists")
            midi_paths.append(midi_path)
            continue

        print(f"  → {mp3.name}")
        try:
            if args.use_demucs:
                audio_path = separate_with_demucs(mp3, tmp_dir)
            else:
                audio_path = mp3

            transcribe(audio_path, midi_path, device)
            midi_paths.append(midi_path)
        except Exception as e:
            import traceback
            print(f"  [FAILED] {mp3.name}")
            traceback.print_exc()

    # clean up demucs tmp files
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)

    if not midi_paths:
        raise RuntimeError("No MIDI files were produced.")

    # split into train / validation
    random.shuffle(midi_paths)
    n_val = max(1, int(len(midi_paths) * args.val_split))
    val_set   = set(midi_paths[:n_val])
    train_set = set(midi_paths[n_val:])

    # CSV root is the parent of the output midi folder
    # midi_filename is relative to that root
    csv_root = output_dir.parent
    csv_path = csv_root / "skryabin.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "midi_filename"])
        writer.writeheader()
        for p in sorted(train_set):
            writer.writerow({"split": "train",
                             "midi_filename": str(p.relative_to(csv_root))})
        for p in sorted(val_set):
            writer.writerow({"split": "validation",
                             "midi_filename": str(p.relative_to(csv_root))})

    print(f"\nDone.")
    print(f"  MIDI files : {output_dir}")
    print(f"  CSV        : {csv_path}")
    print(f"  Train      : {len(train_set)} files")
    print(f"  Val        : {len(val_set)} files")
    print(f"\nFine-tune with:")
    print(f"  csv_path = '{csv_path}'")
    print(f"  root_dir = '{csv_root}'")


if __name__ == "__main__":
    main()
