"""
Prepare GiantMIDI-Piano for combined training with MAESTRO.

Steps:
  1. Go to https://github.com/bytedance/GiantMIDI-Piano and follow the download
     instructions to get the MIDI zip file (the repo uses a form/request system).
  2. Run this script pointing at the downloaded zip:

     python data/prepare_giantmidi.py --zip /path/to/midis.zip --output data/giantmidi

  3. Combine with MAESTRO:

     python data/prepare_giantmidi.py --zip /path/to/midis.zip --output data/giantmidi \\
         --combine data/maestro-v3.0.0.csv

  4. Update train.py:
       csv_path = 'data/combined.csv'
       root_dir = '.'
"""

import argparse
import csv
import random
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--zip", type=str, default=None,
                        help="Path to the downloaded GiantMIDI zip file")
    parser.add_argument("--output", type=str, default="data/giantmidi",
                        help="Directory to extract MIDI files into")
    parser.add_argument("--combine", type=str, default=None,
                        help="Path to an existing CSV (e.g. data/maestro-v3.0.0.csv) "
                             "to merge with — writes data/combined.csv")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of GiantMIDI files held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def composer_from_filename(path: Path) -> str:
    # GiantMIDI filenames: "Lastname, Firstname, Piece, Performer.mid"
    # The composer is everything before the second comma.
    parts = path.stem.split(",")
    if len(parts) >= 2:
        return f"{parts[0].strip()}, {parts[1].strip()}"
    return parts[0].strip()


def build_csv(midi_dir: Path, val_split: float, seed: int) -> list[dict]:
    midi_files = sorted(midi_dir.glob("**/*.mid")) + sorted(midi_dir.glob("**/*.midi"))

    # Group files by composer so no composer appears in both splits
    by_composer: dict[str, list[Path]] = {}
    for p in midi_files:
        c = composer_from_filename(p)
        by_composer.setdefault(c, []).append(p)

    composers = sorted(by_composer.keys())
    random.seed(seed)
    random.shuffle(composers)

    n_val_composers = max(1, int(len(composers) * val_split))
    val_composers = set(composers[:n_val_composers])

    rows = []
    for composer, files in by_composer.items():
        split = "validation" if composer in val_composers else "train"
        for p in files:
            rows.append({"split": split, "midi_filename": str(p.resolve())})
    return rows


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        # MAESTRO rows carry extra columns (composer, title, duration, ...) that we don't need
        writer = csv.DictWriter(f, fieldnames=["split", "midi_filename"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = output_dir / "midi"

    if args.zip:
        print(f"Extracting {args.zip} → {midi_dir}")
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(midi_dir)
        print("Extracted.")
    else:
        print(f"No --zip provided, scanning {midi_dir} for existing MIDI files...")

    midi_files = sorted(midi_dir.glob("**/*.mid")) + sorted(midi_dir.glob("**/*.midi"))
    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found in {midi_dir}. "
            "Pass --zip /path/to/midis.zip to extract them first."
        )
    print(f"Found {len(midi_files)} MIDI files")

    giantmidi_rows = build_csv(midi_dir, args.val_split, args.seed)
    giantmidi_csv = output_dir / "giantmidi.csv"
    write_csv(giantmidi_csv, giantmidi_rows)
    print(f"Wrote {giantmidi_csv}")

    if args.combine:
        maestro_rows = read_csv(args.combine)
        maestro_root = Path(args.combine).parent.resolve()
        for row in maestro_rows:
            p = Path(row["midi_filename"])
            if not p.is_absolute():
                row["midi_filename"] = str(maestro_root / p)

        combined = maestro_rows + giantmidi_rows
        random.shuffle(combined)
        combined_path = Path("data/combined.csv")
        write_csv(combined_path, combined)
        n_train = sum(1 for r in combined if r["split"] == "train")
        n_val   = sum(1 for r in combined if r["split"] == "validation")
        print(f"\nCombined CSV: {combined_path}  (train={n_train}, val={n_val})")
        print("Update train.py:")
        print("  csv_path = 'data/combined.csv'")
        print("  root_dir = '.'")
    else:
        n_train = sum(1 for r in giantmidi_rows if r["split"] == "train")
        n_val   = sum(1 for r in giantmidi_rows if r["split"] == "validation")
        print(f"Train: {n_train}  Val: {n_val}")
        print("Update train.py:")
        print(f"  csv_path = '{giantmidi_csv}'")
        print("  root_dir = '.'")


if __name__ == "__main__":
    main()
