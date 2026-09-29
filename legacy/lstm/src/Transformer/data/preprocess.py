import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


DATASET_URLS = {
    "maestro-v3.0.0": {
        "midi": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "full": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
    },
    "maestro-v2.0.0": {
        "midi": "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip",
        "full": "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip",
    },
    "maestro-v1.0.0": {
        "midi": "https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip",
        "full": "https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip",
    },
}


class _ProgressHook(tqdm):
    def update_to(self, blocks=1, block_size=1, total=None):
        if total is not None:
            self.total = total
        self.update(blocks * block_size - self.n)


def _download(url: str, dest: Path) -> None:
    with _ProgressHook(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:
        urlretrieve(url, filename=dest, reporthook=bar.update_to)


def _extract(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the MAESTRO piano MIDI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="maestro-v3.0.0",
        choices=list(DATASET_URLS.keys()),
        help="MAESTRO version to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root directory where the dataset folder will be created",
    )
    parser.add_argument(
        "--midi-only",
        action="store_true",
        help="Download only MIDI files (~57 MB) instead of full audio+MIDI (~120 GB)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the zip archive after extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_dir / args.dataset
    if dataset_dir.exists():
        print(f"Already exists: {dataset_dir} — skipping download.")
        _print_paths(dataset_dir)
        return

    variant = "midi" if args.midi_only else "full"
    url = DATASET_URLS[args.dataset][variant]
    zip_path = output_dir / url.split("/")[-1]

    print(f"Downloading {args.dataset} ({variant})\n  {url}")
    if not zip_path.exists():
        _download(url, zip_path)
    else:
        print(f"Zip already present at {zip_path}, skipping download.")

    print(f"Extracting → {output_dir}")
    _extract(zip_path, output_dir)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {zip_path}")

    _print_paths(dataset_dir)


def _print_paths(dataset_dir: Path) -> None:
    csv_files = list(dataset_dir.glob("*.csv"))
    if csv_files:
        print(f"\nDataset ready.")
        print(f"  csv_path : {csv_files[0]}")
        print(f"  root_dir : {dataset_dir}")
    else:
        print(f"\nDataset extracted to {dataset_dir}")


if __name__ == "__main__":
    main()
