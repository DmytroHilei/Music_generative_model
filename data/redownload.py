"""
Re-download songs listed in a fetch_songs.py songs.csv whose mp3 is gone (deleted after the piano reduction), by URL,
no search. Same file names as fetch_songs.py (<output>/<artist>/<title> [<id>].mp3), so a song re-downloaded into its
old folder matches its reduction (audio_to_piano.py then only adds what is missing, e.g. --multi).

Resumable: every finished id is appended to <output>/redownloaded.txt and never fetched again (the mp3 itself may be
deleted by the worker afterwards). songs.csv is never written. Waits while the disk is below --min-free-gb and stops
after --max-fails failed downloads in a row (a block, not a broken link).

    .venv-audio/bin/python data/redownload.py --songs-csv /data/songs_round2/songs.csv --output /data/songs_round2 \
        --source soundcloud
"""

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

from fetch_songs import download


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--songs-csv', required=True)
    parser.add_argument('--output', required=True, help='mp3s go to <output>/<artist>/')
    parser.add_argument('--audio-root', default=None,
                        help='where the original mp3s were (default: --output); songs still there are skipped')
    parser.add_argument('--source', choices=['youtube', 'soundcloud', 'all'], default='all')
    parser.add_argument('--exclude', default=None, help='file of ids to skip (first word per line, # comments)')
    parser.add_argument('--quality', default='192')
    parser.add_argument('--sleep', type=float, default=3.0, help='seconds between downloads')
    parser.add_argument('--cookies-from-browser', default=None, metavar='BROWSER[:PROFILE]')
    parser.add_argument('--min-free-gb', type=float, default=6.0, help='wait while the output disk has less free')
    parser.add_argument('--max-fails', type=int, default=5, help='stop after this many failed downloads in a row')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    out_root = Path(args.output)
    audio_root = Path(args.audio_root or args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    done_path = out_root / 'redownloaded.txt'
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    exclude = set()
    if args.exclude:
        exclude = {line.split()[0] for line in open(args.exclude, encoding='utf-8')
                   if line.strip() and not line.startswith('#')}
    on_disk = {p.name.rsplit('[', 1)[-1][:-5] for p in audio_root.glob('*/*.mp3') if p.name.endswith('].mp3')}

    rows, seen = [], set()
    for r in csv.DictReader(open(args.songs_csv, newline='', encoding='utf-8')):
        source = 'soundcloud' if 'soundcloud' in r['url'] else 'youtube'
        if r['id'] in seen or args.source not in ('all', source):
            continue
        seen.add(r['id'])
        if r['id'] not in done and r['id'] not in on_disk and r['id'] not in exclude:
            rows.append(r)
    print(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')}: re-download of {args.songs_csv} ({args.source}) =====")
    print(f'{len(rows)} to download ({len(done)} done before, {len(on_disk)} still on disk, {len(exclude)} excluded)')

    extra = {}
    if args.cookies_from_browser:
        browser, _, profile = args.cookies_from_browser.partition(':')
        extra['cookiesfrombrowser'] = (browser, str(Path(profile).expanduser()) if profile else None, None, None)
    fails = 0
    with open(done_path, 'a') as done_file:
        for i, r in enumerate(rows):
            while shutil.disk_usage(out_root).free / 1e9 < args.min_free_gb:
                print(f'  disk below {args.min_free_gb} GB free, waiting 5 min')
                time.sleep(300)
            print(f"[{i + 1}/{len(rows)}] {r['artist']}: {r['video_title']}")
            if args.dry_run:
                continue
            try:
                download(r, out_root / r['artist'], args.quality, extra)
            except Exception as e:
                fails += 1
                print(f'    FAILED: {e!r}')
                if fails >= args.max_fails:
                    print(f'\nABORTED: {fails} downloads in a row failed (blocked?)')
                    return
                continue
            fails = 0
            done_file.write(r['id'] + '\n')
            done_file.flush()
            time.sleep(args.sleep)
    print(f"RE-DOWNLOAD DONE {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
