"""
Find and download studio songs of the artists in data/artists.txt from YouTube (audio only, mp3).

For each artist: YouTube search -> keep videos whose title looks like a song by that artist (not live / concert /
interview / full album / cover), 1.5-8 min long -> dedupe by normalized song title -> download.
Results: <output>/<artist>/<title> [<id>].mp3 and <output>/songs.csv (artist, title, video id, url, duration).

Private research use only: don't redistribute the audio.

    .venv-audio/bin/python data/fetch_songs.py --per-artist 25 --dry-run     # only list what would be downloaded
    .venv-audio/bin/python data/fetch_songs.py --per-artist 25
Then:
    .venv-audio/bin/python data/audio_to_piano.py --input data/audio/<artist> --output data/finetune/<artist>
"""

import argparse
import csv
import re
import time
from pathlib import Path

import yt_dlp

BAD_WORDS = [
    'live', 'концерт', 'concert', 'interview', 'інтерв', 'интервью', 'full album', 'альбом повністю', 'cover',
    'кавер', 'karaoke', 'караоке', 'мінус', 'минус', 'instrumental', 'reaction', 'реакція', 'tutorial', 'урок',
    'на гітарі', 'акорди', 'chords', 'remix', 'ремікс', 'teaser', 'тизер', 'backstage', 'making of', 'фестиваль',
    'fest', 'unplugged', 'акустика', 'acoustic', 'x factor', 'голос країни', 'shorts', 'trailer', 'трейлер',
    'розбір', 'playlist', 'mix', 'mashup', 'хіти', 'hits', 'змінювались', 'lyrics video збірка', 'найкращі пісні', 'best songs', 'всі пісні',
]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--artists', default='data/artists.txt')
    parser.add_argument('--output', default='data/audio')
    parser.add_argument('--per-artist', type=int, default=25, help='max songs per artist')
    parser.add_argument('--search-size', type=int, default=80, help='search results to scan per query')
    parser.add_argument('--min-sec', type=int, default=90)
    parser.add_argument('--max-sec', type=int, default=480)
    parser.add_argument('--sleep', type=float, default=3.0, help='seconds between downloads (be polite)')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def normalize(text):
    text = text.lower()
    text = re.sub(r'[\(\[].*?[\)\]]', ' ', text)  # (official video), [HD], ...
    text = re.sub(r'official|офіційн\w*|video|відео|audio|аудіо|кліп|clip|lyric\w*|hd|4k|премʼєра|прем\'єра|премьера',
                  ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def song_title(video_title, names):
    """'Скрябін - Місця щасливих людей (Official Video)' -> 'місця щасливих людей', or None if not this artist.
    names: the artist name and its aliases (e.g. 'Танок на Майдані Конго', 'ТНМК')."""
    norm_title = normalize(video_title)
    matched = [normalize(n) for n in names if normalize(n) in norm_title]
    if not matched:
        return None
    rest = norm_title.replace(max(matched, key=len), ' ')
    rest = re.sub(r'\bft\b.*|\bfeat\b.*', ' ', rest)
    rest = re.sub(r'\s+', ' ', rest).strip()
    return rest or None


def search_with_retry(names, n, attempts=4):
    """Network hiccups (DNS, timeouts) shouldn't kill a multi-hour run: retry with backoff, then skip the artist."""
    for attempt in range(attempts):
        try:
            return search(names, n)
        except Exception as e:
            wait = 30 * 2 ** attempt
            print(f'  search failed ({type(e).__name__}), retry {attempt + 1}/{attempts} in {wait}s')
            time.sleep(wait)
    print(f'  SKIPPED {names[0]}: search kept failing')
    return []


def search(names, n):
    opts = {'quiet': True, 'extract_flat': True, 'skip_download': True}
    found = {}
    queries = [q for name in names for q in (f'{name} official audio', f'{name} офіційне відео', f'{name} пісня')]
    with yt_dlp.YoutubeDL(opts) as ydl:
        for query in queries:
            info = ydl.extract_info(f'ytsearch{n}:{query}', download=False)
            for e in info.get('entries') or []:
                if e and e.get('id'):
                    found.setdefault(e['id'], e)
    return list(found.values())


def select(entries, names, args):
    chosen, seen_titles = [], set()
    for e in entries:
        title = e.get('title') or ''
        duration = e.get('duration') or 0
        if not (args.min_sec <= duration <= args.max_sec):
            continue
        if any(w in title.lower() for w in BAD_WORDS):
            continue
        song = song_title(title, names)
        if not song or song in seen_titles:
            continue
        seen_titles.add(song)
        chosen.append({'artist': names[0], 'song': song, 'video_title': title, 'id': e['id'],
                       'url': f"https://www.youtube.com/watch?v={e['id']}", 'duration': duration})
        if len(chosen) >= args.per_artist:
            break
    return chosen


def download(item, out_dir):
    opts = {
        'quiet': True,
        'format': 'bestaudio/best',
        'outtmpl': str(out_dir / '%(title)s [%(id)s].%(ext)s'),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([item['url']])


def main():
    args = parse_args()
    # one artist per line, aliases separated by '|': the first name is the folder / CSV name
    artists = [[n.strip() for n in line.split('|')] for line in open(args.artists, encoding='utf-8')
               if line.strip() and not line.startswith('#')]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / 'songs.csv'
    done = set()
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            done = {row['id'] for row in csv.DictReader(f)}

    fields = ['artist', 'song', 'video_title', 'id', 'url', 'duration']
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for names in artists:
            artist = names[0]
            chosen = select(search_with_retry(names, args.search_size), names, args)
            print(f'\n{artist}: {len(chosen)} songs')
            for item in chosen:
                status = 'have' if item['id'] in done else ('dry' if args.dry_run else 'get ')
                print(f"  [{status}] {item['song']:<40} {item['duration']:>4}s  {item['video_title']}")
                if args.dry_run or item['id'] in done:
                    continue
                try:
                    download(item, out_root / artist)
                except Exception as e:
                    print(f'    FAILED: {e!r}')
                    continue
                writer.writerow(item)
                f.flush()
                time.sleep(args.sleep)


if __name__ == '__main__':
    main()
