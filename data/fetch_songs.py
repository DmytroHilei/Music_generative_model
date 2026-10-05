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

--mode covers: solo piano covers / tutorials of the same artists instead (real human arrangements, transcribed with
data/transcribe_covers.py rather than reduced): the title must name the artist and a piano word, other instruments and
vocal covers are skipped, and a song may come from up to --per-song different videos (different pianists).
    .venv-audio/bin/python data/fetch_songs.py --mode covers --per-artist 60 --dry-run

--source soundcloud searches SoundCloud instead (same filters; a track whose title lacks the artist counts when the
uploader is the artist). Songs already in songs.csv count towards --per-artist, so a second source only tops an
artist up with songs it doesn't have yet.
"""

import argparse
import csv
import fcntl
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yt_dlp

BAD_WORDS = [
    'live', 'концерт', 'concert', 'interview', 'інтерв', 'интервью', 'full album', 'альбом повністю', 'cover',
    'кавер', 'karaoke', 'караоке', 'мінус', 'минус', 'instrumental', 'reaction', 'реакція', 'tutorial', 'урок',
    'на гітарі', 'акорди', 'chords', 'remix', 'ремікс', 'teaser', 'тизер', 'backstage', 'making of', 'фестиваль',
    'fest', 'unplugged', 'акустика', 'acoustic', 'x factor', 'голос країни', 'shorts', 'trailer', 'трейлер',
    'розбір', 'наживо', 'акапел', 'a cappella', 'acapella', 'a capella', 'type beat', 'slowed', 'reverb', 'sped up',
    'nightcore', '8d audio', 'bass boosted', 'playlist', 'mix', 'mashup', 'хіти', 'hits', 'змінювались', 'lyrics video збірка', 'найкращі пісні', 'best songs', 'всі пісні',
]


# covers mode: a piano word is required, covers/tutorials are fine, other instruments and singing are not
PIANO_WORDS = ['piano', 'піаніно', 'фортепіано', 'фортепиано', 'пианино', 'рояль', 'synthesia']
COVER_BAD_WORDS = [
    'live', 'концерт', 'concert', 'interview', 'інтерв', 'karaoke', 'караоке', 'мінус', 'минус', 'reaction', 'реакція',
    'guitar', 'гітар', 'гитар', 'vocal', 'вокал', 'sing', 'співа', 'пою', 'drum', 'барабан', 'violin', 'скрипк',
    'cello', 'віолончел', 'ukulele', 'укулеле', 'accordion', 'акордеон', 'баян', 'bayan', 'sax', 'саксофон', 'flute',
    'флейт', 'бандур', 'bandura', 'orchestra', 'оркестр', 'kalimba', 'калімба', 'harp', 'арф', 'organ', 'орган',
    'hang drum', 'handpan', 'mix', 'mashup', 'медлі', 'medley', 'попурі', 'хіти', 'hits', 'playlist', 'shorts',
    'ft.', 'feat', 'remix', 'ремікс', 'beat', 'бит', 'chords only', 'акорди', 'how to play chords',
]
# a cover/tutorial cue, so a band's own video (Pianoбой, the song 'Fortepiano') isn't taken for a cover
COVER_CUES = re.compile(r'cover|кавер|caver|сover|tutorial|туторіал|урок|synthesia|ноти|ноты|midi|sheet music|'
                        r'piano version|piano solo|piano arrangement|\(piano\)|на піаніно|на фортепіано|на пианино|'
                        r'на фортепиано')
COVER_WORDS = r'piano|піаніно|фортепіано|фортепиано|пианино|рояль|synthesia|cover|кавер|caver|сover|tutorial|' \
              r'урок|туторіал|на|by|easy|легко|ноти|ноты|notes|sheet|music|midi|instrumental|version|версія|' \
              r'arrangement|аранжування|solo|безкоштовні|free|and|the'


RUSSIAN_ONLY = re.compile('[ыэъёЫЭЪЁ]')

# search queries per artist name / alias, by (source, mode); status.py shows them (read with ast, no import)
QUERY_TEMPLATES = {
    ('youtube', 'songs'): ('{} official audio', '{} офіційне відео', '{} пісня'),
    ('youtube', 'covers'): ('{} piano cover', '{} на піаніно', '{} piano tutorial', '{} фортепіано'),
    # SoundCloud: artists upload their own tracks, so the bare name finds most of them
    ('soundcloud', 'songs'): ('{}', '{} пісня'),
    ('soundcloud', 'covers'): ('{} piano cover', '{} піаніно', '{} фортепіано'),
}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--artists', default='data/artists.txt')
    parser.add_argument('--output', default='data/audio')
    parser.add_argument('--per-artist', type=int, default=25, help='max songs per artist')
    parser.add_argument('--search-size', type=int, default=80, help='search results to scan per query')
    parser.add_argument('--min-sec', type=int, default=90)
    parser.add_argument('--max-sec', type=int, default=480)
    parser.add_argument('--quality', default='192', help='mp3 kbps (128 is enough for Demucs + basic-pitch)')
    parser.add_argument('--sleep', type=float, default=3.0, help='seconds between downloads (be polite)')
    parser.add_argument('--search-sleep', type=float, default=0.0, help='seconds between search queries')
    parser.add_argument('--cookies-from-browser', default=None, metavar='BROWSER[:PROFILE]',
                        help="send a logged-in browser's cookies (gets past YouTube's bot block on searches), e.g. "
                             "firefox:~/snap/firefox/common/.mozilla/firefox/<id>.default for snap Firefox")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--mode', choices=['songs', 'covers'], default='songs')
    parser.add_argument('--source', choices=['youtube', 'soundcloud'], default='youtube')
    parser.add_argument('--max-skips', type=int, default=3,
                        help='stop the run after this many artists in a row whose search kept failing (a block, '
                             'not a hiccup): the rest of the list stays untouched for a later re-run')
    parser.add_argument('--max-fails', type=int, default=5,
                        help='stop the run after this many failed downloads in a row (a block or the network down), '
                             'so the rest of the list is not searched while nothing downloads')
    parser.add_argument('--max-minutes', type=float, default=0,
                        help='> 0: start no new artist after this many minutes (the current one finishes)')
    parser.add_argument('--ukrainian-only', action='store_true',
                        help="skip videos whose title has Russian-only letters (ы э ъ ё): keeps Ukrainian-language songs "
                             "of artists who also sing in Russian; titles without telling letters pass")
    parser.add_argument('--songs-csv', default='data/audio/songs.csv', help='covers mode: known song titles')
    parser.add_argument('--per-song-queries', action='store_true',
                        help="covers mode: also one search per known song ('<artist> <song> піаніно', --song-search-size "
                             "results), not only per artist")
    parser.add_argument('--song-search-size', type=int, default=8)
    parser.add_argument('--per-song', type=int, default=2, help='covers mode: max videos of the same song')
    return parser.parse_args()


def normalize(text):
    text = text.lower()
    text = re.sub(r'[\(\[].*?[\)\]]', ' ', text)  # (official video), [HD], ...
    text = re.sub(r'\.(mp3|m4a|wav|flac|ogg)\b', ' ', text)  # SoundCloud uploads named after the file
    text = re.sub(r'official|офіційн\w*|video|відео|audio|аудіо|кліп|clip|lyric\w*|hd|4k|премʼєра|прем\'єра|премьера',
                  ' ', text)
    text = re.sub(r'[^\w\s]|_', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def song_title(video_title, names, covers=False, known=()):
    """'Скрябін - Місця щасливих людей (Official Video)' -> 'місця щасливих людей', or None if not this artist.
    names: the artist name and its aliases (e.g. 'Танок на Майдані Конго', 'ТНМК')."""
    norm_title = normalize(video_title)
    if covers:  # one-word Latin aliases ('Karna') hit unrelated songs in other languages
        names = [n for n in names if re.search('[а-яіїєґ]', n.lower()) or len(normalize(n).split()) > 1]
    matched = [normalize(n) for n in names if normalize(n) in norm_title]
    if not matched:
        return None
    rest = norm_title.replace(max(matched, key=len), ' ')
    rest = re.sub(r'\bft\b.*|\bfeat\b.*', ' ', rest)
    if covers:
        # the artist's known song title when the cover names one, so a song keeps one key (and one split)
        hits = [k for k in known if k and re.search(rf'\b{re.escape(k)}\b', rest)]
        if hits:
            return max(hits, key=len)
        rest = re.sub(rf'\b({COVER_WORDS})\b', ' ', rest)
    rest = re.sub(r'\s+', ' ', rest).strip()
    return rest or None


def search_with_retry(names, n, mode='songs', song_queries=(), song_n=8, source='youtube', extra=None, pause=0.0,
                      attempts=4):
    """Network hiccups (DNS, timeouts) shouldn't kill a multi-hour run: retry with backoff, then skip the artist
    (returns None)."""
    for attempt in range(attempts):
        try:
            return search(names, n, mode, song_queries, song_n, source, extra, pause)
        except Exception as e:
            wait = 30 * 2 ** attempt
            print(f'  search failed ({type(e).__name__}), retry {attempt + 1}/{attempts} in {wait}s')
            time.sleep(wait)
    print(f'  SKIPPED {names[0]}: search kept failing')
    return None


def search(names, n, mode='songs', song_queries=(), song_n=8, source='youtube', extra=None, pause=0.0):
    """extra: more yt-dlp options (cookies); pause: seconds between queries."""
    opts = {'quiet': True, 'extract_flat': True, 'skip_download': True, **(extra or {})}
    found = {}
    templates = QUERY_TEMPLATES[source, mode]
    prefix = 'ytsearch' if source == 'youtube' else 'scsearch'
    queries = [(t.format(name), n) for name in names for t in templates] + [(q, song_n) for q in song_queries]
    with yt_dlp.YoutubeDL(opts) as ydl:
        for query, k in queries:
            info = ydl.extract_info(f'{prefix}{k}:{query}', download=False)
            time.sleep(pause)
            for e in info.get('entries') or []:
                if e and e.get('id'):
                    found.setdefault(e['id'], e)
    return list(found.values())


def select(entries, names, args, known=(), have=None):
    """have: Counter of the artist's songs already downloaded (from any source), so they aren't fetched again."""
    covers = args.mode == 'covers'
    have = have or Counter()
    chosen, seen_titles = [], dict(have)
    budget = args.per_artist - sum(have.values())
    for e in entries:
        if budget <= 0:
            break
        title = e.get('title') or ''
        if args.source == 'soundcloud':
            title = re.sub(r'^\d{1,2}\s*[.)]\s*', '', title)  # '05. Піють півні' (album track number)
        duration = e.get('duration') or 0
        if not (args.min_sec <= duration <= args.max_sec):
            continue
        low = title.lower()
        if covers:
            # piano word and cover cue outside the artist's name ('Pianoбой', 'Pianoboy')
            bare = low
            for n in sorted(names, key=len, reverse=True):
                bare = bare.replace(n.lower(), ' ')
            if not any(w in bare for w in PIANO_WORDS) or not COVER_CUES.search(bare):
                continue
        if args.ukrainian_only and RUSSIAN_ONLY.search(title):
            continue
        if any(w in low for w in (COVER_BAD_WORDS if covers else BAD_WORDS)):
            continue
        match_title = title
        if args.source == 'soundcloud' and not song_title(title, names, covers, known) and \
                normalize(e.get('uploader') or '') in {normalize(n) for n in names}:
            match_title = f'{names[0]} - {title}'  # the artist's own upload: the title is just the song
        song = song_title(match_title, names, covers, known)
        if not song or seen_titles.get(song, 0) >= (args.per_song if covers else 1):
            continue
        seen_titles[song] = seen_titles.get(song, 0) + 1
        url = f"https://www.youtube.com/watch?v={e['id']}" if args.source == 'youtube' else e['url']
        chosen.append({'artist': names[0], 'song': song, 'video_title': title, 'id': e['id'], 'url': url,
                       'duration': duration})
        budget -= 1
    return chosen


def download(item, out_dir, quality, extra=None):
    opts = {
        **(extra or {}),
        'quiet': True,
        'format': 'bestaudio/best',
        'outtmpl': str(out_dir / '%(title)s [%(id)s].%(ext)s'),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': quality}],
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([item['url']])


def main():
    sys.stdout.reconfigure(line_buffering=True)  # the log is a file: status.py reads progress from it live
    args = parse_args()
    # run separator: status.py counts this run's progress from the last one (the logs are appended across re-runs)
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}: {args.source} {args.mode} run of {args.artists} =====")
    # one artist per line, aliases separated by '|': the first name is the folder / CSV name
    artists = [[n.strip() for n in line.split('|')] for line in open(args.artists, encoding='utf-8')
               if line.strip() and not line.startswith('#')]
    extra = {}
    if args.cookies_from_browser:
        browser, _, profile = args.cookies_from_browser.partition(':')
        extra['cookiesfrombrowser'] = (browser, str(Path(profile).expanduser()) if profile else None, None, None)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / 'songs.csv'
    known = {}  # covers mode: the songs we already have per artist (titles as in data/audio/songs.csv)
    raw_titles = {}  # the same titles unnormalized, for per-song queries
    if args.mode == 'covers' and Path(args.songs_csv).exists():
        with open(args.songs_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                known.setdefault(row['artist'], set()).add(normalize(row['song']))
                raw_titles.setdefault(row['artist'], set()).add(row['song'])
    done, have = set(), {}
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add(row['id'])
                have.setdefault(row['artist'], Counter())[row['song']] += 1

    fields = ['artist', 'song', 'video_title', 'id', 'url', 'duration']
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        # one run per output folder: two runs at once wrote every song twice (2026-10-05, 330 duplicate rows)
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f'another fetch_songs.py run is writing {csv_path}: stop it first')
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        skips = fails = 0
        t_start = time.time()
        for names in artists:
            artist = names[0]
            if args.max_minutes > 0 and time.time() - t_start > args.max_minutes * 60:
                print(f'\nSTOPPED before {artist}: --max-minutes {args.max_minutes:g} reached')
                break
            if fails >= args.max_fails:
                print(f'\nABORTED before {artist}: {fails} downloads in a row failed (blocked or network down?)')
                break
            song_queries = [f'{artist} {t} піаніно' for t in sorted(raw_titles.get(artist, ()))] \
                if args.mode == 'covers' and args.per_song_queries else []
            entries = search_with_retry(names, args.search_size, args.mode, song_queries, args.song_search_size,
                                        args.source, extra, args.search_sleep)
            skips = skips + 1 if entries is None else 0
            if skips >= args.max_skips:
                print(f'\nABORTED at {artist}: {skips} artists in a row failed to search (blocked?)')
                break
            chosen = select(entries or [], names, args, known.get(artist, ()), have.get(artist))
            print(f'\n{artist}: {len(chosen)} songs')
            for item in chosen:
                status = 'have' if item['id'] in done else ('dry' if args.dry_run else 'get ')
                print(f"  [{status}] {item['song']:<40} {item['duration']:>4}s  {item['video_title']}")
                if args.dry_run or item['id'] in done:
                    continue
                try:
                    download(item, out_root / artist, args.quality, extra)
                except Exception as e:
                    print(f'    FAILED: {e!r}')
                    fails += 1
                    if fails >= args.max_fails:
                        break
                    continue
                fails = 0
                done.add(item['id'])
                writer.writerow(item)
                f.flush()
                time.sleep(args.sleep)


if __name__ == '__main__':
    main()
