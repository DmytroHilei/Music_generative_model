"""
Candidate Ukrainian artists for the piano-cover search, from Ukrainian Wikipedia categories, ranked by page views.

    python data/artist_candidates.py --out data/artist_candidates.tsv

Sources (Wikipedia API, the intended route for automated access; ~1 request/s, retries, no personal data):
  categories: Українські рок-гурти, Українські попспівачки, Українські попспіваки, Українські рок-співаки
  each page's own categories: drops metal / punk / hardcore / ska acts (the target is pop-rock and pop), flags
  Russian-language acts ('російськомовн' in a category) for review instead of dropping them
  page views (Wikimedia REST API, last 12 months) to rank by how much people look the artist up
Output (TSV): name, views_12m, source category, status (keep / excluded: <reason> / flag: <reason>),
already_used (in data/artists.txt or data/artists_covers_extra.txt). Review it before fetching anything.
"""

import argparse
import csv
import datetime
import re
import time
import urllib.parse
from pathlib import Path

import requests

UA = 'MusicAutoregressiveModel-research/0.1 (non-commercial)'
API = 'https://uk.wikipedia.org/w/api.php'
CATEGORIES = ['Українські рок-гурти', 'Українські попспівачки', 'Українські попспіваки', 'Українські рок-співаки']
EXCLUDE = re.compile(r'метал|панк|хардкор|ска-|ska|грайнд|дум|блек|дез|металкор|нойз', re.I)
FLAG = re.compile(r'російськомовн', re.I)
session = requests.Session()
session.headers['User-Agent'] = UA


def get(url, params=None, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200 and r.text.strip():
                time.sleep(1.0)  # polite pace
                return r.json()
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'failed: {url} {params}')


def category_pages(cat):
    pages, cont = [], {}
    while True:
        d = get(API, dict(action='query', format='json', list='categorymembers', cmtitle=f'Категорія:{cat}',
                          cmtype='page', cmlimit=500, **cont))
        pages += [m['title'] for m in d['query']['categorymembers']]
        if 'continue' not in d:
            return pages
        cont = {'cmcontinue': d['continue']['cmcontinue']}


def page_categories(titles):
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        cont = {}
        while True:
            d = get(API, dict(action='query', format='json', prop='categories', titles='|'.join(batch),
                              cllimit=500, clshow='!hidden', **cont))
            for p in d['query']['pages'].values():
                out.setdefault(p['title'], []).extend(c['title'] for c in p.get('categories', []))
            if 'continue' not in d:
                break
            cont = {k: v for k, v in d['continue'].items() if k != 'continue'}
    return out


def views_12m(title):
    end = datetime.date.today().replace(day=1)
    start = end.replace(year=end.year - 1)
    url = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/uk.wikipedia/all-access/user/'
           f'{urllib.parse.quote(title.replace(" ", "_"), safe="")}/monthly/{start:%Y%m%d}00/{end:%Y%m%d}00')
    d = get(url)
    return sum(i['views'] for i in d.get('items', [])) if d else 0


def clean_name(title):
    return re.sub(r'\s*\((гурт|співачка|співак|музичний гурт|рок-гурт|гурт, Україна)\)\s*$', '', title).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/artist_candidates.tsv')
    args = p.parse_args()
    used = set()
    for f in ('data/artists.txt', 'data/artists_covers_extra.txt'):
        for line in Path(f).read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.startswith('#'):
                used |= {a.strip().lower() for a in line.split('|')}
    src = {}
    for cat in CATEGORIES:
        for t in category_pages(cat):
            src.setdefault(t, cat)
        print(f'{cat}: {sum(1 for v in src.values() if v == cat)} pages', flush=True)
    cats = page_categories(list(src))
    rows = []
    for i, (title, cat) in enumerate(src.items()):
        own = ' '.join(cats.get(title, []))
        status = 'keep'
        m = EXCLUDE.search(own)
        if m:
            status = f'excluded: {m.group(0)}'
        elif FLAG.search(own):
            status = 'flag: Russian-language category'
        v = views_12m(title) if status == 'keep' or status.startswith('flag') else 0
        name = clean_name(title)
        rows.append(dict(name=name, views_12m=v, source=cat, status=status, already_used=name.lower() in used,
                         wiki_title=title))
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(src)} pages', flush=True)
    rows.sort(key=lambda r: -r['views_12m'])
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    keep = [r for r in rows if r['status'] == 'keep']
    print(f'{len(rows)} candidates: {len(keep)} keep, {sum(r["status"].startswith("flag") for r in rows)} flagged, '
          f'{sum(r["status"].startswith("excluded") for r in rows)} excluded -> {args.out}')


if __name__ == '__main__':
    main()
