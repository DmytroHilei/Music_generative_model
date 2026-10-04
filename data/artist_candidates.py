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
EXCLUDE = re.compile(r'метал|хардкор|грайнд|дум-|блек|дез-|металкор|нойз', re.I)  # punk-rock stays (user, 2026-10-04)
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


def stage_names(titles):
    """{wiki title: [search name, aliases...]} from Wikidata: pseudonym (P742) first, then the Ukrainian label
    ('Ім'я Прізвище' for people, unlike the 'Прізвище Ім'я По-батькові' page titles), then a Latin label/alias."""
    qid = {}
    for i in range(0, len(titles), 50):
        d = get(API, dict(action='query', format='json', prop='pageprops', ppprop='wikibase_item',
                          titles='|'.join(titles[i:i + 50])))
        for pg in d['query']['pages'].values():
            if 'pageprops' in pg:
                qid[pg['title']] = pg['pageprops']['wikibase_item']
    ents = {}
    ids = sorted(set(qid.values()))
    for i in range(0, len(ids), 50):
        d = get('https://www.wikidata.org/w/api.php', dict(action='wbgetentities', format='json', ids='|'.join(ids[i:i + 50]),
                                                           props='labels|aliases|claims', languages='uk|en'))
        ents.update(d.get('entities', {}))
    out = {}
    for title, q in qid.items():
        e = ents.get(q, {})
        names = []
        for c in e.get('claims', {}).get('P742', []):  # pseudonym
            v = c.get('mainsnak', {}).get('datavalue', {}).get('value')
            if isinstance(v, str):
                names.append(v)
        uk = e.get('labels', {}).get('uk', {}).get('value')
        en = e.get('labels', {}).get('en', {}).get('value')
        names += [n for n in (uk, en) if n]
        names += [a['value'] for a in e.get('aliases', {}).get('en', [])][:2]
        pseudonyms = {v for c in e.get('claims', {}).get('P742', [])
                      if isinstance(v := c.get('mainsnak', {}).get('datavalue', {}).get('value'), str)}
        out[title] = order_names([clean_name(n) for n in names] + [clean_name(title)], pseudonyms)
    return out


PATRONYMIC = re.compile(r"(ович|евич|йович|івна|ївна|ївич|овна|евна)$")
RUSSIAN = re.compile('[ыэъёЫЭЪЁ]')
CYR = re.compile('[А-Яа-яІіЇїЄєҐґ]')


def order_names(names, pseudonyms=()):
    """Search name first: a Ukrainian-script name, 'Прізвище Ім'я По-батькові' turned into 'Ім'я Прізвище'; a
    pseudonym leads only if it's distinctive (>= 2 words or >= 6 letters) and not Russian-only. Russian-only names go
    last (they would pull Russian-language search results), Latin names after the Cyrillic ones."""
    fixed = []
    for n in names:
        w = n.split()
        if len(w) == 3 and CYR.search(n) and PATRONYMIC.search(w[2]):
            n = f'{w[1]} {w[0]}'
        fixed.append(n.split(' / ')[0].strip())
    fixed = list(dict.fromkeys(f for f in fixed if f))
    def rank(n):
        cyr, rus = bool(CYR.search(n)), bool(RUSSIAN.search(n))
        distinctive = len(n.split()) >= 2 or len(n) >= 6
        pseudo = n in pseudonyms
        return (rus, not cyr, not (pseudo and distinctive), not distinctive)
    return sorted(fixed, key=rank)


def clean_name(title):
    return re.sub(r'\s*\((гурт|співачка|співак|музичний гурт|рок-гурт|гурт, Україна)\)\s*$', '', title).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/artist_candidates.tsv')
    p.add_argument('--top', type=int, default=300, help='artists written to --list (by views)')
    p.add_argument('--list', default='data/artists_covers_next.txt', help='fetch_songs.py artist file to write')
    args = p.parse_args()
    cached = {}  # views from an earlier run (the slow part)
    if Path(args.out).exists():
        with open(args.out, encoding='utf-8') as f:
            cached = {r['wiki_title']: int(r['views_12m']) for r in csv.DictReader(f, delimiter='\t')
                      if int(r['views_12m']) > 0}
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
        v = (cached.get(title) or views_12m(title)) if status == 'keep' or status.startswith('flag') else 0
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
    top = [r for r in keep if not r['already_used']][:args.top]
    names = stage_names([r['wiki_title'] for r in top])
    with open(args.list, 'w', encoding='utf-8') as f:
        f.write('# Piano-cover search, round 2 (2026-10-04): top Ukrainian pop-rock/rock/punk-rock + pop/estrada\n'
                '# artists by uk.wikipedia views (data/artist_candidates.py), not in artists.txt / artists_covers_extra.txt.\n'
                '# Search name first (stage name from Wikidata), aliases after |. Review before fetching.\n')
        for r in top:
            alias = names.get(r['wiki_title']) or [r['name']]
            f.write('|'.join(dict.fromkeys(alias)) + '\n')
    print(f'wrote {len(top)} artists to {args.list}')
    print(f'{len(rows)} candidates: {len(keep)} keep, {sum(r["status"].startswith("flag") for r in rows)} flagged, '
          f'{sum(r["status"].startswith("excluded") for r in rows)} excluded -> {args.out}')


if __name__ == '__main__':
    main()
