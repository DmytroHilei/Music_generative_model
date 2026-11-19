"""
Write census/report.md from the census json outputs (counts*.json, usable*.json), so every number in the report is
read from a measured file. Free text (findings, reconciliation) lives in census/report_notes.md and is appended.

    PYTHONPATH=.:data_prep/datasets:data_prep/census python data_prep/census/report.py

Scenario tags: '' = core three with the full Aria tar; store-pruned / store-deduped = core three with the Aria tier
tars tokenized as their own stores (the tiers cut recordings into different segments than the full tar, so they are
not subsets of it by name). Later rows (sources 4-8) are added by listing their tags in SCENARIOS.
"""

import json
from pathlib import Path

CENSUS = Path('census')
SCENARIOS = [  # (label, tag) in table order
    ('Core three, Aria deduped tier only', 'store-deduped'),
    ('Core three, Aria pruned tier', 'store-pruned'),
    ('Core three, Aria full', ''),
    ('+ sources 4–8', 'plus48'),
    ('+ sources 4–8, without `unclear` licences', 'plus48-no-unclear'),
]
RUNS = [('1B @ 20B', 20e9), ('1.8B @ 35B', 35e9), ('2.5B @ 50B', 50e9)]
ORIGIN = {'gigamidi': 'native (scraped MIDI)', 'discover': 'native (scraped MIDI) + transcribed (contains Aria-MIDI)',
          'aria': 'transcribed (audio to MIDI, piano)', 'aria_pruned': 'transcribed', 'aria_deduped': 'transcribed',
          'pianocore': 'transcribed (Aria, PERiScoPe/Transkun V2, ATEPP) + Disklavier (ASAP)',
          'piast': 'transcribed (piano)', 'beat': 'score (MuseScore MusicXML)', 'pdmx': 'score'}


def load(name, tag):
    p = CENSUS / f'{name}{"_" + tag if tag else ""}.json'
    return json.loads(p.read_text()) if p.exists() else None


def b(x):
    return f'{x / 1e9:.3f}B'


def m(x):
    return f'{x / 1e6:.1f}M'


def headline():
    rows = ['| Scenario | unique notes (loose) | usable @2048 | ' + ' | '.join(f'epochs: {r}' for r, _ in RUNS) + ' |',
            '|---|---|---|' + '---|' * len(RUNS)]
    for label, tag in SCENARIOS:
        u = load('usable', tag)
        if u is None:
            rows.append(f'| {label} | pending | pending |' + ' |' * len(RUNS))
            continue
        lv = u['total']['levels']['loose']
        ep = ' | '.join(f'{seen / lv["usable_notes"]:.2f}' for _, seen in RUNS)
        rows.append(f'| {label} | {b(lv["kept_notes"])} | {b(lv["usable_notes"])} | {ep} |')
    return '\n'.join(rows)


def per_source(tag=''):
    c, u = load('counts', tag), load('usable', tag)
    srcs = [s for s in u['_meta']['sources']]
    out = ['| Source | licence | files listed / parsed / failed | notes raw | md5 | exact | **loose** | loose_t | '
           'usable @2048 (loose) |', '|---|---|---|---|---|---|---|---|---|']
    for s in srcs + ['total']:
        d = c[s]
        lv = d['levels']
        kept = {k: lv[k]['kept_notes'] for k in ('md5', 'exact', 'loose', 'loose_t')}
        us = u[s]['levels']['loose']['usable_notes']
        lic = d.get('licence', '')
        out.append(f'| {s} | {lic} | {d["files_listed"]:,} / {d["files_parsed"]:,} / {d["files_failed"]:,} | '
                   f'{b(d["notes_raw"])} | {b(kept["md5"])} | {b(kept["exact"])} | **{b(kept["loose"])}** | '
                   f'{b(kept["loose_t"])} | {b(us)} |')
    return '\n'.join(out)


def drops(tag=''):
    c = load('counts', tag)
    out = ['| Source | level | dropped within source (files / notes) | dropped vs earlier sources (files / notes) | '
           'first copies in |', '|---|---|---|---|---|']
    for s in load('usable', tag)['_meta']['sources']:
        for lvl, v in c[s]['levels'].items():
            out.append(f'| {s} | {lvl} | {v["dropped_within_files"]:,} / {m(v.get("dropped_within_notes", 0))} | '
                       f'{v["dropped_cross_files"]:,} / {m(v.get("dropped_cross_notes", 0))} | '
                       f'{", ".join(f"{k} {n:,}" for k, n in v["dropped_by_first_source"].items()) or "-"} |')
    return '\n'.join(out)


def shape(tag=''):
    u = load('usable', tag)
    srcs = u['_meta']['sources']
    out = ['| Source | kept notes (loose) | > 2k notes | > 4k | > 8k | piano-only | drums-only | single other '
           'instrument | multi-instrument | origin |', '|---|---|---|---|---|---|---|---|---|---|']
    tot_n, acc = 0, {}
    for s in srcs:
        lv = u[s]['levels']['loose']
        n = lv['kept_notes']
        tot_n += n
        vals = {f'gt{L}': lv[f'long_share_gt_{L}'] for L in (2048, 4096, 8192)} | lv['mix']
        for k, v in vals.items():
            acc[k] = acc.get(k, 0) + v * n
        out.append(f'| {s} | {b(n)} | {vals["gt2048"]:.1%} | {vals["gt4096"]:.1%} | {vals["gt8192"]:.1%} | '
                   f'{vals["piano_only"]:.1%} | {vals["drums_only"]:.1%} | {vals["single_other"]:.1%} | '
                   f'{vals["multi"]:.1%} | {ORIGIN.get(s, "")} |')
    t = {k: v / max(tot_n, 1) for k, v in acc.items()}
    out.append(f'| total | {b(tot_n)} | {t["gt2048"]:.1%} | {t["gt4096"]:.1%} | {t["gt8192"]:.1%} | '
               f'{t["piano_only"]:.1%} | {t["drums_only"]:.1%} | {t["single_other"]:.1%} | {t["multi"]:.1%} | |')
    return '\n'.join(out)


def main():
    notes = (CENSUS / 'report_notes.md').read_text() if (CENSUS / 'report_notes.md').exists() else ''
    lines = notes.split('<!-- HEADLINE -->')
    head, rest = (lines[0], lines[1]) if len(lines) == 2 else ('', notes)
    md = [head.rstrip(), '', headline(), '', rest.strip(), '',
          '## Per source (core three, Aria full)', '', per_source(), '',
          'Levels cascade left to right: each column is the notes kept after that level and all levels before it. '
          'First copy wins in the order GigaMIDI → Discover → Aria; within GigaMIDI and Aria, train before '
          'validation before test; within Discover, archive order (as its own script decided).', '',
          '## Drops by level', '', drops(), '',
          '## Length and instrument mix (loose level, notes-weighted)', '', shape(), '',
          'piano-only: every note in GM programs 0–7; drums-only: every note a drum note; single other: one '
          'non-piano pitched program, no drums; multi: everything else.', '']
    (CENSUS / 'report.md').write_text('\n'.join(md))
    print('census/report.md written')


if __name__ == '__main__':
    main()
