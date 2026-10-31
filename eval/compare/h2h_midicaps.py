"""
The MidiCaps test set in the head-to-head (eval/compare/head2head.py): the home ground of Text2midi and MIDI-LLM, which
trained on the MidiCaps train rows. 500 captions, one file each from F1, Text2midi and MIDI-LLM.

    .venv/bin/python eval/compare/h2h_midicaps.py eligible    # results/head2head/midicaps/eligible.json
    .venv/bin/python eval/compare/h2h_midicaps.py fetch --archive data/midicaps/archive/midicaps.tar.gz
    .venv/bin/python eval/compare/h2h_midicaps.py prompts     # prompts.json, real/, real_all/
    .venv/bin/python eval/compare/h2h_midicaps.py score       # scores.md / scores.json

eligible  test_set rows F1 cannot have seen: in GigaMIDI's validation or test split, not in the store the 107M pilots and
  f1_local trained on (gigamidi_replay_train), and with notes that occur in no train file of GigaMIDI or Discover (F1
  starts from long_450m, which pretrained on Discover). Both note hashes of prepare_discover.py count: exact, and loose =
  the same onsets and pitches under other instruments, velocities or durations. Rows outside GigaMIDI have no hash and
  stay out. Order: sorted by md5, shuffled with --seed. `excluded` = rows of that order whose notes are in the replay
  store under another md5, found after generation started: they keep their place (a prompt's seed and file name go by
  it) and no score or Frechet pool reads them.
fetch  only the eligible names are read from the archive, by their bytes, and kept only if the md5 of the bytes is the
  name: real_all/<md5>.mid. The archive's own paths never reach the disk.
prompts  the first --rows eligible rows with a parseable file, a tempo and a key: ids m000.., real/<id>_0.mid; 495 of the
  500 are scored. programs = the instruments the caption names: instrument_summary is unique by name and
  instrument_numbers_sorted is not (it repeats a program per track, and Piano covers programs 0-7), so each name gets its
  first program, through the program -> name table of the rows where the two lists line up.

Scores. The definitions of Text2midi (arXiv 2412.16526, Table 2): TB = the tempo falls in the bin of the row's tempo,
bin borders 40, 60, 70, 90, 110, 140, 160, 210 bpm; TBT = in that bin or a neighbouring one; CK = the key is the row's;
CKD = the key or its relative major / minor. The authors take tempo and key from music21; our MIDI files carry no tempo
metadata, so here both come from the notes (musicar.data.features: autocorrelation of the onsets, 40-200 bpm;
Krumhansl-Schmuckler), on every row alike and on the first SECONDS seconds as in the paper. The target is the MidiCaps
field, which the caption states. real = the files of the same captions through the same estimators: the ceiling, what
these estimators reach on the true file, and real_shuffled = the real files against another prompt's targets: the floor.
Also: recall of the instruments the caption lists, the same by GM family, Jaccard overlap, and caption match (caption
match.match, our caption builder's thresholds, so it favours ours). A missing or unreadable file scores 0, so every
row has the same denominator; a prompt that lists no instruments has no instrument scores on any row. Bootstrap over prompts as in h2h_clamp3. Text2midi's compression ratio is left out (it
needs COSIATEC), as are CLAP and FAD (audio rendering); CLaMP 3 and Frechet come from h2h_clamp3.py --task midicaps.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_prep' / 'datasets'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2h_clamp3 import bootstrap, deranged, stat, win_table  # noqa: E402
from head2head import COPY_LEN, OUT, open_store, read_midi  # noqa: E402
from musicar.data.features import DT, FIELDS, PITCH_CLASSES, file_features  # noqa: E402
from musicar.generation.caption_match import match, note_attributes, parse_caption  # noqa: E402
from musicar.generation.repetition import copy_share  # noqa: E402
from prepare_discover import note_hashes  # noqa: E402

TASK_DIR = OUT / 'midicaps'
CAPTIONS = ROOT / 'data' / 'midicaps' / 'train.json'
ROWS, SECONDS = 500, 40.0
TEMPO_EDGES = (40, 60, 70, 90, 110, 140, 160, 210)
FLATS = dict(Db='C#', Eb='D#', Gb='F#', Ab='G#', Bb='A#')
DRUMS = 128
SCORES = ('tb', 'tbt', 'ck', 'ckd', 'recall', 'family', 'jaccard', 'match')
LABELS = dict(tb='TB', tbt='TBT', ck='CK', ckd='CKD', recall='instrument recall', family='family recall',
              jaccard='instrument Jaccard', match='caption match (ours)')
SHAPE = ('seconds', 'notes', 'nps', 'runaway', 'copies')


def md5_of(row):
    return row['location'].rsplit('/', 1)[1].removesuffix('.mid')


def caption_rows():
    with open(CAPTIONS, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def test_rows(rows=None):
    return {md5_of(r): r for r in rows or caption_rows() if r.get('test_set')}


def program_names(rows):
    """{GM program: its MidiCaps name}, the most common one where the program and name lists have the same length."""
    seen = defaultdict(Counter)
    for r in rows:
        if len(r['instrument_numbers_sorted']) == len(r['instrument_summary']):
            for program, name in zip(r['instrument_numbers_sorted'], r['instrument_summary']):
                seen[program][name] += 1
    return {program: names.most_common(1)[0][0] for program, names in seen.items()}


def asked_programs(numbers, summary, names):
    """One program per name of the caption's instrument list, the first of the file's programs with that name."""
    first = {}
    for program in numbers:
        first.setdefault(names.get(program), program)
    return [first[name] for name in summary if name in first]


def tempo_bin(bpm):
    """The bin of TEMPO_EDGES; an edge belongs to the bin below it, as in MidiCaps' tempo words (Largo 41-60)."""
    return int(np.digitize(bpm, TEMPO_EDGES, right=True))


def key_index(name):
    """'F# minor' -> file_features' key: pitch class, + 12 for minor."""
    tonic, mode = name.rsplit(' ', 1)
    return PITCH_CLASSES.index(FLATS.get(tonic, tonic)) + (12 if mode == 'minor' else 0)


def relative(key):
    """The relative minor of a major key and the other way round."""
    return 12 + (key + 9) % 12 if key < 12 else (key - 12 + 3) % 12


def family(program):
    return 16 if program == DRUMS else program // 8


def instrument_scores(asked, present):
    """(recall, family recall, Jaccard) of the asked programs in the present ones; 0 when none is present."""
    asked, present = set(asked), set(present)
    if not asked:
        return np.nan, np.nan, np.nan
    fam = {family(p) for p in present}
    return (len(asked & present) / len(asked), np.mean([family(p) in fam for p in asked]),
            len(asked & present) / len(asked | present))


def first_seconds(tok, prog, seconds=SECONDS):
    keep = np.cumsum(tok[:, 3]) * DT <= seconds
    return tok[keep], prog[keep]


def file_row(tok, prog, target):
    """SCORES and SHAPE of a piece against a prompt's target, on its first SECONDS seconds."""
    tok, prog = first_seconds(tok, prog)
    if not len(tok):
        return None
    f = dict(zip(FIELDS, file_features(tok, prog)[0]))
    a = note_attributes(tok, prog)
    bpm, key = float(f['tempo']), int(f['key'])
    gap = abs(tempo_bin(bpm) - tempo_bin(target['tempo'])) if bpm > 0 else 99
    recall, fam, jaccard = instrument_scores(target['programs'], a['present'])
    return dict(tb=float(gap == 0), tbt=float(gap <= 1), ck=float(key == target['key']),
                ckd=float(key >= 0 and key in (target['key'], relative(target['key']))), recall=recall, family=fam,
                jaccard=jaccard, match=match(parse_caption(target['caption']), a)[0], seconds=a['seconds'], notes=a['n'],
                nps=a['nps'], runaway=a['runaway'], copies=copy_share(tok[:, 0].tolist(), tok[:, 3], prog, COPY_LEN))


def eligible_md5s(rows, hashes, seen_exact, seen_loose, replay):
    """(md5s in the shared order's input, sorted; counts of where the rest went). hashes = md5 -> (split, exact, loose),
    split 0 = train."""
    counts = dict(test_rows=len(rows), outside_gigamidi=0, train_split=0, replay=0, same_notes=0)
    keep = []
    for md5 in sorted(rows):
        if md5 not in hashes:
            counts['outside_gigamidi'] += 1
        elif hashes[md5][0] == 0:
            counts['train_split'] += 1
        elif md5 in replay:
            counts['replay'] += 1
        elif hashes[md5][1] in seen_exact or hashes[md5][2] in seen_loose:
            counts['same_notes'] += 1
        else:
            keep.append(md5)
    return keep, dict(counts, eligible=len(keep))


def eligible(args):
    rows = test_rows()
    z = np.load(ROOT / 'data' / 'gigamidi' / 'gigamidi_hashes.npz')
    split, exact, loose = z['split'], z['exact'], z['loose']
    place = {m.tobytes().hex(): i for i, m in enumerate(z['md5'].astype('V16'))}  # S16 items lose trailing zero bytes
    hashes = {m: (int(split[place[m]]), int(exact[place[m]]), int(loose[place[m]])) for m in rows if m in place}
    seen_exact, seen_loose = set(exact[split == 0].tolist()), set(loose[split == 0].tolist())
    with open(ROOT / 'data' / 'discover' / 'discover.csv', encoding='utf-8') as f:
        columns = next(f).rstrip('\n').split(',')
        for line in f:
            c = dict(zip(columns, line.rstrip('\n').split(',')))
            if c['split'] == 'train' and c['status'] == 'kept':
                seen_exact.add(int(c['exact'], 16))
                seen_loose.add(int(c['loose'], 16))
    with open(ROOT / 'data' / 'labels' / 'gigamidi_replay_train.jsonl', encoding='utf-8') as f:
        replay = {json.loads(line)['md5'] for line in f}
    keep, counts = eligible_md5s(rows, hashes, seen_exact, seen_loose, replay)
    np.random.default_rng(args.seed).shuffle(keep)
    offsets, read = open_store('gigamidi_replay_train')
    store = [read(a, b) for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())]
    store = [note_hashes(tok.astype(np.uint16), prog.astype(np.uint8)) for tok, prog in store]
    exact_in, loose_in = {h[0] for h in store}, {h[1] for h in store}
    excluded = [m for m in keep if hashes[m][1] in exact_in or hashes[m][2] in loose_in]
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    (TASK_DIR / 'eligible.json').write_text(json.dumps(dict(seed=args.seed, counts=dict(counts, excluded=len(excluded)),
                                                            md5=keep, excluded=excluded), indent=1))
    print(counts, f'{len(excluded)} in the replay store by notes')


def fetch(args):
    wanted = set(json.loads((TASK_DIR / 'eligible.json').read_text())['md5'])
    out = TASK_DIR / 'real_all'
    out.mkdir(exist_ok=True)
    found, changed = 0, []
    with tarfile.open(args.archive, 'r:gz') as tar:
        for member in tar:
            stem = member.name.rsplit('/', 1)[-1].removesuffix('.mid')
            if not member.isfile() or stem not in wanted:
                continue
            data = tar.extractfile(member).read()
            if hashlib.md5(data).hexdigest() != stem:
                changed.append(stem)
                continue
            (out / f'{stem}.mid').write_bytes(data)
            found += 1
    print(f'{found} of {len(wanted)} files -> {out}' + (f', bytes differ from the name: {changed}' if changed else ''))


def export_prompts(args):
    pool = json.loads((TASK_DIR / 'eligible.json').read_text())
    everything = caption_rows()
    rows, names = test_rows(everything), program_names(everything)
    prompts = []
    (TASK_DIR / 'real').mkdir(exist_ok=True)
    for md5 in pool['md5']:
        r, path = rows[md5], TASK_DIR / 'real_all' / f'{md5}.mid'
        if not (r.get('tempo') and r.get('key') and path.exists()) or read_midi(path) is None:
            continue
        pid = f'm{len(prompts):03d}'
        programs = asked_programs(r['instrument_numbers_sorted'], r['instrument_summary'], names)
        prompts.append(dict(id=pid, name=pid, md5=md5, caption=r['caption'], tempo=r['tempo'], key=key_index(r['key']),
                            programs=programs, **({'excluded': True} if md5 in pool['excluded'] else {})))
        shutil.copyfile(path, TASK_DIR / 'real' / f'{pid}_0.mid')
        if len(prompts) == args.rows:
            break
    tmp = TASK_DIR / 'prompts.tmp'
    tmp.write_text(json.dumps(dict(seed=args.seed, rows=len(prompts), prompts=prompts), indent=1, ensure_ascii=False))
    os.replace(tmp, TASK_DIR / 'prompts.json')  # a queue reads this file when each of its steps starts
    print(f'{len(prompts)} prompts -> {TASK_DIR}')


def missing_row(target):
    """A file that is not there: 0 on every score it could be judged on, no SHAPE."""
    row = {k: 0.0 for k in SCORES}
    if not target['programs']:
        row.update(recall=np.nan, family=np.nan, jaccard=np.nan)
    return row


def system_rows(folder, prompts, targets):
    """(n prompts, SCORES + SHAPE) of a folder's <id>_0.mid files against targets[i]. nan = not judged: a prompt without
    instruments on the instrument scores, a missing file on SHAPE."""
    keys = (*SCORES, *SHAPE)
    table = np.full((len(prompts), len(keys)), np.nan)
    for i, (p, t) in enumerate(zip(prompts, targets)):
        res = read_midi(folder / f"{p['id']}_0.mid") if (folder / f"{p['id']}_0.mid").exists() else None
        row = (file_row(*res, t) if res is not None else None) or missing_row(t)
        table[i] = [row.get(k, np.nan) for k in keys]
    return table


def report(res):
    lines = [f"# MidiCaps test set ({res['date']})", '',
             f"{res['prompts']} captions, one file each, first {SECONDS:.0f} s, tempo and key estimated from the notes "
             f"(h2h_midicaps.py). {res['draws']} bootstrap draws over prompts, 95% percentile interval. A missing file "
             'scores 0. real = the same estimators on the true files (ceiling); real_shuffled = against another '
             "prompt's targets (floor). Caption match uses our caption builder's thresholds.", '',
             '| system | files | ' + ' | '.join(LABELS[k] for k in SCORES) + ' |', '|---' * (len(SCORES) + 2) + '|']
    for name, row in res['systems'].items():
        lines.append(f"| {name} | {row['files']} | " + ' | '.join(stat(res['scores'][k]['rows'][name]) for k in SCORES)
                     + ' |')
    lines += ['', '| system | ' + ' | '.join(SHAPE) + ' |', '|---' * (len(SHAPE) + 1) + '|']
    lines += [f'| {n} | ' + ' | '.join(f'{r[k]:.3g}' for k in SHAPE) + ' |' for n, r in res['systems'].items()]
    lines.append('')
    for k in SCORES:
        lines += win_table(LABELS[k], res['scores'][k])
    return '\n'.join(lines)


def score(args):
    prompts = [p for p in json.loads((TASK_DIR / 'prompts.json').read_text())['prompts'] if not p.get('excluded')]
    names = sorted(d.name for d in TASK_DIR.iterdir()
                   if d.is_dir() and d.name not in ('real_all', 'clamp3_tmp', 'clamp3_emb') and next(d.glob('*.mid'), None))
    tables = {n: system_rows(TASK_DIR / n, prompts, prompts) for n in names}
    swap = deranged(len(prompts), args.seed)
    tables['real_shuffled'] = system_rows(TASK_DIR / 'real', prompts, [prompts[j] for j in swap])
    res = dict(date=time.strftime('%Y-%m-%d %H:%M'), prompts=len(prompts), draws=args.draws, seed=args.seed, scores={},
               systems={})
    columns = (*SCORES, *SHAPE)
    for name, t in tables.items():
        res['systems'][name] = dict(files=int((~np.isnan(t[:, len(SCORES)])).sum()),
                                    **{k: float(np.nanmean(t[:, j])) for j, k in enumerate(columns) if j >= len(SCORES)})
    for j, k in enumerate(SCORES):
        res['scores'][k] = bootstrap({n: t[:, j] for n, t in tables.items()}, args.draws, args.seed)
    (TASK_DIR / 'scores.json').write_text(json.dumps(res, indent=1))
    (TASK_DIR / 'scores.md').write_text(report(res))
    print(report(res))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=1234)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('eligible', help='the test rows F1 cannot have seen')
    f = sub.add_parser('fetch', help='their files from midicaps.tar.gz')
    f.add_argument('--archive', required=True)
    r = sub.add_parser('prompts', help='freeze the prompts')
    r.add_argument('--rows', type=int, default=ROWS)
    s = sub.add_parser('score', help='scores.md / scores.json')
    s.add_argument('--draws', type=int, default=10000)
    args = p.parse_args()
    {'eligible': eligible, 'fetch': fetch, 'prompts': export_prompts, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    main()
