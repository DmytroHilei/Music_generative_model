"""
Head-to-head against other open MIDI models (roadmap 8b, context/claude/music_models.md): every system writes MIDI
files for one frozen prompt set, and every file is scored the same way after being read back from the MIDI
(tokenize_tracks), so the tokenizers never enter the comparison: per note and per second, never per token.

    python eval/compare/head2head.py prompts                         # freeze the prompt sets (once)
    python eval/compare/head2head.py ours --system f1 --task cont    # our checkpoints with their generate presets
    python eval/compare/head2head.py ours --system f1 --task text --set length_seconds=30
    python eval/compare/head2head.py score                           # every system folder -> scores.md / scores.json
    python eval/compare/head2head.py prompts --task cont96           # a 96-prompt superset of a published set
    python eval/compare/head2head.py reuse --task cont96             # its p00-p31: hard links to the published files
    python eval/compare/head2head.py score --task cont96 piano96 --out scores_96
    python eval/compare/head2head.py boot --task cont piano          # paired bootstrap of D -> bootstrap_<task>.*

Tasks (results/head2head/<task>/):
  cont  the 32 clean GigaMIDI val prompts of musicar.eval.sample_eval (the same selection: 128 notes, >= 3
        instruments, evenly spaced files), continued by --notes notes. prompts/<id>.mid; real/<id>_0.mid = the prompt +
        the file's real next 1,000 notes, scored like a system ('real' = what good looks like; its scores must match
        musicar/eval/sample_reference.json). The store is read from data/cache, else from the Hub data repo by range
        requests (song_features.HubStore).
  piano the same for solo piano: 32 prompts of aria_validation (our held-out split of Aria-MIDI, which Aria itself
        may have seen in training) with at least 128 + 1,000 notes, so every real continuation is complete, from 64
        evenly spaced files.
  cont96, piano96  the same stores at 96 prompts, supersets of the published sets: p00-p31 are cont's / piano's rows,
        byte for byte, p32-p95 are new. cont96 goes on along cont's grid; piano's grid ends at its 32 rows, so piano96
        adds 64 files evenly spaced over the eligible files left. `reuse` hard-links the published files of the systems
        in CONT_TASKS (prompt i has seed + i in both sets, so they are the same runs) and a run writes p32+ only. Never
        write through such a link: it is the published file. Not part of a bare `prompts` or `score`.
  text  text_eval's prompts (EN + UK pairs and '' = unconditioned) from the start. prompts.json.

A system = a folder <task>/<system>/ of MIDI files <prompt id>_<k>.mid (cont: the prompt + the continuation in one
file; text: the piece). Other models get adapters in their own environments that write the same names. Ours run here
through musicar.generation.piece.Generator with the presets of config/generate/ (SYSTEMS) plus the task's options;
the resolved options go to <system>/config.json. Existing files are skipped, so a run resumes.

Scores (mean over rows):
  cont, piano  sample_eval.row_metrics and consistency_metrics of the first --notes new notes, verbatim copies (share of the
        new pitched notes inside exact copies of >= 16 chords), prompt_ok (share of files that open with the prompt's
        notes, its last onset left out; an adapter check), and D = sample_eval.distance to the real continuations scored here. A file is split
        after the prompt's note count; continuation notes on the prompt's last onset can swap places with prompt notes.
        boot: per task and --notes, D of every system under one resampling of the prompts, the 95% interval of the
        draws, and per system of ours the share of draws closer than each baseline with the interval of the difference
        (bootstrap_<task>.md / .json; BOOT_NOTE: what these intervals are not).
  text  text_eval's follow score F over the EN pairs and over the UK pairs, caption match (caption_match.match of the
        caption's checkable wants), runaway share, copies, notes/s, instruments, top instrument share.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.data.loader import tokenize_tracks  # noqa: E402
from musicar.data.midi_io import tokens_to_midi  # noqa: E402
from musicar.eval import sample_eval  # noqa: E402
from musicar.generation import piece  # noqa: E402
from musicar.generation.caption_match import match, note_attributes, parse_caption  # noqa: E402
from musicar.generation.repetition import copy_share  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
import text_eval  # noqa: E402
from song_features import HubStore  # noqa: E402  (text_eval puts data_prep/labels on the path)

OUT = ROOT / 'results' / 'head2head'
# sample_eval's selection and length (musicar/eval/sample_reference.json was written with rows 32, new_notes 1000)
ROWS, PROMPT_NOTES, REAL_NOTES = 32, 128, 1000
# continuation tasks: store, fewest instruments in the prompt, fewest notes in the file, evenly spaced files tried
# (aria_validation has 3,735 files, so 4,000 would take the first 32 long ones), rows
CONT_TASKS = {'cont': dict(store='gigamidi_clean_validation', min_instruments=3, min_notes=PROMPT_NOTES,
                           candidates=4000, n_rows=ROWS),
              'piano': dict(store='aria_validation', min_instruments=1, min_notes=PROMPT_NOTES + REAL_NOTES,
                            candidates=64, n_rows=ROWS)}
BASE_TASKS = list(CONT_TASKS)  # the published sets: a bare `prompts` and `score` stay on these
# supersets: base = the published set of their first rows, systems = what `reuse` links from it, grid_rows = rows taken
# from the grid, the rest evenly spaced over the eligible files (piano's 64 candidates end at 32 rows)
CONT_TASKS.update(
    cont96=dict(CONT_TASKS['cont'], n_rows=96, base='cont',
                systems=('long_450m', 'f1', 'skytnt_medium', 'skytnt_large', 'amt')),
    piano96=dict(CONT_TASKS['piano'], n_rows=96, grid_rows=ROWS, base='piano',
                 systems=('long_450m', 'ua_2048_v2', 'piano', 'aria')))
COPY_LEN = 16

# system -> generate presets (config/generate/*.toml, exactly as the app / generate.py use them) and options on top:
# 'all' for every task, then the task's own; --set overrides them all
SYSTEMS = {
    'f1': dict(presets=['f1'], text=dict(length_seconds=60)),  # the app's default: f1_cap, guidance 2.5, T 1.1, DRY 3
    'long_450m': dict(presets=[], all=dict(checkpoint='checkpoints/long_450m_final', temperature=0.9,
                                           sample_instruments=True)),  # musicar-L at sample_eval's temperature
    's50': dict(presets=['s50']),  # the 107M caption pilot
    'piano': dict(presets=['piano']),  # the piano-cover fine-tune of the 450M (overfit_adamw8_1e-4 it349)
    'ua_2048_v2': dict(presets=[], all=dict(checkpoint='checkpoints/ua_2048_v2', temperature=0.9)),  # 108M piano
    # text only, own rows: they start from a real 128-note opening picked by the caption (not written to the file)
    'f1_retrieve': dict(presets=['f1'], text=dict(length_seconds=60, retrieve='gigamidi_replay_train')),
    'caption': dict(presets=['caption']),  # multi_control: no text input, the caption picks the opening and ranks
}


def open_store(name):
    """(offsets, read(a, b) -> (tokens (n, 4), programs (n,)) int64) of a token store: data/cache/<name> if present,
    else the Hub data repo by range requests."""
    local = ROOT / 'data' / 'cache' / name
    if (local / 'offsets.npy').exists():
        offsets = np.load(local / 'offsets.npy')
        n = int(offsets[-1])
        tok = np.memmap(local / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
        if not (local / 'programs.u8').exists():  # piano-only stores (aria_*) have no program file
            return offsets, lambda a, b: (np.asarray(tok[a:b], np.int64), np.zeros(b - a, np.int64))
        prog = np.memmap(local / 'programs.u8', dtype=np.uint8, mode='r', shape=(n,))
        return offsets, lambda a, b: (np.asarray(tok[a:b], np.int64), np.asarray(prog[a:b], np.int64))
    hub = HubStore(name, ROOT / 'data' / 'cache' / 'tmp_head2head')
    print(f'{name}: not in data/cache, reading {hub.repo} by range requests')

    def read(a, b):
        tok = np.frombuffer(hub.fetch('tokens.u16', 8 * a, 8 * b, parts=1), np.uint16).reshape(-1, 4)
        prog = np.frombuffer(hub.fetch('programs.u8', a, b, parts=1), np.uint8)
        return tok.astype(np.int64), prog.astype(np.int64)
    return hub.offsets, read


def write_midi(path, tok, prog):
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(*tok.T, programs=prog.tolist()).write(str(path))


def read_midi(path):
    """(tokens (n, 4), programs (n,)) int64 of a MIDI file as the stores tokenize it, or None."""
    res = tokenize_tracks(str(path))
    return None if res is None else (res[0].astype(np.int64), res[1].astype(np.int64))


def note_set(tok, prog):
    """Sorted (onset slot, program, pitch) of every note."""
    return sorted(zip(np.cumsum(tok[:, 3]).tolist(), prog.tolist(), tok[:, 0].tolist()))


def export_prompts(args):
    """cont: sample_eval.fixed_prompts' rows and reference's real continuations as MIDI, piano the same from
    aria_validation; text: text_eval.PROMPTS."""
    for task in [args.task] if args.task in CONT_TASKS else [] if args.task else BASE_TASKS:
        export_cont(task, **CONT_TASKS[task])
    if args.task in (None, 'text'):
        text = [dict(id=text_eval.slug(name), name=name, caption=caption) for name, caption in text_eval.PROMPTS]
        (OUT / 'text').mkdir(parents=True, exist_ok=True)
        (OUT / 'text' / 'prompts.json').write_text(json.dumps(dict(prompts=text), indent=1, ensure_ascii=False))
        print(f'{len(text)} text prompts -> {OUT / "text"}')


def spaced_files(lengths, min_notes, used, n):
    """n files evenly spaced, in store order, over the files with >= min_notes notes that are not in `used`."""
    eligible = np.setdiff1d(np.flatnonzero(np.asarray(lengths) >= min_notes), sorted(used))
    return eligible[np.linspace(0, len(eligible) - 1, n).astype(int)]


def export_cont(task, store, min_instruments, min_notes, candidates, n_rows, grid_rows=None, **_):
    out = OUT / task
    offsets, read = open_store(store)
    rows = []

    def add(i):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if b - a < min_notes:
            return
        tok, prog = read(a, min(b, a + PROMPT_NOTES + REAL_NOTES))
        if len(set(prog[:PROMPT_NOTES].tolist())) < min_instruments:
            return
        pid = f'p{len(rows):02d}'
        write_midi(out / 'prompts' / f'{pid}.mid', tok[:PROMPT_NOTES], prog[:PROMPT_NOTES])
        write_midi(out / 'real' / f'{pid}_0.mid', tok, prog)
        back = read_midi(out / 'prompts' / f'{pid}.mid')
        # the same notes on the same 20-ms slots; the order inside a slot may change (the store sorted the unquantized
        # onsets, the MIDI's onsets are quantized, so ties go by program), which no score depends on
        same = note_set(*back) == note_set(tok[:PROMPT_NOTES], prog[:PROMPT_NOTES])
        rows.append(dict(id=pid, store=store, file=int(i), notes=len(back[0]), real_notes=len(tok) - PROMPT_NOTES,
                         seconds=round(float(back[0][:, 3].sum()) * 0.02, 2), instruments=sorted(set(prog.tolist())),
                         round_trip=same))
        print(f'{pid}: file {i}, {len(tok) - PROMPT_NOTES} real notes, '
              f'{piece.instrument_names(prog[:PROMPT_NOTES])}' + ('' if same else '  (MIDI round trip differs)'))

    for i in np.linspace(0, len(offsets) - 2, candidates).astype(int):  # sample_eval.fixed_prompts' order and filters
        add(i)
        if len(rows) == (grid_rows or n_rows):
            break
    if grid_rows:
        for i in spaced_files(np.diff(offsets), min_notes, {r['file'] for r in rows}, n_rows - grid_rows):
            add(i)
    meta = dict(store=store, rows=n_rows, prompt_notes=PROMPT_NOTES, min_instruments=min_instruments,
                min_notes=min_notes, candidates=candidates, real_notes=REAL_NOTES, prompts=rows)
    (out / 'prompts.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(rows)} continuation prompts -> {out}')


def shared_prompts(task_dir, base_dir):
    """The rows of base_dir's prompts.json, after checking that task_dir starts with the same rows and that their
    prompt and real MIDI files are byte-identical."""
    base = json.loads((base_dir / 'prompts.json').read_text())['prompts']
    rows = json.loads((task_dir / 'prompts.json').read_text())['prompts']
    if rows[:len(base)] != base:
        raise SystemExit(f'{task_dir.name}: its first rows are not those of {base_dir.name}')
    for r in base:
        for name in (f"prompts/{r['id']}.mid", f"real/{r['id']}_0.mid"):
            if (task_dir / name).read_bytes() != (base_dir / name).read_bytes():
                raise SystemExit(f'{task_dir.name}/{name} differs from {base_dir.name}/{name}')
    return base


def reuse(args):
    """The base set's files of a superset task's systems, as hard links: the same prompts at the same seed + i, so
    the same runs. Existing files stay, and a later run skips them. Nothing is written to the base set."""
    spec = CONT_TASKS[args.task]
    task_dir, base_dir = OUT / args.task, OUT / spec['base']
    rows = shared_prompts(task_dir, base_dir)
    print(f"{len(rows)} prompts of {args.task} are {spec['base']}'s, byte for byte")
    for system in spec['systems']:
        (task_dir / system).mkdir(exist_ok=True)
        linked, kept, missing = 0, 0, []
        for r in rows:
            src, dst = base_dir / system / f"{r['id']}_0.mid", task_dir / system / f"{r['id']}_0.mid"
            if dst.exists():
                kept += 1
            elif src.exists():
                os.link(src, dst)
                linked += 1
            else:
                missing.append(r['id'])
        print(f'{system}: {linked} linked, {kept} already there' + (f', missing {missing}' if missing else ''))


def parse_set(items):
    """--set key=value pairs -> dict, values read as JSON where they parse ('2.5', 'true', '"x"'), else strings."""
    out = {}
    for item in items:
        key, _, value = item.partition('=')
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def run_ours(args):
    """One of SYSTEMS on one task: every prompt's --samples files, skipping those already written."""
    spec = SYSTEMS[args.system]
    task_dir = OUT / args.task
    prompts = json.loads((task_dir / 'prompts.json').read_text())['prompts'][:args.prompts or None]
    out = task_dir / (args.name or args.system)
    options = {**spec.get('all', {}), **spec.get(args.task, {}), **parse_set(args.set),
               'num_samples': args.samples, 'no_mp3': True}
    base = piece.settings(spec['presets'], **options)
    gen = piece.Generator(base.checkpoint, args.device, base.dtype)
    out.mkdir(parents=True, exist_ok=True)
    resolved = {k: v for k, v in vars(base).items() if k not in ('prompt', 'text', 'seed')}
    (out / 'config.json').write_text(json.dumps(dict(system=args.system, task=args.task, iter=gen.iter,
                                                     checkpoint=str(gen.checkpoint), options=resolved), indent=1))
    job = JobStatus('head2head', len(prompts), f'{out.name} on {args.task}')
    t0, done = time.time(), 0
    for i, p in enumerate(prompts):
        names = [out / f"{p['id']}_{k}.mid" for k in range(args.samples)]
        if all(n.exists() for n in names):
            continue
        task = (dict(prompt=str(task_dir / 'prompts' / f"{p['id']}.mid"), prompt_notes=p['notes'], keep_prompt=True,
                     max_new_tokens=args.notes) if args.task in CONT_TASKS else dict(text=p['caption'] or None))
        if base.retrieve and not p['caption']:
            unconditioned_retrieve(gen, spec, options, base, names, out, args.seed + i)
            done += 1
            job.update(i + 1, f"{p['id']}: {done} prompts in {time.time() - t0:.0f} s")
            continue
        batch = args.batch or args.samples
        for b in range(0, args.samples, batch):  # the first batch keeps seed + i, as without --batch
            chunk = names[b:b + batch]
            if all(n.exists() for n in chunk):
                continue
            run = piece.settings(spec['presets'], **{**options, 'num_samples': len(chunk)}, **task,
                                 seed=args.seed + i + 1000 * (b // batch))
            for name, pc in zip(chunk, gen.run(run)):
                pc.midi(run.max_polyphony, run.max_delta).write(str(name))
        done += 1
        job.update(i + 1, f"{p['id']}: {done} prompts in {time.time() - t0:.0f} s")
    job.finish(f'{out}')


def unconditioned_retrieve(gen, spec, options, base, names, out, seed):
    """The '' prompt of a --retrieve system (retrieval needs a caption): what a caption that matches every file would
    pick, a random opening of the indexed files (>= retrieve_min_notes notes) of the same store per row, continued
    without text and not written to the file, as retrieved openings are not. The openings go to <system>/openings/."""
    offsets, read = open_store(base.retrieve)
    rng = np.random.default_rng(seed)
    files = rng.choice(np.flatnonzero(np.diff(offsets) >= base.retrieve_min_notes), len(names), replace=False)
    for k, (name, f) in enumerate(zip(names, files)):
        if name.exists():
            continue
        opening = out / 'openings' / name.name
        write_midi(opening, *read(int(offsets[f]), int(offsets[f]) + PROMPT_NOTES))
        run = piece.settings(spec['presets'], **{**options, 'num_samples': 1, 'retrieve': None},
                             prompt=str(opening), prompt_notes=PROMPT_NOTES, seed=seed + 1000 * k)
        pc = gen.run(run)[0]
        pc.midi(run.max_polyphony, run.max_delta).write(str(name))


def cont_rows(system_dir, prompts, notes):
    """Per file: sample_eval's row and consistency metrics of its first `notes` new notes, copies, prompt_ok."""
    rows = []
    for p in prompts:
        prompt = read_midi(system_dir.parent / 'prompts' / f"{p['id']}.mid")
        n0 = len(prompt[0])
        for f in sorted(system_dir.glob(f"{p['id']}_*.mid")):
            res = read_midi(f)
            if res is None or len(res[0]) <= n0:
                continue
            tok, prog = res[0][:n0 + notes], res[1][:n0 + notes]
            new_tok, new_prog = tok[n0:], prog[n0:]
            r = sample_eval.row_metrics(new_tok[:, 0], new_tok[:, 3], new_prog, set(prompt[1].tolist()))
            if r is None:
                continue
            r['ended'] = float(len(new_tok) < notes)  # MIDI has no EOS: fewer notes than asked = it ended
            r.update(sample_eval.consistency_metrics(new_tok[:, 0], new_tok[:, 1], new_tok[:, 3], new_prog,
                                                     prompt[1].tolist()) or {})
            r['copies'] = copy_share(tok[:, 0].tolist(), tok[:, 3], prog, COPY_LEN, n0)
            last = int(prompt[0][:, 3].sum())  # the prompt's last onset slot: continuation notes there sort among it
            r['prompt_ok'] = float([n for n in note_set(tok, prog) if n[0] < last]
                                   == [n for n in note_set(*prompt) if n[0] < last])
            r['new_notes'] = len(new_tok)
            r['id'] = p['id']
            rows.append(r)
    return rows


def text_rows(system_dir, prompts):
    """Per file: text_eval's row metrics, caption match, runaway share, copies, sample_eval's instrument metrics."""
    rows = []
    for p in prompts:
        wants = parse_caption(p['caption']) if p['caption'] else None
        for f in sorted(system_dir.glob(f"{p['id']}_*.mid")):
            res = read_midi(f)
            if res is None:
                continue
            tok, prog = res
            r = dict(prompt=p['name'], **text_eval.row_metrics(tok[:, 0], tok[:, 1], tok[:, 2], tok[:, 3], prog))
            a = note_attributes(tok, prog)
            se = sample_eval.row_metrics(tok[:, 0], tok[:, 3], prog, set()) or {}
            r.update(match=match(wants, a)[0] if wants is not None else np.nan, runaway=a['runaway'],
                     seconds=a['seconds'], copies=copy_share(tok[:, 0].tolist(), tok[:, 3], prog, COPY_LEN),
                     instruments=se.get('instruments', np.nan), top_share=se.get('top_share', np.nan))
            rows.append(r)
    return rows


def mean(rows, keys):
    return {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) if rows else np.nan for k in keys}


CONT_KEYS = ('instruments', 'out_of_band', 'notes_per_s', 'top_share', 'runaway', 'takeover', 'pc_entropy',
             'same_onset', 'ended', 'density_var', 'density_drift', 'mix_change', 'prompt_kept', 'copies', 'prompt_ok')
TEXT_KEYS = ('match', 'runaway', 'copies', 'nps', 'instruments', 'top_share', 'seconds')
PUBLISHED = ('scores', 'scores_first200')  # the output names `score --out` refuses
BOOT_NOTE = ('D is a sum of absolute gaps: it stops at 0 and a gap near 0 can only grow under resampling, so the draws '
             'lean above D (mean draw - D) and D need not be central in their 95%. That range is the spread over the '
             'prompts, not a confidence interval for the true D, and its low end is no lower bound. A difference of '
             'two Ds keeps what the two inflations do not cancel (mean draw - difference), so read it with that '
             'column; the share of draws closer is no p-value. Only the files this prompt set drew from are covered.')


def score_name(tasks, out):
    """The output name of a `score` run: scores with neither --task nor --out, else --out, which may not be a published
    name: a filtered run would leave the other tasks out of them."""
    if out in PUBLISHED or (tasks and not out):
        raise SystemExit(f"score: --task needs --out, not one of {PUBLISHED}")
    return out or 'scores'


def score(args):
    stem = score_name(args.task, args.out)
    tasks = args.task or [*BASE_TASKS, 'text']
    results = {}
    for task in CONT_TASKS:
        cont_dir = OUT / task
        if task not in tasks or not (cont_dir / 'prompts.json').exists():
            continue
        prompts = json.loads((cont_dir / 'prompts.json').read_text())['prompts']
        by_system = {d.name: cont_rows(d, prompts, args.notes) for d in sorted(cont_dir.iterdir())
                     if d.is_dir() and d.name != 'prompts'}
        real = by_system.get('real')
        ref = None
        if real:
            ref = mean(real, sample_eval.DIST_WEIGHTS)
            ref['sd'] = {k: float(np.std([r[k] for r in real], ddof=1)) for k in sample_eval.DIST_WEIGHTS}
        results[task] = {}
        for name, rows in by_system.items():
            m = dict(mean(rows, CONT_KEYS), rows=len(rows))
            m['D'] = sample_eval.distance(m, ref) if ref and rows else np.nan
            results[task][name] = m
    text_dir = OUT / 'text'
    if 'text' in tasks and (text_dir / 'prompts.json').exists():
        prompts = json.loads((text_dir / 'prompts.json').read_text())['prompts']
        results['text'] = {}
        for d in sorted(x for x in text_dir.iterdir() if x.is_dir()):
            rows = text_rows(d, prompts)
            contrasts = text_eval.pair_contrasts(text_eval.prompt_means(rows))
            m = dict(mean(rows, TEXT_KEYS), rows=len(rows))
            m['F_en'] = text_eval.follow_score({s: c for s, c in contrasts.items() if not s.startswith('uk_')})
            m['F_uk'] = text_eval.follow_score({s: c for s, c in contrasts.items() if s.startswith('uk_')})
            results['text'][d.name] = m
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{stem}.json').write_text(json.dumps(results, indent=1))
    lines = [f'# Head-to-head scores ({time.strftime("%Y-%m-%d %H:%M")}, first {args.notes} new notes per '
             'continuation)', '']
    for task, systems in results.items():
        if not systems:
            continue
        keys = list(next(iter(systems.values())))
        lines += [f'## {task}', '', '| system | ' + ' | '.join(keys) + ' |', '|---' * (len(keys) + 1) + '|']
        lines += [f'| {name} | ' + ' | '.join(f'{m[k]:.3g}' for k in keys) + ' |' for name, m in systems.items()]
        lines.append('')
    (OUT / f'{stem}.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def prompt_table(rows):
    """{prompt id: the DIST_WEIGHTS metrics, mean over its files} of cont_rows' rows."""
    by_id = {}
    for r in rows:
        by_id.setdefault(r['id'], []).append([r[k] for k in sample_eval.DIST_WEIGHTS])
    return {i: np.mean(v, 0) for i, v in by_id.items()}


def bootstrap(tables, draws, seed):
    """Paired prompt-level bootstrap of D. tables = {system: prompt_table}, with 'real'; rows meet by prompt id, at the
    ids every system has. One draw = one resampling of those ids for every system and for real (ref = the resampled
    real rows' means); sd stays that of all real rows. -> ids, {system: D of the rows as they are}, {system: D per
    draw}; 'real' is left out of both."""
    keys = list(sample_eval.DIST_WEIGHTS)
    ids = sorted(set.intersection(*map(set, tables.values())))
    rows = {s: np.array([t[i] for i in ids]) for s, t in tables.items()}
    sd = dict(zip(keys, np.array(list(tables['real'].values())).std(0, ddof=1).tolist()))
    real = rows.pop('real')
    point = {s: sample_eval.distance(dict(zip(keys, v.mean(0).tolist())), dict(zip(keys, real.mean(0).tolist()), sd=sd))
             for s, v in rows.items()}
    idx = np.random.default_rng(seed).integers(0, len(ids), (draws, len(ids)))
    refs = [dict(zip(keys, r), sd=sd) for r in real[idx].mean(1).tolist()]
    per_draw = {s: np.array([sample_eval.distance(dict(zip(keys, m)), ref)
                             for m, ref in zip(v[idx].mean(1).tolist(), refs)]) for s, v in rows.items()}
    return ids, point, per_draw


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def boot_table(point, per_draw, ours):
    """(system rows, pair rows) of one bootstrap: pairs = each of `ours` against each other system."""
    systems = {s: dict(D=point[s], low=interval(d)[0], high=interval(d)[1], bias=float(d.mean() - point[s]))
               for s, d in per_draw.items()}
    pairs = {}
    for a in ours:
        for b in (s for s in per_draw if s not in ours):
            diff = per_draw[a] - per_draw[b]
            pairs[f'{a} vs {b}'] = dict(diff=point[a] - point[b], low=interval(diff)[0], high=interval(diff)[1],
                                        bias=float(diff.mean() - (point[a] - point[b])),
                                        closer=float((diff < 0).mean()))
    return systems, pairs


def boot(args):
    for task in args.task:
        task_dir = OUT / task
        prompts = json.loads((task_dir / 'prompts.json').read_text())['prompts']
        names = [d.name for d in sorted(task_dir.iterdir()) if d.is_dir() and d.name not in ('prompts', 'real')]
        res = dict(task=task, draws=args.draws, seed=args.seed, limitation=BOOT_NOTE, notes={})
        lines = [f'# Head-to-head bootstrap of D: {task} ({time.strftime("%Y-%m-%d %H:%M")}, {args.draws:,} draws, '
                 f'seed {args.seed})', '', BOOT_NOTE, '']
        for notes in args.notes:
            tables = {s: prompt_table(cont_rows(task_dir / s, prompts, notes)) for s in [*names, 'real']}
            ids, point, per_draw = bootstrap(tables, args.draws, args.seed)
            systems, pairs = boot_table(point, per_draw, [s for s in names if s in SYSTEMS])
            res['notes'][notes] = dict(prompts=len(ids), of=len(prompts), systems=systems, pairs=pairs)
            lines += [f'## First {notes} new notes, {len(ids)} of {len(prompts)} prompts', '',
                      '| system | D | 95% of draws | mean draw - D |', '|---|---|---|---|']
            lines += [f"| {s} | {m['D']:.3f} | {m['low']:.3f} to {m['high']:.3f} | {m['bias']:+.3f} |"
                      for s, m in systems.items()]
            lines += ['', '| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |',
                      '|---|---|---|---|---|']
            lines += [f"| {p} | {m['diff']:+.3f} | {m['low']:+.3f} to {m['high']:+.3f} | {m['bias']:+.3f} | "
                      f"{m['closer']:.3f} |" for p, m in pairs.items()]
            lines.append('')
        (OUT / f'bootstrap_{task}.json').write_text(json.dumps(res, indent=1))
        (OUT / f'bootstrap_{task}.md').write_text('\n'.join(lines))
        print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pr = sub.add_parser('prompts', help='freeze the prompt sets')
    pr.add_argument('--task', choices=[*CONT_TASKS, 'text'], default=None, help='one task only (default: all)')
    o = sub.add_parser('ours', help='generate with one of SYSTEMS')
    o.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    o.add_argument('--task', choices=[*CONT_TASKS, 'text'], required=True)
    o.add_argument('--name', default=None, help='output folder name (default: the system)')
    o.add_argument('--prompts', type=int, default=0, help='> 0: only the first N prompts (smoke tests)')
    o.add_argument('--samples', type=int, default=1, help='files per prompt')
    o.add_argument('--batch', type=int, default=0, help='rows per generate call (default: all --samples at once; '
                   'f1 text at 16 rows x guidance x block 2048 needs > 7.5 GB)')
    o.add_argument('--notes', type=int, default=REAL_NOTES, help='cont / piano: new notes per file')
    o.add_argument('--seed', type=int, default=1234, help='prompt i gets seed + i')
    o.add_argument('--device', default='auto')
    o.add_argument('--set', action='append', default=[], help='generate option override, key=value (repeatable)')
    s = sub.add_parser('score', help='score every system folder')
    s.add_argument('--notes', type=int, default=REAL_NOTES, help='cont: new notes scored per file')
    s.add_argument('--task', nargs='+', choices=[*CONT_TASKS, 'text'], default=None,
                   help=f'these tasks only (default: {", ".join(BASE_TASKS)}, text); needs --out')
    s.add_argument('--out', default=None, help='writes <out>.md / <out>.json (default: scores)')
    r = sub.add_parser('reuse', help='hard-link the published files into a superset task')
    r.add_argument('--task', choices=[t for t, spec in CONT_TASKS.items() if 'base' in spec], required=True)
    b = sub.add_parser('boot', help='paired prompt-level bootstrap of D')
    b.add_argument('--task', nargs='+', choices=list(CONT_TASKS), required=True)
    b.add_argument('--notes', type=int, nargs='+', default=[REAL_NOTES, 200], help='new notes scored per file')
    b.add_argument('--draws', type=int, default=10000)
    b.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    {'prompts': export_prompts, 'ours': run_ours, 'score': score, 'reuse': reuse, 'boot': boot}[args.cmd](args)


if __name__ == '__main__':
    main()
