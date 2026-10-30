"""
Head-to-head against other open MIDI models (roadmap 8b, context/claude/music_models.md): every system writes MIDI
files for one frozen prompt set, and every file is scored the same way after being read back from the MIDI
(tokenize_tracks), so the tokenizers never enter the comparison: per note and per second, never per token.

    python eval/compare/head2head.py prompts                         # freeze the prompt sets (once)
    python eval/compare/head2head.py ours --system f1 --task cont    # our checkpoints with their generate presets
    python eval/compare/head2head.py ours --system f1 --task text --set length_seconds=30
    python eval/compare/head2head.py score                           # every system folder -> scores.md / scores.json

Tasks (results/head2head/<task>/):
  cont  the 32 clean GigaMIDI val prompts of musicar.eval.sample_eval (the same selection: 128 notes, >= 3
        instruments, evenly spaced files), continued by --notes notes. prompts/<id>.mid; real/<id>_0.mid = the prompt +
        the file's real next 1,000 notes, scored like a system ('real' = what good looks like; its scores must match
        musicar/eval/sample_reference.json). The store is read from data/cache, else from the Hub data repo by range
        requests (song_features.HubStore).
  piano the same for solo piano: 32 prompts of aria_validation (our held-out split of Aria-MIDI, which Aria itself
        may have seen in training) with at least 128 + 1,000 notes, so every real continuation is complete, from 64
        evenly spaced files.
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
  text  text_eval's follow score F over the EN pairs and over the UK pairs, caption match (caption_match.match of the
        caption's checkable wants), runaway share, copies, notes/s, instruments, top instrument share.
"""

import argparse
import json
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
# (aria_validation has 3,735 files, so 4,000 would take the first 32 long ones)
CONT_TASKS = {'cont': dict(store='gigamidi_clean_validation', min_instruments=3, min_notes=PROMPT_NOTES,
                           candidates=4000),
              'piano': dict(store='aria_validation', min_instruments=1, min_notes=PROMPT_NOTES + REAL_NOTES,
                            candidates=64)}
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
    for task in [args.task] if args.task in CONT_TASKS else [] if args.task else CONT_TASKS:
        export_cont(task, **CONT_TASKS[task])
    if args.task in (None, 'text'):
        text = [dict(id=text_eval.slug(name), name=name, caption=caption) for name, caption in text_eval.PROMPTS]
        (OUT / 'text').mkdir(parents=True, exist_ok=True)
        (OUT / 'text' / 'prompts.json').write_text(json.dumps(dict(prompts=text), indent=1, ensure_ascii=False))
        print(f'{len(text)} text prompts -> {OUT / "text"}')


def export_cont(task, store, min_instruments, min_notes, candidates):
    out = OUT / task
    offsets, read = open_store(store)
    rows = []
    for i in np.linspace(0, len(offsets) - 2, candidates).astype(int):  # sample_eval.fixed_prompts' order and filters
        a, b = int(offsets[i]), int(offsets[i + 1])
        if b - a < min_notes:
            continue
        tok, prog = read(a, min(b, a + PROMPT_NOTES + REAL_NOTES))
        if len(set(prog[:PROMPT_NOTES].tolist())) < min_instruments:
            continue
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
        if len(rows) == ROWS:
            break
    meta = dict(store=store, rows=ROWS, prompt_notes=PROMPT_NOTES, min_instruments=min_instruments,
                min_notes=min_notes, candidates=candidates, real_notes=REAL_NOTES, prompts=rows)
    (out / 'prompts.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(rows)} continuation prompts -> {out}')


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


def score(args):
    results = {}
    for task in CONT_TASKS:
        cont_dir = OUT / task
        if not (cont_dir / 'prompts.json').exists():
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
    if (text_dir / 'prompts.json').exists():
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
    (OUT / 'scores.json').write_text(json.dumps(results, indent=1))
    lines = [f'# Head-to-head scores ({time.strftime("%Y-%m-%d %H:%M")}, first {args.notes} new notes per '
             'continuation)', '']
    for task, systems in results.items():
        if not systems:
            continue
        keys = list(next(iter(systems.values())))
        lines += [f'## {task}', '', '| system | ' + ' | '.join(keys) + ' |', '|---' * (len(keys) + 1) + '|']
        lines += [f'| {name} | ' + ' | '.join(f'{m[k]:.3g}' for k in keys) + ' |' for name, m in systems.items()]
        lines.append('')
    (OUT / 'scores.md').write_text('\n'.join(lines))
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
    args = p.parse_args()
    {'prompts': export_prompts, 'ours': run_ours, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    main()
