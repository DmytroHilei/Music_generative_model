"""
Augmentation search on a repeated-data run (census doc "Data augmentation ideas", hypotheses/aug_*): 42M with the
scale ladder's recipe (config/pretrain/scale_ladder.py) on a fixed quarter of the GigaMIDI replay store's files
(--train_file_frac=0.25: 25,383 files, 41.5M notes), 4,883 iterations = 160M notes seen = ~3.9 passes. The arm
'fresh' is the fresh-data reference on the same machine: the whole store (167.9M notes) for the same iterations,
under one pass (the laptop's scale ladder 42M_160M run: CE 2.426). Repeat penalty = control - fresh; an arm's "won
back" = the share of it the arm recovers.

control and fresh run first (every seed). If the repeat penalty is under --min-gap, the queue stops: there is nothing
to win back at this many passes, so the arms could not separate (then --frac 0.125 = ~8 passes, or no search). Arms
then run seed by seed, so a stopped queue still compares every arm on the first seed. A failed run is logged, the
queue goes on, and the next start tries it again. Each run's evaluations go to results/aug_sweep/runs.json and the
table to logs/aug_sweep.md; its checkpoint directory is deleted. --micro 16 = the same 32,768 notes per step as 16 x 1
(a 5090) instead of the config's 4 x 4; eval_iters follows, so validation keeps the same 1,200 windows.

    .venv/bin/python tools/sweeps/aug_sweep.py --list
    .venv/bin/python tools/sweeps/aug_sweep.py                 # control + fresh, the gate, then every arm missing
    .venv/bin/python tools/sweeps/aug_sweep.py --smoke         # 40 iterations of every arm, nothing recorded
    .venv/bin/python tools/sweeps/aug_sweep.py --only family   # arms whose name contains the text
    .venv/bin/python tools/sweeps/aug_sweep.py --micro 16      # on the 5090 (cloud/run/aug_all.sh)
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from scale_ladder import CONFIG, NOTES_PER_STEP, RUNS as LADDER_RUNS, SIZES, parse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/aug_sweep'
RUNS = OUT / 'runs.json'
LOGS = ROOT / 'logs/aug_sweep'
TABLE = ROOT / 'logs/aug_sweep.md'
SIZE = '42M'
ITERS = round(160e6 / NOTES_PER_STEP)  # 4,883, the ladder's 160M budget
ROWS, VAL_WINDOWS = 16, 1200  # rows per step (config: 4 x 4) and validation windows (300 x 4)
SEEDS = (1337, 2)
REF = 'fresh'
ARMS = {  # census doc idea numbers
    REF: [],  # the whole store (file_frac 1): the fresh-data reference
    'control': [],
    'tempo10': ['--aug_tempo=0.1'],  # idea 2: the fine-tune setting, off in pretraining so far
    'tempo20': ['--aug_tempo=0.2'],  # idea 2: the wider range
    'dyn': ['--hypothesis=aug_dyn:shift=4'],  # idea 4: one velocity shift per window
    'octave': ['--hypothesis=aug_octave:rows=0.5'],  # idea 7: one track of a multi-track window moves +-12
    'family': ['--hypothesis=aug_family:rows=0.5'],  # idea 6: programs permuted within their GM family
    'keys12': ['--aug_transpose=-5,6'],  # all 12 keys instead of +-5
}


def load():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def save(name, rec):
    runs = load()
    runs[name] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RUNS.with_suffix('.tmp')
    tmp.write_text(json.dumps(runs, indent=1))
    tmp.replace(RUNS)


def run_name(arm, seed, frac):
    return f'{arm}_s{seed}' + ('' if frac == 0.25 or arm == REF else f'_f{frac:g}')


def micro_args(micro, eval_iters=None):
    assert ROWS % micro == 0 and VAL_WINDOWS % micro == 0, micro
    return [f'--batch_size={micro}', f'--gradient_accumulation_steps={ROWS // micro}',
            f'--eval_iters={eval_iters or VAL_WINDOWS // micro}']


def train(arm, seed, file_frac, iters, out, log, extra=()):
    """One train.py run -> its exit code; the log holds everything. The checkpoint directory is deleted."""
    LOGS.mkdir(parents=True, exist_ok=True)
    print(time.strftime('%H:%M'), log.name, flush=True)
    cmd = ['.venv/bin/python', 'train.py', CONFIG, *[f'--{k}={v}' for k, v in SIZES[SIZE].items()],
           f'--train_file_frac={file_frac}', f'--seed={seed}', f'--max_iters={iters}', f'--lr_decay_iters={iters}',
           f'--out_dir={out.relative_to(ROOT)}', *ARMS[arm], *extra]
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    shutil.rmtree(out, ignore_errors=True)
    return code


def unique_notes(text):
    m = re.search(r'file_frac \S+ keeps [\d,]+ files, ([\d,]+) notes', text)
    return int(m.group(1).replace(',', '')) if m else None


def run(arm, seed, frac, micro):
    name = run_name(arm, seed, frac)
    file_frac = 1.0 if arm == REF else frac
    log = LOGS / f'{name}.log'
    t0 = time.time()
    code = train(arm, seed, file_frac, ITERS, ROOT / 'checkpoints' / f'aug_{name}', log,
                 [*micro_args(micro), f'--eval_interval={ITERS // 4}'])
    evals, params = parse(log)
    if code or not evals or evals[-1]['step'] < ITERS - 1:
        print(f'  failed (exit {code}), see {log}', flush=True)
        save(name, dict(arm=arm, seed=seed, frac=file_frac, micro=micro, failed=True, code=code))
        return
    save(name, dict(arm=arm, seed=seed, frac=file_frac, micro=micro, params=params, notes=ITERS * NOTES_PER_STEP,
                    iters=ITERS, unique_notes=unique_notes(log.read_text(errors='replace')),
                    minutes=round((time.time() - t0) / 60, 1), curve=[(e['step'], e['ce']) for e in evals],
                    **evals[-1]))
    print(f'  CE {evals[-1]["ce"]:.3f}', flush=True)


def finished(runs, arm, frac):
    """{seed: record} of the finished runs of an arm."""
    return {s: r for s in SEEDS if (r := runs.get(run_name(arm, s, frac))) and not r.get('failed')}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def penalty(runs, frac):
    """control mean - fresh mean, once both finished every seed, else None."""
    ctrl, ref = finished(runs, 'control', frac), finished(runs, REF, frac)
    if len(ctrl) < len(SEEDS) or len(ref) < len(SEEDS):
        return None
    return mean(r['ce'] for r in ctrl.values()) - mean(r['ce'] for r in ref.values())


def table(frac, note=''):
    runs = load()
    ctrl = finished(runs, 'control', frac)
    gap = penalty(runs, frac)
    unique = next((r['unique_notes'] for r in runs.values() if r.get('frac') == frac and r.get('unique_notes')), None)
    passes = f' (~{ITERS * NOTES_PER_STEP / unique:.1f} passes over {unique / 1e6:.1f}M notes)' if unique else ''
    micro = next((r['micro'] for r in runs.values() if r.get('micro')), None)
    try:
        ladder = f' The laptop\'s scale ladder 42M_160M (micro 4): CE {json.loads(LADDER_RUNS.read_text())["42M_160M"]["ce"]:.3f}.'
    except (OSError, KeyError, ValueError):
        ladder = ''
    fmt = lambda v, f: '' if v is None else format(v, f)  # noqa: E731
    lines = [f'# Augmentation search (tools/sweeps/aug_sweep.py): {SIZE}, {frac:g} of the GigaMIDI replay files, '
             f'{ITERS:,} iterations = 160M notes seen{passes}, micro-batch {micro}', '',
             f'{REF} = the whole store under one pass. Repeat penalty = control mean - {REF} mean = '
             f'{fmt(gap, "+.3f") or "(unfinished)"}.{ladder}',
             '', '| arm | ' + ' | '.join(f's{s}' for s in SEEDS) + ' | mean | vs control | won back | prog CE |',
             '|---' * (len(SEEDS) + 5) + '|']
    for arm in ARMS:
        done = finished(runs, arm, frac)
        if not any(run_name(arm, s, frac) in runs for s in SEEDS):
            continue
        cells = [fmt(done[s]['ce'], '.3f') if s in done else 'failed' if run_name(arm, s, frac) in runs else ''
                 for s in SEEDS]
        common = [s for s in done if s in ctrl]
        vs = mean(done[s]['ce'] - ctrl[s]['ce'] for s in common) if arm not in ('control', REF) else None
        won = -vs / gap if vs is not None and gap and gap > 0 else None
        row = [arm, *cells, fmt(mean(r['ce'] for r in done.values()), '.3f'), fmt(vs, '+.3f'), fmt(won, '.0%'),
               fmt(mean(r.get('prog') for r in done.values()), '.3f')]
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', 'vs control: mean difference over the seeds both finished (negative = better). won back = '
              '-(vs control) / repeat penalty.'] + (['', note] if note else [])
    TABLE.write_text('\n'.join(lines) + '\n')


def smoke(arms, frac, micro):
    """40 iterations of each arm with fp8 + compile: the hook loads, training runs, the eval is finite."""
    for arm in arms:
        if arm == REF:  # the plain config on the whole store: control covers it
            continue
        log = LOGS / f'smoke_{arm}.log'
        code = train(arm, SEEDS[0], frac, 40, ROOT / 'checkpoints' / f'aug_smoke_{arm}', log,
                     [*micro_args(micro, eval_iters=20), '--warmup_iters=5', '--eval_interval=40'])
        evals, _ = parse(log)
        text = log.read_text(errors='replace')
        hyp = next((ln for ln in text.splitlines() if ln.startswith('hypothesis:')), '')
        wants_hyp = any(x.startswith('--hypothesis') for x in ARMS[arm])
        ok = (code == 0 and evals and evals[-1]['step'] >= 39 and unique_notes(text) is not None
              and bool(hyp) == wants_hyp)
        print(f'  {"ok" if ok else "FAILED"}  exit {code}  CE {evals[-1]["ce"] if evals else "-"}  {hyp}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', action='store_true')
    p.add_argument('--only', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--frac', type=float, default=0.25)
    p.add_argument('--micro', type=int, default=4, help='rows per micro-batch (16 / micro accumulation steps)')
    p.add_argument('--min-gap', type=float, default=0.01)
    p.add_argument('--force', action='store_true', help='run the arms even if the gate fails')
    a = p.parse_args()
    micro_args(a.micro)  # checks it divides 16 and 1,200
    arms = [arm for arm in ARMS if a.only in arm]
    if a.smoke:
        return smoke(arms, a.frac, a.micro)
    if a.list:
        runs = load()
        for seed in SEEDS:
            for arm in arms:
                r = runs.get(run_name(arm, seed, a.frac))
                print(f'{run_name(arm, seed, a.frac):20s}', '-' if not r else 'failed' if r.get('failed') else
                      f'CE {r["ce"]:.3f}  prog {r["prog"]:.3f}  {r["minutes"]} min')
        return

    def todo(arm, seed):  # missing or failed (a failed run is tried again on the next start)
        r = load().get(run_name(arm, seed, a.frac))
        return arm in arms and (r is None or r.get('failed', False))

    for seed in SEEDS:
        for arm in ('control', REF):
            if todo(arm, seed):
                run(arm, seed, a.frac, a.micro)
                table(a.frac)
    gap = penalty(load(), a.frac)
    if gap is None or (gap < a.min_gap and not a.force):
        note = f'Stopped at the gate: repeat penalty {gap if gap is None else round(gap, 3)}, needed {a.min_gap}.'
        table(a.frac, note)
        print(note)
        return
    print(f'gate passed: repeat penalty {gap:+.3f}', flush=True)
    for seed in SEEDS:
        for arm in ARMS:
            if arm not in ('control', REF) and todo(arm, seed):
                run(arm, seed, a.frac, a.micro)
                table(a.frac)


if __name__ == '__main__':
    main()
