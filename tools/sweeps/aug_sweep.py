"""
Augmentation search on a repeated-data run (census doc "Data augmentation ideas", hypotheses/aug_*): 42M with the
scale ladder's recipe (config/pretrain/scale_ladder.py) on a fixed quarter of the GigaMIDI replay store's files
(--train_file_frac=0.25: 25,383 files, 41.5M notes), 4,883 iterations = 160M notes seen = ~3.9 passes. The arm
'fresh' is the fresh-data reference on the same machine: the whole store (167.9M notes) for the same iterations,
under one pass (the laptop's scale ladder 42M_160M run: CE 2.426). Repeat penalty = control - fresh; an arm's "won
back" = the share of it the arm recovers (over 100% = better than fresh data).
'synth25' takes a quarter of its windows from musicar-L's continuations of real openings
(data_prep/datasets/synth_store.py -> data/cache/synth_musicar_l_train; prompts from the same quarter of the replay
files), the rest from the replay quarter as in control. --size 108M runs the same arms on the ladder's 108M: with
42M it shows whether what musicar-L's samples add shrinks as the student nears the teacher's size.

control and fresh run first (every seed of --seeds). If the repeat penalty (seeds 1337 and 2) is under --min-gap, the
queue stops (--force runs the arms anyway): there is nothing to win back at this many passes, so the arms could not
separate (then --frac 0.125 = ~8 passes, or no search). Arms then run seed by seed, so a stopped queue still compares
every arm on the first seed. A failed run is logged, the queue goes on, and the next start tries it again; a run past
2.2x its expected 5090 time (MS_PER_STEP, --timeout-mult) is killed and recorded as failed. Each run's evaluations go
to results/aug_sweep/runs.json and the table to logs/aug_sweep.md (table_path: logs/aug_sweep_<size>_f<frac>_n<M>M.md
for the parts that differ from 42M / 0.25 / 160M); its checkpoint directory is deleted (--keep: kept under
checkpoints/keep/, with --always_save_checkpoint=True and --save-iters). --micro 16 = the same 32,768 notes per step
as 16 x 1 (a 5090) instead of the config's 4 x 4; eval_iters follows, so validation keeps the same 1,200 windows.

Arm settings beyond flags (OPTS): a fixed train_file_frac ('frac': recorded as the run's frac, no _f suffix in the
name), its own notes seen ('notes'), rows per step ('rows'; iterations = notes / (rows x 2,048), micro-batch
min(--micro, rows)), 'warmup' and 'eval' (interval; default iterations / 4), 'ref' (the arm it is compared with) and
'twin' (fresh_X: the repeated arm X; P_X = X - fresh_X). fresh_* arms run on the whole replay store (frac 1) and are
compared with fresh. Families with their own control/fresh pair: Discover (disc_*, census Discover stores and PANEL)
and Aria (aria_*). --arm-json adds arms from a JSON file {name: [flags] | {"args": [flags], "frac", "notes", "rows",
"warmup", "eval", "ref", "twin", "synth": [[stores], share]}}; a fresh_<X> name defaults to frac 1, ref fresh, twin X.

    .venv/bin/python tools/sweeps/aug_sweep.py --list
    .venv/bin/python tools/sweeps/aug_sweep.py                 # control + fresh, the gate, then every arm missing
    .venv/bin/python tools/sweeps/aug_sweep.py --smoke         # 40 iterations of every arm, nothing recorded
    .venv/bin/python tools/sweeps/aug_sweep.py --only family   # arms whose name contains the text
    .venv/bin/python tools/sweeps/aug_sweep.py --micro 16      # on the 5090 (cloud/run/aug_all.sh)
    .venv/bin/python tools/sweeps/aug_sweep.py --size 108M --arms fresh,control,synth25 --force --micro 16
    .venv/bin/python tools/sweeps/aug_sweep.py --size 217M --arms fresh,control --seeds 1337 --force --keep --micro 16
    .venv/bin/python tools/sweeps/aug_sweep.py --size 6M --notes 320e6 --arms control,fresh_census --seeds 1337 --force
    .venv/bin/python tools/sweeps/aug_sweep.py --arm-json logs/reg_arms.json --arms BEST,fresh_BEST --micro 16
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from scale_ladder import CONFIG, NOTES_PER_STEP, RUNS as LADDER_RUNS, SIZES as LADDER_SIZES, parse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/aug_sweep'
RUNS = OUT / 'runs.json'
LOGS = ROOT / 'logs/aug_sweep'
NOTES = 160e6  # the ladder's 160M budget
ITERS = round(NOTES / NOTES_PER_STEP)  # 4,883
ROWS, VAL_WINDOWS = 16, 1200  # rows per step (config: 4 x 4) and validation windows (300 x 4)
BLOCK = NOTES_PER_STEP // ROWS  # 2,048 notes per row
SEEDS = (1337, 2)  # the default --seeds and the gate's seeds
REF = 'fresh'
SIZES = {**LADDER_SIZES,  # head size 64
         '3M': dict(n_layer=4, n_embd=192, n_head=3),
         '217M': dict(n_layer=16, n_embd=1024, n_head=16)}
MS_PER_STEP = {'3M': 25, '6M': 35, '20M': 65, '42M': 95, '108M': 189, '217M': 350}  # 5090, 32,768 notes per step
BIG = ('108M', '217M')  # 1.5 min start-up overhead instead of 0.7
REPLAY = 'data/cache/gigamidi_replay'
SYNTH_STORE = 'data/cache/synth_musicar_l'  # data_prep/datasets/synth_store.py --out
SYNTH = {  # arm: ([stores], share of the windows from them, split evenly), the rest from the replay store
    'synth25': ([SYNTH_STORE], 0.25),
    'synth_clean25': (['data/cache/synth_clean'], 0.25),  # generated by the 42M member control_s3
}
POOL = 'data/cache/gigamidi_pool'  # census GigaMIDI pool (data_prep/datasets/census_pool.py), 1.897B notes
DISC_T1, DISC_VN, DISC_VA, DISC_VD = (f'data/cache/cdisc_{x}' for x in ('t1', 'vn', 'va', 'vd'))  # discover_sets.py
ARIA = 'data/cache/aria'
HELDOUT = 'data/cache/gigamidi_heldout'
PANEL = f'--val_extra=disc_aria=store:{DISC_VA};disc_dense=store:{DISC_VD};giga=store:{HELDOUT}'
DISC = [f'--csv_path=store:{DISC_T1}', f'--val_csv_path=store:{DISC_VN}', PANEL]
ARIA_ARGS = [f'--csv_path=store:{ARIA}', f'--val_csv_path=store:{ARIA}',
             f'--val_extra=disc=store:{DISC_VN};giga=store:{HELDOUT}']
CENSUS = [f'--csv_path=store:{POOL}', '--boundary_frac=0']
MEMBERS = ','.join(f'checkpoints/members/control_s{s}' for s in (3, 4, 5, 6))


def synth_args(stores, share):
    return ['--csv_path=' + ','.join([f'store:{REPLAY}'] + [f'store:{s}' for s in stores]),
            '--source_weights=' + ','.join([f'{1 - share:g}'] + [f'{share / len(stores):g}'] * len(stores))]


ARMS = {  # census doc idea numbers; plan ids from the 2026-11-20 scaling plan (§3)
    REF: [],  # the whole store (file_frac 1): the fresh-data reference
    'control': [],
    'tempo10': ['--aug_tempo=0.1'],  # idea 2: the fine-tune setting, off in pretraining so far
    'tempo20': ['--aug_tempo=0.2'],  # idea 2: the wider range
    'dyn': ['--hypothesis=aug_dyn:shift=4'],  # idea 4: one velocity shift per window
    'octave': ['--hypothesis=aug_octave:rows=0.5'],  # idea 7: one track of a multi-track window moves +-12
    'family': ['--hypothesis=aug_family:rows=0.5'],  # idea 6: programs permuted within their GM family
    'keys12': ['--aug_transpose=-5,6'],  # all 12 keys instead of +-5
    **{arm: synth_args(*SYNTH[arm]) for arm in ('synth25',)},  # idea 12: musicar-L's samples
    # D: regularization (weight decay 0.1 and dropout 0 in the config), LR, batch
    'wd3x': ['--weight_decay=0.3'],  # D-E1
    'wd10x': ['--weight_decay=1.0'],  # D-E2
    'wd30x': ['--weight_decay=3.0'],  # D-E3
    'wd60x': ['--weight_decay=6.0'],  # D-O2
    'wd06': ['--weight_decay=0.6'],  # D-L5 (6M)
    'drop10': ['--dropout=0.1'],  # D-E4
    'drop20': ['--dropout=0.2'],  # D-O1
    'drop10_late': ['--hypothesis=drop_late:p=0.1:start=1266'],  # D-R2: dropout from 26% of the run on
    'wd30x_trunk': ['--hypothesis=wd_split:muon=3.0'],  # D-R3: Muon's decay only, AdamW stays at 0.1
    'embwd0': ['--hypothesis=wd_split:emb=0.0'],  # D-O4: no decay on the embeddings
    'lr2e3': ['--learning_rate=2e-3'],  # A-L1, D-L4
    'lr05': ['--learning_rate=5e-4'],  # A-L3
    'b16k': [],  # D-R10: 8 rows per step (OPTS)
    'b16k_lr7': ['--learning_rate=7.07e-4'],  # D-R11
    'b16k_wd05': ['--weight_decay=0.05'],  # D-O7
    'b8k': [],  # D-O6: 4 rows per step
    # A: census pool and the whole replay store
    'fresh_census': CENSUS,  # A-R2, A-B1, A-B2, A-C2: the pool's files, under one pass
    'sub168_census': CENSUS,  # A-B1: a ~168M-note subset of the pool
    'full': [],  # A-C2: the whole replay store at --notes (3.8 passes at 640M)
    # B: Discover / Aria
    'disc_control': DISC,  # B-T1: 41.5M notes of the census Discover train store
    'disc_fresh': DISC,
    'disc_fresh120': DISC,
    'disc_dyn': [*DISC, '--hypothesis=aug_dyn:shift=4'],  # B-T1aug
    'giga_fresh120': [],  # B-Gf120 / B-T3: the replay store at 120M notes
    'aria_control': ARIA_ARGS,  # B-T2: aria_train (550.8M notes)
    'aria_fresh': ARIA_ARGS,
    'aria_fresh120': ARIA_ARGS,
    # E: synthetic data, distillation
    **{arm: synth_args(*SYNTH[arm]) for arm in ('synth_clean25',)},  # E-S6
    'distill4': [f'--hypothesis=distill:members={MEMBERS}:alpha=0.0'],  # E-E5 (hypotheses/distill.py)
}
B16K = dict(rows=8, warmup=200)
OPTS = {  # see the docstring; arms not listed: CLI frac and notes, 16 rows, compared with control
    REF: dict(frac=1.0, ref=None),
    'control': dict(ref=None),
    'b16k': B16K, 'b16k_lr7': B16K, 'b16k_wd05': B16K,
    'b8k': dict(rows=4, warmup=400),
    'fresh_census': dict(frac=1.0, ref=REF),
    'sub168_census': dict(frac=0.088, ref='fresh_census'),
    'full': dict(frac=1.0, ref='fresh_census'),
    'disc_control': dict(frac=0.247, ref=None),
    'disc_fresh': dict(frac=1.0, ref=None),
    'disc_fresh120': dict(frac=1.0, notes=120e6, ref='disc_fresh'),
    'disc_dyn': dict(frac=0.247, ref='disc_control'),
    'giga_fresh120': dict(frac=1.0, notes=120e6, ref=REF),
    'aria_control': dict(frac=0.0753, ref=None),
    'aria_fresh': dict(frac=0.305, ref=None),
    'aria_fresh120': dict(frac=0.305, notes=120e6, ref='aria_fresh'),
}
FAMILIES = {'control': REF, 'disc_control': 'disc_fresh', 'aria_control': 'aria_fresh'}  # control: its fresh ref
DEFAULTS = dict(frac=None, notes=None, rows=ROWS, warmup=None, eval=None, ref='control', twin=None)
TWINS = {'fresh_wd03': 'wd3x', 'fresh_wd1': 'wd10x', 'fresh_wd06': 'wd06', 'fresh_dyn': 'dyn',  # fresh_X: X
         'fresh_lr2e3': 'lr2e3', 'fresh_lr05': 'lr05', 'fresh_b16k': 'b16k', 'fresh_embwd0': 'embwd0'}


def add_arm(name, args, **opts):
    """One more arm; a fresh_<X> name defaults to the whole replay store, compared with fresh, paired with X."""
    if name.startswith('fresh_'):
        opts = {'frac': 1.0, 'ref': REF, 'twin': name[len('fresh_'):], **opts}
    assert set(opts) <= set(DEFAULTS), f'{name}: unknown settings {set(opts) - set(DEFAULTS)}'
    if name in ARMS:
        assert ARMS[name] == list(args) and OPTS.get(name, {}) == opts, f'{name} is already a different arm'
    ARMS[name] = list(args)
    if opts:
        OPTS[name] = opts


for _fresh, _arm in TWINS.items():
    add_arm(_fresh, ARMS[_arm], **{k: v for k, v in OPTS.get(_arm, {}).items() if k not in ('ref', 'twin')},
            twin=_arm)


def load_arms(path):
    """--arm-json: {name: [flags] | {"args": [...], "synth": [[stores], share], <OPTS keys>}} -> the arms added."""
    names = []
    for name, spec in json.loads(Path(path).read_text()).items():
        spec = {'args': spec} if isinstance(spec, list) else dict(spec)
        args = list(spec.pop('args', []))
        if 'synth' in spec:
            stores, share = spec.pop('synth')
            SYNTH[name] = (list(stores), share)
            args = synth_args(stores, share) + args
        add_arm(name, args, **spec)
        names.append(name)
    return names


def opt(arm, key):
    return OPTS.get(arm, {}).get(key, DEFAULTS[key])


def load():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def save(name, rec):
    runs = load()
    runs[name] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RUNS.with_suffix('.tmp')
    tmp.write_text(json.dumps(runs, indent=1))
    tmp.replace(RUNS)


def run_name(arm, seed, frac, size='42M', notes=NOTES):
    """<arm>_s<seed>[_f<frac>][_<size>][_n<M>M]: frac only for arms on the CLI frac, notes only for arms on the CLI
    notes (HEAD's names for the 160M runs)."""
    fixed = arm == REF or opt(arm, 'frac') is not None
    return (f'{arm}_s{seed}' + ('' if frac == 0.25 or fixed else f'_f{frac:g}')
            + ('' if size == '42M' else f'_{size}')
            + ('' if notes == NOTES or opt(arm, 'notes') is not None else f'_n{notes / 1e6:g}M'))


def table_path(size, frac=0.25, notes=NOTES):
    parts = ([] if size == '42M' else [size]) + ([] if frac == 0.25 else [f'f{frac:g}']) + (
        [] if notes == NOTES else [f'n{notes / 1e6:g}M'])
    return ROOT / ('logs/aug_sweep' + ''.join('_' + p for p in parts) + '.md')


def file_frac(arm, frac):
    """--train_file_frac of an arm: its own share (the whole store for the fresh arms), all of the synthetic stores."""
    if arm in SYNTH:
        return f'{frac:g}' + ',1' * len(SYNTH[arm][0])
    return frac if opt(arm, 'frac') is None else opt(arm, 'frac')


def arm_frac(arm, frac):
    """The frac a run records."""
    return frac if opt(arm, 'frac') is None else opt(arm, 'frac')


def arm_iters(arm, notes=NOTES):
    """(iterations, rows per step) of an arm: its notes seen over rows x 2,048 notes per step."""
    rows = opt(arm, 'rows')
    return round((opt(arm, 'notes') or notes) / (rows * BLOCK)), rows


def micro_args(micro, eval_iters=None, rows=ROWS):
    m = min(micro, rows)
    assert rows % m == 0 and VAL_WINDOWS % m == 0, (micro, rows)
    return [f'--batch_size={m}', f'--gradient_accumulation_steps={rows // m}',
            f'--eval_iters={eval_iters or VAL_WINDOWS // m}']


def expected_minutes(size, iters, rows=ROWS):
    """5090 minutes of a run: MS_PER_STEP at 16 rows (sqrt(rows / 16) of it for smaller steps) + start-up."""
    return MS_PER_STEP[size] * (rows / ROWS) ** 0.5 * iters / 60000 + (1.5 if size in BIG else 0.7)


def out_dir(name, keep=False):
    return ROOT / 'checkpoints' / ('keep' if keep else '') / f'aug_{name}'


def command(arm, seed, frac, size, iters, out, extra=()):
    return ['.venv/bin/python', 'train.py', CONFIG, *[f'--{k}={v}' for k, v in SIZES[size].items()],
            f'--train_file_frac={file_frac(arm, frac)}', f'--seed={seed}', f'--max_iters={iters}',
            f'--lr_decay_iters={iters}', f'--out_dir={out.relative_to(ROOT)}', *ARMS[arm], *extra]


def run_args(arm, micro, iters, keep=False, save_iters=''):
    """The flags after the arm's: micro-batch, eval interval, warmup, checkpoints."""
    rows = opt(arm, 'rows')
    return ([*micro_args(micro, rows=rows), f'--eval_interval={opt(arm, "eval") or iters // 4}']
            + ([f'--warmup_iters={opt(arm, "warmup")}'] if opt(arm, 'warmup') else [])
            + (['--always_save_checkpoint=True'] if keep else []) + ([f'--save_iters={save_iters}'] if save_iters else []))


def kill_tree(pid):
    """SIGTERM, then SIGKILL, to a process and its descendants (/proc)."""
    kids = {}
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            kids.setdefault(int(stat.read_text().rsplit(')', 1)[1].split()[1]), []).append(int(stat.parent.name))
        except (OSError, IndexError, ValueError):
            continue
    tree, todo = [], [pid]
    while todo:
        tree.append(todo.pop())
        todo += kids.get(tree[-1], [])
    def alive(p):
        try:
            return Path(f'/proc/{p}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except (OSError, IndexError):
            return False

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in tree:
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError):
                pass
        for _ in range(40 if sig == signal.SIGTERM else 0):
            if not any(map(alive, tree)):
                return
            time.sleep(0.5)


def train(cmd, out, log, keep=False, timeout=None):
    """One train.py run -> its exit code ('timeout' past the limit, in minutes); the log holds everything. The
    checkpoint directory is deleted unless kept."""
    LOGS.mkdir(parents=True, exist_ok=True)
    print(time.strftime('%H:%M'), log.name, flush=True)
    with open(log, 'w') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout and timeout * 60)
        except subprocess.TimeoutExpired:
            kill_tree(proc.pid)
            proc.wait()
            code = 'timeout'
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return code


def unique_notes(text):
    m = re.search(r'file_frac \S+ keeps [\d,]+ files, ([\d,]+) notes', text)
    return int(m.group(1).replace(',', '')) if m else None


def run(arm, seed, frac, size, micro, notes=NOTES, keep=False, save_iters='', timeout_mult=2.2):
    name = run_name(arm, seed, frac, size, notes)
    log = LOGS / f'{name}.log'
    iters, rows = arm_iters(arm, notes)
    out = out_dir(name, keep)
    limit = timeout_mult * expected_minutes(size, iters, rows) if timeout_mult else None
    t0 = time.time()
    code = train(command(arm, seed, frac, size, iters, out, run_args(arm, micro, iters, keep, save_iters)), out, log,
                 keep, limit)
    evals, params = parse(log)
    rec = dict(arm=arm, seed=seed, frac=arm_frac(arm, frac), size=size, micro=min(micro, rows))
    if rows != ROWS:
        rec['rows'] = rows
    if code or not evals or evals[-1]['step'] < iters - 1:
        print(f'  failed (exit {code}), see {log}', flush=True)
        save(name, dict(rec, failed=True, code=code) if code != 'timeout' else
             dict(rec, failed=True, code=None, timeout=round(limit, 1)))
        return
    save(name, dict(rec, params=params, notes=iters * rows * BLOCK, iters=iters,
                    unique_notes=unique_notes(log.read_text(errors='replace')),
                    minutes=round((time.time() - t0) / 60, 1), curve=[(e['step'], e['ce']) for e in evals],
                    **({'keep': str(out.relative_to(ROOT))} if keep else {}), **evals[-1]))
    print(f'  CE {evals[-1]["ce"]:.3f}', flush=True)


def finished(runs, arm, frac, size, notes=NOTES, seeds=SEEDS):
    """{seed: record} of the finished runs of an arm."""
    return {s: r for s in seeds if (r := runs.get(run_name(arm, s, frac, size, notes))) and not r.get('failed')}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def penalty(runs, frac, size, notes=NOTES, control='control'):
    """control mean - its fresh reference's mean, once both finished seeds 1337 and 2, else None."""
    ctrl, ref = finished(runs, control, frac, size, notes), finished(runs, FAMILIES[control], frac, size, notes)
    if len(ctrl) < len(SEEDS) or len(ref) < len(SEEDS):
        return None
    return mean(r['ce'] for r in ctrl.values()) - mean(r['ce'] for r in ref.values())


def paired(done, other):
    """Mean CE difference over the seeds both finished."""
    return mean(done[s]['ce'] - other[s]['ce'] for s in done if s in other)


def table(frac, size, note='', notes=NOTES):
    runs = load()
    seeds = list(SEEDS) + sorted({r['seed'] for r in runs.values() if r.get('size', '42M') == size} - set(SEEDS))
    fin = lambda arm: finished(runs, arm, frac, size, notes, seeds)  # noqa: E731
    gap = penalty(runs, frac, size, notes)
    unique = next((r['unique_notes'] for r in runs.values() if r.get('frac') == frac and r.get('unique_notes')
                   and r.get('size', '42M') == size), None)
    passes = f' (~{notes / unique:.1f} passes over {unique / 1e6:.1f}M notes)' if unique else ''
    micro = next((r['micro'] for r in runs.values() if r.get('micro') and r.get('size', '42M') == size), None)
    try:
        ref = json.loads(LADDER_RUNS.read_text())[f'{size}_160M']['ce']
        ladder = f' The laptop\'s scale ladder {size}_160M (micro 4): CE {ref:.3f}.'
    except (OSError, KeyError, ValueError):
        ladder = ''
    twin_of = {**FAMILIES, **{opt(a, 'twin'): a for a in ARMS if opt(a, 'twin')}}
    fmt = lambda v, f: '' if v is None else format(v, f)  # noqa: E731
    lines = [f'# Augmentation search (tools/sweeps/aug_sweep.py): {size}, {frac:g} of the GigaMIDI replay files, '
             f'{round(notes / NOTES_PER_STEP):,} iterations = {notes / 1e6:g}M notes seen{passes}, micro-batch {micro}',
             '', f'{REF} = the whole store under one pass. Repeat penalty = control mean - {REF} mean = '
             f'{fmt(gap, "+.3f") or "(unfinished)"}.{ladder} synth arms: that share of the windows from their stores '
             f'({", ".join(f"{a}: {s[0]}" for a, (s, _) in SYNTH.items())}).',
             '', '| arm | ' + ' | '.join(f's{s}' for s in seeds) + ' | mean | ref | vs ref | won back | P_X | prog CE |',
             '|---' * (len(seeds) + 7) + '|']
    for arm in ARMS:
        names = {s: run_name(arm, s, frac, size, notes) for s in seeds}
        if not any(n in runs for n in names.values()):
            continue
        done = fin(arm)
        cells = [fmt(done[s]['ce'], '.3f') if s in done else 'failed' if names[s] in runs else '' for s in seeds]
        ref = opt(arm, 'ref')
        vs = paired(done, fin(ref)) if ref else None
        fam_gap = penalty(runs, frac, size, notes, ref) if ref in FAMILIES else None
        won = -vs / fam_gap if vs is not None and fam_gap and fam_gap > 0 else None
        p_x = paired(done, fin(twin_of[arm])) if arm in twin_of else None
        row = [arm, *cells, fmt(mean(r['ce'] for r in done.values()), '.3f'), ref or '', fmt(vs, '+.3f'),
               fmt(won, '.0%'), fmt(p_x, '+.3f'), fmt(mean(r.get('prog') for r in done.values()), '.3f')]
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', 'vs ref: mean difference from the arm it is compared with, over the seeds both finished (negative = '
              'better; fresh_* arms against fresh). won back = -(vs ref) / the repeat penalty of the control it is '
              'compared with (control - fresh, disc_control - disc_fresh, aria_control - aria_fresh). P_X = the arm '
              '- its fresh twin (fresh_X; for a control, its fresh reference): the repeat penalty under that recipe.'
              ] + (['', note] if note else [])
    table_path(size, frac, notes).write_text('\n'.join(lines) + '\n')


def stores_missing(arm):
    """The stores an arm's flags name that are not on disk (train: <prefix>_train, validation: <prefix>_validation)."""
    missing = []
    for a in ARMS[arm]:
        key, _, val = a.lstrip('-').partition('=')
        split = 'train' if key == 'csv_path' else 'validation' if key in ('val_csv_path', 'val_extra') else None
        for src in re.findall(r'store:([^,;#]+)', val) if split else []:
            if src != REPLAY and not (ROOT / f'{src}_{split}' / 'meta.json').exists():
                missing.append(f'{src}_{split}')
    return missing


def smoke(arms, frac, size, micro, keep=False):
    """40 iterations of each arm with fp8 + compile: the hook loads, training runs, the eval is finite."""
    for arm in arms:
        if arm == REF:  # the plain config on the whole store: control covers it
            continue
        if missing := stores_missing(arm):
            print(f'  skipped {arm}: no {", ".join(missing)} yet', flush=True)
            continue
        hyp_name = next((x.split('=', 1)[1].split(':')[0] for x in ARMS[arm] if x.startswith('--hypothesis')), None)
        if hyp_name and not (ROOT / 'hypotheses' / f'{hyp_name}.py').exists():
            print(f'  skipped {arm}: no hypotheses/{hyp_name}.py yet', flush=True)
            continue
        log = LOGS / f'smoke_{arm}{"" if size == "42M" else "_" + size}.log'
        out = out_dir(f'smoke_{arm}', keep)
        rows = opt(arm, 'rows')
        cmd = command(arm, SEEDS[0], frac, size, 40, out,
                      [*micro_args(micro, eval_iters=20, rows=rows), '--warmup_iters=5', '--eval_interval=40',
                       *(['--always_save_checkpoint=True'] if keep else [])])
        code = train(cmd, out, log, keep=True)
        kept = any(out.glob('*.pt'))
        shutil.rmtree(out, ignore_errors=True)
        evals, _ = parse(log)
        text = log.read_text(errors='replace')
        hyp = next((ln for ln in text.splitlines() if ln.startswith('hypothesis:')), '')
        ok = (code == 0 and evals and evals[-1]['step'] >= 39  # no file_frac line at frac 1
              and (unique_notes(text) is not None or arm_frac(arm, frac) == 1)
              and bool(hyp) == bool(hyp_name) and (not keep or kept)
              and all(f'{Path(s).name}_train' in text for s in SYNTH.get(arm, ([], 0))[0]))
        print(f'  {"ok" if ok else "FAILED"}  exit {code}  CE {evals[-1]["ce"] if evals else "-"}  {hyp}'
              + (f'  kept {kept}' if keep else ''), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', action='store_true')
    p.add_argument('--only', default='', help='arms whose name contains the text')
    p.add_argument('--arms', default='', help='comma-separated arm names (default: all)')
    p.add_argument('--arm-json', default='', help='more arms from a JSON file (e.g. logs/reg_arms.json)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--frac', type=float, default=0.25)
    p.add_argument('--size', default='42M', choices=list(SIZES))
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='comma-separated (the gate uses 1337 and 2)')
    p.add_argument('--notes', type=float, default=NOTES, help='notes seen per run (iterations = notes / 32,768)')
    p.add_argument('--keep', action='store_true', help='keep the checkpoints (checkpoints/keep/aug_<name>)')
    p.add_argument('--save-iters', default='', help="with --keep: also keep weights at these iterations ('2440,3660')")
    p.add_argument('--timeout-mult', type=float, default=2.2, help='x the expected 5090 time; 0 = no limit')
    p.add_argument('--micro', type=int, default=4, help='rows per micro-batch (16 / micro accumulation steps)')
    p.add_argument('--min-gap', type=float, default=0.01)
    p.add_argument('--force', action='store_true', help='run the arms even if the gate fails')
    a = p.parse_args()
    micro_args(a.micro)  # checks it divides 16 and 1,200
    assert not a.save_iters or a.keep, '--save-iters needs --keep'
    if a.arm_json:
        load_arms(ROOT / a.arm_json)
    seeds = [int(s) for s in a.seeds.split(',') if s]
    wanted = [x for x in a.arms.split(',') if x]
    assert all(x in ARMS for x in wanted), f'unknown arms {set(wanted) - set(ARMS)}'
    arms = [arm for arm in ARMS if a.only in arm and (not wanted or arm in wanted)]
    if a.smoke:
        return smoke(arms, a.frac, a.size, a.micro, a.keep)
    if a.list:
        runs = load()
        fmt = lambda v, f: '-' if v is None else format(v, f)  # noqa: E731
        for seed in seeds:
            for arm in arms:
                name = run_name(arm, seed, a.frac, a.size, a.notes)
                r = runs.get(name)
                print(f'{name:28s}', '-' if not r else 'failed' if r.get('failed') else
                      f'CE {r["ce"]:.3f}  prog {fmt(r.get("prog"), ".3f")}  {r["minutes"]} min')
        return

    def todo(arm, seed):  # missing or failed (a failed run is tried again on the next start)
        r = load().get(run_name(arm, seed, a.frac, a.size, a.notes))
        return arm in arms and (r is None or r.get('failed', False))

    def go(arm, seed):
        run(arm, seed, a.frac, a.size, a.micro, a.notes, a.keep, a.save_iters, a.timeout_mult)
        table(a.frac, a.size, notes=a.notes)

    for seed in seeds:
        for arm in ('control', REF):
            if todo(arm, seed):
                go(arm, seed)
    gap = penalty(load(), a.frac, a.size, a.notes)
    if not a.force and (gap is None or gap < a.min_gap):
        note = f'Stopped at the gate: repeat penalty {gap if gap is None else round(gap, 3)}, needed {a.min_gap}.'
        table(a.frac, a.size, note, a.notes)
        print(note)
        return
    print(f'gate: repeat penalty {gap if gap is None else round(gap, 3)}' + (' (--force)' if a.force else ''),
          flush=True)
    for seed in seeds:
        for arm in ARMS:
            if arm not in ('control', REF) and todo(arm, seed):
                go(arm, seed)


if __name__ == '__main__':
    main()
