"""
Iso-FLOP set on the census GigaMIDI pool, one pass (context/claude/scaling_plan_inputs_2026-11-20/02_brief_isoflop.md,
master plan section 3(C) and 8.3). One arm = one train.py run of config/pretrain/scale_ladder.py with --one_pass.

    .venv/bin/python tools/sweeps/isoflop.py --dry                 # the grid table, minutes, windows needed vs the plan
    .venv/bin/python tools/sweeps/isoflop.py --queue               # queue lines (plan_queue.sh format) of the ESS rows
    .venv/bin/python tools/sweeps/isoflop.py --run 07              # one arm (chained arms copy the stem's pre_cooldown.pt)
    .venv/bin/python tools/sweeps/isoflop.py --gate 0|1|chain|2    # decisions; --append FILE adds the gated rows' lines
    .venv/bin/python tools/sweeps/isoflop.py --arms 1,22,lit3      # several arms (gated ones only when switched on)
    .venv/bin/python tools/sweeps/isoflop.py --panel 1-6           # B-T4: finals on the Discover panel
    .venv/bin/python tools/sweeps/isoflop.py --smoke [--s0]        # chained vs independent CE (S0 on the pool when
                                                                   # it is VERIFIED, else tiny on replay; --laptop)

Iterations come from C = 6 N D (N = the plan's non-embedding count incl. the heads, D = iters x 32,768 notes).
A chained arm resumes from its stem's pre_cooldown.pt (every arm passes --save_pre_cooldown=True; the scale ladder
config sets it False); the plan is the same fixed permutation, so the resumed run reads the windows after the stem's.
Results: results/isoflop/runs.json; logs: logs/isoflop/<id>.log; finals (model_bf16.pt) are kept, a stem's
pre_cooldown.pt is deleted once all its children are done or skipped (results/isoflop/skip_ids.txt).
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/isoflop'
RUNS = OUT / 'runs.json'
SKIP_IDS = OUT / 'skip_ids.txt'
LOGS = ROOT / 'logs/isoflop'
CONFIG = 'config/pretrain/scale_ladder.py'
NPS = 16 * 2048                       # notes per step (batch 16 x block 2048)
U = 1_897_348_864                     # usable notes of the pool
POOL, REPLAY = 'data/cache/gigamidi_pool', 'data/cache/gigamidi_replay'
POOL_SKIP = 'data/cache/gigamidi_pool_skip.txt'  # near-duplicate skip list; fixed before the first pool arm
MS = {3e6: 25, 6e6: 35, 20e6: 65, 42e6: 95, 108e6: 189, 217e6: 350}  # 5090 ms/step at 32,768 notes
N_NE = {  # the plan's N per shape (reproduces its iteration counts); other shapes: the model's count
    '6x256': 5_881_432, '8x256': 7_488_241, '7x320': 10_444_715, '7x384': 14_524_295, '10x384': 19_834_898,
    '10x448': 27_183_761, '10x512': 35_687_230, '11x576': 48_221_689, '12x640': 64_845_476, '13x704': 84_893_674,
    '14x768': 106_615_351}
ALT_LR = 2e-3
GATES = OUT / 'gates.json'            # rows each gate switched on: {'1': [...], '2': [...]}
VERIFIED = ROOT / f'{POOL}_train/VERIFIED'  # written by census_pool.py verify
PANEL = 'disc=store:data/cache/cdisc_vn;disc_aria=store:data/cache/cdisc_va'  # B-T4: Discover panel on the finals
ALIASES = {'22': ['22p'], 'lit3': ['LIT3a', 'LIT3b'], 'o1': ['O1'], 'o2': ['O2a', 'O2b'], 'o4': ['O4'], 'g1a': ['G1a']}


def arm(id, shape, C=None, iters=None, stem=None, lr=1e-3, seed=1337, extra=(), tier='ESS', store=POOL, gate=None,
        keep=False, batch=16):
    return dict(id=id, shape=shape, C=C, iters=iters, stem=stem, lr=lr, seed=seed, extra=list(extra), tier=tier,
                store=store, gate=gate, keep=keep, batch=batch)


B8 = ['--batch_size=8', '--eval_iters=150', '--warmup_iters=200']
GRID = [('01', 3e16, '6x256', None), ('02', 3e16, '8x256', None), ('03', 3e16, '7x320', None),
        ('04', 3e16, '7x384', None), ('05', 3e16, '10x384', None), ('06', 3e16, '10x448', None),
        ('07', 1e17, '7x320', '03'), ('08', 1e17, '7x384', '04'), ('09', 1e17, '10x384', '05'),
        ('10', 1e17, '10x448', '06'), ('11', 1e17, '10x512', None), ('12', 1e17, '11x576', None),
        ('13', 3e17, '10x448', '10'), ('14', 3e17, '10x512', '11'), ('15', 3e17, '11x576', '12'),
        ('16', 3e17, '12x640', None), ('17', 3e17, '13x704', None)]
ARMS = {a['id']: a for a in [
    *[arm(i, s, C, stem=st) for i, C, s, st in GRID],
    arm('18', '6x256', 3e16, lr=ALT_LR), arm('19', '7x384', 3e16, lr=ALT_LR, keep=True),
    arm('20', '10x448', 3e16, lr=ALT_LR, keep=True), arm('21', '10x384', 3e16, seed=2),
    arm('22p', '10x448', iters=4678, extra=['--lr_decay_iters=18711', '--eval_interval=4677']),
    arm('13u', '10x448', iters=56132, tier='REC'),  # #13 from scratch (chain check on a 3e17 final); C None: not fitted
    arm('23', '10x448', 3e16, extra=B8, batch=8),
    arm('24', '12x512', iters=4883, store=REPLAY), arm('25', '12x512', iters=4883),
    arm('26', '12x512', iters=4883, store=REPLAY, extra=['--train_file_frac=0.25', '--one_pass_epochs=4']),
    arm('27', '12x512', iters=9766, stem='25'),
    arm('34', '10x384', iters=76_928, stem='09', extra=['--one_pass_epochs=2'], gate='2'),
    arm('28', '6x256', 3e16, extra=B8, batch=8, tier='REC', gate='1c'),
    arm('29', '11x576', 1e17, lr=ALT_LR, tier='REC', gate='1b'), arm('30', '7x320', 1e17, lr=ALT_LR, tier='REC', gate='1b'),
    *[arm(i, s, 3e16, lr=ALT_LR, tier='REC', gate='1b2') for i, s in (('31', '8x256'), ('32', '7x320'), ('33', '10x384'))],
    arm('LIT3a', '6x256', 3e16, extra=['--weight_decay=0.3'], tier='REC'),
    arm('LIT3b', '10x448', 3e16, extra=['--weight_decay=0.3'], tier='REC'),
    arm('G1a', '5x256', 3e16, gate='1a'),
    arm('O1', '14x768', 3e17, tier='OPT'), arm('O2a', '6x256', 3e16, seed=2, tier='OPT', gate='1d'),
    arm('O2b', '10x448', 3e16, seed=2, tier='OPT', gate='1d'),
    arm('O4', '13x704', 3e17, lr=ALT_LR, tier='OPT', gate='1b2')]}
CHILD_EXTRA = {'10': ['--eval_dense=4677', '--eval_dense_until=4677']}  # #10 also evaluates at 4,677 (chain check)
GATES_AFTER = {'24': ['0'], '21': ['1'], '22p': ['chain'], '12': ['2']}  # gate lines after these rows in --queue
QUEUE_FILE = 'logs/plan_queue.txt'
ORDER = ['24', '25', '27', '26', '01', '02', '03', '04', '05', '06', '23', '18', '19', '20', '21',
         '07', '08', '09', '10', '22p', '11', '12', '13', '14', '15', '16', '17']


def dims(shape):
    L, d = (int(x) for x in shape.split('x'))
    return L, d, d // 64


def n_ne(shape):
    if shape in N_NE:
        return N_NE[shape]
    import torch
    from musicar.model.gpt import GPT, MusicConfig
    L, d, h = dims(shape)
    with torch.device('meta'):
        m = GPT(MusicConfig(n_layer=L, n_embd=d, n_head=h, n_programs=129, pos_emb='rope', block_size=2048,
                            norm='rmsnorm', mlp='swiglu', qk_norm=True, cond_inst=True, n_density=16, pitch_size=130))
    skip = ('transformer.music_embeddings', 'transformer.cond_density', 'cascade.cond_emb')
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(skip))


def iters(a):
    return a['iters'] or round(a['C'] / (6 * n_ne(a['shape']) * a['batch'] * 2048))


def start(a):
    """The iteration a chained arm resumes at: its stem's cooldown start (train.py cooldown_start)."""
    if not a['stem']:
        return 0
    i = iters(ARMS[a['stem']])
    return i - int(0.2 * i)


def ms_step(n):
    xs = sorted(MS)
    n = min(max(n, xs[0]), xs[-1])
    for lo, hi in zip(xs, xs[1:]):
        if n <= hi:
            f = math.log(n / lo) / math.log(hi / lo)
            return MS[lo] * (MS[hi] / MS[lo]) ** f


def minutes(a):
    I, s = iters(a), start(a)
    ms = ms_step(n_ne(a['shape'])) * a['batch'] / 16
    evals = len(eval_steps(a))
    return round(((I - s) * ms + evals * 25 * ms_step(n_ne(a['shape']))) / 60000 + 1.0, 1)


def eval_steps(a):
    I, s = iters(a), start(a)
    every = max(I // 8, 1)
    return [i for i in range(s, I) if i % every == 0 or i == I - 1 or (i <= 3000 and i % 250 == 0)]


def out_dir(a):
    return ROOT / 'checkpoints' / f"iso_{a['id']}_{a['shape']}"


def skip_hash(store):
    p = ROOT / POOL_SKIP
    return hashlib.md5(p.read_bytes()).hexdigest()[:12] if store == POOL and p.exists() else ''


def command(a):
    L, d, h = dims(a['shape'])
    I = iters(a)
    cmd = ['.venv/bin/python', 'train.py', CONFIG, f'--n_layer={L}', f'--n_embd={d}', f'--n_head={h}',
           f"--csv_path=store:{a['store']}", '--one_pass=True', '--boundary_frac=0', '--batch_size=16',
           '--gradient_accumulation_steps=1', '--eval_iters=75', f'--max_iters={I}', f'--lr_decay_iters={I}',
           '--save_pre_cooldown=True', '--always_save_checkpoint=True', f'--eval_interval={max(I // 8, 1)}',
           '--eval_dense=250',
           '--eval_dense_until=3000', f"--learning_rate={a['lr']:g}", f"--seed={a['seed']}",
           f'--out_dir={out_dir(a).relative_to(ROOT)}']
    if skip_hash(a['store']):
        cmd.append(f'--one_pass_skip={POOL_SKIP}')
    if a['stem']:
        cmd.append('--init_from=resume')
    return cmd + a['extra'] + CHILD_EXTRA.get(a['id'], [])


def load():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def save(rec):
    runs = load()
    runs[rec['id']] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RUNS.with_suffix('.tmp')
    tmp.write_text(json.dumps(runs, indent=1))
    tmp.replace(RUNS)


def skipped():
    return set(SKIP_IDS.read_text().split()) if SKIP_IDS.exists() else set()


def parse(text):
    """'step N:' evaluation lines -> [(step, ce, nocond ce or None)]."""
    out = []
    for m in re.finditer(r'step (\d+): train loss \S+ val loss [\d.]+ \| \S+ CE ([\d.]+)[^\n\r]*', text):
        nc = re.search(r'\| nocond CE ([\d.]+)', m.group(0))
        out.append((int(m.group(1)), float(m.group(2)), float(nc.group(1)) if nc else None))
    return out


def cleanup_stems():
    """Delete a stem's pre_cooldown.pt once every child is done or skipped (keep=True stems: by hand / gates)."""
    runs, skip = load(), skipped()
    for sid, s in ARMS.items():
        kids = [a['id'] for a in ARMS.values() if a['stem'] == sid]
        p = out_dir(s) / 'pre_cooldown.pt'
        if p.exists() and sid in runs and not s['keep'] and all(k in runs or k in skip for k in kids):
            p.unlink()
            print(f'deleted {p.relative_to(ROOT)}')


def run(id, dry=False):
    a = ARMS[id]
    out, log = out_dir(a), LOGS / f'{id}.log'
    cmd = command(a)
    if dry:
        print(' '.join(cmd))
        return
    if a['stem']:
        src = out_dir(ARMS[a['stem']]) / 'pre_cooldown.pt'
        assert src.exists(), f'{id}: stem {a["stem"]} has no {src}'
        stem_rec = load().get(a['stem'], {})
        assert stem_rec.get('skip', '') == skip_hash(a['store']), f'{id}: skip list changed since stem {a["stem"]}'
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out / 'ckpt.pt')
    LOGS.mkdir(parents=True, exist_ok=True)
    print(time.strftime('%H:%M'), id, ' '.join(cmd[2:]), flush=True)
    t0 = time.time()
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    (out / 'ckpt.pt').unlink(missing_ok=True)
    if code:
        sys.exit(f'{id}: train.py failed (exit {code}), see {log}')
    curve = parse(log.read_text(errors='replace'))
    I = iters(a)
    assert curve and curve[-1][0] == I - 1, f'{id}: last evaluation at {curve[-1][0] if curve else None} of {I}'
    save(dict(id=id, shape=a['shape'], C=a['C'], n_ne=n_ne(a['shape']), iters=I, start=start(a), stem=a['stem'],
              lr=a['lr'], seed=a['seed'], batch=a['batch'], notes=I * a['batch'] * 2048, store=a['store'],
              extra=a['extra'], ce=curve[-1][1], nocond=curve[-1][2], curve=curve, skip=skip_hash(a['store']),
              minutes=round((time.time() - t0) / 60, 1)))
    if not any(b['stem'] == id for b in ARMS.values()):
        (out / 'pre_cooldown.pt').unlink(missing_ok=True)
    cleanup_stems()
    print(f'{id}: CE {curve[-1][1]:.4f} in {(time.time() - t0) / 60:.1f} min')


def resolve(ids):
    """'1,22,lit3' -> ['01', '22p', 'LIT3a', 'LIT3b']."""
    out = []
    for x in str(ids).split(','):
        x = x.strip()
        if x in ARMS:
            out.append(x)
        elif x.lower() in ALIASES:
            out += ALIASES[x.lower()]
        elif x.isdigit() and f'{int(x):02d}' in ARMS:
            out.append(f'{int(x):02d}')
        elif x:
            sys.exit(f'unknown arm {x}')
    return out


def gated_off(id):
    """A gated row runs only after its gate switched it on (results/isoflop/gates.json)."""
    g = ARMS[id]['gate']
    if not g:
        return False
    dec = json.loads(GATES.read_text()) if GATES.exists() else {}
    return id not in dec.get(g[0], [])


def panel(spec):
    """B-T4: the finals (model_bf16.pt) of grid rows a-b scored on the Discover panel (eval only) -> runs.json."""
    a, _, b = spec.partition('-')
    for n in range(int(a), int(b or a) + 1):
        id = f'{n:02d}'
        rec, ck = load().get(id), out_dir(ARMS[id]) / 'model_bf16.pt'
        if rec is None or not ck.exists() or 'panel' in rec:
            print(f'{id}: ' + ('already scored' if rec and 'panel' in rec else 'no final'))
            continue
        L, d, h = dims(ARMS[id]['shape'])
        log = LOGS / f'panel_{id}.log'
        with open(log, 'w') as f:
            code = subprocess.run(['.venv/bin/python', 'train.py', CONFIG, f'--n_layer={L}', f'--n_embd={d}',
                                   f'--n_head={h}', f'--csv_path=store:{REPLAY}', '--init_from=finetune',
                                   f'--init_ckpt={ck.relative_to(ROOT)}', '--eval_only=True', '--compile=False',
                                   '--batch_size=16', '--eval_iters=75', f'--val_extra={PANEL}',
                                   '--out_dir=logs/isoflop/panel_tmp'], cwd=ROOT, stdout=f,
                                  stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors='replace')
        if code:
            print(f'{id}: panel eval failed (exit {code}), see {log}')
            continue
        line = re.search(r'step \d+: train loss[^\n]*', text).group(0)
        rec['panel'] = {k: float(v) for k, v in re.findall(r'\| (\w+) CE ([\d.]+)', line)}
        save(rec)
        print(f'{id}: {rec["panel"]}')


def queue_line(a):
    need = out_dir(ARMS[a['stem']]) / 'pre_cooldown.pt' if a['stem'] else ROOT / f"{a['store']}_train/meta.json"
    return (f"min={minutes(a)} need={need.relative_to(ROOT)} tier={a['tier']} id=iso_{a['id']} :: "
            f".venv/bin/python tools/sweeps/isoflop.py --run {a['id']}")


def append(path, lines):
    """Add lines to a queue file: written as a new file, then renamed over it."""
    path = Path(path)
    old = path.read_text() if path.exists() else ''
    tmp = path.with_name(path.name + '.iso_tmp')
    tmp.write_text(old + ('' if not old or old.endswith('\n') else '\n') + ''.join(f'{x}\n' for x in lines))
    tmp.replace(path)


def plan_windows(store, skip=''):
    import numpy as np
    from musicar.data.loader import one_pass_plan
    off = np.load(ROOT / f'{store}_train/offsets.npy')
    n = np.diff(off)
    keep = n >= 64
    mask = None
    if skip:
        mask = np.isin(np.flatnonzero(keep), np.loadtxt(ROOT / skip, dtype=np.int64, ndmin=1))
    return len(one_pass_plan(n[keep] + 2, 2048, 1337, 1, 0.05, mask)['plan'])


def dry():
    print(f"{'id':>5} {'C':>6} {'shape':>7} {'N_ne':>7} {'iters':>7} {'resume':>7} {'D':>7} {'passes':>6} "
          f"{'r':>6} {'min':>5} tier")
    tot, need = {}, 0
    for id in [*ORDER, *[i for i in ARMS if i not in ORDER]]:
        a = ARMS[id]
        I, N = iters(a), n_ne(a['shape'])
        D = I * a['batch'] * 2048
        print(f"{id:>5} {a['C'] or 0:6.0e} {a['shape']:>7} {N / 1e6:6.2f}M {I:7,} {start(a) or '-':>7} "
              f"{D / 1e6:6.0f}M {D / U:6.3f} {D / N:6.1f} {minutes(a):5.1f} {a['tier']}{'*' if a['gate'] else ''}"
              f"{' <- ' + a['stem'] if a['stem'] else ''}")
        key = a['tier'] + ('*' if a['gate'] else '')
        tot[key] = tot.get(key, 0) + minutes(a)
        if a['store'] == POOL and '--one_pass_epochs=2' not in a['extra']:
            need = max(need, I * a['batch'])
    print('minutes by tier (* = gated): ' + ', '.join(f'{k} {v:.1f}' for k, v in tot.items()))
    print(f'pool windows needed in one epoch: >= {need:,} (#13: 56,132 x 16 = 898,112)')
    if (ROOT / f'{POOL}_train/offsets.npy').exists():
        n = plan_windows(POOL, POOL_SKIP if skip_hash(POOL) else '')
        print(f'pool plan: {n:,} windows ({"OK" if n >= need else "TOO FEW"})')
    else:
        print('pool plan: 927,744 windows (no skip list; computed from the census offsets on the laptop)')


def ce(runs, id, step=None):
    r = runs.get(id)
    if r is None:
        return None
    if step is None:
        return r['ce']
    return next((c for s, c, _ in r['curve'] if s == step), None)


def gate(which, args):
    from tools.sweeps.isoflop_fit import parabola
    runs, add, ok = load(), [], True
    if which == '0':
        import numpy as np
        meta = json.loads((ROOT / f'{POOL}_train/meta.json').read_text())
        ok &= (meta['n_files'], meta['n_notes']) == (1_574_692, 1_910_871_953)
        n = int((lambda d: d[d >= 64].sum())(np.diff(np.load(ROOT / f'{POOL}_train/offsets.npy'))))
        ok &= n == U
        w = plan_windows(POOL, POOL_SKIP if skip_hash(POOL) else '')
        ok &= w >= 898_112
        print(f"pool {meta['n_files']:,} files, {meta['n_notes']:,} notes, {n:,} usable; plan {w:,} windows")
        r24 = runs.get('24')
        print(f"#24: {'CE %.4f' % r24['ce'] if r24 else 'not run'}")
        ok &= bool(r24) and math.isfinite(r24['ce'])
        speed = loader_speed()
        print(f'loader: {speed:,.0f} windows/s at micro 16 (gate >= 730)')
        ok &= speed >= 730
    elif which == '1':
        arms3 = ['01', '02', '03', '04', '05', '06'] + [i for i in ('G1a',) if i in runs]
        fit = parabola([n_ne(ARMS[i]['shape']) for i in arms3], [ce(runs, i) for i in arms3])
        left = sum(n_ne(ARMS[i]['shape']) < fit['n_star'] for i in arms3)
        bracketed = fit['A'] > 0 and left >= 2 and len(arms3) - left >= 2
        print(f"(a) 3e16 vertex N* {fit['n_star'] / 1e6:.2f}M, A {fit['A']:.4f}, {left} arms left of it: "
              f"{'bracketed' if bracketed else 'NOT bracketed -> G1a (5x256)'}")
        if not bracketed and 'G1a' not in runs:
            add.append('G1a')
        t = (ce(runs, '18') - ce(runs, '01')) - (ce(runs, '20') - ce(runs, '06'))
        print(f'(b) LR tilt t = {t:+.4f}' + (' -> #29-30' if abs(t) > 0.01 else '') +
              (' + #31-33 + O4 (per-size best LR)' if abs(t) > 0.02 else ''))
        if abs(t) > 0.01:
            add += ['29', '30']
        if abs(t) > 0.02:
            add += ['31', '32', '33', 'O4']
        gain = ce(runs, '06') - ce(runs, '23')
        print(f'(c) batch 16k gain {gain:+.4f}' + (' -> #28' if gain > 0.01 else ''))
        if gain > 0.01:
            add.append('28')
        noise = abs(ce(runs, '21') - ce(runs, '05'))
        print(f'(d) seed spread {noise:.4f}' + (' -> seed 2 of #1 and #6' if noise > 0.02 else ''))
        if noise > 0.02:
            add += ['O2a', 'O2b']
        for sid in ('19', '20'):  # their stems were kept until Gate 1
            p = out_dir(ARMS[sid]) / 'pre_cooldown.pt'
            if p.exists() and args.delete_kept:
                p.unlink()
    elif which == 'chain':
        a, b = ce(runs, '22p', 4677), ce(runs, '10', 4677)
        d = abs(a - b)
        ok = d <= 0.01
        print(f'chain check: #22p {a:.4f} vs #10 {b:.4f} at 4,677: |diff| {d:.4f} '
              f"{'OK' if ok else '> 0.01: FIX THE RESUME PATH before 3e17 (or run the rest unchained)'}")
    elif which == '2':
        arms = ['07', '08', '09', '10', '11', '12']
        fit = parabola([n_ne(ARMS[i]['shape']) for i in arms], [ce(runs, i) for i in arms])
        r = 1e17 / (6 * fit['n_star']) / fit['n_star']
        print(f"1e17 vertex N* {fit['n_star'] / 1e6:.2f}M, r* {r:.1f}: " + ('skip #34' if r < 15 else 'run #34'))
        if r >= 15:
            add.append('34')
        else:
            with open(SKIP_IDS, 'a') as f:
                f.write('34\n')
            cleanup_stems()
    if which in ('1', '2'):
        dec = json.loads(GATES.read_text()) if GATES.exists() else {}
        dec[which] = add
        OUT.mkdir(parents=True, exist_ok=True)
        GATES.write_text(json.dumps(dec, indent=1))
    lines = [queue_line(ARMS[i]) for i in add if i not in runs and i not in skipped()]
    for x in lines:
        print(x)
    if args.append and lines:
        append(args.append, lines)
        print(f'appended {len(lines)} lines to {args.append}')
    return ok


def loader_speed(n=3000):
    from functools import partial
    from torch.utils.data import DataLoader
    from musicar.data.loader import NoteWindowDataset, PlanSampler, collate_windows
    ds = NoteWindowDataset(f'store:{POOL}', '.', 'train', 2048, augment=True, special_tokens=True, pack_short=True,
                           programs=True, one_pass=True, one_pass_skip=POOL_SKIP if skip_hash(POOL) else '')
    dl = DataLoader(ds, batch_size=16, sampler=PlanSampler(len(ds)), num_workers=8,
                    collate_fn=partial(collate_windows, extra_names=('program',)))
    t0, seen = time.time(), 0
    for i, _ in enumerate(dl):
        if i == 2:
            t0, seen = time.time(), 0
        seen += 16
        if seen >= n:
            break
    return seen / (time.time() - t0)


def smoke(args):
    """Chained vs independent: an independent run of I iterations, a stem of I/2 (its pre_cooldown.pt at
    I/2 - I/10), a chained run from it to I; final CE within 0.02. Laptop default: tiny model on the replay store;
    --s0: 6x256 and 13x704 on the pool, 40 iterations, the instance's defaults."""
    if args.s0:
        shapes, I, store, extra = ['6x256', '13x704'], 40, POOL, []
    else:
        shapes, I, store = ['4x128'], args.smoke_iters, REPLAY
        extra = ['--batch_size=4', '--gradient_accumulation_steps=4', '--compile=False', '--fp8=False']
    ok = True
    for shape in shapes:
        L, d, h = dims(shape)
        base = ['.venv/bin/python', 'train.py', CONFIG, f'--n_layer={L}', f'--n_embd={d}', f'--n_head={h}',
                f'--csv_path=store:{store}', '--one_pass=True', '--boundary_frac=0', '--eval_iters=75',
                '--save_pre_cooldown=True', '--eval_interval=100000', '--batch_size=16',
                '--gradient_accumulation_steps=1'] + extra
        tag = ROOT / 'checkpoints' / f'iso_smoke_{shape}'
        shutil.rmtree(tag, ignore_errors=True)
        res = {}
        for name, it, more in (('indep', I, []), ('stem', I // 2, []), ('chain', I, ['--init_from=resume'])):
            out = tag / name
            if name == 'chain':
                out.mkdir(parents=True, exist_ok=True)
                shutil.copy(tag / 'stem' / 'pre_cooldown.pt', out / 'ckpt.pt')
            log = LOGS / f'smoke_{shape}_{name}.log'
            LOGS.mkdir(parents=True, exist_ok=True)
            with open(log, 'w') as f:
                code = subprocess.run(base + [f'--max_iters={it}', f'--lr_decay_iters={it}',
                                              f'--out_dir={out.relative_to(ROOT)}'] + more,
                                      cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            text = log.read_text(errors='replace')
            assert code == 0, f'{shape} {name}: exit {code}, see {log}'
            res[name] = parse(text)[-1][1]
            if name == 'chain':
                want = f'one_pass: windows {(I // 2 - int(0.2 * (I // 2))) * 16:,} ..'
                assert want in text, f'{shape}: the chained run did not start at the stem window ({want})'
        diff = abs(res['chain'] - res['indep'])
        ok &= diff <= 0.02
        print(f"smoke {shape}: independent {res['indep']:.4f}, chained {res['chain']:.4f} (stem {res['stem']:.4f}), "
              f"|diff| {diff:.4f} {'OK' if diff <= 0.02 else 'FAIL'}")
        shutil.rmtree(tag, ignore_errors=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry', action='store_true')
    p.add_argument('--queue', action='store_true', help='queue lines of the ungated ESS rows not done / skipped')
    p.add_argument('--tier', default='ESS', help='with --queue: tiers to include, e.g. ESS,REC')
    p.add_argument('--run')
    p.add_argument('--arms', help='comma list (1, 22, lit3, o2 ...): run each not done; gated rows only when switched on')
    p.add_argument('--panel', help="grid rows 'a-b': score their finals on the Discover panel")
    p.add_argument('--laptop', action='store_true', help='with --smoke: the tiny replay smoke even if the pool exists')
    p.add_argument('--print-cmd', action='store_true', help='with --run: print the train.py command only')
    p.add_argument('--gate', choices=['0', '1', 'chain', '2'])
    p.add_argument('--append', help='with --gate / --queue: append the lines to this queue file')
    p.add_argument('--delete-kept', action='store_true', help='with --gate 1: delete the kept stems of #19-20')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--s0', action='store_true')
    p.add_argument('--smoke-iters', type=int, default=400)
    a = p.parse_args()
    if a.dry:
        dry()
    elif a.queue:
        runs, skip, tiers = load(), skipped(), a.tier.split(',')
        ids = [i for i in ORDER + [i for i in ARMS if i not in ORDER]
               if ARMS[i]['tier'] in tiers and not ARMS[i]['gate'] and i not in runs and i not in skip]
        lines = []
        for i in ids:
            lines.append(queue_line(ARMS[i]))
            for g in GATES_AFTER.get(i, []):
                lines.append(f"min=2 need=results/isoflop/runs.json tier=ESS id=iso_gate{g} :: .venv/bin/python "
                             f"tools/sweeps/isoflop.py --gate {g} --append {QUEUE_FILE}")
        print('\n'.join(lines))
        if a.append:
            append(a.append, lines)
    elif a.run:
        run(a.run, dry=a.print_cmd)
    elif a.arms:
        runs, skip = load(), skipped()
        for id in resolve(a.arms):
            if id in runs or id in skip:
                print(f'{id}: done / skipped')
            elif gated_off(id):
                print(f'{id}: gated off (gate {ARMS[id]["gate"]})')
            else:
                run(id, dry=a.print_cmd)
    elif a.panel:
        panel(a.panel)
    elif a.gate:
        sys.exit(0 if gate(a.gate, a) else 1)
    elif a.smoke:
        a.s0 = a.s0 or (VERIFIED.exists() and not a.laptop)
        sys.exit(0 if smoke(a) else 1)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
