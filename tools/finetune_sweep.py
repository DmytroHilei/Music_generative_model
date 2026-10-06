"""
Data-mix / LR sweep for the 450M Ukrainian piano fine-tune, three phases run one after another on the laptop GPU:

    1. hypotheses: short runs that test why ua_450_s1 (mix 0.35 / 0.45 / 0.2, LR 3e-4) never beat step 0 on covers
       val (5.450 -> 5.57-5.69, with or without the window conditions): (a) the reductions (Demucs + basic-pitch,
       val CE ~10 vs ~5.5 for covers) pull the model away from the covers -> no reductions; (b) LR 3e-4 (from the
       107M ua-A sweep) is too high for the 450M -> 1e-4; (c) both; (d) + dropout 0 (long_450m was pretrained
       with dropout 0.0, the fine-tune turns on 0.1: the no-reductions run at 3e-4 still rose 5.449 -> 5.578 by
       step 50, with train loss 6.9 vs val 5.58).
       Gate: if none beats step 0 (the base model), the sweep stops: no mix is worth comparing then.
    2. grid: the other mixes at the LR (and variant) of the phase-1 winner, the same short run each.
    3. long: the run with the lowest best covers val CE, LONG_ITERS steps.

    .venv/bin/python tools/finetune_sweep.py               # all phases; finished runs (complete log) are skipped
    .venv/bin/python tools/finetune_sweep.py --dry-run     # the plan and a time estimate
    .venv/bin/python tools/finetune_sweep.py --long-run c60r20a20_lr1e-4_do0   # phase 3 only, with this run's settings

Every run is train.py config/finetune_ua450.py config/finetune_ua450_full.py + its mix, LR and length, log
logs/ua450_mix_<name>.log, out_dir checkpoints/ua450_mix_<name> (best-val bf16 weights only: no resumable state for
the short runs, ~0.9 GB each), name = <mix>_lr<LR><variant>. status.py finds the running one by itself.
Results: logs/ua450_sweep.txt.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ['config/finetune_ua450.py', 'config/finetune_ua450_full.py']
COVERS = 'data/finetune/ukrainian_covers_v4.csv'
REDUCTIONS = 'data/finetune/ukrainian_reduction_v4.csv'
ARIA = 'store:data/cache/aria_poprock'
# name -> (covers, reductions, Aria replay)
MIXES = {
    'c35r45a20': (0.35, 0.45, 0.20),  # finetune_ua450.py as is
    'c60r20a20': (0.60, 0.20, 0.20),  # covers-heavy
    'c80r00a20': (0.80, 0.00, 0.20),  # covers + replay only, no reductions
    'c50r40a10': (0.50, 0.40, 0.10),  # less replay
    'c25r45a30': (0.25, 0.45, 0.30),  # more replay (less forgetting of the base)
}
# variant suffix -> extra train.py overrides
VARIANTS = {
    '': [],
    '_do0': ['--dropout=0.0'],  # as the base was pretrained
}
# phase 1: (mix, LR, variant); ua_450_s1 = c35r45a20 at 3e-4 is the failed reference
HYPOTHESES = [
    ('c80r00a20', '3e-4', ''),      # (a) no reductions
    ('c35r45a20', '1e-4', ''),      # (b) lower LR
    ('c80r00a20', '1e-4', ''),      # (c) both
    ('c80r00a20', '1e-4', '_do0'),  # (d) both + dropout 0
]
SHORT_ITERS = 300   # 12 windows/step: ~4.4 covers passes at 0.35
LONG_ITERS = 900
SEC_PER_IT = 4.3    # ua_450_s1 on the laptop incl. evals every 50 steps
STARTUP_SEC = 120   # data load + torch.compile
GATE_MARGIN = 0.005 # the phase-1 winner must beat step 0 by this much on covers val CE
STEP = re.compile(r'^[ \t]*step (\d+): .*?\| covers_v4 CE ([\d.]+) \(.*?(?:\| nocond CE ([\d.]+))?$', re.M)
SUMMARY = ROOT / 'logs/ua450_sweep.txt'


def run_name(mix, lr, variant=''):
    return f'{mix}_lr{lr}{variant}'


def parse_name(name):
    m = re.fullmatch(r'(c\d+r\d+a\d+)_lr([^_]+)(_.+)?', name)
    if not m or m[1] not in MIXES or (m[3] or '') not in VARIANTS:
        sys.exit(f'bad run name {name!r}: <mix>_lr<LR><variant>, mixes {list(MIXES)}, variants {list(VARIANTS)}')
    return m[1], m[2], m[3] or ''


def train_args(name, iters, long=False):
    mix, lr, variant = parse_name(name)
    c, r, a = MIXES[mix]
    sources = [(COVERS, c, 1), (REDUCTIONS, r, 1), (ARIA, a, 0)]  # (path, weight, tempo aug)
    sources = [s for s in sources if s[1] > 0]
    args = [f'--csv_path={",".join(s[0] for s in sources)}',
            f'--source_weights={",".join(str(s[1]) for s in sources)}',
            f'--aug_stores={",".join(str(s[2]) for s in sources)}',
            f'--learning_rate={lr}', *VARIANTS[variant],
            f'--max_iters={iters}', f'--lr_decay_iters={iters}',
            f'--out_dir=checkpoints/ua450_mix_{name}{"_long" if long else ""}',
            f'--wandb_run_name=ua450-mix-{name}{"-long" if long else ""}']
    if not long:
        args += ['--ckpt_interval_min=0', '--save_pre_cooldown=False']
    return args


def log_path(name, long=False):
    return ROOT / f'logs/ua450_mix_{name}{"_long" if long else ""}.log'


def results(log):
    """[(step, covers val CE, nocond CE or None)] of a run's evals."""
    if not log.exists():
        return []
    text = log.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    return [(int(s), float(v), float(n) if n else None) for s, v, n in STEP.findall(text)]


def finished(log, iters):
    return any(s == iters - 1 for s, _, _ in results(log))


def run(name, iters, long=False):
    log = log_path(name, long)
    if finished(log, iters):
        print(f'{log.name}: already done, skipped')
        return
    print(f'{time.strftime("%H:%M")} {log.name}: {iters} steps, ~{(iters * SEC_PER_IT + STARTUP_SEC) / 60:.0f} min',
          flush=True)
    cmd = ['.venv/bin/python', 'train.py', *CONFIGS, *train_args(name, iters, long)]
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code or not finished(log, iters):
        sys.exit(f'{log.name}: train.py failed (exit {code}), see the log')


def summarize(names, long_name=None):
    lines = [f'ua450 mix/LR sweep, {time.strftime("%Y-%m-%d %H:%M")} (covers v4 val CE; lower is better)',
             f'{"run":24s} {"step 0":>7s} {"best":>7s} {"@step":>6s} {"final":>7s} {"nocond@best":>12s}']
    best = {}
    runs = [(n, log_path(n)) for n in names] + ([(long_name + ' long', log_path(long_name, True))] if long_name else [])
    for n, log in runs:
        res = results(log)
        if not res:
            continue
        s_best, v_best, nc = min(res, key=lambda x: x[1])
        best[n] = (v_best, res[0][1])
        lines.append(f'{n:24s} {res[0][1]:7.3f} {v_best:7.3f} {s_best:6d} {res[-1][1]:7.3f} '
                     f'{nc if nc is not None else float("nan"):12.3f}')
    SUMMARY.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--long-run', default='', help='skip phases 1-2 and run the long run, e.g. c60r20a20_lr1e-4')
    args = ap.parse_args()
    hyp = [run_name(*h) for h in HYPOTHESES]
    if args.dry_run:
        short = SHORT_ITERS * SEC_PER_IT + STARTUP_SEC
        long = LONG_ITERS * SEC_PER_IT + STARTUP_SEC
        n_grid = len(MIXES) - 1  # at most: the mixes not yet run at the winner's LR and variant
        print(f'phase 1: {", ".join(hyp)}, {SHORT_ITERS} steps each, ~{len(hyp) * short / 3600:.1f} h')
        print(f'phase 2: <= {n_grid} mixes at the winning LR + variant, {SHORT_ITERS} steps each, '
              f'<= ~{n_grid * short / 3600:.1f} h')
        print(f'phase 3: best run, {LONG_ITERS} steps, ~{long / 60:.0f} min')
        print(f'total <= ~{((len(hyp) + n_grid) * short + long) / 3600:.1f} h')
        print('example:', ' '.join(['train.py', *CONFIGS, *train_args(hyp[-1], SHORT_ITERS)]))
        return
    if args.long_run:
        parse_name(args.long_run)
        run(args.long_run, LONG_ITERS, long=True)
        summarize(hyp, args.long_run)
        return

    for name in hyp:
        run(name, SHORT_ITERS)
    best = summarize(hyp)
    winner = min(best, key=lambda n: best[n][0])
    v_best, step0 = best[winner]
    if v_best > step0 - GATE_MARGIN:
        sys.exit(f'GATE: no hypothesis run beat step 0 on covers val ({step0:.3f}; best {winner} {v_best:.3f}): '
                 'sweep stopped (check the data / the base before comparing mixes)')
    _, lr, variant = parse_name(winner)
    names = hyp + [n for n in (run_name(m, lr, variant) for m in MIXES) if n not in hyp]
    for name in names[len(hyp):]:
        run(name, SHORT_ITERS)
    best = summarize(names)
    winner = min(best, key=lambda n: best[n][0])
    print(f'phase 3: {winner} (best {best[winner][0]:.3f})', flush=True)
    run(winner, LONG_ITERS, long=True)
    summarize(names, winner)
    print('SWEEP DONE')


if __name__ == '__main__':
    main()
