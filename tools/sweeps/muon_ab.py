"""
Muon variants at 42M (todo c7cb78dd; optimizations.md "Paper search before F1"): the abl_2048 recipe (12L x 512, new
block, Muon 1e-3 WSD, 1,500 it x 65,536 notes) on the local stores (config/pretrain/abl_2048_local.py), one arm per
variant of musicar/training/optim.py Muon, seeds 1337 / 2 / 3 run round-robin (every arm at a seed before the next
seed, so a stopped sweep still compares all arms). Arm rms = the step-size control for normuon (both update at RMS
0.2 exactly; base Muon's Moonlight scale gives RMS 0.2 only when every orthogonalized singular value is 1). Final held-out GigaMIDI CE and Aria CE per run in logs/muon_ab.md;
a run whose log has its last eval is skipped. ckpt.pt and pre_cooldown.pt (optimizer state) deleted after each
run, model_bf16.pt kept.

    .venv/bin/python tools/sweeps/muon_ab.py --dry-run
    .venv/bin/python tools/sweeps/muon_ab.py                       # 12 runs, ~25 min each
    .venv/bin/python tools/sweeps/muon_ab.py --seeds 1337          # a 1-seed screen
    .venv/bin/python tools/sweeps/muon_ab.py --table-only
"""

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.jobstatus import JobStatus  # noqa: E402

CONFIGS = ['config/pretrain/abl_2048.py', 'config/pretrain/abl_2048_local.py']
ARMS = {
    'base': [],
    'polar': ['--muon_ns=polar'],
    'normuon': ['--muon_normuon=True'],
    'cautious': ['--muon_cautious=True'],
    'rms': ['--muon_rms=True'],  # NorMuon's step size (RMS 0.2 exactly) without its per-neuron division
}
SEEDS = (1337, 2, 3)
LAST = 'step 1499:'
TABLE = ROOT / 'logs/muon_ab.md'


def name(arm, seed):
    return f'muon_{arm}_s{seed}'


def log_path(arm, seed):
    return ROOT / f'logs/{name(arm, seed)}.log'


def result(arm, seed):
    """(held-out GigaMIDI CE, Aria CE) of the last eval, None if the run has not finished."""
    p = log_path(arm, seed)
    if not p.exists():
        return None
    line = next((ln for ln in reversed(p.read_text(errors='replace').replace('\r', '\n').splitlines())
                 if LAST in ln), None)
    if line is None:
        return None
    ces = dict(re.findall(r'(\S+) CE ([\d.]+)', line))
    return float(ces['gigamidi_heldout']), float(ces['aria'])


def on_battery():
    try:
        return any(p.read_text().strip() == '0' for p in Path('/sys/class/power_supply').glob('A*/online'))
    except OSError:
        return False


def table(arms, seeds):
    lines = ['# Muon variants at 42M (tools/sweeps/muon_ab.py; abl_2048 recipe on the local stores, 1,500 it)', '',
             '| arm | ' + ' | '.join(f'held-out s{s}' for s in seeds) + ' | held-out mean | vs base | Aria mean | vs base |',
             '|---|' + '---|' * (len(seeds) + 4)]
    means = {}
    for arm in arms:
        rs = [result(arm, s) for s in seeds]
        done = [r for r in rs if r]
        means[arm] = (statistics.mean(r[0] for r in done), statistics.mean(r[1] for r in done)) if done else None
    for arm in arms:
        rs = [result(arm, s) for s in seeds]
        m, b = means[arm], means.get('base')
        # vs base over the seeds both arms finished
        both = [s for s in seeds if result(arm, s) and result('base', s)]
        d = [statistics.mean(result(arm, s)[i] - result('base', s)[i] for s in both) if both else None for i in (0, 1)]
        fmt = lambda x: f'{x:.4f}' if x is not None else ''  # noqa: E731
        sign = lambda x: f'{x:+.4f}' if x is not None and arm != 'base' else ''  # noqa: E731
        lines.append(f'| {arm} | ' + ' | '.join(fmt(r[0]) if r else '' for r in rs)
                     + f' | {fmt(m[0]) if m else ""} | {sign(d[0])} | {fmt(m[1]) if m else ""} | {sign(d[1])} |')
    lines += ['', 'vs base: mean difference over the seeds both arms finished (negative = better).']
    TABLE.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--table-only', action='store_true')
    args = p.parse_args()
    arms, seeds = args.arms.split(','), [int(s) for s in args.seeds.split(',')]
    assert all(a in ARMS for a in arms), arms
    if args.table_only:
        return table(arms, seeds)
    todo = [(a, s) for s in seeds for a in arms if result(a, s) is None]
    print(f'{len(todo)} runs')
    job = None if args.dry_run else JobStatus('muon_ab', len(todo), f'{len(todo)} runs')
    for i, (arm, seed) in enumerate(todo):
        n = name(arm, seed)
        cmd = ['.venv/bin/python', 'train.py', *CONFIGS, f'--out_dir=checkpoints/{n}', f'--wandb_run_name={n}',
               f'--seed={seed}', *ARMS[arm]]
        print(f'{time.strftime("%H:%M")}   $ {" ".join(cmd)} > logs/{n}.log', flush=True)
        if args.dry_run:
            continue
        while on_battery():
            print('on battery: waiting for AC', flush=True)
            time.sleep(300)
        job.update(i, f'{arm} seed {seed}', force=True)
        with open(log_path(arm, seed), 'w') as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f'{time.strftime("%H:%M")}   {n}: exit {code}', flush=True)
        for f in ('ckpt.pt', 'pre_cooldown.pt'):
            (ROOT / f'checkpoints/{n}/{f}').unlink(missing_ok=True)
        table(arms, seeds)
    if job:
        job.finish(str(TABLE))


if __name__ == '__main__':
    main()
