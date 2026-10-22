"""
Attention -> linear attention conversion test on the 107M section model (context/claude/optimizations.md "Paper
search before F1"; musicar/model/linear_attn.py). v2_A0's attention in 10 of its 14 blocks becomes a Gated DeltaNet
(linear_attn='hybrid4': full attention stays in blocks 3, 7, 11, 13, as Qwen3-Next's 3 : 1), c_attn / c_proj carried
over, then the forgetting-sweep recipe for 1,500 it (config/pilots/section_forget.py without cf / splice / repeat), so
the full-attention reference is the existing arm forget_ctrl (v2_A0 + the same 1,500 it).

Arms:
  h4          stage 1 = attention transfer (LoLCATs): only the linear layers train, 300 it at LR 1e-3, on the relative
              MSE to the attention they replace (the rest of the model sees the original states; no compile / fp8),
              then stage 2 = the 1,500 it recipe on everything
  h4_direct   stage 2 only, straight from the copied weights (does stage 1 matter?)

Each stage runs through tools/sweeps/section_long.py (one chunk + the change probe); an arm whose stage-2 probe is at
the last iteration is not trained again (restartable). Table logs/sections/linear.md: held-out / Aria CE, val section
gains, the change probe, s/it (from the training logs; the GDN layers are plain PyTorch, not a fused kernel).
Disk (low, user 2026-10-22): when an arm is done its stage-1 directory and its resumable ckpt.pt are deleted; the
stage-2 model_bf16.pt (probes, generation) and all logs / probe results stay.

    .venv/bin/python tools/sweeps/linear_convert.py --dry-run
    .venv/bin/python tools/sweeps/linear_convert.py                 # ~3-4 h on the laptop
    .venv/bin/python tools/sweeps/linear_convert.py --table-only
    .venv/bin/python tools/sweeps/linear_convert.py --arms h4 --quick   # ~30 min check (GPU path, speed, early CE)

--quick: stage 1 = 150 it (at most 10 min), stage 2 = one 12-minute chunk with evaluations every 200 it, runs named
linear_<arm>_quick*, no clean-up; table logs/sections/linear_quick.md with forget_ctrl at step 500 as the reference.
"""

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ['config/pilots/section_v2.py', 'config/pilots/section_forget.py']
MAX_ITERS = 1500
S1_ITERS = 300
S1_ARGS = ['--attn_transfer=True', '--compile=False', '--fp8=False', '--learning_rate=1e-3', '--warmup_iters=20',
           '--eval_interval=100']
ARMS = {'h4': dict(linear_attn='hybrid4', transfer=True),
        'h4_direct': dict(linear_attn='hybrid4', transfer=False)}
REFERENCE = 'forget_ctrl'  # v2_A0 + the same 1,500 it with full attention (tools/sweeps/section_forget.py)
TABLE = ROOT / 'logs/sections/linear.md'
STEP = re.compile(r'step (\d+): train loss ([\d.]+).*?\| \S+ CE ([\d.]+) \(.*?\| aria_labeled CE ([\d.]+).*?'
                  r'section gain ([-+.\d]+) / none ([-+.\d]+) / prev ([-+.\d]+)')
SPEED = re.compile(r'([\d.]+)s/it')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--minutes', type=float, default=240, help='training minutes per stage (one chunk)')
    p.add_argument('--quick', action='store_true', help='~30-minute check instead of the full arms')
    p.add_argument('--table-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def run_stage(name, iters, extra, args, minutes=None):
    chunks = '1' if args.quick else '0'
    cmd = ['.venv/bin/python', 'tools/sweeps/section_long.py', '--name', name, '--out-dir', f'checkpoints/{name}',
           '--max-iters', str(iters), '--minutes', f'{minutes or args.minutes:g}', '--chunks', chunks,
           '--stop-at', '100000']
    for c in CONFIGS:
        cmd += ['--config', c]
    cmd += ['--', *extra, f'--wandb_run_name=sec-{name.replace("_", "-")}']
    print(f'{time.strftime("%H:%M")} ' + ' '.join(cmd[1:]), flush=True)
    return 0 if args.dry_run else subprocess.run(cmd, cwd=ROOT).returncode


def probe_row(name):
    path = ROOT / 'logs/sections' / f'{name}.json'
    rows = json.loads(path.read_text()).get('probes', []) if path.exists() else []
    return max(rows, key=lambda r: r['iter']) if rows else None


def log_text(name):
    path = ROOT / 'logs/sections' / f'{name}.log'
    return path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''


def run_name(arm, args):
    return f'linear_{arm}_quick' if args.quick else f'linear_{arm}'


def train_arm(arm, args):
    spec = ARMS[arm]
    name = run_name(arm, args)
    s1_iters = 150 if args.quick else S1_ITERS
    extra = [f'--linear_attn={spec["linear_attn"]}']
    if spec['transfer']:
        s1 = probe_row(f'{name}_s1')
        if not (s1 and s1['iter'] >= s1_iters - 1):
            code = run_stage(f'{name}_s1', s1_iters, extra + S1_ARGS, args, minutes=10 if args.quick else None)
            if code != 0:
                return code
        extra.append(f'--init_ckpt=checkpoints/{name}_s1/ckpt.pt')
    if args.quick:
        extra.append('--eval_interval=200')
    return run_stage(name, MAX_ITERS, extra, args, minutes=12 if args.quick else None)


def clean_up(arm, args):
    """A finished arm keeps only checkpoints/linear_<arm>/model_bf16.pt: stage 1 was only its init, ckpt.pt (weights +
    optimizer, ~1.4 GB) only resumes training."""
    name = f'linear_{arm}'
    for path in (ROOT / 'checkpoints' / f'{name}_s1', ROOT / 'checkpoints' / name / 'ckpt.pt'):
        if path.exists():
            print(f'{time.strftime("%H:%M")} delete {path.relative_to(ROOT)}', flush=True)
            if not args.dry_run:
                shutil.rmtree(path) if path.is_dir() else path.unlink()


def write_table(arms, args):
    table = TABLE.with_name('linear_quick.md') if args.quick else TABLE
    lines = ['# Attention -> Gated DeltaNet conversion (107M v2_A0, 1,500 it of the forgetting-sweep recipe)'
             + (': 30-minute check, reference at step 500' if args.quick else ''), '',
             f'Updated {time.strftime("%Y-%m-%d %H:%M")}. hybrid4 = linear attention in 10 of 14 blocks (full in 3, 7, '
             '11, 13). Reference: forget_ctrl (full attention, same data and iterations). Val: last evaluation (200 '
             'held-out windows). Probe: 512 held-out windows around section changes of >= 2 points, gain on the first '
             '32 notes of the changed section (nats per note). Stage 1 (transfer) MSE = relative output error of the '
             'linear layers vs their attention, summed over the 10 layers, last training step. s/it: last tqdm '
             'reading of the run (laptop, 65,536 notes/it).', '',
             '| arm | stage 1: MSE start -> end, held-out CE | iter | held-out CE | Aria CE | val sec gain vs shuffled / '
             'none / prev | probe vs shuffled | vs none | vs previous | s/it |', '|---' * 10 + '|']
    for arm in [REFERENCE] + arms:
        name = arm if arm == REFERENCE else run_name(arm, args)
        found = STEP.findall(log_text(name))
        if args.quick and arm == REFERENCE:
            found = [f for f in found if f[0] == '500']
        ev = found[-1] if found else None
        s1 = STEP.findall(log_text(f'{name}_s1')) if arm != REFERENCE and ARMS[arm]['transfer'] else []
        stage1 = f'{s1[1][1]} -> {s1[-1][1]}, {s1[-1][2]}' if len(s1) > 1 else ''
        pr = probe_row(name)
        speed = SPEED.findall(log_text(name))
        lines.append(f"| {arm} | {stage1} | {ev[0] if ev else ''} | {ev[2] if ev else ''} | {ev[3] if ev else ''} | "
                     f"{' / '.join(ev[4:]) if ev else ''} | "
                     + (f"{pr['gain_shuffled']:+.4f} | {pr['gain_none']:+.4f} | {pr.get('gain_prev', float('nan')):+.4f}"
                        if pr else ' | | ') + f" | {speed[-1] if speed else ''} |")
    table.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    args = parse_args()
    arms = [a for a in args.arms.split(',') if a]
    assert all(a in ARMS for a in arms), arms
    if not args.table_only:
        for arm in arms:
            row = probe_row(run_name(arm, args))
            if args.quick or not (row and row['iter'] >= MAX_ITERS - 1):
                code = train_arm(arm, args)
                print(f'{time.strftime("%H:%M")} {arm}: exit {code}', flush=True)
                row = probe_row(run_name(arm, args))
            if not args.quick and row and row['iter'] >= MAX_ITERS - 1:
                clean_up(arm, args)
            if not args.dry_run:
                write_table(arms, args)
    if not args.dry_run:
        write_table(arms, args)


if __name__ == '__main__':
    main()
