"""
MHA -> GQA conversion layer by layer on the 107M section model (musicar/model/gqa.py): v2_A0's 12 key / value heads
become 4 (groups of 3, as F1's 18 -> 6), then the forgetting-sweep recipe for 1,500 it (config/pilots/section_forget.py
without cf / splice / repeat), the same as arm gqa4 of tools/sweeps/section_forget.py (mean-pooled K / V straight
into the recipe: held-out CE 3.166 / 3.037 / 2.972 at it 500 / 1000 / 1499 vs forget_ctrl 2.790 / 2.790 / 2.772).

Arms:
  lw        stage 1 = gqa_init 'pca' (data-aware merge from each layer's input moments, 16 calibration batches through
            the teacher stream) + attention transfer: only the attention layers train, 300 it at LR 1e-3, each on
            the relative MSE to its own full attention (the rest of the model sees the original states; no compile /
            fp8); then stage 2 = the 1,500 it recipe from stage 1's ckpt.pt
  pca       stage 2 only, from the pca merge (does the transfer add anything?)
  mean_lw   stage 1 from the mean-pool instead of the merge (does the merge matter once there is a transfer?)

Each stage runs through tools/sweeps/section_long.py (one chunk + the change probe), as in linear_convert.py; an arm
whose stage-2 probe is at the last iteration is not trained again. Table logs/sections/gqa.md: held-out CE at it 0 /
500 / 1000 / 1499 and the gap to forget_ctrl, Aria CE, val section gains, the probe, s/it. Disk: a finished arm keeps
only its stage-2 model_bf16.pt (and the logs).

    .venv/bin/python tools/sweeps/gqa_convert.py --dry-run
    .venv/bin/python tools/sweeps/gqa_convert.py                    # lw + pca, ~2 h on the laptop
    .venv/bin/python tools/sweeps/gqa_convert.py --arms lw --stage1-only --s1-iters 150   # ~10 min: the conversion
    .venv/bin/python tools/sweeps/gqa_convert.py --table-only
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear_convert import log_text, probe_row, run_stage  # noqa: E402

MAX_ITERS = 1500
S1_ITERS = 300
S1_ARGS = ['--attn_transfer=True', '--compile=False', '--fp8=False', '--learning_rate=1e-3', '--warmup_iters=20',
           '--eval_interval=50']
KV = ['--n_kv_head=4']
ARMS = {'lw': dict(args=KV + ['--gqa_init=pca'], transfer=True),
        'pca': dict(args=KV + ['--gqa_init=pca'], transfer=False),
        'mean_lw': dict(args=KV, transfer=True)}
REFERENCES = ('forget_ctrl', 'forget_gqa4')  # full attention / mean-pool straight into the recipe (section_forget.py)
TABLE = ROOT / 'logs/sections/gqa.md'
STEP = re.compile(r'step (\d+): train loss ([\d.na]+).*?\| \S+ CE ([\d.]+) \(.*?\| aria_labeled CE ([\d.]+).*?'
                  r'section gain ([-+.\d]+) / none ([-+.\d]+) / prev ([-+.\d]+)')
SPEED = re.compile(r'([\d.]+)s/it')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--arms', default='lw,pca')
    p.add_argument('--minutes', type=float, default=240, help='training minutes per stage (one chunk)')
    p.add_argument('--s1-iters', type=int, default=S1_ITERS)
    p.add_argument('--stage1-only', action='store_true', help='stop after stage 1 (the conversion itself)')
    p.add_argument('--table-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    args.quick = False  # linear_convert.run_stage's flag
    return args


def train_arm(arm, args):
    spec, name = ARMS[arm], f'gqa_{arm}'
    extra = list(spec['args'])
    if spec['transfer']:
        s1 = probe_row(f'{name}_s1')
        if not (s1 and s1['iter'] >= args.s1_iters - 1):
            code = run_stage(f'{name}_s1', args.s1_iters, extra + S1_ARGS, args)
            if code != 0:
                return code
        if args.stage1_only:
            return 0
        extra.append(f'--init_ckpt=checkpoints/{name}_s1/ckpt.pt')
    elif args.stage1_only:
        return 0
    return run_stage(name, MAX_ITERS, extra, args)


def clean_up(arm, args):
    name = f'gqa_{arm}'
    for path in (ROOT / 'checkpoints' / f'{name}_s1', ROOT / 'checkpoints' / name / 'ckpt.pt'):
        if path.exists():
            print(f'{time.strftime("%H:%M")} delete {path.relative_to(ROOT)}', flush=True)
            if not args.dry_run:
                shutil.rmtree(path) if path.is_dir() else path.unlink()


def write_table(arms):
    lines = ['# MHA -> GQA conversion (107M v2_A0, 12 -> 4 KV heads, 1,500 it of the forgetting-sweep recipe)', '',
             f'Updated {time.strftime("%Y-%m-%d %H:%M")}. Held-out CE at it 0 / 500 / 1000 / 1499 (200 held-out '
             'windows) and the gap to forget_ctrl (full attention, same data and iterations) at 1499. forget_gqa4 = '
             'mean-pooled K / V straight into the recipe. Stage 1: transfer loss (relative output MSE summed over the '
             '14 layers) first -> last training step, held-out CE at its start (after the merge) and end. Probe: 512 '
             'held-out windows around section changes of >= 2 points, gain on the first 32 notes (nats per note).', '',
             '| arm | stage 1: transfer, CE start -> end | CE @0 | @500 | @1000 | @1499 | gap @1499 | Aria CE | val sec '
             'gain vs shuffled / none / prev | probe vs shuffled | vs none | s/it |', '|---' * 12 + '|']
    ref = {it: ce for it, _, ce, *_ in STEP.findall(log_text('forget_ctrl'))}
    for arm in list(REFERENCES) + arms:
        name = arm if arm in REFERENCES else f'gqa_{arm}'
        found = STEP.findall(log_text(name))
        at = {f[0]: f for f in found}
        ev = found[-1] if found else None
        s1 = STEP.findall(log_text(f'{name}_s1')) if arm not in REFERENCES and ARMS[arm]['transfer'] else []
        train = [f[1] for f in s1 if f[1] != 'nan']
        stage1 = f'{train[0]} -> {train[-1]}, CE {s1[0][2]} -> {s1[-1][2]}' if s1 and train else ''
        gap = f'{float(ev[2]) - float(ref[ev[0]]):+.3f}' if ev and ev[0] in ref else ''
        pr = probe_row(name)
        speed = SPEED.findall(log_text(name))
        cells = [at[i][2] if i in at else '' for i in ('0', '500', '1000', '1499')]
        lines.append(f"| {arm} | {stage1} | {' | '.join(cells)} | {gap} | {ev[3] if ev else ''} | "
                     f"{' / '.join(ev[4:]) if ev else ''} | "
                     + (f"{pr['gain_shuffled']:+.4f} | {pr['gain_none']:+.4f}" if pr else ' | ')
                     + f" | {speed[-1] if speed else ''} |")
    TABLE.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    args = parse_args()
    arms = [a for a in args.arms.split(',') if a]
    assert all(a in ARMS for a in arms), arms
    if not args.table_only:
        for arm in arms:
            row = probe_row(f'gqa_{arm}')
            if args.stage1_only or not (row and row['iter'] >= MAX_ITERS - 1):
                print(f'{time.strftime("%H:%M")} {arm}: exit {train_arm(arm, args)}', flush=True)
                row = probe_row(f'gqa_{arm}')
            if not args.stage1_only and row and row['iter'] >= MAX_ITERS - 1:
                clean_up(arm, args)
            if not args.dry_run:
                write_table(arms)
    if not args.dry_run:
        write_table(arms)


if __name__ == '__main__':
    main()
