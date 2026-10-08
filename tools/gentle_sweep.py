"""
Gentle fine-tunes of the 450M base on the Ukrainian piano data (to-do 15d, 2026-10-08), judged by generation as well
as by covers val CE. Every earlier fine-tune (LR 1e-4..3e-4, full model) raised covers val CE from step ~50 while
the train loss kept falling, but the 107M fine-tunes didn't lower covers CE either and still generated much
closer to real covers, so each arm keeps bf16 snapshots at fixed note budgets and eval/seed_compare.py scores them
(27 Ukrainian val covers continued from their first 128 notes, at the sampling settings the H3 sweep picked).

Arms (all: finetune_ua450.py + finetune_ua450_full.py data and mix, dropout 0 as the base was pretrained, the same
notes seen = ITERS x 12 windows ~ 1.2 passes over the covers):
    muon_3e-5      H5: LR 3e-5 instead of 1e-4..3e-4
    muon_1e-5      H5: LR 1e-5
    adamw8_3e-5    H6: AdamW (torchao 8-bit states, fp32 AdamW doesn't fit 8 GB at 450M) instead of Muon
    muon_3e-5_frz  H7: input embeddings + condition tables frozen
    muon_3e-5_b4   H9: 4x the batch (48 windows/step), 1/4 of the steps

    .venv/bin/python tools/gentle_sweep.py   # every arm, then its scores at the H3 pick
    .venv/bin/python tools/gentle_sweep.py --dry-run
    .venv/bin/python tools/gentle_sweep.py --arms muon_3e-5 adamw8_3e-5 --settings ...

Logs logs/gentle/<arm>.log, generation scores logs/gentle/<arm>_gen.{txt,json}, out_dir checkpoints/gentle_<arm>
(model_bf16_it<N>.pt per snapshot, ~0.9 GB each). Finished parts are skipped on a re-run. Summary (CE and generation
distance per snapshot): logs/gentle/summary.txt.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ['config/finetune_ua450.py', 'config/finetune_ua450_full.py']
ITERS = 200          # x 12 windows x ~0.35 covers share x ~1,600 notes ~ 1.3M covers notes ~ 1.2 passes
SNAPSHOTS = (0.125, 0.25, 0.5, 1.0)  # fractions of the run (1.0 = the last step)
FREEZE = 'transformer.music_embeddings,transformer.cond_inst,transformer.cond_density,transformer.program'
ARMS = {
    'muon_3e-5': ['--learning_rate=3e-5'],
    'muon_1e-5': ['--learning_rate=1e-5'],
    'adamw8_3e-5': ['--learning_rate=3e-5', '--optimizer_name=adamw8bit'],
    'muon_3e-5_frz': ['--learning_rate=3e-5', f'--freeze={FREEZE}'],
    'muon_3e-5_b4': ['--learning_rate=3e-5', '--gradient_accumulation_steps=48'],
}
BATCH_MULT = {'muon_3e-5_b4': 4}
SEC_PER_WINDOW = 2.9 / 12  # laptop, micro 1 (ua450 sweep 2026-10-07)
STARTUP_SEC = 150
STEP = re.compile(r'^[ \t]*step (\d+): .*?\| covers_v4 CE ([\d.]+) ', re.M)
OUT = ROOT / 'logs/gentle'


def plan(arm):
    """(iters, eval_interval, snapshot iters) of an arm: the same notes seen in every arm."""
    m = BATCH_MULT.get(arm, 1)
    iters = ITERS // m
    snaps = sorted({max(1, round(f * iters)) if f < 1 else iters - 1 for f in SNAPSHOTS})
    return iters, max(1, 25 // m), snaps


def ce_curve(log):
    if not log.exists():
        return {}
    text = log.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    return {int(s): float(v) for s, v in STEP.findall(text)}


def train(arm):
    iters, every, snaps = plan(arm)
    log = OUT / f'{arm}.log'
    out_dir = ROOT / f'checkpoints/gentle_{arm}'
    if all((out_dir / f'model_bf16_it{s}.pt').exists() for s in snaps):
        print(f'{arm}: trained already, skipped')
        return
    print(f'{time.strftime("%H:%M")} {arm}: {iters} steps, snapshots {snaps}, '
          f'~{(ITERS * 12 * SEC_PER_WINDOW + STARTUP_SEC) / 60:.0f} min', flush=True)
    cmd = ['.venv/bin/python', 'train.py', *CONFIGS, '--dropout=0.0', *ARMS[arm],
           f'--max_iters={iters}', f'--lr_decay_iters={iters}', f'--warmup_iters={max(2, 10 // BATCH_MULT.get(arm, 1))}',
           f'--eval_interval={every}', f'--save_iters={",".join(map(str, snaps))}',
           '--ckpt_interval_min=0', '--save_pre_cooldown=False',
           f'--out_dir={out_dir.relative_to(ROOT)}', f'--wandb_run_name=ua450-gentle-{arm}']
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f'{arm}: train.py failed (exit {code}), see {log}')


def generate(arm, settings, seeds):
    _, _, snaps = plan(arm)
    js = OUT / f'{arm}_gen.json'
    if js.exists():
        print(f'{arm}: scored already, skipped')
        return
    cks = [f'checkpoints/gentle_{arm}/model_bf16_it{s}.pt' for s in snaps]
    cmd = ['.venv/bin/python', 'eval/seed_compare.py', '--checkpoints', *cks, '--settings', *settings,
           '--seeds', *map(str, seeds), '--out', str(OUT / f'{arm}_gen.txt'), '--json', str(js)]
    print(f'{time.strftime("%H:%M")} {arm}: scoring {len(cks)} snapshots', flush=True)
    with open(OUT / f'{arm}_gen.log', 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f'{arm}: seed_compare failed (exit {code}), see {OUT / f"{arm}_gen.log"}')


def summarize(arms):
    lines = [f'gentle 450M fine-tunes, {time.strftime("%Y-%m-%d %H:%M")}: covers v4 val CE (step 0 = 5.449) and '
             f'seed_compare distance to the real continuations (lower = closer) per snapshot',
             f'{"arm":16s}{"iter":>6s}{"notes":>7s}{"CE":>8s}  {"setting":22s}{"dist":>7s}{"nps":>7s}{"melRep":>8s}'
             f'{"recur":>7s}{"keySim":>8s}']
    for arm in arms:
        ce = ce_curve(OUT / f'{arm}.log')
        js = OUT / f'{arm}_gen.json'
        recs = json.loads(js.read_text()) if js.exists() else []
        m = BATCH_MULT.get(arm, 1)
        for r in recs:
            it = r['iter']
            near = min(ce, key=lambda s: abs(s - it)) if ce else None  # CE at the nearest eval
            lines.append(f'{arm:16s}{it:6d}{it * m * 12:7d}{ce[near] if near is not None else float("nan"):8.3f}  '
                         f'{r["setting"]:22s}{r["dist"]:7.3f}{r["nps"]:7.2f}{r["mel_rep"]:8.1f}{r["recur"]:7.3f}'
                         f'{r["keySim"]:8.3f}')
    (OUT / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--settings', nargs='+', default=['T=0.9,dt=0.1,dens=prompt'],
                    help='seed_compare sampling settings (H3 pick for the 450M base, 2026-10-09)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--train-only', action='store_true', help='train every arm, score none')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for arm in args.arms:
            iters, every, snaps = plan(arm)
            print(f'{arm:16s} {iters} steps, eval every {every}, snapshots {snaps}: {" ".join(ARMS[arm])}')
        print(f'~{len(args.arms) * (ITERS * 12 * SEC_PER_WINDOW + STARTUP_SEC) / 60:.0f} min of training, '
              f'{len(args.arms) * len(SNAPSHOTS) * 0.9:.0f} GB of snapshots, plus scoring')
        return
    for arm in args.arms:
        train(arm)
        if not args.train_only:
            generate(arm, args.settings, args.seeds)
    summarize(args.arms)


if __name__ == '__main__':
    main()
