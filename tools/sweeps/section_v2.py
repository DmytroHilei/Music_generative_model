"""
Section captions v2 (config/pilots/section_v2.py: change-aware section captions, the new label tables, Aria): a quick
check of three arms on the same data and LR schedule, then the winner trained on to the end (context/claude/
text_control.md "Section captions").

  1. arms (one chunk of --arm-minutes each, enough for 1,000 iterations; evaluations every 500 write the weights),
     each probed around section changes by tools/sweeps/section_long.py (eval/probes/section_gain.py --at-changes):
       A0    prefix, the section's caption arrives with its first note (usable live)
       A256  prefix, the caption arrives a whole section early (the long_A256 setting)
       B0    gated cross-attention after every 2nd block, caption with the first note
  2. winner: the highest change-start gain vs shuffled captions (first 32 notes of sections that changed by >= 2
     points) at 1,000 iterations; below --tie-gain for every arm -> A0 (live, the simplest).
     Written to logs/sections/v2_decision.md.
  3. the winner resumes from its ckpt.pt (the arms already run the full schedule: WSD, constant LR until 8,000 of
     10,000) in --minutes chunks to 10,000 iterations, probed after each chunk; early stop when the gain vs shuffled
     is < --min-gain from --stop-at on.
  4. --then-queue: tools/sweeps/gen_queue.py (the generation re-runs after the drum fix).
Arms already probed at >= 1,000 iterations are not trained again (the script can be restarted).

    .venv/bin/python tools/sweeps/section_v2.py --dry-run
    .venv/bin/python tools/sweeps/section_v2.py --then-queue      # the night of 2026-10-20: ~1.8 + ~4.6 + ~1.3 h
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'config/pilots/section_v2.py'
MAX_ITERS = 10000
DECIDE_AT = 1000
ARMS = {
    'A0': ['--section_lead=0'],
    'A256': ['--section_lead=256'],
    'B0': ['--section_mode=xattn', '--section_lead=0'],
}
DECISION = ROOT / 'logs/sections/v2_decision.md'


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--arms', default=','.join(ARMS), help='arms of the quick check')
    p.add_argument('--arm-minutes', type=float, default=32, help='training minutes per arm (>= 1,000 iterations)')
    p.add_argument('--tie-gain', type=float, default=0.002, help='below this for every arm: A0')
    p.add_argument('--minutes', type=float, default=90, help='chunk minutes of the winner')
    p.add_argument('--stop-at', type=int, default=4000)
    p.add_argument('--min-gain', type=float, default=0.002)
    p.add_argument('--then-queue', action='store_true', help='run tools/sweeps/gen_queue.py at the end')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def driver(arm, extra, dry):
    """tools/sweeps/section_long.py for an arm: the same name, out dir, config and train.py arguments every time (the
    driver refuses a chunk with other ones)."""
    cmd = ['.venv/bin/python', 'tools/sweeps/section_long.py', '--name', f'v2_{arm}', '--out-dir',
           f'checkpoints/v2_{arm}', '--config', CONFIG, '--max-iters', str(MAX_ITERS), *extra,
           '--', *ARMS[arm], f'--wandb_run_name=sec-v2-{arm}']
    print(f'{time.strftime("%H:%M")} ' + ' '.join(cmd[1:]), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def probe_at(arm, at=DECIDE_AT):
    """The probe row of an arm's latest weights at >= `at` iterations, or None."""
    path = ROOT / 'logs/sections' / f'v2_{arm}.json'
    if not path.exists():
        return None
    rows = [r for r in json.loads(path.read_text()).get('probes', []) if r['iter'] >= at]
    return min(rows, key=lambda r: r['iter']) if rows else None


def main():
    args = parse_args()
    arms = [a for a in args.arms.split(',') if a]
    for arm in arms:
        if probe_at(arm):
            print(f'{arm}: probed at {probe_at(arm)["iter"]} already')
            continue
        driver(arm, ['--minutes', f'{args.arm_minutes:g}'], args.dry_run)
    rows = {arm: probe_at(arm) for arm in arms}
    if args.dry_run:
        return
    missing = [a for a, r in rows.items() if r is None]
    if missing:
        sys.exit(f'no probe at >= {DECIDE_AT} iterations for {missing} (see logs/sections/v2_<arm>.log)')
    best = max(arms, key=lambda a: rows[a]['gain_shuffled'])
    winner = best if rows[best]['gain_shuffled'] >= args.tie_gain else 'A0'
    lines = [f'# Section captions v2: quick check ({time.strftime("%Y-%m-%d %H:%M")})', '',
             'Change-start gain (first 32 notes of sections after a change of >= 2 points, nats per note).', '',
             '| arm | settings | weights @iter | vs shuffled | vs none | no-change sections vs none | excess |',
             '|---|---|---|---|---|---|---|']
    for a in arms:
        r = rows[a]
        lines.append(f"| {a} | {' '.join(ARMS[a])} | {r['iter']} | {r['gain_shuffled']:+.4f} | {r['gain_none']:+.4f} "
                     f"| {r['gain_none_nochange']:+.4f} | {r['excess']:+.3f} |")
    lines += ['', f'Winner: **{winner}**' + ('' if winner == best else
                                             f' (no arm reached {args.tie_gain} vs shuffled: A0 by the tie rule)')]
    DECISION.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    code = driver(winner, ['--minutes', f'{args.minutes:g}', '--chunks', '0', '--stop-at', str(args.stop_at),
                           '--min-gain', f'{args.min_gain:g}'], False)
    print(f'{time.strftime("%H:%M")} {winner}: driver exit {code}', flush=True)
    if args.then_queue:
        print(f'{time.strftime("%H:%M")} gen_queue', flush=True)
        subprocess.run(['.venv/bin/python', 'tools/sweeps/gen_queue.py'], cwd=ROOT)


if __name__ == '__main__':
    main()
