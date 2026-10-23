"""
Section-caption forgetting (config/pilots/section_forget.py): arms continuing v2_A0 for 1,500 iterations, each run by
tools/sweeps/section_long.py (one chunk, then the probe around section changes: change-start gain vs shuffled / none /
previous section caption), then section_follow hold + switch (cfg 2, section prompts, switch at note 1024). Arms whose
probe exists are not trained again; arms whose section_follow summary exists are not generated again (the script can
be restarted).

Table logs/sections/forget.md: held-out / Aria CE and the val section gains (last evaluation in the training log),
the probe, and per switch pair the switched share = effect / |d before the switch| (0 = keeps the first prompt's
style, 1 = follows the new prompt fully; v2_A0 @10k: ~0).

    .venv/bin/python tools/sweeps/section_forget.py --dry-run
    .venv/bin/python tools/sweeps/section_forget.py                      # ~6.3 h
    .venv/bin/python tools/sweeps/section_forget.py --arms cf,all --table-only
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ['config/pilots/section_v2.py', 'config/pilots/section_forget.py']
MAX_ITERS = 1500
CF = ['--section_cf_weight=1.0', '--section_cf_eta=0.05', '--section_cls_weight=0.1']
ARMS = {
    'cf': CF,
    'all': CF + ['--section_splice_frac=0.25', '--section_repeat=64'],
    'ctrl': [],
    'splice': ['--section_splice_frac=0.25'],
    'repeat': ['--section_repeat=64'],
    # iso-time arms (2026-10-22): the wall time of 'all' at 1,500 it (94 min on the laptop: 3.68 s/it; no cf 2.08 s/it)
    'nocf': ['--section_splice_frac=0.25', '--section_repeat=64'],
    'nocf_s50': ['--section_splice_frac=0.5', '--section_repeat=64'],
    'cfcool': CF + ['--section_splice_frac=0.25', '--section_repeat=64', '--section_cf_from=0.8'],
    'cf1': CF + ['--section_splice_frac=0.25', '--section_repeat=64', '--section_cf_rows=1'],
    # overshoot fix B3 + B5 (2026-10-23): 'all' with section attributes v2 (section-level thresholds, rhythm + texture)
    'all_v2': CF + ['--section_splice_frac=0.25', '--section_repeat=64', '--section_attrs=v2'],
}
FOLLOW_PROMPTS = {'all_v2': 'section_v2'}  # section_follow --prompts (default 'section')
ITERS = {'nocf': 2700, 'nocf_s50': 2700, 'cfcool': 2350, 'cf1': 2200}  # others MAX_ITERS
TABLE = ROOT / 'logs/sections/forget.md'
BASELINE = 'checkpoints/v2_A0'
PAIRS = ('loud', 'fast', 'drums', 'high', 'strings', 'uk_loud')
STEP = re.compile(r'step (\d+): .*?\| \S+ CE ([\d.]+) \(.*?\| aria_labeled CE ([\d.]+).*?section gain ([-+.\d]+) / none '
                  r'([-+.\d]+) / prev ([-+.\d]+)')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--minutes', type=float, default=150, help='training minutes per arm (one chunk)')
    p.add_argument('--no-follow', action='store_true', help='skip the section_follow generation')
    p.add_argument('--table-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def follow_dir(arm):
    return ROOT / 'results/section_follow' / (f'forget_{arm}' if arm != 'v2_A0' else 'v2_A0_section')


def train(arm, args):
    cmd = ['.venv/bin/python', 'tools/sweeps/section_long.py', '--name', f'forget_{arm}', '--out-dir',
           f'checkpoints/forget_{arm}', '--max-iters', str(ITERS.get(arm, MAX_ITERS)), '--minutes', f'{args.minutes:g}',
           '--chunks', '0', '--stop-at', '100000']
    for c in CONFIGS:
        cmd += ['--config', c]
    cmd += ['--', *ARMS[arm], f'--wandb_run_name=sec-forget-{arm}']
    print(f'{time.strftime("%H:%M")} ' + ' '.join(cmd[1:]), flush=True)
    return 0 if args.dry_run else subprocess.run(cmd, cwd=ROOT).returncode


def follow(arm, args):
    out = follow_dir(arm)
    cmd = ['.venv/bin/python', 'eval/compare/section_follow.py', '--checkpoints', f'checkpoints/forget_{arm}',
           '--cfg', '2', '--prompts', FOLLOW_PROMPTS.get(arm, 'section'), '--parts', 'hold,switch', '--mp3', '--out',
           str(out)]
    print(f'{time.strftime("%H:%M")} ' + ' '.join(cmd[1:]), flush=True)
    if args.dry_run:
        return 0
    with open(ROOT / 'logs/sections' / f'forget_{arm}_follow.log', 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def probe_row(arm):
    path = ROOT / 'logs/sections' / f'forget_{arm}.json'
    rows = json.loads(path.read_text()).get('probes', []) if path.exists() else []
    return max(rows, key=lambda r: r['iter']) if rows else None


def last_eval(arm):
    path = ROOT / 'logs/sections' / f'forget_{arm}.log'
    found = STEP.findall(path.read_text(encoding='utf-8', errors='ignore')) if path.exists() else []
    return found[-1] if found else None


def switched(arm):
    """{pair: effect / |mean d before the switch|} from the switch table of section_follow's summary.md."""
    path = follow_dir(arm) / 'summary.md'
    if not path.exists():
        return {}
    out, in_switch = {}, False
    for line in path.read_text().splitlines():
        if line.startswith('## '):
            in_switch = 'switch' in line
        cells = [c.strip() for c in line.strip('|').split('|')]
        if in_switch and cells and cells[0] in PAIRS and len(cells) >= 12:
            before = [float(c) for c in cells[2:6]]
            d = sum(before) / len(before)
            out[cells[0]] = float(cells[10]) / abs(d) if d else float('nan')
    return out


def write_table(arms):
    lines = ['# Section-caption forgetting: arms from v2_A0 (1,500 it; iso-time arms nocf / nocf_s50 / cfcool / cf1: '
             'the wall time of all)', '',
             f'Updated {time.strftime("%Y-%m-%d %H:%M")}. Val gains: the last evaluation (200 held-out windows, all '
             'positions). Probe: 512 held-out windows around changes of >= 2 points, gain on the first 32 notes of the '
             'changed section (nats per note, 4 heads). Switch: share of the full flip at note 1024 (section_follow '
             'cfg 2, 16 rows per prompt; 0 = keeps the first style, 1 = follows the new prompt).', '',
             '| arm | iter | held-out CE | Aria CE | val sec gain vs shuffled / none / prev | probe vs shuffled | '
             'vs none | vs previous | ' + ' | '.join(f'switch {p}' for p in PAIRS) + ' | switch mean |',
             '|---' * (9 + len(PAIRS)) + '|']
    for arm in ['v2_A0'] + arms:
        ev = last_eval(arm) if arm != 'v2_A0' else None
        pr = probe_row(arm) if arm != 'v2_A0' else \
            json.loads((ROOT / 'logs/sections/v2_A0.json').read_text())['probes'][-1]
        sw = switched(arm)
        vals = [sw.get(p, float('nan')) for p in PAIRS]
        ok = [v for v in vals if v == v]
        lines.append(f"| {arm} | {ev[0] if ev else (pr['iter'] if pr else '')} | {ev[1] if ev else ''} | "
                     f"{ev[2] if ev else ''} | {' / '.join(ev[3:]) if ev else ''} | "
                     + (f"{pr['gain_shuffled']:+.4f} | {pr['gain_none']:+.4f} | {pr.get('gain_prev', float('nan')):+.4f}"
                        if pr else ' | | ') + ' | '
                     + ' | '.join(f'{v:+.2f}' for v in vals) + f" | {sum(ok) / len(ok) if ok else float('nan'):+.2f} |")
    TABLE.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    args = parse_args()
    arms = [a for a in args.arms.split(',') if a]
    assert all(a in ARMS for a in arms), arms
    if not args.table_only:
        for arm in arms:
            row = probe_row(arm)
            if not (row and row['iter'] >= ITERS.get(arm, MAX_ITERS) - 1):
                code = train(arm, args)
                if code != 0:
                    print(f'{arm}: driver exit {code}, next arm', flush=True)
                    continue
            if not args.no_follow and not (follow_dir(arm) / 'summary.md').exists():
                print(f'{time.strftime("%H:%M")} {arm}: section_follow exit {follow(arm, args)}', flush=True)
            if not args.dry_run:
                write_table(arms)
    if not args.dry_run:
        write_table(arms)


if __name__ == '__main__':
    main()
