"""
Section captions: short arms at the same notes seen (config/pilots/section_pilot.py: 480 it x 65,536 notes from
text_t2_v3_long, ~15 min each), then the teacher-forced probe (eval/probes/section_gain.py) and optionally the
generation test (eval/compare/section_follow.py). context/claude/text_control.md "Section captions".

Stage 1 (mode, and current vs next = section_lead), section_len 256, 4 slots / cross-attention after every 2nd block:
    ctrl   no section captions (T2 continued on the same data: what the extra steps alone do)
    A0     'prefix', caption with the section's first note
    A128   'prefix', caption 128 notes early (halfway through the previous section)
    A256   'prefix', caption a whole section early (= the next section's caption)
    B0     'xattn', caption with the section's first note
    B128   'xattn', caption 128 notes early
Stage 2 (--base <stage-1 arm>, its mode and lead fraction kept): section_len 128 / 512, and 2 / 8 slots ('prefix') or
cross-attention after every / every 4th block ('xattn').

    .venv/bin/python tools/sweeps/section_sweep.py                      # stage 1: train, probe, summary
    .venv/bin/python tools/sweeps/section_sweep.py --gen                # + the generation test (~15 min per arm)
    .venv/bin/python tools/sweeps/section_sweep.py --base B0            # stage 2 around B0
    .venv/bin/python tools/sweeps/section_sweep.py --arms A0 B0 --dry-run
    .venv/bin/python tools/sweeps/section_sweep.py --overnight --iters 1100   # everything below, unattended

--overnight: stage 1 -> probe + generation test (cfg 2, the arms + text_t2_v3_long; MIDI, MP3s at the end for the winner + baseline) -> winner -> stage 2 around it
-> probe + generation test -> final winner among the stage-1 winner and its stage-2 arms; report
logs/sections/overnight.md. Every arm trains --iters steps (same notes seen). A failed arm is logged and skipped.
Winner (pick()): score = 0.5 x mean rank of the switch effects (6 pairs; the thing T2 can't do) + 0.25 x mean rank
of the hold shifts in the last segment (10 prompts) + 0.25 x rank of the teacher-forced late section gain vs none;
arms whose runaway share (> 40 notes/s segments) is > 0.15 above text_t2_v3_long's are out; ctrl is the reference,
never the winner.

Logs logs/sections/<arm>.log, out_dir checkpoints/sec_<arm> (bf16 weights, ~0.25 GB), probe results/section_gain/
<stage>, generation results/section_follow/<stage>, summary logs/sections/summary_<stage>.md. Finished arms are
skipped on a re-run.
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'eval/compare'))
CONFIGS = ['config/pilots/text_pilot.py', 'config/pilots/section_pilot.py']
SECTION_LEN = 256
STAGE1 = {
    'ctrl': ['--section_len=0'],
    'A0': ['--section_mode=prefix', '--section_lead=0'],
    'A128': ['--section_mode=prefix', '--section_lead=128'],
    'A256': ['--section_mode=prefix', '--section_lead=256'],
    'B0': ['--section_mode=xattn', '--section_lead=0'],
    'B128': ['--section_mode=xattn', '--section_lead=128'],
}
ITERS = 480
SEC_PER_IT = 1.75      # laptop, 8 x 8 x 1,024 notes per step (measured 2026-10-19: prefix 1.67, xattn 1.80)
STARTUP_MIN = 1.5      # data, LaBSE, compile + the first evaluation
EVAL_MIN = 0.25        # one evaluation (6 text val sets x 200 windows, ~13 s)
OUT = ROOT / 'logs/sections'
BASELINE = 'text_t2_v3_long'
GEN_CFG = 2.0
STEP = re.compile(r'step (\d+): .*?heldout_labeled CE ([\d.]+) .*?\| text gain ([-+.\d]+) \(vs none ([-+.\d]+)\)'
                  r' \| late ([-+.\d]+) / none ([-+.\d]+)'
                  r'(?: \| section gain ([-+.\d]+) / none ([-+.\d]+), late ([-+.\d]+) / ([-+.\d]+) \| all text late'
                  r' ([-+.\d]+))?')
FIELDS = ('iter', 'ce', 'text_gain', 'text_gain_none', 'text_late', 'text_none_late', 'sec_gain', 'sec_none',
          'sec_late', 'sec_none_late', 'all_late')


def stage2(base):
    """Arms around a stage-1 arm: other section lengths (same lead fraction), other slots / cross-attention spacing."""
    args = STAGE1[base]
    mode = 'xattn' if '--section_mode=xattn' in args else 'prefix'
    lead = next(int(a.split('=')[1]) for a in args if a.startswith('--section_lead='))
    arms = {}
    for n in (128, 512):
        arms[f'{base}_len{n}'] = [f'--section_mode={mode}', f'--section_len={n}',
                                  f'--section_lead={round(lead / SECTION_LEN * n)}']
    for k in ((2, 8) if mode == 'prefix' else (1, 4)):
        key = '--section_slots' if mode == 'prefix' else '--section_xattn_every'
        arms[f'{base}_{"slots" if mode == "prefix" else "every"}{k}'] = args + [f'{key}={k}']
    return arms


def last_step(log):
    if not log.exists():
        return None
    text = log.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    found = STEP.findall(text)
    return dict(zip(FIELDS, (float(x) if x else float('nan') for x in found[-1]))) if found else None


def minutes(iters):
    return STARTUP_MIN + iters * SEC_PER_IT / 60 + 3 * EVAL_MIN


def train(arm, args, dry, iters=ITERS, ckpt_root=None):
    """ckpt_root: write the weights there (another disk), linked from checkpoints/sec_<arm>."""
    log, out_dir = OUT / f'{arm}.log', f'checkpoints/sec_{arm}'
    if ckpt_root and not dry and not (ROOT / out_dir).exists():
        (Path(ckpt_root) / f'sec_{arm}').mkdir(parents=True, exist_ok=True)
        (ROOT / out_dir).symlink_to(Path(ckpt_root) / f'sec_{arm}')
    done = last_step(log)
    if done and done['iter'] >= iters - 1 and (ROOT / out_dir / 'model_bf16.pt').exists():
        print(f'{arm}: trained already, skipped')
        return
    cmd = ['.venv/bin/python', 'train.py', *CONFIGS, *args, f'--max_iters={iters}', f'--lr_decay_iters={iters}',
           f'--eval_interval={iters // 2}', f'--out_dir={out_dir}', f'--wandb_run_name=sec-{arm}']
    print(f'{time.strftime("%H:%M")} {arm}: ~{minutes(iters):.0f} min  ' + ' '.join(cmd[2:]), flush=True)
    if dry:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def summary(arms, stage):
    lines = [f'# Section arms, {stage} (last evaluation; gains in nats, late = window positions >= 256)', '',
             '| arm | settings | val CE | file caption late (vs other / none) | sections (vs other / none) | '
             'sections late (vs other / none) | all text late |', '|---|---|---|---|---|---|---|']
    for arm, args in arms.items():
        r = last_step(OUT / f'{arm}.log')
        if r is None:
            lines.append(f'| {arm} | {" ".join(args)} | not run | | | | |')
            continue
        sec = f"{r['sec_gain']:+.4f} / {r['sec_none']:+.4f}"
        sec_late = f"{r['sec_late']:+.4f} / {r['sec_none_late']:+.4f}"
        all_late = r['all_late'] if r['all_late'] == r['all_late'] else r['text_none_late']  # ctrl: file caption only
        lines.append(f"| {arm} | {' '.join(a.lstrip('-') for a in args)} | {r['ce']:.4f} @{int(r['iter'])} | "
                     f"{r['text_late']:+.4f} / {r['text_none_late']:+.4f} | {sec} | {sec_late} | {all_late:+.4f} |")
    return '\n'.join(lines) + '\n'


def read_rows(path):
    with open(path) as f:
        return [{k: v if k in ('checkpoint', 'part', 'pool', 'prompt') else float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def rank(values):
    """{arm: value} -> {arm: rank} (0 = highest; NaN last; ties share the mean rank)."""
    keys = list(values)
    v = np.array([values[k] if values[k] == values[k] else -math.inf for k in keys], float)
    order = (-v).argsort(kind='stable')
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for x in set(v.tolist()):  # ties
        r[v == x] = r[v == x].mean()
    return dict(zip(keys, r.tolist()))


def pick(arms, rows, seg=256, switch=1024):
    """(winner, report lines) among `arms` (not ctrl) from the generation rows and the training logs."""
    from section_follow import SWITCH, hold_late, switch_d
    sw = switch_d(rows, seg, switch)
    hl = hold_late(rows)
    runaway = {}
    for r in rows:
        if r['cfg'] == GEN_CFG and r['runaway'] == r['runaway']:
            runaway.setdefault(r['checkpoint'], []).append(r['runaway'])
    runaway = {k: float(np.mean(v)) for k, v in runaway.items()}
    cand = [a for a in arms if a != 'ctrl' and (f'sec_{a}', GEN_CFG) in sw]
    if not cand:
        return None, ['no arm with generation results']
    sw_rank = {a: 0.0 for a in cand}
    for pair in SWITCH:
        for a, r in rank({a: sw[(f'sec_{a}', GEN_CFG)].get(pair, ([], math.nan, 0))[1] for a in cand}).items():
            sw_rank[a] += r / len(SWITCH)
    prompts = sorted({p for a in cand for p in hl.get((f'sec_{a}', GEN_CFG), {})})
    hold_rank = {a: 0.0 for a in cand}
    for pr in prompts:
        for a, r in rank({a: hl.get((f'sec_{a}', GEN_CFG), {}).get(pr, math.nan) for a in cand}).items():
            hold_rank[a] += r / len(prompts)
    logs = {a: last_step(OUT / f'{a}.log') or {} for a in cand}
    tf_rank = rank({a: logs[a].get('sec_none_late', math.nan) for a in cand})
    score = {a: 0.5 * sw_rank[a] + 0.25 * hold_rank[a] + 0.25 * tf_rank[a] for a in cand}
    limit = runaway.get(BASELINE, 1.0) + 0.15
    ok = [a for a in cand if runaway.get(f'sec_{a}', 1.0) <= limit] or cand
    winner = min(ok, key=lambda a: (score[a], logs[a].get('ce', math.inf)))
    lines = [f'| arm | score | switch rank | hold rank | teacher rank | ' + ' | '.join(SWITCH) +
             ' | runaway | val CE | late section gain vs none |', '|---|' + '---|' * (7 + len(SWITCH))]
    for a in sorted(cand, key=lambda a: score[a]):
        eff = sw[(f'sec_{a}', GEN_CFG)]
        lines.append(f'| {a}{" (winner)" if a == winner else "" if a in ok else " (out: runaway)"} | '
                     f'{score[a]:.2f} | {sw_rank[a]:.2f} | {hold_rank[a]:.2f} | {tf_rank[a]:.1f} | '
                     + ' | '.join(f'{eff[p][1]:+.2f}' if p in eff else '' for p in SWITCH)
                     + f' | {runaway.get(f"sec_{a}", math.nan):.0%} | {logs[a].get("ce", math.nan):.4f} | '
                     f'{logs[a].get("sec_none_late", math.nan):+.4f} |')
    lines.append(f'\nrunaway share of {BASELINE}: {runaway.get(BASELINE, math.nan):.0%}; ctrl: '
                 f'{runaway.get("sec_ctrl", math.nan):.0%}')
    return winner, lines


def probe_lines(path):
    """Early-caption check from a probe result: gain vs none in the last 32 notes of a section (before the next
    caption's section starts) and in its first 8 notes."""
    if not Path(path).exists():
        return []
    lines = ['| arm | lead | last 32 notes of a section | first note | notes 1-8 | all |', '|---|---|---|---|---|---|']
    for r in json.loads(Path(path).read_text()):
        b = {k: g for k, _, g in r['none']['in_section']}
        last = [g for k, g in b.items() if k.endswith(f"-{r['section_len']}")][0]
        name = Path(r['checkpoint']).name
        lines.append(f"| {name} | {r['section_lead']} | {last:+.4f} | {b['0-1']:+.4f} | {b['1-8']:+.4f} | "
                     f"{r['none']['all']:+.4f} |")
    return lines


def run(cmd, what):
    """A subprocess that may fail without stopping the night."""
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return True
    except Exception:
        print(f'{time.strftime("%H:%M")} {what} FAILED:\n{traceback.format_exc()}', flush=True)
        return False


def evaluate(arms, stage, extra_ckpts=()):
    ckpts = [f'checkpoints/sec_{a}' for a in arms if (ROOT / f'checkpoints/sec_{a}/model_bf16.pt').exists()]
    run(['.venv/bin/python', 'eval/probes/section_gain.py', '--checkpoints', ','.join(ckpts + list(extra_ckpts)),
         '--out', f'results/section_gain/{stage}'], f'probe {stage}')
    gen = [c for c in ckpts if not c.endswith('sec_ctrl') or stage == 'stage1']
    run(['.venv/bin/python', 'eval/compare/section_follow.py', '--checkpoints',
         ','.join(gen + ([f'checkpoints/{BASELINE}'] if stage == 'stage1' else [])), '--cfg', f'{GEN_CFG:g}',
         '--out', f'results/section_follow/{stage}'], f'generation test {stage}')


def render(dirs):
    """MP3s of the MIDI files in dirs (only for the winner and the baseline: the disk is nearly full)."""
    from musicar.data.midi_io import render_mp3
    from text_follow import SOUNDFONT
    for d in dirs:
        for mid in sorted(Path(d).glob('*.mid')):
            try:
                render_mp3(mid, SOUNDFONT)
            except Exception as e:
                print(f'{mid}: {e}', flush=True)


def overnight(iters, ckpt_root=None):
    """Progress on the dashboard (logs/jobs, 'Sampling jobs'): 12 units = 6 + 4 arms trained, 2 evaluations."""
    from musicar.jobstatus import JobStatus
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    job = JobStatus('section_overnight', 12, 'stage 1')
    unit = [0]

    def phase(detail):
        job.update(unit[0], detail, force=True)
        unit[0] += 1
    report = [f'# Section captions overnight ({time.strftime("%Y-%m-%d %H:%M")}, {iters} it x 65,536 notes per arm)',
              '']
    for i, (arm, a) in enumerate(STAGE1.items()):
        phase(f'stage 1: training {arm} ({i + 1}/6): {" ".join(x.lstrip("-") for x in a)}')
        try:
            train(arm, a, False, iters, ckpt_root)
        except Exception:
            print(f'{arm} FAILED:\n{traceback.format_exc()}', flush=True)
    phase('stage 1: probe + generation test (6 arms + T2)')
    evaluate(STAGE1, 'stage1')
    rows1 = read_rows(ROOT / 'results/section_follow/stage1/rows.csv') \
        if (ROOT / 'results/section_follow/stage1/rows.csv').exists() else []
    winner1, lines = pick(STAGE1, rows1)
    report += ['## Stage 1', '', summary(STAGE1, 'stage1'), *lines, '',
               'Early captions (probe, gain vs null sections):', '', *probe_lines(ROOT / 'results/section_gain/stage1/gain.json'), '']
    print('\n'.join(report), flush=True)
    (OUT / 'overnight.md').write_text('\n'.join(report))
    if winner1 is None:
        job.finish('no stage-1 winner: logs/sections/overnight.md')
        return
    arms2 = stage2(winner1)
    for i, (arm, a) in enumerate(arms2.items()):
        phase(f'stage 2 (stage-1 winner {winner1}): training {arm} ({i + 1}/4): {" ".join(x.lstrip("-") for x in a)}')
        try:
            train(arm, a, False, iters, ckpt_root)
        except Exception:
            print(f'{arm} FAILED:\n{traceback.format_exc()}', flush=True)
    stage = f'stage2_{winner1}'
    phase('stage 2: probe + generation test (4 arms)')
    evaluate(arms2, stage, extra_ckpts=[f'checkpoints/sec_{winner1}'])
    rows2 = read_rows(ROOT / f'results/section_follow/{stage}/rows.csv') \
        if (ROOT / f'results/section_follow/{stage}/rows.csv').exists() else []
    rows = rows2 + [r for r in rows1 if r['checkpoint'] in (f'sec_{winner1}', BASELINE)]
    final, lines = pick({winner1: STAGE1[winner1], **arms2}, rows)
    report += [f'## Stage 2 around {winner1}', '', summary(arms2, stage), *lines, '',
               *probe_lines(ROOT / f'results/section_gain/{stage}/gain.json'), '',
               f'**Winner: {final}** ({" ".join(({winner1: STAGE1[winner1], **arms2})[final])})', '',
               f'Total {(time.time() - t0) / 3600:.1f} h. Tables: results/section_follow/stage1/summary.md, '
               f'results/section_follow/{stage}/summary.md; probe: results/section_gain/*/summary.md; MP3s next to them.']
    (OUT / 'overnight.md').write_text('\n'.join(report))
    print('\n'.join(report[-12:]), flush=True)
    win_stage = 'stage1' if final == winner1 else stage
    cfg_dir = f'cfg{GEN_CFG:g}'
    render([ROOT / f'results/section_follow/{win_stage}/sec_{final}/{cfg_dir}',
            ROOT / f'results/section_follow/stage1/{BASELINE}/{cfg_dir}'])
    job.finish(f'winner {final}: logs/sections/overnight.md')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--arms', nargs='*', help='only these arms (default: all of the stage)')
    p.add_argument('--base', help='stage 2 around this stage-1 arm')
    p.add_argument('--gen', action='store_true', help='also run the generation test on the arms + text_t2_v3_long')
    p.add_argument('--gen-cfg', default='2', help='guidance scales of the generation test')
    p.add_argument('--iters', type=int, default=ITERS, help='steps per arm (65,536 notes each)')
    p.add_argument('--overnight', action='store_true', help='stage 1, pick, stage 2, pick (see the docstring)')
    p.add_argument('--ckpt-root', help='write the weights under this dir (linked from checkpoints/sec_<arm>)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.overnight:
        return overnight(args.iters, args.ckpt_root)
    stage = f'stage2_{args.base}' if args.base else 'stage1'
    arms = stage2(args.base) if args.base else STAGE1
    if args.arms:
        arms = {k: v for k, v in arms.items() if k in args.arms}
    for arm, a in arms.items():
        train(arm, a, args.dry_run, args.iters, args.ckpt_root)
    if args.dry_run:
        return
    ckpts = ','.join(f'checkpoints/sec_{a}' for a in arms)
    subprocess.run(['.venv/bin/python', 'eval/probes/section_gain.py', '--checkpoints', ckpts,
                    '--out', f'results/section_gain/{stage}'], cwd=ROOT, check=True)
    if args.gen:
        subprocess.run(['.venv/bin/python', 'eval/compare/section_follow.py', '--checkpoints',
                        ckpts + ',checkpoints/text_t2_v3_long', '--cfg', args.gen_cfg, '--mp3',
                        '--out', f'results/section_follow/{stage}'], cwd=ROOT, check=True)
    text = summary(arms, stage)
    (OUT / f'summary_{stage}.md').write_text(text)
    print(text)


if __name__ == '__main__':
    sys.exit(main())
