"""
AdamW tuning of the 450M Ukrainian piano fine-tune (to-do 15d, plan agreed 2026-10-12). The plain AdamW-8bit run
(LR 1e-4, 350 it, dropout 0, `checkpoints/overfit_adamw8_1e-4`) beat ua_2048_v2, the base and the WiSE blend by ear
"with a big gap" (4-seed dist 0.132), so this sweeps around it. Every run is judged by generation
(eval/compare/seed_compare.py on the 27 Ukrainian val covers at the H3 setting), not by covers val CE (it rises while the
samples improve, H1).

Phases (each skips work that is already done, so a re-run after an interruption continues):
    P0  score the existing run's snapshots @175 / @349 (seeds 1 2), no training
    P1  350 it per arm: LR 5e-5 / 2e-4 / 3e-4, dropout 0.1 at LR 1e-4 (the fine-tune configs' value; the 450M was
        pretrained with 0), optionally 6 windows/step (`--arms ... lr1e-4_b6`: 700 it, the same notes)
    P2  the best P1 arm (any snapshot, seeds 1 2) trained 2x as long, snapshots at 1/2, 3/4, end (2 / 3 / 4 cover passes;
        mid-run snapshots aren't cooled down)
    P3  the 3 best snapshots re-scored on fresh seeds 3 4 (winner's curse), the best 4-seed mean = finalist; its run
        repeated with training seed 2 and scored on seeds 1-4; listening samples of the finalist next to the others
        in samples/listen5/

    .venv/bin/python tools/sweeps/adamw_sweep.py --dry-run          # plan + time estimate
    .venv/bin/python tools/sweeps/adamw_sweep.py                    # P0-P3, ~3.3 h on the laptop (refuses on battery)
    .venv/bin/python tools/sweeps/adamw_sweep.py --phases 0 1       # only some phases
    .venv/bin/python tools/sweeps/adamw_sweep.py --arms lr2e-4 lr1e-4_b6 --phases 1

Recipe of every run = config/finetune/finetune_ua450.py + finetune_ua450_full.py (v4 covers/reductions + Aria pop-rock),
adamw8bit, dropout 0 unless the arm says otherwise, 12 windows/step, warmup 10, WSD with a 20 % cooldown. Runs go to
checkpoints/adamw_<run>/model_bf16_it<N>.pt (~0.9 GB each), logs and scores to logs/adamw/, the table to
logs/adamw/summary.txt.
"""

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/adamw'
CONFIGS = ['config/finetune/finetune_ua450.py', 'config/finetune/finetune_ua450_full.py']
COMMON = ['--optimizer_name=adamw8bit', '--dropout=0.0', '--warmup_iters=10', '--eval_interval=25',
          '--ckpt_interval_min=0', '--save_pre_cooldown=False']
SETTING = 'T=0.9,dt=0.1,dens=prompt'  # H3 pick, used for every 450M score so far
ITERS = 350                           # at 12 windows/step ~ 2 passes over the covers
WINDOWS = 12
ARMS = {
    'lr1e-4': ['--learning_rate=1e-4'],  # = the existing run (2026-10-12), not retrained
    'lr5e-5': ['--learning_rate=5e-5'],
    'lr2e-4': ['--learning_rate=2e-4'],
    'lr3e-4': ['--learning_rate=3e-4'],
    'lr1e-4_do0.1': ['--learning_rate=1e-4', '--dropout=0.1'],
    'lr1e-4_b6': ['--learning_rate=1e-4', '--gradient_accumulation_steps=6'],
}
DEFAULT_ARMS = ['lr5e-5', 'lr2e-4', 'lr3e-4', 'lr1e-4_do0.1']
EXISTING = {'lr1e-4': ('checkpoints/overfit_adamw8_1e-4', 'logs/overfit_adamw8_1e-4.log')}
LISTEN_SONGS = ('04', '09', '17', '21', '24')  # the samples/listen5 set
SEC_PER_WINDOW = 2.9 / 12
STARTUP_SEC = 150
SCORE_SEC = 100                        # one snapshot, 2 seeds x 27 songs
STEP = re.compile(r'^[ \t]*step (\d+): .*?\| covers_v4 CE ([\d.]+) ', re.M)


def windows(flags):
    acc = [f for f in flags if f.startswith('--gradient_accumulation_steps=')]
    return int(acc[-1].split('=')[1]) if acc else WINDOWS


def run_iters(flags, mult=1):
    return ITERS * WINDOWS // windows(flags) * mult


def snaps_of(iters, fracs):
    return sorted({round(f * iters) if f < 1 else iters - 1 for f in fracs})


def paths(run):
    if run in EXISTING:
        d, log = EXISTING[run]
        return ROOT / d, ROOT / log
    return ROOT / f'checkpoints/adamw_{run}', OUT / f'{run}.log'


def ckpt(run, it):
    return paths(run)[0] / f'model_bf16_it{it}.pt'


def sh(cmd, log):
    print(f'{time.strftime("%H:%M")}   $ {" ".join(cmd)}', flush=True)
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f'failed (exit {code}), see {log}')


def train(run, flags, iters, snaps, seed=None):
    out_dir, log = paths(run)
    if all(ckpt(run, s).exists() for s in snaps):
        print(f'{run}: trained already, skipped')
        return
    print(f'{time.strftime("%H:%M")} train {run}: {iters} it, snapshots {snaps}, '
          f'~{(iters * windows(flags) * SEC_PER_WINDOW + STARTUP_SEC) / 60:.0f} min', flush=True)
    cmd = ['.venv/bin/python', 'train.py', *CONFIGS, *COMMON, *flags, f'--max_iters={iters}',
           f'--lr_decay_iters={iters}', f'--save_iters={",".join(map(str, snaps))}',
           f'--out_dir={out_dir.relative_to(ROOT)}', f'--wandb_run_name=ua450-adamw-{run}']
    if seed is not None:
        cmd.append(f'--seed={seed}')
    sh(cmd, log)


def score(run, its, seeds):
    tag = ''.join(map(str, seeds))
    js = OUT / f'{run}_seeds{tag}.json'
    if js.exists():
        return
    print(f'{time.strftime("%H:%M")} score {run} @{its}, seeds {seeds}', flush=True)
    sh(['.venv/bin/python', 'eval/compare/seed_compare.py', '--batch-size', '14',
        '--checkpoints', *[str(ckpt(run, i).relative_to(ROOT)) for i in its], '--settings', SETTING,
        '--seeds', *map(str, seeds), '--out', str(OUT / f'{run}_seeds{tag}.txt'), '--json', str(js)],
       OUT / f'{run}_seeds{tag}.log')


def scores():
    """{(run, iter): {seeds tag: record}} from every scored file."""
    s = {}
    for f in glob.glob(str(OUT / '*_seeds*.json')):
        run, tag = Path(f).stem.rsplit('_seeds', 1)
        for r in json.load(open(f)):
            s.setdefault((run, r['iter']), {})[tag] = r
    return s


def ranked(tag='12', runs=None):
    s = scores()
    return sorted(((v[tag]['dist'], k) for k, v in s.items() if tag in v and (runs is None or k[0] in runs)))


def mean4(v):
    return (v['12']['dist'] + v['34']['dist']) / 2 if '12' in v and '34' in v else float('nan')


def ce_curve(run):
    log = paths(run)[1]
    if not log.exists():
        return {}
    text = log.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    return {int(a): float(b) for a, b in STEP.findall(text)}


def flags_of(run):
    base = run.removesuffix('_seed2').removesuffix('_long')
    return ARMS[base]


def summarize():
    s = scores()
    lines = [f'AdamW 450M fine-tunes, {time.strftime("%Y-%m-%d %H:%M")}: seed_compare distance to the real continuations '
             f'at {SETTING} (lower = closer; ua_2048_v2 0.245, base 0.245, 4-seed means); covers v4 val CE (step 0 = '
             f'5.449)',
             f'{"run":22s}{"iter":>6s}{"CE":>8s}{"d s1+2":>9s}{"d s3+4":>9s}{"d mean":>9s}{"d s1-4":>9s}{"nps":>7s}'
             f'{"melRep":>8s}{"recur":>7s}']
    def order(kv):  # 4-seed rows first by their mean, then the rest by their own distance
        v = kv[1]
        return (0, mean4(v)) if '34' in v else (1, (v.get('12') or v.get('1234'))['dist'])

    for (run, it), v in sorted(s.items(), key=order):
        ce = ce_curve(run)
        near = ce[min(ce, key=lambda x: abs(x - it))] if ce else float('nan')
        r = v.get('12') or v.get('1234')
        g = lambda t: f'{v[t]["dist"]:9.3f}' if t in v else f'{"":9s}'
        m = mean4(v)
        lines.append(f'{run:22s}{it:6d}{near:8.3f}{g("12")}{g("34")}{m:9.3f}{g("1234")}{r["nps"]:7.2f}'
                     f'{r["mel_rep"]:8.1f}{r["recur"]:7.3f}')
    (OUT / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def on_ac():
    sup = Path('/sys/class/power_supply')
    for d in sup.glob('*'):
        t = (d / 'type').read_text().strip() if (d / 'type').exists() else ''
        if t != 'Battery' and (d / 'online').exists() and (d / 'online').read_text().strip() == '1':
            return True
    return not any(sup.glob('BAT*'))  # desktops without a battery


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--phases', type=int, nargs='+', default=[0, 1, 2, 3], choices=[0, 1, 2, 3])
    ap.add_argument('--arms', nargs='+', default=DEFAULT_ARMS, choices=[a for a in ARMS if a not in EXISTING])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-battery', action='store_true', help='run even when the laptop is not on AC power')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        tot = 0
        print('P0  lr1e-4 (existing): score @175 @349')
        tot += 2 * SCORE_SEC
        for a in args.arms:
            it = run_iters(ARMS[a])
            sec = it * windows(ARMS[a]) * SEC_PER_WINDOW + STARTUP_SEC + 2 * SCORE_SEC
            tot += sec
            print(f'P1  {a:14s} {it} it, snapshots {snaps_of(it, (0.5, 1.0))}: {" ".join(ARMS[a])}  ~{sec / 60:.0f} min')
        long_sec = 2 * ITERS * WINDOWS * SEC_PER_WINDOW + STARTUP_SEC + 3 * SCORE_SEC
        p3 = 3 * SCORE_SEC + 2 * ITERS * WINDOWS * SEC_PER_WINDOW + STARTUP_SEC + 2 * SCORE_SEC + 3 * 60
        tot += long_sec + p3
        print(f'P2  best P1 arm x2 ({2 * ITERS} it at 12 windows), snapshots at 1/2, 3/4, end  ~{long_sec / 60:.0f} min')
        print(f'P3  top 3 on seeds 3 4, finalist retrained with seed 2 (<= {2 * ITERS} it), seeds 1-4, listening '
              f'samples  <= ~{p3 / 60:.0f} min')
        print(f'total <= ~{tot / 3600:.1f} h, <= {2 * len(args.arms) + 6} snapshots x 0.9 GB '
              f'(free on /home: {shutil.disk_usage(ROOT).free / 2**30:.0f} GB); AC power now: {on_ac()}')
        return
    if not on_ac() and not args.allow_battery:
        sys.exit('on battery: plug in, or pass --allow-battery')

    if 0 in args.phases:
        print('== P0')
        score('lr1e-4', [175, 349], (1, 2))
    p1 = ['lr1e-4', *args.arms]
    if 1 in args.phases:
        print('== P1')
        for a in args.arms:
            it = run_iters(ARMS[a])
            sn = snaps_of(it, (0.5, 1.0))
            train(a, ARMS[a], it, sn)
            score(a, sn, (1, 2))
    if 2 in args.phases:
        print('== P2')
        r = ranked('12', set(p1))
        if not r:
            sys.exit('P2 needs P0/P1 scores')
        best = r[0][1][0]
        it = run_iters(ARMS[best], 2)
        sn = snaps_of(it, (0.5, 0.75, 1.0))
        print(f'best P1 arm: {best} (@{r[0][1][1]}, {r[0][0]:.3f}) -> {best}_long')
        train(f'{best}_long', ARMS[best], it, sn)
        score(f'{best}_long', sn, (1, 2))
    if 3 in args.phases:
        print('== P3')
        top = [k for _, k in ranked('12') if not k[0].endswith('_seed2')][:3]
        by_run = {}
        for run, it in top:
            by_run.setdefault(run, []).append(it)
        for run, its in by_run.items():
            score(run, sorted(its), (3, 4))
        s = scores()
        cands = [k for k in top if '34' in s[k]]
        run, it = min(cands, key=lambda k: mean4(s[k]))
        print(f'finalist: {run} @{it}, 4-seed mean {mean4(s[(run, it)]):.3f}')
        full = run_iters(flags_of(run), 2 if run.endswith('_long') else 1)
        frac = (0.5, 0.75, 1.0) if run.endswith('_long') else (0.5, 1.0)
        train(f'{run}_seed2', flags_of(run), full, snaps_of(full, frac), seed=2)
        score(f'{run}_seed2', [it], (1, 2, 3, 4))
        name = f'5_adamw_{run}_it{it}'
        dest = ROOT / 'samples/listen5' / name
        if not dest.exists():
            tmp = ROOT / 'samples/adamw_final'
            sh(['.venv/bin/python', 'eval/compare/seed_compare.py', '--batch-size', '14', '--checkpoints',
                str(ckpt(run, it).relative_to(ROOT)), '--settings', SETTING, '--seeds', '1', '--out-dir', str(tmp)],
               OUT / 'listen.log')
            src = tmp / f'{ckpt(run, it).parent.name}_{ckpt(run, it).stem}'
            dest.mkdir(parents=True)
            for n in LISTEN_SONGS:
                for f in src.glob(f'{n}_*'):
                    shutil.copy(f, dest)
            print(f'listening samples: {dest.relative_to(ROOT)} (all 27 songs: {src.relative_to(ROOT)})')
    summarize()


if __name__ == '__main__':
    main()
