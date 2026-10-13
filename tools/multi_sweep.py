"""
Length / LR / replay sweep of the Ukrainian multi-track fine-tune of the 450M base (2026-10-14), judged by generation.
ua_450_multi_long showed that multi_v1 val CE (best ~300 it into the continuation) and the generation check (best
~2,000 it) disagree, and that 10 songs x 2 seeds can't separate the checkpoints, so every snapshot is scored on 24
validation songs and several seeds.

Phases (each skips work already done, so a re-run after an interruption continues):
    P0  score the existing checkpoints (base, ua_450_multi it750 / it1499, ua_450_multi_long's snapshots) on seeds
        1 2 3: the seed noise and the reference distances
    P1  per arm: 2,000 it from long_450m_final with snapshots at 500 / 1,000 / 1,500 / 1,999, then each snapshot
        scored on seeds 1 2 3. Arms: the ua_450_multi recipe (LR 2e-4, replay 0.25) as the control, LR 1e-4, replay 0.5
    P2  listening samples (MP3, the 9 listening songs, seed 1) of the 2 best P1 snapshots

    .venv/bin/python tools/multi_sweep.py --dry-run       # plan, time and disk estimate
    .venv/bin/python tools/multi_sweep.py                 # P0-P2, ~4.7 h on the laptop (refuses on battery)
    .venv/bin/python tools/multi_sweep.py --phases 0      # only some phases

Score = eval/multi_generate.py's mean distance to each song's real continuation (lower = closer) at the ua_multi
settings, 24 evenly spaced multi_v1 validation songs, 128-note prompt + 1,000 new notes, no end before 1,000.
Snapshots before the WSD cooldown (it 1,600 of 2,000) are not cooled down. Runs go to checkpoints/multi_<arm>/,
scores to samples/multi_sweep/<run>_it<N>_s<seed>/scores.json (MIDI only), logs and the table to logs/multi_sweep/.
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'logs/multi_sweep'
SCORES = ROOT / 'samples/multi_sweep'
CSV = 'data/finetune/ukrainian_multi_v1.csv'
CONFIGS = ['config/finetune_ua450.py', 'config/finetune_ua450_multi.py']
COMMON = ['--ckpt_interval_min=0', '--save_pre_cooldown=False', '--eval_interval=100']
ITERS = 2000
SNAPS = (500, 1000, 1500, 1999)
ARMS = {
    'control': [],                                   # = ua_450_multi's recipe, 2,000 it
    'lr1e-4': ['--learning_rate=1e-4'],
    'replay0.5': ['--source_weights=0.5,0.5'],
}
# P0: existing runs -> (checkpoint per label, iterations into the fine-tune for the table)
EXISTING = {
    'base': {0: 'checkpoints/long_450m_final/model_bf16.pt'},
    'ua_450_multi': {750: 'checkpoints/ua_450_multi/model_bf16_it750.pt',
                     1499: 'checkpoints/ua_450_multi/model_bf16_it1499.pt'},
    'ua_450_multi_long': {750 + 750: 'checkpoints/ua_450_multi_long/model_bf16_it750.pt',
                          750 + 1350: 'checkpoints/ua_450_multi_long/model_bf16_it1350.pt',
                          750 + 2249: 'checkpoints/ua_450_multi_long/model_bf16_it2249.pt'},
}
LISTEN = ('скрябін - говорили,скрябін - хай буде так,воплі відоплясова - горіла сосна,соловій - юність,'
          'вона тарас чубай,плач єремії - червона рута,один в каное - пообіцяй,файне місто,кольорова')
SEC_PER_IT = 2.45                       # ua_450_multi_long on the laptop, evals every 100 it included
STARTUP_SEC = 150
SCORE_SEC = 30                          # one checkpoint, one seed, 24 songs x 1,000 notes
SEEDS = (1, 2, 3)
STEP = re.compile(r'^[ \t]*step (\d+): .*?\| multi_v1 CE ([\d.]+) ', re.M)


def ckpt(run, it):
    if run in EXISTING:
        return ROOT / EXISTING[run][it]
    return ROOT / f'checkpoints/multi_{run}/model_bf16_it{it}.pt'


def sh(cmd, log):
    print(f'{time.strftime("%H:%M")}   $ {" ".join(cmd)}', flush=True)
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f'failed (exit {code}), see {log}')


def train(arm):
    if all(ckpt(arm, s).exists() for s in SNAPS):
        print(f'{arm}: trained already, skipped')
        return
    print(f'{time.strftime("%H:%M")} train {arm}: {ITERS} it, snapshots {SNAPS}, '
          f'~{(ITERS * SEC_PER_IT + STARTUP_SEC) / 60:.0f} min', flush=True)
    sh(['.venv/bin/python', 'train.py', *CONFIGS, *COMMON, *ARMS[arm], f'--max_iters={ITERS}',
        f'--lr_decay_iters={ITERS}', f'--save_iters={",".join(map(str, SNAPS))}', f'--out_dir=checkpoints/multi_{arm}',
        f'--wandb_run_name=ua450-multi-{arm}'], OUT / f'{arm}.log')


def score(run, it, seed):
    out = SCORES / f'{run}_it{it}_s{seed}'
    if (out / 'scores.json').exists():
        return
    sh(['.venv/bin/python', 'eval/multi_generate.py', '--csv', CSV, '--songs', '24', '--checkpoint',
        str(ckpt(run, it).relative_to(ROOT)), '--seed', str(seed), '--min-new', '1000', '--no-mp3',
        '--out-dir', str(out.relative_to(ROOT))], OUT / f'score_{run}_it{it}_s{seed}.log')


def scores():
    """{(run, it): {seed: (mean distance, mean consistency, median notes/s ratio to real)}}"""
    s = {}
    for f in SCORES.glob('*_it*_s*/scores.json'):
        run, rest = f.parent.name.rsplit('_it', 1)
        it, seed = rest.split('_s')
        g = json.load(open(f))['groups']['ukrainian_multi_v1']
        ratio = statistics.median((x['metrics']['notes_per_s'] if x['metrics'] else 0) / x['real']['notes_per_s']
                                  for x in g['songs'])
        s.setdefault((run, int(it)), {})[int(seed)] = (g['mean_dist'], g.get('mean_cons', float('nan')), ratio)
    return s


def mean_dist(v, seeds):
    return statistics.mean(v[k][0] for k in seeds) if all(k in v for k in seeds) else float('nan')


def ce_curve(run):
    log = OUT / f'{run}.log'
    if not log.exists():
        return {}
    text = log.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    return {int(a): float(b) for a, b in STEP.findall(text)}


def summarize():
    s = scores()
    sds = [statistics.stdev(x[0] for x in v.values()) for v in s.values() if len(v) >= 3]
    lines = [f'Multi-track fine-tunes of the 450M, {time.strftime("%Y-%m-%d %H:%M")}: mean distance to the real '
             f'continuations of 24 multi_v1 val songs (lower = closer), ua_multi settings; nps = median notes/s '
             f'ratio to the real song; CE = multi_v1 val CE at the nearest eval (sweep arms only). '
             f'Seed sd per checkpoint: {statistics.mean(sds):.3f}' if sds else '',
             f'{"run":20s}{"it":>6s}{"CE":>8s}{"s1":>8s}{"s2":>8s}{"s3":>8s}{"mean":>8s}{"C":>7s}{"nps":>6s}']

    for (run, it), v in sorted(s.items(), key=lambda kv: mean_dist(kv[1], SEEDS)):
        ce = ce_curve(run)
        near = ce[min(ce, key=lambda x: abs(x - it))] if ce else float('nan')
        d = lambda k: f'{v[k][0]:8.3f}' if k in v else f'{"":8s}'
        m = mean_dist(v, SEEDS)
        c = statistics.mean(x[1] for x in v.values())
        r = statistics.mean(x[2] for x in v.values())
        lines.append(f'{run:20s}{it:6d}{near:8.3f}{d(1)}{d(2)}{d(3)}{m:8.3f}{c:7.2f}{r:6.2f}')
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
    ap.add_argument('--phases', type=int, nargs='+', default=[0, 1, 2], choices=[0, 1, 2])
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-battery', action='store_true', help='run even when the laptop is not on AC power')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        p0 = sum(len(v) for v in EXISTING.values()) * len(SEEDS) * SCORE_SEC
        p1 = len(args.arms) * (ITERS * SEC_PER_IT + STARTUP_SEC + len(SNAPS) * len(SEEDS) * SCORE_SEC)
        p2 = 2 * 3 * SCORE_SEC
        print(f'P0  {sum(len(v) for v in EXISTING.values())} existing checkpoints x {len(SEEDS)} seeds  ~{p0 / 60:.0f} min')
        for a in args.arms:
            print(f'P1  {a:10s} {ITERS} it, snapshots {SNAPS}, {len(SEEDS)} seeds each: {" ".join(ARMS[a]) or "(recipe)"}')
        print(f'    ~{p1 / 60:.0f} min for {len(args.arms)} arms')
        print(f'P2  listening samples of the top 2  ~{p2 / 60:.0f} min')
        print(f'total ~{(p0 + p1 + p2) / 3600:.1f} h; disk ~{len(args.arms) * (len(SNAPS) + 1) * 0.93:.1f} GB '
              f'(free: {shutil.disk_usage(ROOT).free / 2**30:.0f} GB); AC power now: {on_ac()}')
        return
    if not on_ac() and not args.allow_battery:
        sys.exit('on battery: plug in, or pass --allow-battery')

    if 0 in args.phases:
        print('== P0', flush=True)
        for run, its in EXISTING.items():
            for it in its:
                for seed in SEEDS:
                    score(run, it, seed)
        summarize()
    if 1 in args.phases:
        print('== P1', flush=True)
        for arm in args.arms:
            train(arm)
            for it in SNAPS:
                for seed in SEEDS:
                    score(arm, it, seed)
            summarize()
    if 2 in args.phases:
        print('== P2', flush=True)
        s = scores()
        top = sorted((mean_dist(v, SEEDS), k) for k, v in s.items() if k[0] in ARMS and all(x in v for x in SEEDS))
        for _, (run, it) in top[:2]:
            out = ROOT / f'samples/multi_sweep_listen/{run}_it{it}'
            if not (out / 'scores.json').exists():
                sh(['.venv/bin/python', 'eval/multi_generate.py', '--csv', CSV, '--checkpoint',
                    str(ckpt(run, it).relative_to(ROOT)), '--match', LISTEN, '--seed', '1', '--min-new', '1000',
                    '--out-dir', str(out.relative_to(ROOT))], OUT / f'listen_{run}_it{it}.log')
        summarize()
    print(f'MULTI SWEEP DONE {time.strftime("%F %T")}', flush=True)


if __name__ == '__main__':
    main()
