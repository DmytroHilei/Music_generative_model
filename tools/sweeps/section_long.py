"""
The long section arm (config/pilots/section_long.py: A256, change oversampling, 8,000 it) in chunks, with the
teacher-forced probe around section changes after each chunk and an early stop. context/claude/text_control.md
"Section captions".

Each chunk: train.py for --minutes (resumes from <out_dir>/ckpt.pt when it exists), then eval/probes/section_gain.py
--at-changes on the latest weights (model_bf16.pt, written at every evaluation, i.e. every 500 it). Target: the
caption gain on the first 32 notes of sections that change by >= 2 points vs the section before (the base model
predicts them 0.2-0.3 nats worse than the rest of the section; the 1,200-it sweep arms: +0.0005).
Early stop: weights at >= --stop-at iterations and gain vs shuffled (another file's section captions: the content)
< --min-gain -> no further chunks (--force overrides). The gain vs none is shown, not used: early on it is as large on
sections without a change (the fresh null caption, not content).

    .venv/bin/python tools/sweeps/section_long.py --minutes 60                # one chunk of an hour
    .venv/bin/python tools/sweeps/section_long.py --minutes 45 --duty 0.6     # quieter: the GPU idles 40% of the time
    .venv/bin/python tools/sweeps/section_long.py --chunks 0                  # to the end (overnight), probe per chunk
    .venv/bin/python tools/sweeps/section_long.py --probe-only                # probe the current weights
    .venv/bin/python tools/sweeps/section_long.py -- --section_change_frac=0.7   # extra train.py arguments
    # the change-aware caption arm (config/pilots/section_long_change.py), comparable with long_A256:
    .venv/bin/python tools/sweeps/section_long.py --name long_A256_change --out-dir checkpoints/long_A256_change \
        --config config/pilots/section_long_change.py --minutes 90

Every chunk of a run must train with the same config files and extra train.py arguments (else e.g. a chunk without
section_style=change would train on plain captions): the first chunk records them in the state json, a later chunk
with others is refused (--force overrides).
Ctrl+C during training: train.py saves ckpt.pt after the current step and exits; the driver stops too.
Log logs/sections/long_A256.log (appended), probe results/section_gain/long_A256/it<N>, table logs/sections/
long_A256.md, state logs/sections/long_A256.json.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ['config/pilots/text_pilot.py', 'config/pilots/section_pilot.py', 'config/pilots/section_long.py']
NAME = 'long_A256'
OUT_DIR = Path('/var/tmp/musicar_sections/long_A256')  # = out_dir of section_long.py
MAX_ITERS = 8000
RESUMABLE = re.compile(r'saved resumable checkpoint at iter (\d+)')


def parse_args():
    global LOG, STATE, TABLE, PROBE, OUT_DIR, MAX_ITERS
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--minutes', type=float, default=60, help='training minutes per chunk')
    p.add_argument('--chunks', type=int, default=1, help='chunks to run (0 = until done or stopped)')
    p.add_argument('--duty', type=float, default=1.0, help='train.py --gpu_duty (< 1 = quieter, slower)')
    p.add_argument('--stop-at', type=int, default=3000, help='early-stop check from weights of this iteration on')
    p.add_argument('--min-gain', type=float, default=0.005, help='nats per note on the first 32 notes after changes')
    p.add_argument('--windows', type=int, default=512, help='probe windows around changes')
    p.add_argument('--probe-only', action='store_true')
    p.add_argument('--config', action='append', default=[], help='extra train.py config file(s) after CONFIGS')
    p.add_argument('--force', action='store_true',
                   help='train on after an early stop, or with other settings than the first chunk')
    p.add_argument('--dry-run', action='store_true', help='print the commands only')
    p.add_argument('--name', default=NAME, help='log / table / probe names (another name = a separate run)')
    p.add_argument('--out-dir', default=str(OUT_DIR), help="train.py out_dir (passed on when it isn't the default)")
    p.add_argument('--max-iters', type=int, default=MAX_ITERS, help='= max_iters (passed on when not the default)')
    p.add_argument('train_args', nargs='*', help='after --: extra train.py arguments')
    args = p.parse_args()
    LOG, STATE, TABLE = (ROOT / 'logs/sections' / f'{args.name}{ext}' for ext in ('.log', '.json', '.md'))
    PROBE = ROOT / 'results/section_gain' / args.name
    if Path(args.out_dir) != OUT_DIR:
        args.train_args.append(f'--out_dir={args.out_dir}')
    if not Path(args.out_dir).is_absolute():  # train.py runs in ROOT
        args.out_dir = str(ROOT / args.out_dir)
    if args.max_iters != MAX_ITERS:
        args.train_args += [f'--max_iters={args.max_iters}', f'--lr_decay_iters={args.max_iters}']
    OUT_DIR, MAX_ITERS = Path(args.out_dir), args.max_iters
    return args


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {'probes': [], 'stopped': ''}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def trained_iters():
    """The next iteration to run, from the last resumable checkpoint in the log (0 = not started)."""
    if not LOG.exists():
        return 0
    found = RESUMABLE.findall(LOG.read_text(encoding='utf-8', errors='ignore'))
    return int(found[-1]) if found else 0


def train_chunk(args):
    cmd = ['.venv/bin/python', 'train.py', *CONFIGS, *args.config, f'--max_minutes={args.minutes:g}',
           f'--gpu_duty={args.duty:g}', *args.train_args]
    if (OUT_DIR / 'ckpt.pt').exists():
        cmd.append('--init_from=resume')
    print(f'{time.strftime("%H:%M")} train from iter {trained_iters()}: ' + ' '.join(cmd[2:]), flush=True)
    if args.dry_run:
        return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(f'\n===== chunk {time.strftime("%Y-%m-%d %H:%M")}: {" ".join(cmd[2:])}\n')
        f.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:  # train.py got the SIGINT too: wait while it saves ckpt.pt and exits
                print('interrupted: waiting for train.py to save ckpt.pt', flush=True)


def change_gain(table, buckets=('2-3', '3-6')):
    """Note-weighted mean gain over the change-score buckets >= 2 of a probe table [(bucket, notes, gain)]."""
    rows = [(n, g) for b, n, g in table if b in buckets and n]
    return sum(n * g for n, g in rows) / max(1, sum(n for n, _ in rows))


def probe(args, state):
    """Probe the current weights unless their iteration is probed already; returns the new row or None."""
    out = PROBE / 'latest'
    cmd = ['.venv/bin/python', 'eval/probes/section_gain.py', '--checkpoints', str(OUT_DIR), '--at-changes',
           '--windows', str(args.windows), '--out', str(out)]
    print(f'{time.strftime("%H:%M")} probe: ' + ' '.join(cmd[2:]), flush=True)
    if args.dry_run or not (OUT_DIR / 'model_bf16.pt').exists():
        return None
    subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    r = json.loads((out / 'gain.json').read_text())[0]
    if any(p['iter'] == r['iter'] for p in state['probes']):
        print(f"weights of iter {r['iter']} probed already")
        return None
    (out / 'gain.json').replace(out.with_name(f"it{r['iter']}.json"))
    (out / 'summary.md').replace(out.with_name(f"it{r['iter']}.md"))
    ce = {b: c for b, _, c in r['ce']['change_start']}
    rest = {b: c for b, _, c in r['ce']['change_rest']}
    row = {'iter': r['iter'], 'time': time.strftime('%Y-%m-%d %H:%M'),
           'gain_none': change_gain(r['none']['change_start']),
           'gain_shuffled': change_gain(r['shuffled']['change_start']),
           'gain_none_nochange': change_gain(r['none']['change_start'], ('0-1',)),
           'excess': change_gain([(b, n, ce[b] - rest[b]) for b, n, _ in r['ce']['change_start']]),
           'all_none': r['none']['all']}
    state['probes'].append(row)
    return row


def write_table(state, args):
    lines = [f'# {args.name}: caption gain on the first 32 notes of sections after a change of >= 2 points', '',
             f'Probe: {args.windows} held-out windows around changes; gain in nats per note; excess = CE of those '
             'notes minus the rest of their section (what a caption could remove).', '',
             '| weights @iter | probed | gain vs none | vs shuffled | no-change sections vs none | all notes vs none '
             '| excess |', '|---|---|---|---|---|---|---|']
    for p in state['probes']:
        lines.append(f"| {p['iter']} | {p['time']} | {p['gain_none']:+.4f} | {p['gain_shuffled']:+.4f} | "
                     f"{p['gain_none_nochange']:+.4f} | {p['all_none']:+.4f} | {p['excess']:+.3f} |")
    if state['stopped']:
        lines += ['', f"**Stopped:** {state['stopped']}"]
    TABLE.write_text('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    state = load_state()
    if args.probe_only:
        row = probe(args, state)
        save_state(state)
        write_table(state, args)
        print(json.dumps(row, indent=1) if row else '', f'-> {TABLE}')
        return
    settings = {'configs': CONFIGS + args.config, 'train_args': args.train_args}
    if state.get('settings', settings) != settings and not args.force:
        sys.exit(f"{args.name} was trained with {state['settings']}, not {settings}: every chunk needs the same "
                 f"config files and train.py arguments (--force to train on anyway)")
    if 'settings' not in state and not args.dry_run:
        state['settings'] = settings
        save_state(state)
    n = 0
    while args.chunks == 0 or n < args.chunks:
        if state['stopped'] and not args.force:
            sys.exit(f"stopped: {state['stopped']} (--force to train on)")
        if trained_iters() >= MAX_ITERS:
            print(f'trained to {MAX_ITERS} iterations: done')
            break
        code = train_chunk(args)
        n += 1
        row = probe(args, state)
        if row:
            print(f"iter {row['iter']}: change-start gain vs none {row['gain_none']:+.4f}, "
                  f"vs shuffled {row['gain_shuffled']:+.4f} (excess {row['excess']:+.3f})")
            if row['iter'] >= args.stop_at and row['gain_shuffled'] < args.min_gain:
                state['stopped'] = (f"iter {row['iter']}: change-start gain vs shuffled {row['gain_shuffled']:+.4f} "
                                    f"< {args.min_gain}")
        if not args.dry_run:
            save_state(state)
            write_table(state, args)
        if code != 0:
            sys.exit(f'train.py exited with {code}' + (' (interrupted, ckpt.pt saved)' if code in (130, 143) else ''))
        if args.dry_run:
            break
    print(f'-> {TABLE}')


if __name__ == '__main__':
    main()
