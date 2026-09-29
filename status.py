"""
Live dashboard of everything running in this project: training runs, song download, piano reduction, GPU, disk.

    .venv/bin/python status.py            # refreshes every 5 s, Ctrl+C to quit
    .venv/bin/python status.py --once     # print once

Everything is read from files (logs/*.log, data/audio, data/finetune) and nvidia-smi, so it never touches
the running jobs.
"""

import argparse
import csv
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'

# training runs in the order they execute: (label, log file, params)
RUNS = [
    ('ab-indep', 'ab_indep.log', '6M'),
    ('ab-cascade-v2', 'ab_cascade_v2.log', '6M'),
    ('ladder-S', 'ladder_S.log', '6M'),
    ('ladder-M', 'ladder_M.log', '20M'),
    ('ladder-L', 'ladder_L.log', '42M'),
    ('ladder-XL', 'ladder_XL.log', '65M'),
    ('ladder-L-fp8', 'ladder_L_fp8.log', '42M fp8'),
    ('ladder-L-pitchhead', 'ladder_L_pitchhead.log', '43M'),
    ('ladder-L-moe8', 'ladder_L_moe8.log', '117M/42M'),
    ('iso-M', 'iso_M.log', '20M 400Mt'),
    ('iso-L', 'iso_L.log', '42M 194Mt'),
    ('iso-XL', 'iso_XL.log', '65M 125Mt'),
    ('ladder-L-lr3e-4', 'ladder_L_lr3e-4.log', '42M'),
    ('ladder-L-lr1e-3', 'ladder_L_lr1e-3.log', '42M'),
    ('ladder-L-muon', 'ladder_L_muon.log', '42M muon'),
    ('ladder-L-muon-lr1e-3', 'ladder_L_muon_lr1e-3.log', '42M muon'),
    ('iso-S-muon', 'iso_S_muon.log', '6M 1.33Bt'),
]

TQDM = re.compile(r'Training:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)(it/s|s/it)')
STEP = re.compile(r'step (\d+): train loss ([\d.na]+), val loss [\d.]+ \| val CE ([\d.]+) '
                  r'\(pit ([\d.]+) vel ([\d.]+) dur ([\d.]+) del ([\d.]+)\)(?: \| val2 CE ([\d.]+))?')


def tail_text(path, n_bytes=200_000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n_bytes))
            return f.read().decode('utf-8', 'ignore').replace('\r', '\n')
    except FileNotFoundError:
        return None


def running(pattern):
    return subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


def training_table():
    t = Table(title='Training runs', expand=True, title_justify='left')
    for col, kw in [('run', {}), ('params', {'justify': 'right'}), ('state', {}), ('progress', {'ratio': 2}),
                    ('step', {'justify': 'right'}), ('speed', {'justify': 'right'}), ('ETA', {'justify': 'right'}),
                    ('val CE', {'justify': 'right'}), ('aria CE', {'justify': 'right'}),
                    ('pit/vel/dur/dt', {'justify': 'right'})]:
        t.add_column(col, **kw)
    active = [line.split('--out_dir=')[-1].split()[0] for line in
              subprocess.run(['pgrep', '-af', 'train.py'], capture_output=True, text=True).stdout.splitlines()
              if '--out_dir=' in line]
    for label, log, params in RUNS:
        text = tail_text(LOGS / log)
        if not text or not text.strip():
            t.add_row(label, params, Text('queued', style='dim'), '', '', '', '', '', '', '')
            continue
        bars = TQDM.findall(text)
        steps = STEP.findall(text)
        pct, cur, tot, _, eta, rate, unit = bars[-1] if bars else ('0', '0', '1', '', '?', '?', 'it/s')
        cur, tot = int(cur), int(tot)
        is_running = any(Path(a).name == log[:-4] for a in active)
        done = steps and int(steps[-1][0]) >= tot - 1
        state = (Text('done', style='green') if done else Text('running', style='bold yellow') if is_running
                 else Text('stopped', style='red'))
        if 'Traceback' in text[-5000:] or 'out of memory' in text[-5000:]:
            state = Text('CRASHED', style='bold red')
        speed = f'{rate} {unit}' if bars else ''
        s = steps[-1] if steps else None
        val = s[2] if s else ''
        aria = s[7] if s and s[7] else ''
        heads = f'{s[3]}/{s[4]}/{s[5]}/{s[6]}' if s else ''
        prog = ProgressBar(total=tot, completed=tot if done else cur, width=None)
        t.add_row(label, params, state, prog, f'{tot if done else cur}/{tot}', '' if done else speed,
                  '' if done else eta, val, aria, heads)
    return t


def pipeline_table():
    t = Table(title='Fine-tune data pipeline', expand=True, title_justify='left')
    for col in ('stage', 'state', 'progress', 'count', 'detail'):
        t.add_column(col, ratio=2 if col == 'progress' else None)

    # download
    songs_csv = ROOT / 'data/audio/songs.csv'
    per_artist = Counter()
    if songs_csv.exists():
        with open(songs_csv, newline='', encoding='utf-8') as f:
            per_artist = Counter(row['artist'] for row in csv.DictReader(f))
    artists = [line.split('|')[0].strip() for line in open(ROOT / 'data/artists.txt', encoding='utf-8')
               if line.strip() and not line.startswith('#')]
    fetch_running = running('data/fetch_songs.py')
    # an artist counts as done when it has songs and no download is running (some bands have < 25 uploads)
    done_artists = sum(1 for a in artists if per_artist.get(a, 0) >= (20 if fetch_running else 1))
    dl_state = Text('running', style='bold yellow') if running('data/fetch_songs.py') else Text('idle', style='dim')
    missing = [a for a in artists if per_artist.get(a, 0) < (20 if fetch_running else 1)]
    t.add_row('download', dl_state, ProgressBar(total=len(artists), completed=done_artists),
              f'{sum(per_artist.values())} songs', f"{done_artists}/{len(artists)} artists; "
              f"left: {', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}")

    # reduction
    audio = [p for p in (ROOT / 'data/audio').glob('*/*.mp3') if p.parent.name != 'skryabin_test']
    audio += list((ROOT / 'Skryabin').glob('*.mp3'))
    reduced = [p for p in (ROOT / 'data/finetune').glob('*/midi/*.mid') if p.parts[-3] != 'skryabin_test']
    red_state = Text('running', style='bold yellow') if running('run_reduce.sh') else Text('idle', style='dim')
    rate = ''
    recent = [p.stat().st_mtime for p in reduced if time.time() - p.stat().st_mtime < 900]
    if len(recent) >= 2:
        per_song = (max(recent) - min(recent)) / (len(recent) - 1)
        left = max(0, len(audio) - len(reduced))
        rate = f'{per_song:.0f} s/song, ~{left * per_song / 60:.0f} min left for {left} songs'
    failed = (tail_text(LOGS / 'reduce.log') or '').count('FAILED')
    t.add_row('piano reduction', red_state, ProgressBar(total=max(1, len(audio)), completed=len(reduced)),
              f'{len(reduced)}/{len(audio)}', rate + (f'; {failed} failed' if failed else ''))
    return t


def system_panel():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,'
                              'temperature.gpu,clocks.sm', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(', ')
        util, mem, mem_tot, power, temp, clock = out
        gpu = (f'GPU {util:>3}%   mem {int(mem) / 1024:.1f}/{int(mem_tot) / 1024:.1f} GB   '
               f'{float(power):.0f} W   {temp} °C   {clock} MHz')
    except Exception:
        gpu = 'GPU: nvidia-smi unavailable'
    du = shutil.disk_usage(ROOT)
    disk = f'disk free {du.free / 1e9:.1f} GB of {du.total / 1e9:.0f} GB'
    style = 'bold red' if du.free < 2e9 else ''
    return Panel(Text(f'{gpu}\n') + Text(disk, style=style), title='System', title_align='left')


def render():
    return Group(Text(time.strftime('%H:%M:%S'), style='dim'), system_panel(), training_table(), pipeline_table())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if args.once:
        from rich.console import Console
        Console().print(render())
        return
    with Live(render(), refresh_per_second=1, screen=False) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(render())
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
