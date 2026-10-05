"""
Live dashboard of everything running in this project: training runs, song download, piano reduction, GPU, disk.

    .venv/bin/python status.py            # refreshes every 5 s, Ctrl+C to quit
    .venv/bin/python status.py --once     # print once

Everything is read from files (logs/*.log, data/audio, data/finetune) and nvidia-smi, so it never touches
the running jobs.
"""

import argparse
import ast
import csv
import json
import os
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

from jobstatus import read_jobs

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
    ('iso-M-muon', 'iso_M_muon.log', '20M 400Mt'),
    ('iso-L-muon', 'iso_L_muon.log', '42M 194Mt'),
    ('big-107M', 'big_107M.log', '107M 2.33Bt'),
    ('ua-A lr1e-4', 'ua_A_lr1e-4.log', 'FT 107M'),
    ('ua-A lr3e-4', 'ua_A_lr3e-4.log', 'FT 107M'),
    ('ua-A lr1e-3', 'ua_A_lr1e-3.log', 'FT 107M'),
    ('poprock-B1', 'poprock_B1.log', '107M 203Mt'),
    ('ua-A2 (fixed split)', 'ua_A2.log', 'FT 107M'),
    ('ua-B2', 'ua_B2.log', 'FT from B1'),
    ('ua-B3 style x10', 'ua_B3_x10.log', 'FT from B1'),
    ('ua-B3 style x30', 'ua_B3_x30.log', 'FT from B1'),
    ('ua-C tempo aug', 'ua_C_tempo.log', 'FT from B1'),
    ('ua-C tempo+vel aug', 'ua_C_tempovel.log', 'FT from B1'),
    ('ua-C seed 2 (noise)', 'ua_C_seed2.log', 'FT from B1'),
    ('ua-C tempo aug, UA only', 'ua_C_tempo_ft.log', 'FT from B1'),
    ('ua-C tempo ±20%, UA only', 'ua_C_tempo20_ft.log', 'FT from B1'),
    ('ua-C winner, 1200 steps', 'ua_C_long1200.log', 'FT from B1'),
    ('ua-D full Ukrainian set, tempo ±10%', 'ua_D_big.log', 'FT from B1'),
    ('pilot-mi grown (RoPE 2048, 129 prog)', 'pilot_mi_grown.log', '108M 197Mt'),
    ('pilot-mi scratch (RoPE 2048, 129 prog)', 'pilot_mi_scratch.log', '108M 197Mt'),
    ('pilot-mi grown (stretched wpe 2048, 129 prog)', 'pilot_mi_wpe.log', '108M 197Mt'),
    ('ua-2048 covers + reductions (whole songs)', 'ua_2048.log', 'FT from pilot grown'),
    ('ua-2048 covers only (whole songs)', 'ua_2048_covers.log', 'FT from pilot grown'),
    ('mix grid s2 (0.5/0.3/0.2 seed 2)', 'mix_s2.log', 'FT from pilot grown'),
    ('mix grid c6r2 (0.6/0.2/0.2)', 'mix_c6r2.log', 'FT from pilot grown'),
    ('mix grid c4r4 (0.4/0.4/0.2)', 'mix_c4r4.log', 'FT from pilot grown'),
    ('mix grid c3r5 (0.3/0.5/0.2)', 'mix_c3r5.log', 'FT from pilot grown'),
    ('mix grid c5r4a1 (0.5/0.4/0.1)', 'mix_c5r4a1.log', 'FT from pilot grown'),
    ('mix grid c6r3a1 (0.6/0.3/0.1)', 'mix_c6r3a1.log', 'FT from pilot grown'),
    ('mix grid p12 (0.5/0.3/0.2, 12 passes)', 'mix_p12.log', 'FT from pilot grown'),
    ('ua-2048 v2 (bigger cover set)', 'ua_2048_v2.log', 'FT from pilot grown'),
    ('future heads 0.0 seed 1337 (ua-D recipe, 400 it)', 'fut_w0.0_s1337.log', 'FT from B1'),
    ('future heads 0.2 seed 1337 (ua-D recipe, 400 it)', 'fut_w0.2_s1337.log', 'FT from B1'),
    ('future heads 0.0 seed 2 (ua-D recipe, 400 it)', 'fut_w0.0_s2.log', 'FT from B1'),
    ('future heads 0.2 seed 2 (ua-D recipe, 400 it)', 'fut_w0.2_s2.log', 'FT from B1'),
    ('future heads 1.0 seed 1337 (ua-D recipe, 400 it)', 'fut_w1.0_s1337.log', 'FT from B1'),
    ('future heads 1.0 seed 2 (ua-D recipe, 400 it)', 'fut_w1.0_s2.log', 'FT from B1'),
    ('abl base (L, LayerNorm+GELU, lr 1e-3)', 'abl_base.log', '42M 98Mt'),
    ('abl newblock (RMSNorm+SwiGLU+QK-norm)', 'abl_newblock.log', '42M 98Mt'),
    ('abl newblock lr 2e-3', 'abl_newblock_lr2e-3.log', '42M 98Mt'),
    ('abl iso-M, old block (C=3.2e16)', 'abl_iso_M.log', '19M 209Mt'),
    ('abl iso-XL, old block (C=3.2e16)', 'abl_iso_XL.log', '16Lx640 54Mt'),
    ('abl iso-M, new block (C=3.2e16)', 'abl_iso_M_nb.log', '19M 209Mt'),
    ('abl iso-XL, new block (C=3.2e16)', 'abl_iso_XL_nb.log', '16Lx640 54Mt'),
    ('mix G .7 / A .3 (new block, lr 2e-3, 500 it)', 'abl_mix_g7a3.log', '42M 33Mt'),
    ('mix G .4 / A .3 / D .3', 'abl_mix_g4a3d3.log', '42M 33Mt'),
    ('mix G .25 / A .25 / D .5', 'abl_mix_g25a25d5.log', '42M 33Mt'),
    ('mix G .45 / A .1 / D .45', 'abl_mix_g45a1d45.log', '42M 33Mt'),
    ('crash-resume test (SIGKILL at ~4 min)', 'abl_resume_test.log', '42M 26Mt'),
    ('seed 2: base (old block)', 'abl_base_s2.log', '42M 98Mt'),
    ('seed 2: newblock', 'abl_newblock_s2.log', '42M 98Mt'),
    ('seed 3: base (old block)', 'abl_base_s3.log', '42M 98Mt'),
    ('seed 3: newblock', 'abl_newblock_s3.log', '42M 98Mt'),
    ('AdamW newblock seed 1337', 'abl_adamw_nb_s1337.log', '42M 98Mt'),
    ('AdamW newblock seed 2', 'abl_adamw_nb_s2.log', '42M 98Mt'),
    ('AdamW newblock seed 3', 'abl_adamw_nb_s3.log', '42M 98Mt'),
    ('abl iso-S, new block (C=3.2e16)', 'abl_iso_S_nb.log', '6Lx256 681Mt'),
    ('conditioning A/B: newblock + instruments + density', 'abl_cond.log', '42M 98Mt'),
]

TQDM = re.compile(r'Training:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)(it/s|s/it)')
STEP = re.compile(r'step (\d+): train loss ([\d.na]+), val loss [\d.]+ \| (\S+) CE ([\d.]+) '
                  r'\(pit ([\d.]+) vel ([\d.]+) dur ([\d.]+) del ([\d.]+)\)(?: \| (?!prog |future )(\S+) CE ([\d.]+))?')


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


def training_table(done_keep=None):
    """All unfinished runs plus the last done_keep finished ones (None = all)."""
    t = Table(title='Training runs', expand=True, title_justify='left')
    for col, kw in [('run', {}), ('params', {'justify': 'right'}), ('state', {}), ('progress', {'ratio': 2}),
                    ('step', {'justify': 'right'}), ('speed', {'justify': 'right'}), ('ETA', {'justify': 'right'}),
                    ('main val CE', {'justify': 'right'}), ('2nd val CE', {'justify': 'right'}),
                    ('pit/vel/dur/dt', {'justify': 'right'})]:
        t.add_column(col, **kw)
    # a run is live if some train.py process has its stdout on that log (works for --out_dir and config-file runs)
    active = set()
    for pid in subprocess.run(['pgrep', '-f', 'python train.py'], capture_output=True, text=True).stdout.split():
        try:
            active.add(Path(os.readlink(f'/proc/{pid}/fd/1')).name)
        except OSError:
            pass
    rows = []  # (finished?, cells)
    for label, log, params in RUNS:
        text = tail_text(LOGS / log)
        if not text or not text.strip():
            rows.append((False, (label, params, Text('queued', style='dim'), '', '', '', '', '', '', '')))
            continue
        bars = TQDM.findall(text)
        steps = STEP.findall(text)
        pct, cur, tot, _, eta, rate, unit = bars[-1] if bars else ('0', '0', '1', '', '?', '?', 'it/s')
        cur, tot = int(cur), int(tot)
        is_running = log in active
        done = steps and int(steps[-1][0]) >= tot - 1
        state = (Text('done', style='green') if done else Text('running', style='bold yellow') if is_running
                 else Text('stopped', style='red'))
        if 'Traceback' in text[-5000:] or 'out of memory' in text[-5000:]:
            state = Text('CRASHED', style='bold red')
        speed = f'{rate} {unit}' if bars else ''
        s = steps[-1] if steps else None
        # older logs print 'val CE' / 'val2 CE', newer ones the set's name (e.g. 'gigamidi_clean CE')
        named = lambda name, v: v if name in ('val', 'val2') else f'{v} {name}'
        val = named(s[2], s[3]) if s else ''
        aria = named(s[8], s[9]) if s and s[9] else ''
        heads = f'{s[4]}/{s[5]}/{s[6]}/{s[7]}' if s else ''
        prog = ProgressBar(total=tot, completed=tot if done else cur, width=None)
        rows.append((state.plain == 'done', (label, params, state, prog, f'{tot if done else cur}/{tot}',
                                             '' if done else speed, '' if done else eta, val, aria, heads)))
    finished = [i for i, (fin, _) in enumerate(rows) if fin]
    hidden = set(finished[:-done_keep] if done_keep else finished) if done_keep is not None else set()
    for i, (_, cells) in enumerate(rows):
        if i not in hidden:
            t.add_row(*cells)
    if hidden:
        t.caption = f'{len(hidden)} older finished runs hidden (status.py --all shows them)'
    return t


GIGA_BAR = re.compile(r'^(\w+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)file/s'
                      r'(?:, notes=(\w+), per_file=\d+, proj_gb=([\d.]+))?', re.M)
GIGA_DONE = re.compile(r'^(\w+): ([\d,]+) files, ([\d,]+) notes, (\d+) failed', re.M)
DISC_BAR = re.compile(r'discover:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)file/s'
                      r'(?:, kept=(\d+)%, notes=(\w+), proj_gb=([\d.]+))?')


def gigamidi_row(t):
    """data/prepare_gigamidi.py: splits finished (from the progress file) and the one being tokenized (from its log)."""
    progress_path = ROOT / 'data/gigamidi/gigamidi_progress.json'
    if not progress_path.exists():
        return
    done = [split for split, st in json.loads(progress_path.read_text()).items() if st.get('done')]
    is_running = running('data/prepare_gigamidi.py')
    text = tail_text(LOGS / 'prepare_gigamidi.log') or ''
    bars = GIGA_BAR.findall(text)
    stopped = 'STOP:' in text[-2000:]
    state = (Text('running', style='bold yellow') if is_running else Text('stopped (disk)', style='red') if stopped
             else Text('done', style='green') if 'train' in done else Text('idle', style='dim'))
    detail = f"done: {', '.join(done) or '—'}"
    if bars and is_running:
        split, _, cur, tot, _, eta, rate, notes, proj = bars[-1]
        detail += f"; {split}: {rate} files/s, ETA {eta}" + (f", {notes} notes, ~{proj} GB projected" if proj else '')
        t.add_row('GigaMIDI tokenize', state, ProgressBar(total=int(tot), completed=int(cur)),
                  f'{split} {int(cur):,}/{int(tot):,}', detail)
        return
    finished = GIGA_DONE.findall(text)
    if finished:
        detail += '; ' + ', '.join(f'{sp} {nf} files / {int(nn.replace(",", "")) / 1e6:.0f}M notes'
                                   for sp, nf, nn, _ in finished)
    t.add_row('GigaMIDI tokenize', state, ProgressBar(total=3, completed=len(done)), f'{len(done)}/3 splits', detail)


def discover_row(t):
    """Discover MIDI: download (logs/download_discover.log) then data/prepare_discover.py (logs/prepare_discover.log)."""
    text = tail_text(LOGS / 'prepare_discover.log')
    if text is None:
        return
    tar = Path('/data/discover/Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz')
    is_running = running('data/prepare_discover.py')
    bars = DISC_BAR.findall(text)
    if 'prepare done' in text:
        done = re.findall(r'^(train|validation): ([\d,]+) files, ([\d,]+) notes', text, re.M)
        t.add_row('Discover tokenize+dedupe', Text('done', style='green'), ProgressBar(total=1, completed=1), '',
                  ', '.join(f'{sp} {nf} files / {int(nn.replace(",", "")) / 1e6:.0f}M notes' for sp, nf, nn in done))
    elif is_running and bars:
        pct, cur, tot, _, eta, rate, kept, notes, proj = bars[-1]
        t.add_row('Discover tokenize+dedupe', Text('running', style='bold yellow'),
                  ProgressBar(total=int(tot), completed=int(cur)), f'{int(cur):,}/{int(tot):,}',
                  f'{rate} files/s, ETA {eta}' + (f', kept {kept}%, {notes} notes, ~{proj} GB projected' if kept else ''))
    elif not tar.exists():
        partial = list(Path('/data/discover/.cache/huggingface/download').glob('*.incomplete'))
        size = max((p.stat().st_size for p in partial), default=0)
        t.add_row('Discover download', Text('running' if running('hf download') else 'idle', style='bold yellow'),
                  ProgressBar(total=29_378, completed=size // 1_000_000), f'{size / 1e9:.1f}/29.4 GB', '')
    else:
        t.add_row('Discover tokenize+dedupe', Text('waiting' if is_running else 'idle', style='dim'),
                  ProgressBar(total=1, completed=0), '', 'starting')


def hf_upload_row(t):
    """cloud/data.py upload (logs/hf_upload.log): stores done of the manifest, GB, current store, final verification."""
    text = tail_text(LOGS / 'hf_upload.log', 1_000_000)
    manifest_path = ROOT / 'data/hf_upload/manifest.json'
    if text is None or not manifest_path.exists():
        return
    text = text.replace('\r', '\n')
    sizes = {}
    for rel, e in json.loads(manifest_path.read_text()).items():
        sizes[rel.split('/')[0]] = sizes.get(rel.split('/')[0], 0) + e['size']
    started = re.findall(r'^(\w+): uploading [\d.]+ GB \(attempt (\d+)\)', text, re.M)
    skipped = re.findall(r'^(\w+): already on the Hub', text, re.M)
    order = [s for s, _ in started]
    is_running = running('cloud/data.py upload')
    verified = 'upload verified' in text
    # a started store is finished once a later store has started (or the whole upload verified)
    done = set(skipped) | set(order[:-1]) | (set(order) if verified else set())
    gb_done = sum(sizes.get(s, 0) for s in done) / 1e9
    gb_total = sum(sizes.values()) / 1e9
    failed = re.findall(r'failed: (\w+)', text)
    if verified:
        state, detail = Text('done', style='green'), re.search(r'upload verified.*', text).group(0)
    elif is_running:
        cur, attempt = started[-1] if started else ('hashing', '1')
        state = Text('running', style='bold yellow')
        detail = f'now: {cur} ({sizes.get(cur, 0) / 1e9:.1f} GB, attempt {attempt})' + \
                 (f'; {len(failed)} failed attempts so far' if failed else '')
    else:
        state = Text('stopped', style='red')
        detail = (text.strip().splitlines() or [''])[-1][:120] + ' (re-run: python cloud/data.py upload)'
    t.add_row('HF data upload', state, ProgressBar(total=round(gb_total * 10), completed=round(gb_done * 10)),
              f'{len(done)}/{len(sizes)} stores, {gb_done:.1f}/{gb_total:.1f} GB', detail)


def query_templates():
    """fetch_songs.QUERY_TEMPLATES, read from the source (status.py's venv has no yt_dlp to import it with)."""
    tree = ast.parse((ROOT / 'data/fetch_songs.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'QUERY_TEMPLATES':
            return ast.literal_eval(node.value)
    return {}


def artist_states(log):
    """Latest state per artist over all runs in a fetch log: 'done' or 'skipped' (search kept failing)."""
    states, last_skipped = {}, None
    for line in log.splitlines():
        m = re.match(r'\s+SKIPPED (.+): search kept failing$', line)
        if m:
            last_skipped = m.group(1)
            continue
        m = re.match(r'^(.+): \d+ songs$', line)
        if m:
            states[m.group(1)] = 'skipped' if m.group(1) == last_skipped else 'done'
            last_skipped = None
    return states


def fetch_info(output_dir, log_path):
    """A round-2 fetch_songs.py run writing to output_dir: its settings (from the command line), the current run's
    progress (the log part after the last '=====' line), the songs downloaded so far (songs.csv) and an ETA."""
    log = (tail_text(log_path, 30_000_000) or '')
    procs = [l.split(None, 1) for l in subprocess.run(['pgrep', '-af', 'data/fetch_songs.py'], capture_output=True,
                                                      text=True).stdout.splitlines()
             if f'--output {output_dir}' in l and 'pgrep' not in l]
    info = {'running': bool(procs), 'paused': False, 'log': log}
    sep = list(re.finditer(r'^===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\n]*$', log, re.M))
    run = log[sep[-1].end():] if sep else log
    run_start = time.mktime(time.strptime(sep[-1].group(1), '%Y-%m-%d %H:%M:%S')) if sep else None
    args = {}
    if procs:
        pid, cmd = procs[0]
        args = dict(re.findall(r'--([\w-]+)(?:\s+(?!--)(\S+))?', cmd))
        info['paused'] = subprocess.run(['ps', '-o', 'stat=', '-p', pid], capture_output=True,
                                        text=True).stdout.strip().startswith('T')
        # a run started without a separator line (older fetch_songs.py) begins at its process start
        proc_start = time.time() - int(subprocess.run(['ps', '-o', 'etimes=', '-p', pid], capture_output=True,
                                                      text=True).stdout or 0)
        run_start = max(run_start or 0, proc_start)
    source, mode = args.get('source') or 'youtube', args.get('mode') or 'songs'
    info.update(source=source, mode=mode, cookies='cookies-from-browser' in args,
                search_size=args.get('search-size') or '80', sleep=float(args.get('sleep') or 3),
                search_sleep=float(args.get('search-sleep') or 0), templates=query_templates().get((source, mode), ()))
    names = []
    if args.get('artists') and (ROOT / args['artists']).exists():
        names = [l for l in (ROOT / args['artists']).read_text(encoding='utf-8').splitlines()
                 if l.strip() and not l.startswith('#')]
    info['aliases'] = sum(len(l.split('|')) for l in names) / len(names) if names else 0
    if names:  # a previous run without a separator line: this run begins where its list's first artist last started
        first = list(re.finditer(rf'^{re.escape(names[0].split("|")[0].strip())}: \d+ songs$', run, re.M))
        run = run[first[-1].start():] if first else run
    started = re.findall(r'^(.+): \d+ songs$', run, re.M)
    info['current'] = started[-1] if procs and started else None
    done = max(0, len(started) - (1 if procs else 0))
    info.update(run_total=len(names), run_done=done, run_left=max(0, len(names) - done),
                n403=run.count('HTTP Error 403'), skipped=run.count('SKIPPED'), aborted='ABORTED' in run)
    info['eta'] = None
    if procs and run_start and done:
        per_artist = (time.time() - run_start) / done
        info.update(per_artist=per_artist, eta=info['run_left'] * per_artist)
    csv_path = Path(output_dir) / 'songs.csv'
    info['downloaded'] = max(0, sum(1 for _ in open(csv_path, encoding='utf-8')) - 1) if csv_path.exists() else 0
    run_rows = run.count('[get ]') - run.count('FAILED')  # '[get ]' can follow yt-dlp's progress on a line
    info['per_artist_songs'] = run_rows / done if done else None
    return info


def eta_text(secs):
    return f'~{secs / 3600:.1f} h left (≈ {time.strftime("%a %H:%M", time.localtime(time.time() + secs))})'


def fetch_lines(info):
    """Two detail lines for a round-2 download: where it is and what it searches."""
    src = info['source'] + (' + browser cookies' if info['cookies'] else '')
    if info['running']:
        now = 'PAUSED (reduction catching up)' if info['paused'] else f"now: {info['current'] or 'searching'}"
        line1 = f"{src} · {now} · this run {info['run_done']}/{info['run_total']} artists, {info['run_left']} left"
        if info['eta'] is not None:
            line1 += f" · {info['per_artist'] / 60:.1f} min/artist, {eta_text(info['eta'])}"
    else:
        line1 = f'{src} · not running' + (' · last run ABORTED (search blocked)' if info['aborted'] else '')
    queries = ', '.join(f'"{t.format("<artist>")}"' for t in info['templates'])
    line2 = (f"queries ×{info['aliases']:.1f} names/artist: {queries} · {info['search_size']} results each, "
             f"{info['search_sleep']:g} s between searches, {info['sleep']:g} s between downloads") \
        if info['running'] else ''
    return line1, line2


def round2_artists():
    return [l for l in (ROOT / 'data/artists_covers_next.txt').read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')]


def run_state(info, busy, pending):
    if info['running'] or busy:
        return Text('running', style='bold yellow')
    if pending or info['run_left'] or info['aborted']:
        return Text('stopped', style='red')
    return Text('done', style='green')


def covers_round2_row(t):
    """Piano covers of data/artists_covers_next.txt: download (fetch_covers2.log, appended re-runs), transcribed by
    logs/run_gpu_worker_round2.sh."""
    if not (LOGS / 'fetch_covers2.log').exists():
        return
    out = Path('/data/covers_round2')
    info = fetch_info(str(out), LOGS / 'fetch_covers2.log')
    states = artist_states(info['log'])
    total = len(round2_artists())
    handled = sum(1 for v in states.values() if v == 'done') - (1 if info['running'] else 0)
    redo = sum(1 for v in states.values() if v == 'skipped')
    midis = len(list(out.glob('*/midi/*.mid')))
    pending = len(list(out.glob('*/*.mp3')))
    line1, line2 = fetch_lines(info)
    line3 = (f"{info['downloaded']} covers downloaded, {midis} transcribed, {pending} mp3 waiting · this run: "
             f"{info['n403']} × 403, {info['skipped']} artists skipped" + (f' · {redo} skipped to redo' if redo else ''))
    t.add_row('piano covers round 2', run_state(info, False, pending), ProgressBar(total=total, completed=handled),
              f'{handled}/{total} artists', '\n'.join(l for l in (line1, line2, line3) if l))


def songs_round2_rows(t):
    """Original songs of data/artists_covers_next.txt: download (fetch_songs2.log, appended re-runs) and piano
    reduction by logs/run_gpu_worker_round2.sh."""
    if not (LOGS / 'fetch_songs2.log').exists():
        return
    songs, red = Path('/data/songs_round2'), Path('/data/finetune_round2')
    info = fetch_info(str(songs), LOGS / 'fetch_songs2.log')
    states = artist_states(info['log'])
    total = len(round2_artists())
    handled = sum(1 for v in states.values() if v == 'done') - (1 if info['running'] else 0)
    redo = sum(1 for v in states.values() if v == 'skipped')
    waiting = len(unreduced_mp3s(songs, red))  # the worker may keep the mp3s (DELETE=0), so not every mp3 is waiting
    line1, line2 = fetch_lines(info)
    line3 = (f"{info['downloaded']} songs downloaded" +
             (f", {info['per_artist_songs']:.0f}/artist this run" if info['per_artist_songs'] else '') +
             f" · this run: {info['n403']} × 403, {info['skipped']} artists skipped" +
             (f' · {redo} skipped to redo' if redo else ''))
    t.add_row('songs round 2: download', run_state(info, False, 0), ProgressBar(total=total, completed=handled),
              f'{handled}/{total} artists', '\n'.join(l for l in (line1, line2, line3) if l))

    reduced = [p.stat().st_mtime for p in red.glob('*/midi/*.mid')]
    worker = running('run_gpu_worker_round2.sh')
    recent = [m for m in reduced if time.time() - m < 3600]
    detail = f'{waiting} mp3 waiting'
    if worker and len(recent) >= 2:
        per_song = (max(recent) - min(recent)) / (len(recent) - 1)
        detail = f'{3600 / per_song:.0f} songs/h over the last hour · ' + detail
        # still to come: the songs waiting + the remaining artists at this run's songs per artist
        to_come = waiting + (info['run_left'] * info['per_artist_songs'] if info['per_artist_songs'] else 0)
        eta = max(info['eta'] or 0, to_come * per_song)
        detail += f' · ~{to_come:.0f} songs still to reduce, {eta_text(eta)}'
    state = Text('running', style='bold yellow') if worker else \
        Text('done', style='green') if not waiting and not info['running'] else Text('stopped', style='red')
    t.add_row('songs round 2: piano reduction', state,
              ProgressBar(total=max(1, info['downloaded']), completed=len(reduced)),
              f'{len(reduced)}/{info["downloaded"]} songs', detail)


def unreduced_mp3s(songs, red):
    """mp3s under songs/<artist>/ that red/<artist>/ has neither reduced (songs.csv) nor failed on (failed.txt)."""
    out = []
    for d in songs.glob('*/'):
        done = set()
        if (red / d.name / 'songs.csv').exists():
            done = {r['source_audio'] for r in csv.DictReader(open(red / d.name / 'songs.csv', encoding='utf-8'))}
        if (red / d.name / 'failed.txt').exists():
            done |= set((red / d.name / 'failed.txt').read_text(encoding='utf-8').splitlines())
        out += [p for p in d.glob('*.mp3') if str(p) not in done]
    return out


def redownload_rows(t):
    """data/redownload.py streams (lost mp3s fetched again by URL): one row per log, its run after the last '=====' line."""
    for name, log_name in (('SoundCloud', 'redownload_sc.log'), ('YouTube', 'redownload_yt.log')):
        log = tail_text(LOGS / log_name, 30_000_000)
        sep = list(re.finditer(r'^===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d): re-download of (\S+) \((\w+)\) =====$',
                               log or '', re.M))
        if not sep:
            continue
        run = log[sep[-1].end():]
        start = time.mktime(time.strptime(sep[-1].group(1), '%Y-%m-%d %H:%M:%S'))
        which = 'round 1' if sep[-1].group(2).startswith('data/audio') else 'round 2'
        m = re.search(r'^(\d+) to download', run, re.M)
        total = int(m.group(1)) if m else 0
        steps = re.findall(r'^\[(\d+)/\d+\] ', run, re.M)
        at = int(steps[-1]) if steps else 0
        failed = run.count('    FAILED')
        finished = 'RE-DOWNLOAD DONE' in run
        alive = running(f'data/redownload.py .*--source {sep[-1].group(3)}')
        completed = max(0, at - failed - (1 if alive and not finished else 0))
        detail = f'{name} · {which} · {failed} failed'
        if alive:
            if run.rstrip().endswith('waiting 5 min'):
                detail += ' · WAITING: disk below the free-space floor'
            elif at > 1:
                per_song = (time.time() - start) / (at - 1)
                detail += f' · {per_song:.0f} s/song, {eta_text((total - at + 1) * per_song)}'
            if which == 'round 2' and sep[-1].group(3) == 'youtube' and \
                    running('data/redownload.py --songs-csv data/audio/songs.csv'):  # the YouTube chain's 2nd run
                detail += ' · then round 1 (queued)'
        elif 'ABORTED' in run:
            detail += ' · ABORTED (downloads blocked?)'
        state = Text('running', style='bold yellow') if alive else \
            Text('done', style='green') if finished else Text('stopped', style='red')
        t.add_row(f're-download: {name}', state, ProgressBar(total=max(1, total), completed=completed),
                  f'{completed}/{total} songs', detail)


def pipeline_table():
    t = Table(title='Data pipeline', expand=True, title_justify='left')
    for col in ('stage', 'state', 'progress', 'count', 'detail'):
        t.add_column(col, width=24 if col == 'progress' else None, ratio=1 if col == 'detail' else None)

    # download
    songs_csv = ROOT / 'data/audio/songs.csv'
    per_artist = Counter()
    if songs_csv.exists():
        with open(songs_csv, newline='', encoding='utf-8') as f:
            per_artist = Counter(row['artist'] for row in csv.DictReader(f))
    artists = [line.split('|')[0].strip() for line in open(ROOT / 'data/artists.txt', encoding='utf-8')
               if line.strip() and not line.startswith('#')]
    # round 1 only (data/audio): the round-2 downloads write to /data/... and have their own rows
    round1 = [l.split(None, 1) for l in subprocess.run(['pgrep', '-af', 'data/fetch_songs.py'], capture_output=True,
                                                         text=True).stdout.splitlines()
              if '--output /data/' not in l and 'pgrep' not in l]
    fetch_running = bool(round1)
    # the fetch log prints "<artist>: N songs" when it starts an artist, so the artists before the last one are done
    fetch_log = next((LOGS / n for n in ('fetch2.log', 'fetch.log') if (LOGS / n).exists()), None)
    log_text = (tail_text(fetch_log, 10_000_000) or '') if fetch_log else ''
    started = re.findall(r'^(.+): (\d+) songs$', log_text, re.M)
    started = [a for a, _ in started]
    if fetch_running:
        done_artists = max(0, len(started) - 1)
        current = started[-1] if started else 'searching'
        detail = f"{done_artists}/{len(artists)} artists; now: {current}"
        pid = [round1[0][0]] if round1 else []
        etime = subprocess.run(['ps', '-o', 'etimes=', '-p', pid[0]], capture_output=True, text=True).stdout if pid else ''
        if etime.strip() and done_artists:
            per_artist_s = int(etime) / done_artists
            detail += f"; {per_artist_s / 60:.0f} min/artist, ~{(len(artists) - done_artists) * per_artist_s / 3600:.1f} h left"
    else:
        done_artists = sum(1 for a in artists if per_artist.get(a, 0) > 0)
        missing = [a for a in artists if not per_artist.get(a, 0)]
        detail = f"{done_artists}/{len(artists)} artists of data/artists.txt (YouTube)" + \
            (f"; none for: {', '.join(missing[:4])}" if missing else '')
    dl_state = Text('running', style='bold yellow') if fetch_running else Text('done', style='green')
    # this run's new downloads: "[get ]" lines in its log minus the failed ones
    new = log_text.count('[get ]') - log_text.count('FAILED')
    t.add_row('songs round 1: download', dl_state, ProgressBar(total=len(artists), completed=done_artists),
              f'{sum(per_artist.values())} songs ({new} in the last run)', detail)

    # reduction: pending = audio without its MIDI yet (with --delete-audio, reduced songs' mp3s are gone)
    audio = [(p, ROOT / 'data/finetune' / p.parent.name / 'midi' / f'{p.stem}.mid')
             for p in (ROOT / 'data/audio').glob('*/*.mp3') if p.parent.name != 'skryabin_test']
    audio += [(p, ROOT / 'data/finetune/skryabin_local/midi' / f'{p.stem}.mid') for p in (ROOT / 'Skryabin').glob('*.mp3')]
    pending = sum(1 for _, midi in audio if not midi.exists())
    reduced = [p for p in (ROOT / 'data/finetune').glob('*/midi/*.mid') if p.parts[-3] != 'skryabin_test']
    red_running = running('run_reduce.sh') or running('run_fetch_reduce2.sh')
    red_state = Text('running', style='bold yellow') if red_running else Text('done', style='green')
    rate = ''
    recent = [p.stat().st_mtime for p in reduced if time.time() - p.stat().st_mtime < 900]
    if len(recent) >= 2:
        per_song = (max(recent) - min(recent)) / (len(recent) - 1)
        rate = f'{per_song:.0f} s/song, ~{pending * per_song / 60:.0f} min left for {pending} downloaded songs'
    elif pending:
        rate = f'{pending} downloaded songs waiting'
    # reduce2.log also receives the round-2 reductions (appended): count round 1 only, up to its DONE line
    failed = sum((tail_text(LOGS / n) or '').split('REDUCTION DONE')[0].count('FAILED') for n in ('reduce.log', 'reduce2.log'))
    t.add_row('songs round 1: piano reduction', red_state, ProgressBar(total=max(1, len(reduced) + pending), completed=len(reduced)),
              f'{len(reduced)}/{len(reduced) + pending}', rate + (f'; {failed} failed' if failed else ''))
    gigamidi_row(t)
    discover_row(t)
    hf_upload_row(t)
    covers_round2_row(t)
    songs_round2_rows(t)
    redownload_rows(t)
    return t


def fmt_secs(s):
    s = int(s)
    return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}' if s >= 3600 else f'{s // 60}:{s % 60:02d}'


def jobs_table(keep=6):
    """generate.py / sample_sweep.py runs (logs/jobs/*.json): all live ones plus the most recent finished ones."""
    t = Table(title='Sampling jobs', expand=True, title_justify='left')
    for col, kw in [('job', {}), ('pid', {'justify': 'right'}), ('state', {}), ('progress', {'ratio': 2}),
                    ('notes', {'justify': 'right'}), ('speed', {'justify': 'right'}), ('ETA', {'justify': 'right'}),
                    ('detail / result', {'ratio': 2})]:
        t.add_column(col, **kw)
    jobs = read_jobs()
    alive = lambda j: Path(f"/proc/{j['pid']}").exists()
    live = [j for j in jobs if not j['finished'] and alive(j)]
    rest = [j for j in jobs if j not in live][-max(0, keep - len(live)):] if keep > len(live) else []
    for j in rest + live:
        elapsed = (j['updated'] if j['finished'] or not alive(j) else time.time()) - j['started']
        rate = j['done'] / elapsed if elapsed > 0 else 0
        if j['finished']:
            state = Text('done', style='green')
        elif alive(j):
            state = Text('running', style='bold yellow')
        else:
            state = Text('died', style='red')
        running_now = state.plain == 'running'
        eta = fmt_secs((j['total'] - j['done']) / rate) if running_now and rate > 0 else ''
        t.add_row(j['name'], str(j['pid']), state, ProgressBar(total=max(1, j['total']), completed=j['done']),
                  f"{j['done']}/{j['total']}", f'{rate:.1f}/s' if rate else '', eta,
                  j['result'] if j['finished'] else j['detail'])
    if not jobs:
        t.add_row('—', '', Text('none yet', style='dim'), '', '', '', '', 'generate.py / sample_sweep.py')
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


def render(show_all=False):
    # what is running comes first: the live view is cut at the terminal height
    return Group(Text(time.strftime('%H:%M:%S'), style='dim'), system_panel(), pipeline_table(),
                 training_table(None if show_all else 4), jobs_table(keep=6 if show_all else 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--all', action='store_true', help='also show old finished runs and sampling jobs')
    args = parser.parse_args()
    if args.once:
        from rich.console import Console
        Console().print(render(args.all))
        return
    with Live(render(args.all), refresh_per_second=1, screen=False) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(render(args.all))
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
