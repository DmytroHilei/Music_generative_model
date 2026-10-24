"""
Live dashboard of everything running in this project: training runs, song download, piano reduction, GPU, disk.

    .venv/bin/python tools/status.py            # refreshes every 5 s, Ctrl+C to quit
    .venv/bin/python tools/status.py --once     # print once
    .venv/bin/python tools/status.py --web 8765 # no terminal view: serve it as a web page (phone on the same network:
                                                # http://<laptop ip>:8765/?t=<token>, the token is printed);
                                                # the project to-do list at /todo?t=<token> (tools/todo_web.py),
                                                # the CI reports at /ci/?t=<token> (tools/ci_web.py)
    tools/dashboard_service.sh install          # the same, always on: systemd user service on 127.0.0.1, fixed token,
                                                # https://<machine>.<tailnet>.ts.net through tailscale serve

Everything is read from files (logs/*.log, data/audio, data/finetune) and nvidia-smi, so it never touches
the running jobs. Long section runs (tools/sweeps/section_long.py, train.py in chunks) get their own table: the chunk
in progress and when it ends, the last evaluation, the probes after each chunk and the early stop.
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

from musicar.jobstatus import read_jobs

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / 'logs'
COOKIE = 'musicar_t'  # web page: the access token as a cookie (set by any ?t=<token> request)

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
    ('ua-450 stage 1 (round 1+2 data, 450M)', 'ua_450_s1.log', 'FT 450M'),
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


def head_until(path, marker, chunk=1 << 20):
    """The start of a file up to the first marker (the whole file if it never appears)."""
    out = b''
    try:
        with open(path, 'rb') as f:
            while (block := f.read(chunk)):
                out += block
                if marker.encode() in out:
                    break
    except FileNotFoundError:
        return ''
    return out.decode('utf-8', 'ignore').split(marker)[0]


def running(pattern):
    """PID of the oldest process matching pattern (the top of its tree: a runner script, not its children), or None."""
    out = subprocess.run(['pgrep', '-o', '-f', pattern], capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def running_text(pid, label='running'):
    """State cell of a live stage, with the PID to kill it by."""
    return Text(f'{label}\npid {pid}' if pid else label, style='bold yellow')


def active_train_logs():
    """{log name: (pid, log path, config)} of the live train.py runs. A run is live if a train.py process has its
    stdout on that log (works for --out_dir and config-file runs); pid is the main process, not a data worker."""
    pids = subprocess.run(['pgrep', '-f', 'python train.py'], capture_output=True, text=True).stdout.split()
    active = {}
    for pid in pids:
        try:
            with open(f'/proc/{pid}/stat') as f:
                ppid = f.read().rsplit(')', 1)[1].split()[1]
            if ppid in pids:  # a DataLoader worker shares the parent's stdout
                continue
            path = Path(os.readlink(f'/proc/{pid}/fd/1'))
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                args = f.read().decode().split('\0')
        except OSError:
            continue
        if not path.is_file():  # stdout on a terminal or pipe: no log to read
            continue
        config = next((Path(a).stem for a in args if a.startswith('config/')), '')
        active[path.name] = (pid, path, config)
    return active


SKIP_SETTINGS = {'out_dir', 'wandb_run_name', 'lr_decay_iters', 'eval_interval'}
SEC_LATE = re.compile(r'section gain [-+.\d]+ / none [-+.\d]+(?: / prev [-+.\d]+)?, late [-+.\d]+ / ([-+.\d]+)')


def run_settings(path):
    """The command-line overrides a train.py log printed ('Overriding: key = value'), e.g. 'section_mode=xattn
    section_lead=128', plus the latest late section gain vs none if the run logs one."""
    head = head_until(path, 'compiling the model')
    items = [f'{k}={v}' for k, v in re.findall(r'^Overriding: (\w+) = (.*)$', head, re.M) if k not in SKIP_SETTINGS]
    gains = SEC_LATE.findall(tail_text(path) or '')
    return ' '.join(items) + (f'\nsection gain late {gains[-1]}' if gains else '')


def training_table(done_keep=None):
    """All unfinished runs plus the last done_keep finished ones (None = all)."""
    t = Table(title='Training runs', expand=True, title_justify='left')
    for col, kw in [('run', {}), ('params', {'justify': 'right'}), ('state', {}), ('progress', {'ratio': 2}),
                    ('step', {'justify': 'right'}), ('speed', {'justify': 'right'}), ('ETA', {'justify': 'right'}),
                    ('main val CE', {'justify': 'right'}), ('2nd val CE', {'justify': 'right'}),
                    ('pit/vel/dur/dt', {'justify': 'right'})]:
        t.add_column(col, **kw)
    active = active_train_logs()
    # live runs not listed in RUNS are picked up from their process: label = log name, params = config file
    known = {log for _, log, _ in RUNS}
    runs = [(label, LOGS / log, params) for label, log, params in RUNS]
    # section-caption arms (tools/sweeps/section_sweep.py), live or finished, with their settings: train.py logs only
    # (not the probe / driver logs next to them); the chunked long runs have their own table (long_runs_view)
    logs = [p for p in (LOGS / 'sections').glob('*.log') if p.name != 'overnight.log']
    heads = {p: log_head(p) for p in logs}
    sections = sorted((p for p in logs if b'Overriding' in heads[p] and b'===== chunk ' not in heads[p]),
                      key=lambda p: p.stat().st_mtime)
    runs += [(f'sec {p.stem}', p, run_settings(p)) for p in sections]
    known |= {p.name for p in logs}
    runs += [(path.stem, path, run_settings(path) or config) for name, (_, path, config) in sorted(active.items())
             if name not in known]
    rows = []  # (finished?, cells)
    for label, path, params in runs:
        log = path.name
        text = tail_text(path)
        if not text or not text.strip():
            rows.append((False, (label, params, Text('queued', style='dim'), '', '', '', '', '', '', '')))
            continue
        bars = TQDM.findall(text)
        steps = STEP.findall(text)
        pct, cur, tot, _, eta, rate, unit = bars[-1] if bars else ('0', '0', '1', '', '?', '?', 'it/s')
        cur, tot = int(cur), int(tot)
        is_running = log in active
        done = steps and int(steps[-1][0]) >= tot - 1
        state = (Text('done', style='green') if done
                 else Text(f'running\npid {active[log][0]}', style='bold yellow') if is_running
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
        t.caption = f'{len(hidden)} older finished runs hidden (tools/status.py --all shows them)'
    return t


GIGA_BAR = re.compile(r'^(\w+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)file/s'
                      r'(?:, notes=(\w+), per_file=\d+, proj_gb=([\d.]+))?', re.M)
GIGA_DONE = re.compile(r'^(\w+): ([\d,]+) files, ([\d,]+) notes, (\d+) failed', re.M)
DISC_BAR = re.compile(r'discover:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)file/s'
                      r'(?:, kept=(\d+)%, notes=(\w+), proj_gb=([\d.]+))?')


def gigamidi_row(t):
    """data_prep/datasets/prepare_gigamidi.py: splits finished (from the progress file) and the one being tokenized (from its
    log)."""
    progress_path = ROOT / 'data/gigamidi/gigamidi_progress.json'
    if not progress_path.exists():
        return
    done = [split for split, st in json.loads(progress_path.read_text()).items() if st.get('done')]
    is_running = running('data_prep/datasets/prepare_gigamidi.py')
    text = tail_text(LOGS / 'prepare_gigamidi.log') or ''
    bars = GIGA_BAR.findall(text)
    stopped = 'STOP:' in text[-2000:]
    state = (running_text(is_running) if is_running else Text('stopped (disk)', style='red') if stopped
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
    """Discover MIDI: download (logs/download_discover.log) then data_prep/datasets/prepare_discover.py
    (logs/prepare_discover.log)."""
    text = tail_text(LOGS / 'prepare_discover.log')
    if text is None:
        return
    tar = Path('/data/discover/Discover-MIDI-Dataset-CC-BY-NC-SA.tar.gz')
    is_running = running('data_prep/datasets/prepare_discover.py')
    bars = DISC_BAR.findall(text)
    if 'prepare done' in text:
        done = re.findall(r'^(train|validation): ([\d,]+) files, ([\d,]+) notes', text, re.M)
        t.add_row('Discover tokenize+dedupe', Text('done', style='green'), ProgressBar(total=1, completed=1), '',
                  ', '.join(f'{sp} {nf} files / {int(nn.replace(",", "")) / 1e6:.0f}M notes' for sp, nf, nn in done))
    elif is_running and bars:
        pct, cur, tot, _, eta, rate, kept, notes, proj = bars[-1]
        t.add_row('Discover tokenize+dedupe', running_text(is_running),
                  ProgressBar(total=int(tot), completed=int(cur)), f'{int(cur):,}/{int(tot):,}',
                  f'{rate} files/s, ETA {eta}' + (f', kept {kept}%, {notes} notes, ~{proj} GB projected' if kept else ''))
    elif not tar.exists():
        partial = list(Path('/data/discover/.cache/huggingface/download').glob('*.incomplete'))
        size = max((p.stat().st_size for p in partial), default=0)
        hf_pid = running('hf download')
        t.add_row('Discover download', running_text(hf_pid) if hf_pid else Text('idle', style='dim'),
                  ProgressBar(total=29_378, completed=size // 1_000_000), f'{size / 1e9:.1f}/29.4 GB', '')
    else:
        t.add_row('Discover tokenize+dedupe', Text('waiting' if is_running else 'idle', style='dim'),
                  ProgressBar(total=1, completed=0), '', 'starting')


def hf_upload_row(t):
    """cloud/sync/data.py upload (logs/hf_upload.log): stores done of the manifest, GB, current store, final verification."""
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
    is_running = running('cloud/sync/data.py upload')
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
        state = running_text(is_running)
        detail = f'now: {cur} ({sizes.get(cur, 0) / 1e9:.1f} GB, attempt {attempt})' + \
                 (f'; {len(failed)} failed attempts so far' if failed else '')
    else:
        state = Text('stopped', style='red')
        detail = (text.strip().splitlines() or [''])[-1][:120] + ' (re-run: python cloud/sync/data.py upload)'
    t.add_row('HF data upload', state, ProgressBar(total=round(gb_total * 10), completed=round(gb_done * 10)),
              f'{len(done)}/{len(sizes)} stores, {gb_done:.1f}/{gb_total:.1f} GB', detail)


def query_templates():
    """fetch_songs.QUERY_TEMPLATES, read from the source (status.py's venv has no yt_dlp to import it with)."""
    tree = ast.parse((ROOT / 'data_prep/songs/fetch_songs.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'QUERY_TEMPLATES':
            return ast.literal_eval(node.value)
    return {}


_artist_states = {}  # log path -> (bytes read, states, last skipped artist)


def artist_states(path):
    """Latest state per artist over all runs in a fetch log: 'done' or 'skipped' (search kept failing). Reads the
    whole log (tens of MB of download progress, more than tail_text keeps), only the new part on each refresh."""
    offset, states, last_skipped = _artist_states.get(path, (0, {}, None))
    if path.stat().st_size < offset:
        offset, states, last_skipped = 0, {}, None
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read()
    data = data[:data.rfind(b'\n') + 1]  # whole lines only: the rest is read next time
    for line in data.decode('utf-8', 'ignore').replace('\r', '\n').splitlines():
        m = re.match(r'\s+SKIPPED (.+): search kept failing$', line)
        if m:
            last_skipped = m.group(1)
            continue
        m = re.match(r'^(.+): \d+ songs$', line)
        if m:
            states[m.group(1)] = 'skipped' if m.group(1) == last_skipped else 'done'
            last_skipped = None
    _artist_states[path] = (offset + len(data), states, last_skipped)
    return states


def fetch_info(output_dir, log_path):
    """A round-2 fetch_songs.py run writing to output_dir: its settings (from the command line), the current run's
    progress (the log part after the last '=====' line), the songs downloaded so far (songs.csv) and an ETA."""
    log = (tail_text(log_path, 30_000_000) or '')
    procs = [l.split(None, 1) for l in subprocess.run(['pgrep', '-af', 'data_prep/songs/fetch_songs.py'], capture_output=True,
                                                      text=True).stdout.splitlines()
             if f'--output {output_dir}' in l and 'pgrep' not in l]
    info = {'running': bool(procs), 'pid': procs[0][0] if procs else None, 'paused': False, 'log': log}
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
                n403=run.count('HTTP Error 403'), skipped=run.count('SKIPPED'), aborted='ABORTED' in run,
        failed=run.count('    FAILED'))
    info['eta'] = None
    if procs and run_start and done:
        per_artist = (time.time() - run_start) / done
        info.update(per_artist=per_artist, eta=info['run_left'] * per_artist)
    csv_path = Path(output_dir) / 'songs.csv'
    # unique songs: overlapping runs once wrote rows twice
    info['downloaded'] = len({r['id'] for r in csv.DictReader(open(csv_path, encoding='utf-8'))}) \
        if csv_path.exists() else 0
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
        line1 = f'{src} · not running' + (' · last run ABORTED (blocked or network down?)' if info['aborted'] else '')
    queries = ', '.join(f'"{t.format("<artist>")}"' for t in info['templates'])
    line2 = (f"queries ×{info['aliases']:.1f} names/artist: {queries} · {info['search_size']} results each, "
             f"{info['search_sleep']:g} s between searches, {info['sleep']:g} s between downloads") \
        if info['running'] else ''
    return line1, line2


def round2_artists():
    return [l for l in (ROOT / 'data/artists/covers_next.txt').read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')]


def run_state(info, busy, pending):
    if info['running'] or busy:
        return running_text(info['pid'])
    if pending or info['run_left'] or info['aborted']:
        return Text('stopped', style='red')
    return Text('done', style='green')


def covers_round2_row(t):
    """Piano covers of data/artists/covers_next.txt: download (fetch_covers2.log, appended re-runs), transcribed by
    logs/run_gpu_worker_round2.sh."""
    if not (LOGS / 'fetch_covers2.log').exists():
        return
    out = Path('/data/covers_round2')
    info = fetch_info(str(out), LOGS / 'fetch_covers2.log')
    states = artist_states(LOGS / 'fetch_covers2.log')
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
    """Original songs of data/artists/covers_next.txt: download (fetch_songs2.log, appended re-runs) and piano
    reduction by logs/run_gpu_worker_round2.sh."""
    if not (LOGS / 'fetch_songs2.log').exists():
        return
    songs, red = Path('/data/songs_round2'), Path('/data/finetune_round2')
    info = fetch_info(str(songs), LOGS / 'fetch_songs2.log')
    states = artist_states(LOGS / 'fetch_songs2.log')
    total = len(round2_artists())
    handled = sum(1 for v in states.values() if v == 'done') - (1 if info['running'] else 0)
    redo = sum(1 for v in states.values() if v == 'skipped')
    waiting = len(unreduced_mp3s(songs, red))  # the worker may keep the mp3s (DELETE=0), so not every mp3 is waiting
    line1, line2 = fetch_lines(info)
    line3 = (f"{info['downloaded']} songs downloaded" +
             (f", {info['per_artist_songs']:.0f}/artist this run" if info['per_artist_songs'] else '') +
             f" · this run: {info['failed']} downloads failed, {info['n403']} × 403, {info['skipped']} artists skipped" +
             (f' · {redo} skipped to redo' if redo else ''))
    t.add_row('songs round 2: download', run_state(info, False, 0), ProgressBar(total=total, completed=handled),
              f'{handled}/{total} artists', '\n'.join(l for l in (line1, line2, line3) if l))

    # by song id: a duet is downloaded under both artists and reduced in both folders, but is one song of songs.csv
    reduced = {p.stem.rsplit('[', 1)[-1].rstrip(']'): p.stat().st_mtime for p in red.glob('*/midi/*.mid')}
    # failed.txt also lists broken partial downloads (.m4a, never in songs.csv): only failed mp3s are songs
    failed = len({line.rsplit('[', 1)[-1] for f in red.glob('*/failed.txt')
                  for line in f.read_text(encoding='utf-8').splitlines() if line.endswith('].mp3')}
                 - {k + '].mp3' for k in reduced})
    worker = running('run_gpu_worker_round2.sh')
    recent = [m for m in reduced.values() if time.time() - m < 3600]
    detail = f'{waiting} mp3 waiting' + (f' · {failed} failed' if failed else '')
    if worker and len(recent) >= 2:
        per_song = (max(recent) - min(recent)) / (len(recent) - 1)
        detail = f'{3600 / per_song:.0f} songs/h over the last hour · ' + detail
        # still to come: the songs waiting + the remaining artists at this run's songs per artist
        to_come = waiting + (info['run_left'] * info['per_artist_songs'] if info['per_artist_songs'] else 0)
        eta = max(info['eta'] or 0, to_come * per_song)
        detail += f' · ~{to_come:.0f} songs still to reduce, {eta_text(eta)}'
    state = running_text(worker) if worker else \
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
    """data_prep/songs/redownload.py streams (lost mp3s fetched again by URL): one row per log, its run after the last
    '=====' line."""
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
        alive = running(f'data_prep/songs/redownload.py .*--source {sep[-1].group(3)}')
        completed = max(0, at - failed - (1 if alive and not finished else 0))
        detail = f'{name} · {which} · {failed} failed'
        if alive:
            if run.rstrip().endswith('waiting 5 min'):
                detail += ' · WAITING: disk below the free-space floor'
            elif at > 1:
                per_song = (time.time() - start) / (at - 1)
                detail += f' · {per_song:.0f} s/song, {eta_text((total - at + 1) * per_song)}'
            if which == 'round 2' and sep[-1].group(3) == 'youtube' and \
                    running('data_prep/songs/redownload.py --songs-csv data/audio/songs.csv'):  # the YouTube chain's 2nd run
                detail += ' · then round 1 (queued)'
        elif 'ABORTED' in run:
            detail += ' · ABORTED (downloads blocked?)'
        state = running_text(alive) if alive else \
            Text('done', style='green') if finished else Text('stopped', style='red')
        t.add_row(f're-download: {name}', state, ProgressBar(total=max(1, total), completed=completed),
                  f'{completed}/{total} songs', detail)


def multi_pass_rows(t):
    """Multi-track pass (logs/run_multi_pass.sh): one row per round, songs in multi.csv / failed.txt vs mp3s kept."""
    worker = running('run_multi_pass.sh')
    for name, songs, out, log in (('round 1', Path('/data/songs_round1'), ROOT / 'data/finetune', 'multi_round1.log'),
                                  ('round 2', Path('/data/songs_round2'), Path('/data/finetune_round2'),
                                   'multi_round2.log')):
        if not (LOGS / log).exists():
            continue
        total = len(list(songs.glob('*/*.mp3')))
        done = failed = 0
        for d in out.iterdir() if out.exists() else []:
            if (d / 'multi.csv').exists():
                with open(d / 'multi.csv', newline='', encoding='utf-8') as f:
                    done += sum(1 for r in csv.DictReader(f) if r['source_audio'].startswith(str(songs)))
            if (d / 'failed.txt').exists():
                failed += sum(1 for l in (d / 'failed.txt').read_text(encoding='utf-8').splitlines() if l.strip())
        alive = running(f'audio_to_piano.py --input-root {songs} ')
        recent = sorted(p.stat().st_mtime for p in out.glob('*/multi/*.mid') if time.time() - p.stat().st_mtime < 900)
        detail = f'{failed} failed (all passes)' if failed else ''
        if alive and len(recent) >= 2:
            per_song = (recent[-1] - recent[0]) / (len(recent) - 1)
            detail = '; '.join(x for x in (f'{per_song:.0f} s/song, {eta_text((total - done) * per_song)}', detail) if x)
        state = running_text(alive) if alive else running_text(worker, 'queued') if worker and done < total else \
            Text('done', style='green') if done >= total else Text('stopped', style='red')
        t.add_row(f'multi-track: {name}', state, ProgressBar(total=max(1, total), completed=done),
                  f'{done}/{total} songs', detail)


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
    artists = [line.split('|')[0].strip() for line in open(ROOT / 'data/artists/songs.txt', encoding='utf-8')
               if line.strip() and not line.startswith('#')]
    # round 1 only (data/audio): the round-2 downloads write to /data/... and have their own rows
    round1 = [l.split(None, 1) for l in subprocess.run(['pgrep', '-af', 'data_prep/songs/fetch_songs.py'],
                                                         capture_output=True, text=True).stdout.splitlines()
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
        detail = f"{done_artists}/{len(artists)} artists of data/artists/songs.txt (YouTube)" + \
            (f"; none for: {', '.join(missing[:4])}" if missing else '')
    dl_state = running_text(round1[0][0]) if fetch_running else Text('done', style='green')
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
    red_state = running_text(red_running) if red_running else Text('done', style='green')
    rate = ''
    recent = [p.stat().st_mtime for p in reduced if time.time() - p.stat().st_mtime < 900]
    if len(recent) >= 2:
        per_song = (max(recent) - min(recent)) / (len(recent) - 1)
        rate = f'{per_song:.0f} s/song, ~{pending * per_song / 60:.0f} min left for {pending} downloaded songs'
    elif pending:
        rate = f'{pending} downloaded songs waiting'
    # reduce2.log also receives the round-2 reductions (appended): count round 1 only, from the start to its DONE line
    failed = sum(head_until(LOGS / n, 'REDUCTION DONE').count('FAILED') for n in ('reduce.log', 'reduce2.log'))
    t.add_row('songs round 1: piano reduction', red_state, ProgressBar(total=max(1, len(reduced) + pending), completed=len(reduced)),
              f'{len(reduced)}/{len(reduced) + pending}', '; '.join(x for x in (rate, f'{failed} failed' if failed else '') if x))
    gigamidi_row(t)
    discover_row(t)
    hf_upload_row(t)
    covers_round2_row(t)
    songs_round2_rows(t)
    redownload_rows(t)
    multi_pass_rows(t)
    t.caption = ('kill <pid> stops a stage; a runner script (.sh) with its children: kill -- -<pid>')
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


# long section runs (tools/sweeps/section_long.py): train.py in chunks, all appended to LONG_DIR/<name>.log, each after
# a '===== chunk <date>: <train.py args>' line; the probe after each chunk and the early stop in LONG_DIR/<name>.json
LONG_DIR = LOGS / 'sections'
LONG_DRIVER = ROOT / 'tools/sweeps/section_long.py'
# on the raw bytes, so offsets are byte offsets (progress bars hold multi-byte characters)
LONG_HEADER = re.compile(rb'^===== chunk (\d{4}-\d\d-\d\d \d\d:\d\d): ([^\r\n]*)', re.M)
LONG_EVAL = re.compile(rb'step \d+: train loss [^\r\n]*')
EVAL_HEAD = re.compile(r'step (\d+): train loss ([\d.na]+), val loss [\d.]+ \| (\S+) CE ([\d.]+)')
SEC_GAIN = re.compile(r'section gain ([-+.\d]+) / none ([-+.\d]+)(?: / prev [-+.\d]+)?, late ([-+.\d]+) / ([-+.\d]+)')
LONG_STARTUP = 90  # seconds from a chunk's header to its training loop (data, LaBSE, model): before its first step
_long_logs = {}  # log path -> what long_log_marks has read of it


def log_head(path, n_bytes=4096):
    try:
        with open(path, 'rb') as f:
            return f.read(n_bytes)
    except OSError:
        return b''


def read_bytes(path, start, end):
    with open(path, 'rb') as f:
        f.seek(start)
        return f.read(max(0, end - start))


def as_text(data):
    """Log bytes as text, progress-bar carriage returns as newlines."""
    return data.decode('utf-8', 'ignore').replace('\r', '\n')


def _last_marks(data, base):
    """The last chunk header (byte offset, date, args) and the last evaluation line (byte offset, line) in data, which
    starts at byte base of its file."""
    heads, evals = list(LONG_HEADER.finditer(data)), list(LONG_EVAL.finditer(data))
    return {'header': (base + heads[-1].start(), as_text(heads[-1].group(1)), as_text(heads[-1].group(2)))
            if heads else None,
            'eval': (base + evals[-1].start(), as_text(evals[-1].group(0))) if evals else None}


def long_log_marks(path, block=1 << 18, overlap=4096):
    """The last chunk header and the last evaluation line of a long-run log (see _last_marks), reading only what wasn't
    read before: the first call goes back from the end until it has both (about one chunk), later calls only read
    what was appended since."""
    size = path.stat().st_size
    marks = _long_logs.get(path)
    if marks is None or size < marks['size']:  # first look, or a new log under the same name
        marks = {'size': size, 'header': None, 'eval': None}
        end = size
        while end > 0 and not (marks['header'] and marks['eval']):
            start = max(0, end - block)
            for k, v in _last_marks(read_bytes(path, start, min(size, end + overlap)), start).items():
                marks[k] = marks[k] or v
            end = start
    elif size > marks['size']:
        lo = max(0, marks['size'] - overlap)
        for k, v in _last_marks(read_bytes(path, lo, size), lo).items():
            if v and (marks[k] is None or v[0] > marks[k][0]):
                marks[k] = v
        marks['size'] = size
    _long_logs[path] = marks
    return marks


def long_drivers():
    """{run name: (pid, command line)} of the running section_long.py drivers (--name, else the script's default)."""
    try:
        default = re.search(r"^NAME = '([^']+)'", LONG_DRIVER.read_text(encoding='utf-8'), re.M).group(1)
    except (OSError, AttributeError):
        default = ''
    out = {}
    for line in subprocess.run(['pgrep', '-af', 'tools/sweeps/section_long.py'], capture_output=True,
                               text=True).stdout.splitlines():
        pid, _, cmd = line.partition(' ')
        if 'python' not in (cmd.split() or [''])[0]:  # the shell that started it
            continue
        m = re.search(r'--name[= ](\S+)', cmd)
        out[m.group(1) if m else default] = (pid, cmd)
    return out


def long_stop_rule(cmd=''):
    """(iteration from which the early stop applies, min gain vs shuffled): the driver's arguments, else its defaults."""
    try:
        src = LONG_DRIVER.read_text(encoding='utf-8')
    except OSError:
        src = ''
    out = []
    for flag, cast in (('--stop-at', int), ('--min-gain', float)):
        m = re.search(rf'{flag}[= ](\S+)', cmd) or re.search(rf"'{flag}', type=\w+, default=([\d.]+)", src)
        out.append(cast(m.group(1)) if m else None)
    return out


def hms(s):
    """tqdm's '12:30' / '1:02:30' -> seconds."""
    secs = 0
    for part in s.split(':'):
        secs = secs * 60 + int(part)
    return secs


def long_run_row(name, active, drivers):
    """Cells of one long run (state, progress, chunk, last evaluation) and its probe table (None before the first
    probe)."""
    log, state_path = LONG_DIR / f'{name}.log', LONG_DIR / f'{name}.json'
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):  # no probe yet (or the driver is writing it)
        state = {}
    probes, stopped = state.get('probes', []), state.get('stopped', '')
    marks = long_log_marks(log) if log.exists() else {'size': 0, 'header': None, 'eval': None}
    tail_at = max(0, marks['size'] - 200_000)
    data = read_bytes(log, tail_at, marks['size']) if log.exists() else b''
    header = marks['header']
    tail = as_text(data)
    chunk = as_text(data[header[0] - tail_at:]) if header and header[0] >= tail_at else tail  # this chunk's part
    bars, bars_now = TQDM.findall(tail), TQDM.findall(chunk)
    it, total = (int(bars[-1][1]), int(bars[-1][2])) if bars else (0, 0)
    saved = re.findall(r'saved resumable checkpoint at iter (\d+)', tail)
    if saved and not bars_now:  # between chunks: the next chunk starts at the saved iteration
        it = int(saved[-1])
    spi = None  # seconds per iteration in this chunk
    if bars_now and bars_now[-1][5] != '?':
        rate = float(bars_now[-1][5])
        spi = rate if bars_now[-1][6] == 's/it' else 1 / rate if rate else None

    pid = active.get(str(log))
    probe_pid = running(f'section_gain.py .*section_gain/{name}/latest')
    driver = drivers.get(name)
    if pid:
        st = running_text(pid, 'training')
        idle = time.time() - log.stat().st_mtime
        if idle > 300:
            st.append(f'\nlog idle {fmt_secs(idle)}', style='bold red')
    elif probe_pid:
        st = running_text(probe_pid, 'probing')
    elif 'Traceback' in chunk[-5000:] or 'out of memory' in chunk[-5000:]:
        st = Text('CRASHED', style='bold red')
    elif stopped:
        st = Text('early stop', style='bold red')
    elif total and it >= total:
        st = Text('done', style='green')
    elif driver:
        st = running_text(driver[0], 'driver')
    else:
        st = Text('paused', style='dim')

    args = header[2] if header else ''
    max_min = float(m.group(1)) if (m := re.search(r'--max_minutes=([\d.]+)', args)) else 0.0
    duty = float(m.group(1)) if (m := re.search(r'--gpu_duty=([\d.]+)', args)) else 1.0
    lines = []
    if header:
        lines.append(f'{"chunk" if pid else "last chunk"} from {header[1][11:]}' +
                     (f', {max_min:g} min' if max_min else '') + (f', GPU duty {duty:g}' if duty < 1 else ''))
    if pid and max_min and header:
        # the time limit counts from the training loop's start: the first step's elapsed time in this chunk, else the
        # header plus the usual startup
        loop_start = log.stat().st_mtime - hms(bars_now[-1][3]) if bars_now else \
            time.mktime(time.strptime(header[1], '%Y-%m-%d %H:%M')) + LONG_STARTUP
        end = loop_start + 60 * max_min
        left = max(0.0, end - time.time())
        lines.append(f'ends ~{time.strftime("%H:%M", time.localtime(end))} (in {fmt_secs(left)})')
        if spi:
            lines.append(f'≈ it {min(total, it + int(left / spi)):,} by then, then a probe')
    elif not pid and saved:
        lines.append(f'ckpt.pt at it {int(saved[-1]):,}')
    if spi and total and it < total:
        lines.append(f'{(total - it) * spi / 3600:.1f} h of training left at this speed')

    ev = marks['eval'][1] if marks['eval'] else ''
    m, g = EVAL_HEAD.search(ev), SEC_GAIN.search(ev)
    last_eval = (f'step {int(m.group(1)):,}: {m.group(3)} CE {m.group(4)}' if m else '') + \
        (f'\nsection gain {g.group(1)} / none {g.group(2)}\nlate {g.group(3)} / none {g.group(4)}' if g else '')
    cells = (name, st, ProgressBar(total=max(1, total), completed=min(it, total)),
             f'{it:,}/{total:,}\n{it / total:.0%}' if total else f'{it:,}/?', f'{spi:.2f} s/it' if spi else '',
             '\n'.join(lines), last_eval)

    if not probes and not stopped:
        return cells, None
    stop_at, min_gain = long_stop_rule(driver[1] if driver else '')
    p = Table(title=f'{name}: gain of the section captions on the first 32 notes after a section change of >= 2 '
                    'points (nats per note)' +
                    (f' · early stop if vs shuffled < {min_gain:g} from it {stop_at:,}' if None not in (stop_at, min_gain)
                     else ''), expand=True, title_justify='left')
    for col in ('weights @it', 'probed', 'vs shuffled', 'vs none', 'no-change vs none', 'all notes vs none',
                'excess CE'):
        p.add_column(col, justify='right')
    for r in probes:
        style = '' if None in (stop_at, min_gain) else 'green' if r['gain_shuffled'] >= min_gain else \
            'red' if r['iter'] >= stop_at else ''
        p.add_row(f"{r['iter']:,}", r['time'][5:], Text(f"{r['gain_shuffled']:+.4f}", style=style),
                  f"{r['gain_none']:+.4f}", f"{r['gain_none_nochange']:+.4f}", f"{r['all_none']:+.4f}",
                  f"{r['excess']:+.3f}")
    if stopped:
        p.caption, p.caption_style = f'stopped: {stopped} (section_long.py --force trains on)', 'bold red'
    return cells, p


def long_runs_view():
    """The long section runs: one row each (state, progress, the chunk in progress and its expected end, the last
    evaluation) and their probe tables; None if there are none. Runs: logs with chunk headers in LONG_DIR, and
    LONG_DIR/long_*.json states."""
    names = sorted({p.stem for p in LONG_DIR.glob('*.log') if b'===== chunk ' in log_head(p)} |
                   {p.stem for p in LONG_DIR.glob('long_*.json')})
    if not names:
        return None
    t = Table(title='Long section runs (tools/sweeps/section_long.py)', expand=True, title_justify='left')
    for col, kw in [('run', {}), ('state', {}), ('progress', {'ratio': 1}), ('step', {'justify': 'right'}),
                    ('speed', {'justify': 'right'}), ('chunk', {'ratio': 2}), ('last evaluation', {'ratio': 2})]:
        t.add_column(col, **kw)
    active = {str(path): pid for pid, path, _ in active_train_logs().values()}
    drivers = long_drivers()
    probe_tables, unprobed = [], []
    for name in names:
        cells, probes = long_run_row(name, active, drivers)
        t.add_row(*cells)
        if probes is None:
            unprobed.append(name)
        else:
            probe_tables.append(probes)
    if unprobed:
        t.caption = f'no probe yet: {", ".join(unprobed)} (the driver probes after each chunk)'
    return Group(t, *probe_tables)


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
    long_runs = long_runs_view()
    return Group(Text(time.strftime('%H:%M:%S'), style='dim'), system_panel(), *([long_runs] if long_runs else []),
                 pipeline_table(), training_table(None if show_all else 4), jobs_table(keep=6 if show_all else 2))


def tailscale_urls(port):
    """https addresses that tailscale serve forwards to this port (none without tailscale or a serve config)."""
    try:
        out = subprocess.run(['tailscale', 'serve', 'status', '--json'], capture_output=True, text=True, timeout=5)
        web = (json.loads(out.stdout or '{}') or {}).get('Web') or {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    urls = []
    for hostport, conf in web.items():
        host, _, p = hostport.rpartition(':')
        for path, handler in (conf.get('Handlers') or {}).items():
            if handler.get('Proxy', '').rstrip('/').endswith(f':{port}'):
                urls.append(f'https://{host}' + ('' if p == '443' else f':{p}') + path.rstrip('/'))
    return urls


def serve(port, interval, show_all, token, host='0.0.0.0'):
    """The dashboard as a self-refreshing HTML page (rich's own HTML export of the same view) on host:port (0.0.0.0:
    every address of this machine), the to-do page (tools/todo_web.py) and the CI reports (tools/ci_web.py); requests
    without ?t=<token> or the cookie it sets get 403 (the cookie keeps the CI pages' relative links working)."""
    import io
    import threading
    from http.cookies import SimpleCookie
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit
    from rich.console import Console
    import ci_web
    import todo_web
    page = [b'starting']

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, ctype, body, headers=()):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            if parse_qs(urlsplit(self.path).query).get('t') == [token]:
                self.send_header('Set-Cookie', f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000')
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            cookie = SimpleCookie(self.headers.get('Cookie') or '').get(COOKIE)
            if parse_qs(urlsplit(self.path).query).get('t') == [token] or (cookie and cookie.value == token):
                return True
            self.reply(403, 'text/plain', b'forbidden')
            return False

        def do_GET(self):
            if not self.allowed():
                return
            if urlsplit(self.path).path == '/ci':
                query = urlsplit(self.path).query
                self.reply(301, 'text/plain', b'', [('Location', '/ci/' + (f'?{query}' if query else ''))])
                return
            self.reply(*(todo_web.handle('GET', self.path) or ci_web.handle('GET', self.path)
                         or (200, 'text/html; charset=utf-8', page[0])))

        def do_POST(self):
            if self.allowed():
                body = self.rfile.read(min(int(self.headers.get('Content-Length') or 0), 1 << 20))
                self.reply(*(todo_web.handle('POST', self.path, body) or (404, 'text/plain', b'not found')))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ips = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.split() if host == '0.0.0.0' else [host]
    urls = tailscale_urls(port) + [f'http://{ip}:{port}' for ip in ips]
    print('dashboard on ' + ', '.join(f'{u}/?t={token}' for u in urls), flush=True)
    head = (f'<meta http-equiv="refresh" content="{max(5, int(interval))}">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">')
    link = (f'<body><p style="font:15px system-ui,sans-serif;margin:8px"><a href="/todo?t={token}" '
            f'style="color:#60a5fa">Project to-do &rarr;</a> &nbsp; <a href="/ci/?t={token}" '
            'style="color:#60a5fa">CI runs &rarr;</a></p>')
    while True:
        console = Console(record=True, width=140, file=io.StringIO(), color_system='truecolor')
        console.print(render(show_all))
        html = console.export_html(inline_styles=True).replace('<head>', '<head>' + head, 1)
        page[0] = html.replace('<body>', link, 1).encode()
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--web', type=int, default=0, metavar='PORT', help='serve the view as a web page instead')
    parser.add_argument('--token', default='', help='access token of the web page (default: a random one)')
    parser.add_argument('--token-file', default='', help='read the token from this file (created with a random one '
                        'if missing): the link stays the same across restarts')
    parser.add_argument('--host', default='0.0.0.0', help='address of the web page (127.0.0.1: this machine only, '
                        'e.g. behind tailscale serve)')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--all', action='store_true', help='also show old finished runs and sampling jobs')
    parser.add_argument('--long-dir', default='', help='directory of the long section runs (default logs/sections)')
    args = parser.parse_args()
    if args.long_dir:
        global LONG_DIR
        LONG_DIR = Path(args.long_dir).resolve()
    if args.web:
        import secrets
        token = args.token
        if args.token_file:
            path = Path(args.token_file).expanduser()
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(secrets.token_urlsafe(12) + '\n')
                path.chmod(0o600)
            token = path.read_text().strip()
        serve(args.web, max(args.interval, 10), args.all, token or secrets.token_urlsafe(8), args.host)
        return
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
