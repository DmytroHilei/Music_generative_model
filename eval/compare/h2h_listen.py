"""
Listening set from the head-to-head files, and the tally of the blind test's answers.

    .venv/bin/python eval/compare/h2h_listen.py clips                 # web/public/compare/clips/*.mp3 + clips.json
    .venv/bin/python eval/compare/h2h_listen.py tally answers.txt     # one result code per line (or issue text)

clips: prompts are drawn with a fixed seed from the 96 continuation and the 96 piano prompts (no listening, no
scores), so the set is a sample of the scored files and not a selection. Each clip = the last PROMPT_S seconds of the
prompt + CONT_S seconds of the system's continuation, FluidR3, loudness-normalised, with a short fade out.
tally: per pair of systems, wins / losses / no preference over all listeners and a 95% Wilson interval of the win
share among the decided answers.
"""
import argparse
import base64
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results/head2head'
DST = ROOT / 'web/public/compare'
SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
SEED = 20261117
PROMPT_S, CONT_S, FADE_S = 8.0, 30.0, 2.0
TASKS = {  # task: (number of prompts, systems in page order)
    'cont96': (8, ['real', 'long_450m', 'amt', 'skytnt_medium']),
    'piano96': (4, ['real', 'long_450m', 'aria']),
}
NAMES = {'real': 'real continuation', 'long_450m': 'musicar-L', 'amt': 'AMT large', 'skytnt_medium': 'SkyTNT medium',
         'aria': 'Aria medium'}
PAIRS = {  # blind test: musicar-L against each other system of the task
    'cont96': [('long_450m', 'amt'), ('long_450m', 'skytnt_medium'), ('long_450m', 'real')],
    'piano96': [('long_450m', 'aria'), ('long_450m', 'real')],
}


def render(mid, mp3, start, seconds):
    """fluidsynth -> raw PCM (stopped at the clip's end: its fast render once never stopped) -> cut, fade, MP3."""
    raw = mp3.with_suffix('.raw')
    rate = 44100
    limit = int((start + seconds + 1.0) * rate * 4)
    proc = subprocess.Popen(['fluidsynth', '-ni', '-r', str(rate), '-O', 's16', '-T', 'raw', '-F', str(raw), SOUNDFONT,
                             str(mid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if raw.exists() and raw.stat().st_size >= limit:
            proc.kill()
            break
        time.sleep(0.05)
    proc.wait()
    have = raw.stat().st_size / (rate * 4) - start
    seconds = max(min(seconds, have), 1.0)
    subprocess.run(['ffmpeg', '-y', '-f', 's16le', '-ar', str(rate), '-ac', '2', '-i', str(raw), '-ss', f'{start:.2f}',
                    '-t', f'{seconds:.2f}', '-af',
                    f'loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=out:st={seconds - FADE_S:.2f}:d={FADE_S}',
                    '-ar', str(rate), '-b:a', '64k', str(mp3)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw.unlink()
    return round(seconds, 2)


def clips(_):
    (DST / 'clips').mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    out = dict(seed=SEED, prompt_s=PROMPT_S, cont_s=CONT_S, names=NAMES, pairs=PAIRS, items=[])
    for task, (n, systems) in TASKS.items():
        prompts = json.loads((SRC / task / 'prompts.json').read_text())['prompts']
        for p in sorted(rng.sample(prompts, n), key=lambda p: p['id']):
            start = max(p['seconds'] - PROMPT_S, 0.0)
            item = dict(id=f'{task}_{p["id"]}', task=task, prompt=p['id'], prompt_s=round(p['seconds'] - start, 2),
                        instruments=p['instruments'], clips={})
            for s in systems:
                mid = SRC / task / s / f'{p["id"]}_0.mid'
                if not mid.exists():
                    mid = SRC / task / s / f'{p["id"]}.mid'
                end = pretty_midi.PrettyMIDI(str(mid)).get_end_time()
                # opaque names: the blind test's page must not show the system in the file name
                name = hashlib.sha1(f'{SEED}_{task}_{p["id"]}_{s}'.encode()).hexdigest()[:12] + '.mp3'
                seconds = render(mid, DST / 'clips' / name, start, min(PROMPT_S + CONT_S, p['seconds'] - start + CONT_S))
                item['clips'][s] = dict(file=f'clips/{name}', seconds=seconds, ended=bool(end < p['seconds'] + CONT_S))
                print(task, p['id'], s, name, seconds, flush=True)
            out['items'].append(item)
    (DST / 'clips.json').write_text(json.dumps(out, indent=1))


def wilson(k, n, z=1.96):
    if not n:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def tally(a):
    """Result codes: base64 JSON {v, listener, answers: [[item, system_a, system_b, choice]]}, choice a | b | none."""
    text = Path(a.file).read_text()
    counts = defaultdict(lambda: [0, 0, 0])  # (task, ours, other): wins, losses, none
    listeners = set()
    for code in re.findall(r'MUSICAR1:([A-Za-z0-9+/=_-]+)', text):
        rec = json.loads(base64.urlsafe_b64decode(code + '=' * (-len(code) % 4)))
        if rec['listener'] in listeners:
            continue
        listeners.add(rec['listener'])
        for item, sa, sb, choice in rec['answers']:
            task = item.rsplit('_', 1)[0]
            ours, other = ('long_450m', sb) if sa == 'long_450m' else ('long_450m', sa)
            picked = None if choice == 'none' else (sa if choice == 'a' else sb)
            counts[(task, ours, other)][2 if picked is None else 0 if picked == ours else 1] += 1
    print(f'{len(listeners)} listeners\n')
    print('| Task | musicar-L vs | Preferred musicar-L | Preferred the other | No preference | Win share (95%) |')
    print('|---|---|---|---|---|---|')
    for (task, _, other), (w, l, n) in sorted(counts.items()):
        lo, hi = wilson(w, w + l)
        share = f'{w / (w + l):.2f} ({lo:.2f}–{hi:.2f})' if w + l else '-'
        print(f'| {task} | {NAMES[other]} | {w} | {l} | {n} | {share} |')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('clips').set_defaults(fn=clips)
    t = sub.add_parser('tally')
    t.add_argument('file')
    t.set_defaults(fn=tally)
    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == '__main__':
    main()
