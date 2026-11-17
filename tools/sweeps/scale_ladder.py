"""
Scaling ladder: model size x notes seen with long_450m's recipe on the laptop stores (config/pretrain/scale_ladder.py).

    .venv/bin/python tools/sweeps/scale_ladder.py            # anchors, then every arm still missing
    .venv/bin/python tools/sweeps/scale_ladder.py --list     # arms and what is done
    .venv/bin/python tools/sweeps/scale_ladder.py --only 20M # arms whose name contains the text

Each arm is one train.py run (its own WSD cooldown); its last evaluation goes to results/scale_ladder/runs.json and
its checkpoint is deleted. Anchors = released checkpoints scored on the same held-out windows (eval only). Budgets run
in rising order, so the file is usable before the queue ends. The fit is tools/sweeps/scale_fit.py.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/scale_ladder'
LOGS = ROOT / 'logs/scale_ladder'
RUNS = OUT / 'runs.json'
CONFIG = 'config/pretrain/scale_ladder.py'
NOTES_PER_STEP = 4 * 4 * 2048

SIZES = {  # head size 64, the block of long_450m
    '6M': dict(n_layer=6, n_embd=256, n_head=4),
    '20M': dict(n_layer=10, n_embd=384, n_head=6),
    '42M': dict(n_layer=12, n_embd=512, n_head=8),
    '108M': dict(n_layer=14, n_embd=768, n_head=12),
}
BUDGETS = {'40M': 40e6, '80M': 80e6, '160M': 160e6}  # notes seen; the store has 167.9M
ANCHORS = {  # name: (configs, checkpoint, notes seen in its own run)
    'musicar-L': (['config/pretrain/long_450m.py'], 'checkpoints/long_450m_final/model_bf16.pt', 126582 * 65536),
}
EVAL = ['--val_csv_path=store:data/cache/gigamidi_heldout', '--val2_csv_path=', '--val_extra=', '--eval_iters=300',
        '--batch_size=4', '--style_eval_songs=0', '--wandb_log=False']


def load():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def save(name, rec):
    runs = load()
    runs[name] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RUNS.with_suffix('.tmp')
    tmp.write_text(json.dumps(runs, indent=1))
    tmp.replace(RUNS)


def parse(log):
    """Every 'step N:' evaluation line of a train.py log -> [dict(step, ce, heads, prog)], and the parameter count."""
    text = log.read_text(errors='replace')
    evals = []
    for m in re.finditer(r'step (\d+): train loss \S+ val loss [\d.]+ \| \S+ CE ([\d.]+) \(([^)]*)\)[^\n\r]*', text):
        heads = {k: float(v) for k, v in re.findall(r'(pit|vel|dur|del) ([\d.]+)', m.group(3))}
        prog = re.search(r'prog CE ([\d.]+)', m.group(0))
        evals.append(dict(step=int(m.group(1)), ce=float(m.group(2)), heads=heads,
                          prog=float(prog.group(1)) if prog else None))
    params = re.search(r'Model parameters : ([\d,]+)', text)
    return evals, int(params.group(1).replace(',', '')) if params else None


def sh(cmd, log):
    LOGS.mkdir(parents=True, exist_ok=True)
    print(time.strftime('%H:%M'), log.name, flush=True)
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f'failed (exit {code}), see {log}')


def run_anchor(name):
    configs, ckpt, notes = ANCHORS[name]
    log = LOGS / f'anchor_{name}.log'
    sh(['.venv/bin/python', 'train.py', *configs, '--csv_path=store:data/cache/gigamidi_replay', '--source_weights=',
        '--init_from=finetune', f'--init_ckpt={ckpt}', '--eval_only=True', '--act_ckpt=0', '--compile=False', *EVAL,
        '--out_dir=logs/scale_ladder/tmp_out'], log)
    evals, params = parse(log)
    save(name, dict(kind='anchor', params=params, notes=notes, **evals[0]))


def run_arm(size, budget):
    name = f'{size}_{budget}'
    iters = round(BUDGETS[budget] / NOTES_PER_STEP)
    out = ROOT / 'checkpoints' / f'scale_{name}'
    log = LOGS / f'{name}.log'
    t0 = time.time()
    sh(['.venv/bin/python', 'train.py', CONFIG, *[f'--{k}={v}' for k, v in SIZES[size].items()],
        f'--max_iters={iters}', f'--lr_decay_iters={iters}', f'--eval_interval={max(iters // 4, 1)}',
        f'--out_dir={out.relative_to(ROOT)}'], log)
    evals, params = parse(log)
    last = evals[-1]
    assert last['step'] >= iters - 1, f'{name}: last evaluation at step {last["step"]} of {iters}'
    save(name, dict(kind='arm', size=size, budget=budget, params=params, notes=iters * NOTES_PER_STEP, iters=iters,
                    minutes=round((time.time() - t0) / 60, 1), curve=[(e['step'], e['ce']) for e in evals], **last))
    shutil.rmtree(out, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', action='store_true')
    p.add_argument('--only', default='')
    a = p.parse_args()
    arms = [(s, b) for b in BUDGETS for s in SIZES]
    done = load()
    if a.list:
        for name in [*ANCHORS, *[f'{s}_{b}' for s, b in arms]]:
            r = done.get(name)
            print(f'{name:14s}', f'CE {r["ce"]:.3f}  params {r["params"] / 1e6:.1f}M  notes {r["notes"] / 1e6:.0f}M'
                  if r else '-')
        return
    for name in ANCHORS:
        if name not in done and a.only in name:
            run_anchor(name)
    for size, budget in arms:
        if f'{size}_{budget}' not in done and a.only in f'{size}_{budget}':
            run_arm(size, budget)


if __name__ == '__main__':
    main()
