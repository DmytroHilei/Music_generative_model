"""
Quality of the two final 450M models under the Pi's planned quantization (context/claude/inference_pi.md §4 #1),
simulated with fake quantization (musicar/model/quant.py): weights / activations / KV cache rounded to what int8 /
int4 kernels hold.

    CE   val CE in float32 (train.py --eval_only --fake_quant=<spec>): piano on covers v4 val (+ covers v2), band on
         ukrainian_multi_v1 val (+ gigamidi_heldout), for every spec in CE_SPECS
    gen  generation in bf16 with the quantized weights, judged like the models were chosen: piano = seed_compare on the
         27 Ukrainian val covers at T=0.9,dt=0.1,dens=prompt, seeds 1-4 (adamw_sweep); band = multi_generate on 24
         multi_v1 val songs, seeds 1-3 (multi_sweep), for every spec in GEN_SPECS ('none' = the unquantized bf16
         baseline, rerun here after the drum-duration fix)

Every result is a file under logs/quant/; a re-run skips what exists. Refuses on battery.

    .venv/bin/python tools/sweeps/quant_check.py --dry-run
    .venv/bin/python tools/sweeps/quant_check.py                  # CE then gen, ~1 h
    .venv/bin/python tools/sweeps/quant_check.py --parts gen --models band
    .venv/bin/python tools/sweeps/quant_check.py --max-minutes 5   # a chunk: no new run after 5 min (runs are 1-3 min)
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.jobstatus import JobStatus  # noqa: E402

OUT = ROOT / 'logs/quant'
MODELS = {
    'piano': dict(ckpt='checkpoints/overfit_adamw8_1e-4/model_bf16_it349.pt',
                  configs=['config/finetune/finetune_ua450.py', 'config/finetune/finetune_ua450_full.py'],
                  seeds=(1, 2, 3, 4)),
    'band': dict(ckpt='checkpoints/multi_control/model_bf16_it1999.pt',
                 configs=['config/finetune/finetune_ua450.py', 'config/finetune/finetune_ua450_multi.py'],
                 seeds=(1, 2, 3)),
}
CE_SPECS = ['none', 'w8', 'w8-a8', 'w8-a8-kv8', 'w4g32', 'w4g32-a8', 'w4g32-a8-kv8', 'w4g128-a8-kv8', 'w4-a8-kv8']
GEN_SPECS = ['none', 'w8-a8-kv8', 'w4g32-a8-kv8']
SETTING = 'T=0.9,dt=0.1,dens=prompt'      # adamw_sweep's piano setting
MULTI_CSV = 'data/finetune/ukrainian_multi_v1.csv'
CE_MIN, GEN_MIN = {'piano': 3, 'band': 3}, {'piano': 3, 'band': 1.5}  # rough minutes per CE run / per seed


def on_battery():
    try:
        return any(p.read_text().strip() == '0' for p in Path('/sys/class/power_supply').glob('A*/online'))
    except OSError:
        return False


def sh(cmd, log):
    print(f'{time.strftime("%H:%M")}   $ {" ".join(cmd)}', flush=True)
    with open(log, 'w') as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f'failed (exit {code}), see {log}')


def ce_file(model, spec):
    return OUT / f'ce_{model}_{spec}.json'


def gen_file(model, spec, seed):
    return OUT / f'gen_{model}_{spec}_s{seed}.json'


def run_ce(model, spec):
    m = MODELS[model]
    log = OUT / f'ce_{model}_{spec}.log'
    sh(['.venv/bin/python', 'train.py', *m['configs'], '--init_from=finetune', f'--init_ckpt={m["ckpt"]}',
        '--eval_only=True', f'--fake_quant={"" if spec == "none" else spec}', '--dtype=float32', '--fp8=False',
        '--compile=False', '--wandb_log=False', '--out_dir=logs/quant/tmp_out'], log)
    line = next(ln for ln in log.read_text().splitlines() if ln.startswith('step 0:'))
    ces = dict(re.findall(r'\| (\S+) CE ([\d.]+)', line))
    heads = dict(re.findall(r'(pit|vel|dur|del) ([\d.]+)', line.split('(', 1)[1].split(')')[0]))
    prog = re.search(r'prog CE ([\d.]+)', line)
    res = dict(ce={k: float(v) for k, v in ces.items() if k != 'prog'}, heads={k: float(v) for k, v in heads.items()},
               prog=float(prog.group(1)) if prog else None, line=line)
    ce_file(model, spec).write_text(json.dumps(res, indent=1))


def run_gen(model, spec, seed):
    m = MODELS[model]
    out = gen_file(model, spec, seed)
    log = OUT / f'gen_{model}_{spec}_s{seed}.log'
    quant = [] if spec == 'none' else ['--quant', spec]
    if model == 'piano':
        sh(['.venv/bin/python', 'eval/compare/seed_compare.py', '--batch-size', '14', '--checkpoints', m['ckpt'],
            '--settings', SETTING, '--seed', str(seed), '--json', str(out.with_suffix('.raw.json')), *quant], log)
        rec = json.loads(out.with_suffix('.raw.json').read_text())
        rec = rec[0] if isinstance(rec, list) else rec
        out.write_text(json.dumps(dict(dist=rec['dist'], record=rec), indent=1))
    else:
        d = ROOT / 'samples/quant' / f'band_{spec}_s{seed}'
        sh(['.venv/bin/python', 'eval/compare/multi_generate.py', '--csv', MULTI_CSV, '--songs', '24', '--checkpoint',
            m['ckpt'], '--seed', str(seed), '--min-new', '1000', '--no-mp3', '--out-dir', str(d.relative_to(ROOT)),
            *quant], log)
        g = next(iter(json.loads((d / 'scores.json').read_text())['groups'].values()))
        out.write_text(json.dumps(dict(dist=g['mean_dist'], cons=g.get('mean_cons')), indent=1))


def summary():
    lines = [f'Fake-quantized 450M models, {time.strftime("%Y-%m-%d %H:%M")} (musicar/model/quant.py; tools/sweeps/'
             f'quant_check.py). CE in float32; generation in bf16 with the rounded weights.', '']
    for model in MODELS:
        base = json.loads(ce_file(model, 'none').read_text()) if ce_file(model, 'none').exists() else None
        lines.append(f'{model}: val CE (Δ vs unquantized)')
        for spec in CE_SPECS:
            if not ce_file(model, spec).exists():
                continue
            r = json.loads(ce_file(model, spec).read_text())
            cells = []
            for name, v in r['ce'].items():
                d = v - base['ce'][name] if base and name in base['ce'] else float('nan')
                cells.append(f'{name} {v:.3f} ({d:+.3f})')
            lines.append(f'  {spec:16s} ' + ' | '.join(cells))
        lines.append(f'{model}: generation distance to the real continuations, mean over seeds {MODELS[model]["seeds"]}'
                     f' (lower = closer)')
        for spec in GEN_SPECS:
            ds = [json.loads(gen_file(model, spec, s).read_text())['dist'] for s in MODELS[model]['seeds']
                  if gen_file(model, spec, s).exists()]
            if ds:
                sd = statistics.stdev(ds) if len(ds) > 1 else float('nan')
                lines.append(f'  {spec:16s} {statistics.mean(ds):.3f} (seeds: {" ".join(f"{d:.3f}" for d in ds)}; '
                             f'sd {sd:.3f})')
        lines.append('')
    (OUT / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--parts', default='ce,gen')
    ap.add_argument('--models', default='piano,band')
    ap.add_argument('--max-minutes', type=float, default=0, help='start no new run after this many minutes (0 = all)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--summary', action='store_true', help='only rewrite logs/quant/summary.txt')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summary:
        return summary()
    parts, models = args.parts.split(','), args.models.split(',')
    todo = []
    for model in models:
        if 'ce' in parts:
            todo += [('ce', model, spec, None, CE_MIN[model]) for spec in CE_SPECS if not ce_file(model, spec).exists()]
        if 'gen' in parts:
            todo += [('gen', model, spec, seed, GEN_MIN[model]) for spec in GEN_SPECS for seed in MODELS[model]['seeds']
                     if not gen_file(model, spec, seed).exists()]
    total = sum(t[-1] for t in todo)
    print(f'{len(todo)} runs, ~{total:.0f} min')
    if args.dry_run:
        for t in todo:
            print('  ', *t[:4])
        return
    if on_battery():
        sys.exit('on battery: plug in first')
    job = JobStatus('quant_check', len(todo), f'{len(todo)} runs, ~{total:.0f} min')
    t0 = time.time()
    for i, (part, model, spec, seed, _) in enumerate(todo):
        if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
            print(f'--max-minutes {args.max_minutes:g} reached after {i} runs; run again to continue')
            break
        job.update(i, f'{part} {model} {spec}' + (f' seed {seed}' if seed else ''), force=True)
        run_ce(model, spec) if part == 'ce' else run_gen(model, spec, seed)
        summary()
    job.finish(f'{OUT}/summary.txt')
    summary()


if __name__ == '__main__':
    main()
