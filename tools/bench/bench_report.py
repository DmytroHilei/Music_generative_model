"""
Markdown report of bench.py --out files: best configuration per shape, FP8 vs BF16, peak memory under the cap, the
speed factor against a reference GPU on the same configuration, and the hours of the compute-request runs at the
measured rate (cloud/run/h100_bench.sh writes the inputs).

    .venv/bin/python tools/bench/bench_report.py logs/h100/bench.jsonl --ref logs/bench_5090.jsonl > logs/h100/bench.md

Hours: notes / notes-per-second of the best FP8 configuration / multi-GPU efficiency (default 0.85); node hours =
GPU hours / 4. The per-GPU rate is taken at tokens_per_step 16,384 when measured (each of 4 data-parallel GPUs then
does a quarter of a 65,536-note step and the whole Muon update), else at 65,536.
"""
import argparse
import json
from pathlib import Path

RUNS = {  # shape: (run, notes low, notes high) - the request's XL / XXL rows (outline section 6.2)
    '28x1664': ('1B (primary)', 20e9, 27e9), '22x1920': ('1B (wide)', 20e9, 27e9),
    '36x2048': ('1.8B (primary)', 35e9, 49e9), '28x2304': ('1.8B (wide)', 35e9, 49e9),
}
REF_SHAPE = '28x1152'  # musicar-L
TRAIN_5090 = 56_000    # musicar-L's measured training rate on the RTX 5090 (8.3B notes / 41 h)


def load(paths):
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def shape(r):
    return f"{r['n_layer']}x{r['n_embd']}"


def best(recs, **want):
    ok = [r for r in recs if 'tok_s' in r and all(r.get(k) == v for k, v in want.items())]
    return max(ok, key=lambda r: r['tok_s']) if ok else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--ref', help='bench.py --out file of the reference GPU (same musicar-L configuration)')
    p.add_argument('--eff', type=float, default=0.85, help='multi-GPU efficiency for the run hours')
    a = p.parse_args()
    recs = load(a.files)
    dev = recs[0]['device'] if recs else '?'
    print(f'# Benchmark: {dev}\n')
    print(f"Memory cap {recs[0].get('mem_cap_gb') or 'none'} GB. tok/s = notes per second on one GPU, synthetic "
          'batches, bench.py (forward + backward + optimizer).\n')
    print('| shape | params | tokens/step | fp8 | micro | act ckpt | notes/s | ms/step | peak GB | reserved GB |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in recs:
        if 'tok_s' in r:
            n = r['params']
            print(f"| {shape(r)} | {f'{n / 1e9:.2f}B' if n >= 1e8 else f'{n / 1e6:.1f}M'} | {r['tokens_per_step']:,} | {r['fp8']} | {r['micro']} | "
                  f"{r['act_ckpt']} | {r['tok_s']:,.0f} | {r['ms_step']:,.0f} | {r['peak_gb']:.1f} | "
                  f"{r['reserved_gb']:.1f} |")
    failed = [r for r in recs if 'error' in r]
    if failed:
        print('\nFailed (OOM under the cap, or other):\n')
        for r in failed:
            print(f"- {shape(r)} fp8 {r['fp8']} micro {r['micro']} ckpt {r['act_ckpt']} tokens/step "
                  f"{r['tokens_per_step']:,}: {r['error']}")

    print('\n## FP8 vs BF16 (best of each, 65,536 notes/step)\n')
    print('| shape | BF16 notes/s | FP8 notes/s | FP8 gain |\n|---|---|---|---|')
    for s in sorted({shape(r) for r in recs}):
        b16 = best([r for r in recs if shape(r) == s], fp8=0, tokens_per_step=65536)
        b8 = best([r for r in recs if shape(r) == s], fp8=1, tokens_per_step=65536)
        if b16 and b8:
            print(f"| {s} | {b16['tok_s']:,.0f} | {b8['tok_s']:,.0f} | {b8['tok_s'] / b16['tok_s'] - 1:+.0%} |")

    ref_h = best([r for r in recs if shape(r) == REF_SHAPE], fp8=1, tokens_per_step=65536)
    print('\n## Speed against the RTX 5090 (musicar-L, 28x1152, FP8, 65,536 notes/step)\n')
    if ref_h:
        line = f"- this GPU: {ref_h['tok_s']:,.0f} notes/s (micro {ref_h['micro']})"
        if a.ref:
            ref_5 = best([r for r in load([a.ref]) if shape(r) == REF_SHAPE], fp8=1, tokens_per_step=65536)
            if ref_5:
                line += f"; RTX 5090 bench {ref_5['tok_s']:,.0f} (micro {ref_5['micro']}): factor " \
                        f"{ref_h['tok_s'] / ref_5['tok_s']:.2f}"
        line += f"; against the 5090's training rate {TRAIN_5090:,}: {ref_h['tok_s'] / TRAIN_5090:.2f}"
        print(line + ' (the request assumed 1.0-1.3)')
    else:
        print('- no musicar-L FP8 measurement')

    print(f'\n## Hours of the request runs at the measured rate (efficiency {a.eff:.0%})\n')
    print('| run | shape | per-GPU notes/s (tokens/step) | H100 hours | node hours | days on one node | fits the cap |')
    print('|---|---|---|---|---|---|---|')
    for s, (run, lo, hi) in RUNS.items():
        rs = [r for r in recs if shape(r) == s]
        b = best(rs, fp8=1, tokens_per_step=16384) or best(rs, fp8=1, tokens_per_step=65536)
        if not b:
            print(f'| {run} | {s} | not measured | | | | {"OOM" if any(shape(r) == s for r in failed) else "-"} |')
            continue
        h_lo, h_hi = (n / b['tok_s'] / 3600 / a.eff for n in (lo, hi))
        print(f"| {run} | {s} | {b['tok_s']:,.0f} ({b['tokens_per_step']:,}) | {h_lo:,.0f}-{h_hi:,.0f} | "
              f"{h_lo / 4:,.0f}-{h_hi / 4:,.0f} | {h_lo / 4 / 24:.1f}-{h_hi / 4 / 24:.1f} | "
              f"yes (peak {b['peak_gb']:.0f} GB) |")


if __name__ == '__main__':
    main()
