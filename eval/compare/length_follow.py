"""
Do pieces end at the requested length? (length control, roadmap item 11; MusicConfig.n_remain.) Generation from BOS
with --length-seconds L for every L in --lengths, plus a 'none' run (no length: where the model ends on its own), rows
in a batch, instruments sampled. Per row: the first EOS, its time (seconds from BOS, the sum of the delta_times up to
it) and end / L. Models without the length input get the same runs, all of them unconditional (the control arm).

    python eval/compare/length_follow.py --checkpoints checkpoints/len_pilot,checkpoints/len_ctrl \\
        --out results/length_follow/pilot --midi 2 --mp3

Writes <out>/rows.csv, <out>/summary.md (per checkpoint and length: share ended, median end / L, share within +-10% /
+-20% of L, median |log2(end / L)|) and MIDI (+ MP3) of the first --midi rows (<out>/<checkpoint>/len<L>_<i>.mid).
"""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.data.loader import BOS_PITCH, EOS_PITCH  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402
from text_follow import SOUNDFONT, write_midi  # noqa: E402

DT_SECONDS = 0.02


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoint files / run dirs')
    p.add_argument('--lengths', default='30,60,120,240', help='requested lengths in seconds')
    p.add_argument('--rows', type=int, default=8, help='rows per length (one batch)')
    p.add_argument('--max-nps', type=float, default=12.0,
                   help='notes per second the note budget allows: max_new = 1.5 x L x this')
    p.add_argument('--temperature', type=float, default=0.85)
    p.add_argument('--top-p', type=float, default=None)
    p.add_argument('--eos-t1', action='store_true', help="generate's eos_t1 (EOS at temperature 1)")
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--midi', type=int, default=0, help='MIDI files of the first N rows per length')
    p.add_argument('--mp3', action='store_true')
    p.add_argument('--soundfont', default=SOUNDFONT)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='results/length_follow/run')
    p.add_argument('--summarize', action='store_true', help='only rewrite summary.md from rows.csv')
    return p.parse_args()


def end_of(pitch, dt):
    """(notes before the first EOS, its time in seconds) of one row; (None, None) without an EOS. Row 0 = BOS."""
    eos = (pitch == EOS_PITCH).nonzero()
    if len(eos) == 0:
        return None, None
    e = int(eos[0])
    return e - 1, float(dt[:e].sum()) * DT_SECONDS


def summarize(rows):
    lines = ['| checkpoint | length (s) | rows | ended | median end (s) | median end / L | within 10% | within 20% '
             '| median abs log2(end / L) |', '|---|---|---|---|---|---|---|---|---|']
    keys = sorted({(r['checkpoint'], r['length']) for r in rows},
                  key=lambda k: (k[0], k[1] != 'none', 0.0 if k[1] == 'none' else float(k[1])))
    for ck, length in keys:
        rs = [r for r in rows if (r['checkpoint'], r['length']) == (ck, length)]
        ended = [r for r in rs if r['end_s'] != '']
        ends = [float(r['end_s']) for r in ended]
        cells = [ck, length, str(len(rs)), f'{len(ended) / len(rs):.0%}', f'{statistics.median(ends):.1f}' if ends else '']
        if length == 'none' or not ends:
            cells += [''] * 4
        else:
            ratio = [e / float(length) for e in ends]
            cells += [f'{statistics.median(ratio):.2f}',
                      f'{sum(abs(x - 1) <= 0.1 for x in ratio) / len(rs):.0%}',
                      f'{sum(abs(x - 1) <= 0.2 for x in ratio) / len(rs):.0%}',
                      f'{statistics.median(abs(math.log2(max(x, 1e-3))) for x in ratio):.2f}']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'within 10% / 20%: share of all rows (rows without an EOS count as misses).']
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        with open(out / 'rows.csv') as f:
            rows = list(csv.DictReader(f))
        (out / 'summary.md').write_text(summarize(rows))
        print(summarize(rows))
        return
    lengths = [float(x) for x in args.lengths.split(',') if x]
    runs = [None] + lengths  # None = no length input ('none')
    ckpts = [c for c in args.checkpoints.split(',') if c]
    job = JobStatus('length_follow', len(ckpts) * len(runs), f'{len(ckpts)} checkpoint(s) -> {out}')
    rows, done = [], 0
    for ck in ckpts:
        model, it = load_model(ck, args.device, torch.bfloat16 if args.device.startswith('cuda') else torch.float32)
        assert model.config.pitch_size > BOS_PITCH, f'{ck}: needs BOS / EOS'
        name = Path(ck).name if Path(ck).is_dir() else Path(ck).parent.name
        if not model.config.n_remain:
            print(f'{name}: no length input (n_remain 0): every run is unconditional')
        for length in runs:
            t0 = time.time()
            n_new = int(1.5 * (length or max(lengths)) * args.max_nps)
            torch.manual_seed(args.seed)
            bos = [torch.full((args.rows, 1), BOS_PITCH, device=args.device)] + \
                [torch.zeros(args.rows, 1, dtype=torch.long, device=args.device)] * 4
            res = generate(model, *bos[:4], max_new_tokens=n_new, temperature=args.temperature, top_p=args.top_p,
                           program=None if model.config.n_programs else 0, programs=bos[4] if model.config.n_programs
                           else None, return_programs=True, eos_t1=args.eos_t1, length_seconds=length)
            arrays = [t.cpu() for t in res]
            label = 'none' if length is None else f'{length:g}'
            for r in range(args.rows):
                notes, end_s = end_of(arrays[0][r], arrays[3][r])
                rows.append(dict(checkpoint=name, iter=it, length=label, row=r, seed=args.seed,
                                 notes='' if notes is None else notes, end_s='' if end_s is None else f'{end_s:.2f}'))
                if r < args.midi:
                    cut = [a[r:r + 1, :notes + 1] if notes is not None else a[r:r + 1] for a in arrays]
                    write_midi([c.numpy() for c in cut], 0, out / name / f'len{label}_{r}.mid', args.mp3,
                               args.soundfont)
            done += 1
            job.update(done, f'{name}: length {label}')
            print(f'{name} length {label}: {n_new} notes x {args.rows} rows in {time.time() - t0:.0f} s')
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (out / 'summary.md').write_text(summary)
    print(summary)
    job.finish(f'{out}/summary.md')


if __name__ == '__main__':
    main()
