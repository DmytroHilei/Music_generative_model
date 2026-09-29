"""
Compare sampling settings without listening: continue the same real prompts under several settings and
measure the continuations against what the piece really does next.

    python sample_sweep.py --checkpoint checkpoints/big_107M --device cpu --threads 6 \\
        --configs "T=1.0" "T=0.9" "T=1.0,density=prompt" "T=1.0,dt_bias=1"

Config keys: T (temperature), top_k, dt_bias, density (notes/s or 'prompt'), anchor.
Prompts: --prompts random Aria val pieces longer than prompt + new notes. Same seed per prompt for every
config. Metrics per config, averaged over prompts (real = the piece's true continuation):
  notes/s, same-onset % (notes stacked on one onset), velocity, duration, key-sim to the real continuation
  (1 - L1/2 of pitch-class histograms), NLL of the real continuation isn't needed: this compares samples only.
Progress shows on the dashboard (status.py, "Sampling jobs"); the table goes to stdout and --out.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from generate import resolve_checkpoint
from jobstatus import JobStatus
from model import GPT, MusicConfig

DT_SECONDS = 0.02


def parse_config(text):
    cfg = dict(T=1.0, top_k=None, dt_bias=0.0, density=None, anchor=0)
    for part in filter(None, text.split(',')):
        k, v = part.split('=')
        k = k.strip()
        if k not in cfg:
            raise SystemExit(f"unknown config key {k!r} in {text!r}")
        cfg[k] = v if (k == 'density' and v == 'prompt') else (int(v) if k in ('top_k', 'anchor') else float(v))
    return cfg


def stats(t, ref_pitch):
    """t: (n, 4) pitch, velocity bin, duration bin, delta bin."""
    dt = t[:, 3] * DT_SECONDS
    h = lambda p: np.bincount(p % 12, minlength=12) / len(p)
    return dict(nps=len(t) / max(dt[1:].sum(), 1e-3), onset=100 * (t[1:, 3] == 0).mean(),
                vel=(t[:, 1] * 4 + 2).mean(), dur=(t[:, 2] * DT_SECONDS).mean(),
                key=1 - 0.5 * np.abs(h(t[:, 0]) - h(ref_pitch)).sum())


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoint', default='checkpoints/big_107M', help='file or run dir')
    ap.add_argument('--configs', nargs='+', default=['T=1.0', 'T=0.9', 'T=1.0,density=prompt'])
    ap.add_argument('--prompts', type=int, default=5)
    ap.add_argument('--prompt-notes', type=int, default=64)
    ap.add_argument('--new-notes', type=int, default=448)
    ap.add_argument('--seed', type=int, default=3, help='picks the prompts; sampling seed is 1 for all')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--out', default=None, help='also write the table here')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    configs = [(c, parse_config(c)) for c in args.configs]

    path = resolve_checkpoint(args.checkpoint)
    ck = torch.load(path, map_location=args.device)
    cfg = MusicConfig(**{k: v for k, v in ck['model_args'].items() if k in MusicConfig.__dataclass_fields__},
                      dropout=0.0)
    model = GPT(cfg)
    model.load_state_dict({k.replace('_orig_mod.', ''): v.float() for k, v in ck['model'].items()})
    model.eval().to(args.device)
    header = f"checkpoint {path} (iter {ck.get('iter_num', '?')}), {args.prompts} Aria val prompts x " \
             f"{args.prompt_notes} notes -> {args.new_notes} new notes"
    print(header)

    off = np.load('data/cache/aria_validation/offsets.npy')
    tok = np.memmap('data/cache/aria_validation/tokens.u16', dtype=np.uint16, mode='r', shape=(int(off[-1]), 4))
    need = args.prompt_notes + args.new_notes
    picks = np.random.default_rng(args.seed).choice(np.where(np.diff(off) > need)[0], args.prompts, replace=False)

    job = JobStatus('sweep', len(picks) * len(configs) * args.new_notes, f'{len(configs)} configs')
    rows = {'real': []}
    done = 0
    for pi, fi in enumerate(picks):
        piece = np.asarray(tok[off[fi]:off[fi] + need], dtype=np.int64)
        prompt, real = piece[:args.prompt_notes], piece[args.prompt_notes:]
        rows['real'].append(stats(real, real[:, 0]))
        prompt_nps = len(prompt) / max(prompt[1:, 3].sum() * DT_SECONDS, 1e-3)
        x = [torch.from_numpy(prompt[:, j]).unsqueeze(0).to(args.device) for j in range(4)]
        for name, c in configs:
            torch.manual_seed(1)
            progress = lambda n, base=done, name=name, pi=pi: job.update(
                base + n, f'prompt {pi + 1}/{len(picks)}: {name}')
            with torch.no_grad():
                out = model.generate(*x, max_new_tokens=args.new_notes, temperature=c['T'], top_k=c['top_k'],
                                     anchor=c['anchor'], dt_bias=c['dt_bias'], progress=progress,
                                     target_nps=prompt_nps if c['density'] == 'prompt' else c['density'],
                                     dt_seconds=DT_SECONDS)
            done += args.new_notes
            gen = torch.stack([o[0, args.prompt_notes:] for o in out], 1).cpu().numpy()
            rows.setdefault(name, []).append(stats(gen, real[:, 0]))

    lines = [header, '', f"{'config':28s}{'notes/s':>9s}{'onset %':>9s}{'velocity':>10s}{'dur s':>8s}{'key-sim':>9s}"]
    for name, r in rows.items():
        m = {k: np.mean([x[k] for x in r]) for k in r[0]}
        lines.append(f"{name:28s}{m['nps']:9.2f}{m['onset']:9.1f}{m['vel']:10.1f}{m['dur']:8.3f}{m['key']:9.3f}")
    table = '\n'.join(lines)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + '\n')
    job.finish(f"best-matching density: {min(((abs(np.log(np.mean([x['nps'] for x in r]) / np.mean([x['nps'] for x in rows['real']]))), n) for n, r in rows.items() if n != 'real'))[1]}")


if __name__ == '__main__':
    main()
