"""
Context longer than training (context/claude/optimizations.md, linear attention; tools/sweeps/linear_convert.py): the
same held-out files of >= max(--blocks) notes, teacher-forced with the model's own file and section captions, once at
each context length in --blocks (one window from the file start), CE summed over the 4 note heads per predicted note,
averaged per position bucket. Reference 'strided <train block>': windows of the training length every block / 2
notes, each note scored in the window where it has the most context (>= block / 2 notes after the first window),
i.e. what generation with cache refills sees. Gain = CE(reference) - CE(long) per bucket: > 0 = the longer context
helps, < 0 = the model breaks past its training length (RoPE in the full-attention layers; the Gated DeltaNet layers
have no position limit).

    python eval/probes/long_context.py --checkpoints checkpoints/forget_ctrl,checkpoints/linear_h4 \\
        --out results/long_context/linear

Writes <out>/summary.md (also printed) and <out>/long_context.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.data.loader import NoteWindowDataset  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.text.encoder import TextEncoder  # noqa: E402
from section_gain import VAL_SOURCE, note_ce, train_setting  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs (RoPE, section_len)')
    p.add_argument('--out', default='results/long_context/run')
    p.add_argument('--blocks', default='1024,2048,4096', help='context lengths, one window from the file start')
    p.add_argument('--files', type=int, default=64, help='held-out files of >= max(--blocks) notes (evenly thinned)')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--section-max-length', type=int, default=48)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def buckets(blocks, train_block):
    edges = sorted({0, 128, train_block // 2, *blocks})
    return list(zip(edges[:-1], edges[1:]))


def dataset(ck, model, block):
    cfg = model.config
    labels = ROOT / (train_setting(ck, 'captions', '') or 'data/labels')
    return NoteWindowDataset(VAL_SOURCE, root_dir=str(ROOT), split='validation', block_size=block, eval_stride=block,
                             special_tokens=True, programs=True, pad_short=True, captions=str(labels), caption_uk=0.3,
                             section_len=cfg.section_len, section_lead=cfg.section_lead,
                             section_style=train_setting(ck, 'section_style', 'plain'),
                             section_attrs=train_setting(ck, 'section_attrs', 'v1'),
                             section_repeat=cfg.section_repeat, section_caption_mode='true')


def scores(model, encoder, ds, windows, args):
    """CE (n windows, block) and valid mask of the given (file, start) windows."""
    ds.windows = np.asarray(windows, dtype=np.int64)
    ce, valid = [], []
    for b in range(0, len(windows), args.batch):
        c, v, _ = note_ce(model, encoder, [ds[i] for i in range(b, min(b + args.batch, len(windows)))], args.device,
                          args.section_max_length)
        ce.append(c)
        valid.append(v)
    return torch.cat(ce), torch.cat(valid)


def probe(ck, encoder, args):
    model, it = load_model(ck, args.device, torch.bfloat16 if args.device.startswith('cuda') else torch.float32)
    cfg = model.config
    assert cfg.pos_emb == 'rope' and cfg.section_len, f'{ck}: needs RoPE and section captions'
    train_block = cfg.block_size
    blocks = [int(b) for b in args.blocks.split(',')]
    L = max(blocks)
    model.config.block_size = L  # positions past training: RoPE tables are computed per call
    ds = dataset(ck, model, L)
    long_files = np.flatnonzero(ds.lengths > L)
    files = long_files[[int(i * len(long_files) / args.files) for i in range(min(args.files, len(long_files)))]]
    # the long passes, one window from the start of each file
    long = {}
    for block in blocks:
        long[block] = scores(model, encoder, dataset(ck, model, block), [(f, 0) for f in files], args)
    # strided reference at the training length
    half = train_block // 2
    ref = torch.full((len(files), L), float('nan'))
    short = dataset(ck, model, train_block)
    for s in range(0, L - train_block + 1, half):
        c, v = scores(model, encoder, short, [(f, s) for f in files], args)
        lo = 0 if s == 0 else half  # first window: every note; later ones: the second half (>= half notes context)
        part = torch.where(v[:, lo:], c[:, lo:], torch.full_like(c[:, lo:], float('nan')))
        ref[:, s + lo:s + train_block] = part
    res = {'checkpoint': str(ck), 'iter': it, 'train_block': train_block, 'files': len(files),
           'linear_attn': getattr(cfg, 'linear_attn', ''), 'rows': []}
    for lo, hi in buckets(blocks, train_block):
        row = {'positions': f'{lo}-{hi}'}
        r = ref[:, lo:hi]
        row['strided'] = float(r[~r.isnan()].mean()) if (~r.isnan()).any() else float('nan')
        for block in blocks:
            if hi > block:
                continue
            c, v = long[block]
            m = v[:, lo:hi] & ~r.isnan()
            row[f'ctx{block}'] = float(c[:, lo:hi][m].mean()) if m.any() else float('nan')
            row[f'gain{block}'] = float((r - c[:, lo:hi])[m].mean()) if m.any() else float('nan')
        res['rows'].append(row)
    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return res


def summarize(results, blocks):
    lines = ['# Context past the training length', '',
             'CE summed over the 4 note heads per predicted note, held-out labeled files of >= '
             f'{max(blocks)} notes, own captions. strided = windows of the training length every half window (each '
             'note with >= half a window of context, as generation with refills). gain = strided - long context '
             '(> 0: the longer context helps).', '']
    for r in results:
        lines += [f"## {Path(r['checkpoint']).name} (iter {r['iter']}, trained at {r['train_block']}, linear_attn "
                  f"{r['linear_attn'] or 'none'}, {r['files']} files)", '',
                  '| positions | strided | ' + ' | '.join(f'ctx {b} | gain {b}' for b in blocks) + ' |',
                  '|---|---|' + '---|---|' * len(blocks)]
        for row in r['rows']:
            cells = [f"{row.get(f'ctx{b}', float('nan')):.3f} | {row.get(f'gain{b}', float('nan')):+.3f}"
                     if f'ctx{b}' in row else ' | ' for b in blocks]
            lines.append(f"| {row['positions']} | {row['strided']:.3f} | " + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = TextEncoder(device=args.device)
    results = [probe(ck, encoder, args) for ck in args.checkpoints.split(',') if ck]
    (out / 'long_context.json').write_text(json.dumps(results, indent=1))
    text = summarize(results, [int(b) for b in args.blocks.split(',')])
    (out / 'summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
