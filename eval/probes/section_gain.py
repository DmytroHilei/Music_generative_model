"""
Where do section captions act? (context/claude/text_control.md, "Section captions"). Teacher-forced CE of held-out
labeled windows with the model's own section captions vs another file's (shuffled) and vs none (null captions), the
file caption kept in all three. Gain = CE(other) - CE(own) per predicted note, averaged per bucket of:
  - its position inside its section (rows k * section_len ..): a caption that arrives early (section_lead > 0)
    should already help in the last notes of the previous section (the model prepares the change), one that comes
    with the section only from its first note on;
  - notes since the latest caption arrived (the decay after arrival; T2's file caption fades ~1,000x by note 128);
  - the window position (comparable with the T2 table in text_control.md);
  - the change score of the note's section vs the one before (musicar.text.captions.section_change: dynamics, pace,
    register, drums, instruments), over the whole section and over its first 32 notes: where the caption says
    something the previous notes don't.
--at-changes: the windows are placed around section boundaries with a change of >= 2 points (the boundary at window
index block / 2), instead of evenly over the held-out files (where ~1 in 5 windows has one).

    python eval/probes/section_gain.py --checkpoints checkpoints/sec_A0,checkpoints/sec_A128 --out results/section_gain/s1
    python eval/probes/section_gain.py --checkpoints checkpoints/sec_A256 --at-changes --out results/section_gain/a256_ch

Writes <out>/summary.md (also printed) and <out>/gain.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.data.loader import NoteWindowDataset, collate_windows, section_change_scores  # noqa: E402
from musicar.model.checkpoint import load_model, resolve_checkpoint  # noqa: E402
from musicar.model.gpt import STREAM_ORDER  # noqa: E402
from musicar.text.encoder import TextEncoder  # noqa: E402

VAL_SOURCE = 'store:data/cache/gigamidi_heldout#labeled'
MODES = ('true', 'shuffled', 'none')
SINCE = [(0, 1), (1, 8), (8, 32), (32, 128), (128, 512), (512, 2048)]
WINDOW = [(0, 8), (8, 32), (32, 128), (128, 512), (512, 1024)]
CHANGE = [(0, 1), (1, 2), (2, 3), (3, 6)]


def in_section(S):
    """Position-in-section buckets; the last 32 notes of a section are where an early caption can act first."""
    return [(0, 1), (1, 8), (8, 32), (32, S // 2), (S // 2, S - 32), (S - 32, S)]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs with section_len')
    p.add_argument('--out', default='results/section_gain/run')
    p.add_argument('--windows', type=int, default=256, help='held-out windows (evenly thinned, same for every mode)')
    p.add_argument('--at-changes', action='store_true', help='windows around section changes of >= 2 points')
    p.add_argument('--section-style', default=None, choices=('plain', 'change'),
                   help="section captions as in training ('change' = name what changed); default: the checkpoint's")
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--block', type=int, default=1024)
    p.add_argument('--section-max-length', type=int, default=48)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.no_grad()
def note_ce(model, encoder, items, device, max_length):
    """Per-position CE summed over the 4 note heads (B, T) and the valid-target mask, for dataset items."""
    (streams, extra), y = collate_windows(items, extra_names=('program', 'text', 'sec_text', 'sec_pos'))
    streams = [t.to(device) for t in streams]
    y = [t.to(device) for t in y]
    text = encoder.encode(extra['text'], tokens=True)
    caps = extra['sec_text']
    states = encoder.encode([c for row in caps for c in row], tokens=True, max_length=max_length)
    states = states.view(len(caps), len(caps[0]), *states.shape[1:])
    ctx = torch.autocast('cuda', torch.bfloat16) if device.startswith('cuda') else torch.autocast('cpu', enabled=False)
    with ctx:
        vecs = model.section_vectors(states)
        h = model.encode(*streams, program=extra['program'].to(device), text=text,
                         sections=(vecs, extra['sec_pos'].to(device)))
        tgt = dict(zip(model.head_names, y))
        logits = model.cascade(h, tgt)
    ce = sum(F.cross_entropy(logits[n].float().transpose(1, 2), tgt[n], ignore_index=-1, reduction='none')
             for n in STREAM_ORDER)
    return ce.cpu(), (tgt['pitch'] != -1).cpu(), extra['sec_pos']


def bucket_means(gain, key, valid, buckets):
    out = []
    for lo, hi in buckets:
        m = valid & (key >= lo) & (key < hi)
        out.append((f'{lo}-{hi}', int(m.sum()), float(gain[m].mean()) if m.any() else float('nan')))
    return out


def train_setting(ck, name, default):
    """A train.py setting saved in the checkpoint's config (default for checkpoints from before it existed)."""
    return torch.load(resolve_checkpoint(ck), map_location='cpu', mmap=True).get('config', {}).get(name, default)


def probe(ck, encoder, args):
    model, it = load_model(ck, args.device, torch.bfloat16 if args.device.startswith('cuda') else torch.float32)
    cfg = model.config
    if not cfg.section_len:
        return None
    S = cfg.section_len
    style = args.section_style or train_setting(ck, 'section_style', 'plain')
    kw = dict(root_dir=str(ROOT), split='validation', block_size=args.block, eval_stride=args.block,
              special_tokens=True, programs=True, pad_short=True, captions=str(ROOT / 'data/labels'), caption_uk=0.3,
              section_len=S, section_lead=cfg.section_lead, section_style=style)
    sets = {m: NoteWindowDataset(VAL_SOURCE, section_caption_mode=m, **kw) for m in MODES}
    ds = sets['true']
    if args.at_changes:
        ds._find_changes(2)
        windows = np.array([(f, min(max(row - args.block // 2, 0), int(ds.lengths[f]) - args.block - 1))
                            for f, row in ds.changes])
        for m in MODES:
            sets[m].windows = windows
    n = len(ds)
    idx = [int(i * n / args.windows) for i in range(min(args.windows, n))]
    # change score of every section of every file: (store, file in store) -> its first section's index in scores
    scores = [section_change_scores(store, S, ds.pad) for store in ds.store_dirs]
    first_sec = []
    for store in ds.store_dirs:
        n_sec = -(-(np.diff(np.load(store / 'offsets.npy')) + 2 * ds.pad) // S)
        first_sec.append(np.cumsum(n_sec) - n_sec)
    ce = {m: [] for m in MODES}
    valid, pin, since, wpos, change = [], [], [], [], []
    for b in range(0, len(idx), args.batch):
        chunk = idx[b:b + args.batch]
        for m in MODES:
            c, v, pos = note_ce(model, encoder, [sets[m][i] for i in chunk], args.device, args.section_max_length)
            ce[m].append(c)
        valid.append(v)
        T = c.size(1)
        tgt_i = torch.arange(1, T + 1)                                      # window index of each predicted note
        starts = torch.tensor([int(sets['true'].windows[i][1]) for i in chunk])
        pin.append((starts[:, None] + tgt_i[None]) % S)
        latest = torch.where(pos[:, None, :] <= tgt_i[None, :, None], pos[:, None, :], -1).amax(-1)
        since.append(tgt_i[None] - latest)
        wpos.append(tgt_i[None].expand(len(chunk), -1) - 1)
        rows = []
        for i, start in zip(chunk, starts.tolist()):
            f = int(ds.windows[i][0])
            sid, fid = int(ds.store_ids[f]), int(ds.file_ids[f])
            last = -(-int(ds.lengths[f]) // S) - 1
            k = np.minimum((start + tgt_i.numpy()) // S, last)  # rows past the file (padding) are masked anyway
            rows.append(scores[sid][first_sec[sid][fid] + k].astype(np.int64))
        change.append(torch.from_numpy(np.stack(rows)))
    cat = {m: torch.cat(v) for m, v in ce.items()}
    valid, pin, since, wpos, change = (torch.cat(v) for v in (valid, pin, since, wpos, change))
    res = {'checkpoint': str(ck), 'iter': it, 'section_len': S, 'section_lead': cfg.section_lead,
           'section_mode': cfg.section_mode, 'windows': len(idx), 'at_changes': args.at_changes, 'section_style': style}
    # the CE itself with the own captions (headroom: how much worse the first notes after a change are predicted)
    # (notes with >= 128 notes of context: a window's first notes are predicted badly whatever their section)
    ctx = valid & (wpos >= 128)
    res['ce'] = {'change': bucket_means(cat['true'], change, ctx, CHANGE),
                 'change_start': bucket_means(cat['true'], change, ctx & (pin < 32), CHANGE),
                 'change_first8': bucket_means(cat['true'], change, ctx & (pin < 8), CHANGE),
                 'change_rest': bucket_means(cat['true'], change, ctx & (pin >= 32), CHANGE)}
    for other in ('shuffled', 'none'):
        gain = cat[other] - cat['true']
        res[other] = {'all': float(gain[valid].mean()),
                      'in_section': bucket_means(gain, pin, valid, in_section(S)),
                      'since_arrival': bucket_means(gain, since, valid, SINCE),
                      'window': bucket_means(gain, wpos, valid, WINDOW),
                      'change': bucket_means(gain, change, valid, CHANGE),
                      'change_start': bucket_means(gain, change, valid & (pin < 32), CHANGE)}
    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return res


def summarize(results):
    lines = []
    for r in results:
        name = Path(r['checkpoint']).name if Path(r['checkpoint']).is_dir() else Path(r['checkpoint']).parent.name
        lines += [f"## {name} ({r['section_mode']}, section {r['section_len']}, lead {r['section_lead']}, "
                  f"@{r['iter']}, {r['windows']} windows{' around changes' if r.get('at_changes') else ''}"
                  f"{', change-aware captions' if r.get('section_style') == 'change' else ''})", '',
                  f"all positions: vs shuffled {r['shuffled']['all']:+.4f}, vs none {r['none']['all']:+.4f}", '']
        for table, title in (('in_section', 'position in section'), ('since_arrival', 'notes since the caption'),
                             ('window', 'window position'), ('change', 'change score of the section'),
                             ('change_start', 'change score, first 32 notes of the section')):
            if table not in r['shuffled']:
                continue
            lines += [f'| {title} | notes | vs shuffled | vs none |', '|---|---|---|---|']
            for (b, cnt, gs), (_, _, gn) in zip(r['shuffled'][table], r['none'][table]):
                lines.append(f'| {b} | {cnt:,} | {gs:+.4f} | {gn:+.4f} |')
            lines.append('')
        if 'ce' in r:
            lines += ['| CE (own captions, >= 128 notes of context) by change score | whole section | first 8 notes | first 32 | rest |',
                      '|---|---|---|---|---|']
            for row in zip(*(r['ce'][t] for t in ('change', 'change_first8', 'change_start', 'change_rest'))):
                lines.append(f'| {row[0][0]} | ' + ' | '.join(f'{ce:.3f} ({cnt:,})' for _, cnt, ce in row) + ' |')
            lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = TextEncoder(device=args.device)
    results = []
    for ck in [c for c in args.checkpoints.split(',') if c]:
        r = probe(ck, encoder, args)
        if r is None:
            print(f'{ck}: no section captions, skipped')
            continue
        results.append(r)
        print(summarize([r]))
    (out / 'gain.json').write_text(json.dumps(results, indent=1))
    (out / 'summary.md').write_text(summarize(results))
    print(f'-> {out}/summary.md')


if __name__ == '__main__':
    main()
