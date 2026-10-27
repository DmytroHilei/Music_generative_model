"""
Where the caption acts: the teacher-forced text gain per note position after BOS (the CE with another file's caption
minus the CE with the own caption, and vs no caption), over the first --notes notes of held-out labeled openings, in
position bands. The window averages (val/text_gain, the early-notes gain over 128 notes) dilute a gain that sits on the
first notes (the 107M's sits in notes ~1-30): the +0.02 plateau of plain512 may be +0.08 there. CPU is fine for a few
hundred openings of 128 notes (the 450M: ~5 min at 200 windows).
    python eval/probes/opening_gain.py checkpoints/f1_plain512_long checkpoints/text_t2_v3_long --windows 200
Prints, per checkpoint and band of positions after BOS (position i predicts note i + 1): the own-caption CE (the 4
note heads summed, as ce/total), the gain vs a shuffled caption and vs none, and the count of targets.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musicar.data.loader import BOS_PITCH, NoteWindowDataset, collate_windows  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.model.gpt import STREAM_ORDER  # noqa: E402

BANDS = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256))


def openings(store, notes, windows, mode):
    """Held-out labeled windows that start at a file's BOS, evenly thinned to `windows`; mode = the caption_mode."""
    ds = NoteWindowDataset(store, root_dir='.', split='validation', block_size=notes, eval_stride=notes, programs=True,
                           special_tokens=True, pad_short=True, captions='data/labels', caption_mode=mode)
    starts = [i for i, (f, v) in enumerate(ds.windows) if int(v) == 0]
    idx = [starts[int(j)] for j in torch.linspace(0, len(starts) - 1, min(windows, len(starts))).tolist()]
    return ds, idx


def per_position_ce(model, streams, program, targets, text):
    """(B, T) CE summed over the 4 note heads, -1 targets 0; (B, T) valid mask."""
    with torch.no_grad():
        kw = dict(program=program) if model.config.n_programs else {}
        x = model.encode(*streams, text=text, **kw)
        tgt = dict(zip(model.head_names, targets))
        logits = model.cascade(x, tgt)
    ce = sum(F.cross_entropy(logits[n].float().transpose(1, 2), tgt[n], ignore_index=-1, reduction='none')
             for n in STREAM_ORDER)
    return ce, tgt['pitch'] != -1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--store', default='store:data/cache/gigamidi_heldout#labeled')
    p.add_argument('--windows', type=int, default=200)
    p.add_argument('--notes', type=int, default=128)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    sets = {mode: openings(args.store, args.notes, args.windows, mode) for mode in ('own', 'shuffled', 'none')}
    idx = sets['own'][1]
    from musicar.text.encoder import TextEncoder
    encoder = TextEncoder('sentence-transformers/LaBSE', device='cpu', dtype=torch.float32)
    batches = []
    for b in range(0, len(idx), args.batch):
        got = {mode: collate_windows([ds[i] for i in idx[b:b + args.batch]], extra_names=('program', 'text'))
               for mode, (ds, _) in sets.items()}
        (X, extra), Y = got['own']
        assert (X[0][:, 0] == BOS_PITCH).all()
        states = {mode: encoder.encode(got[mode][0][1]['text'], tokens=True) for mode in got}
        batches.append((X, extra['program'], Y, states))
    print(f'{len(idx)} openings x {args.notes} notes from {args.store}')
    for path in args.checkpoints:
        model, it = load_model(path, device=args.device)
        model.pad_multiple = 0
        assert model.config.text_prefix, f'{path}: not a T2 caption model'
        sums = {mode: torch.zeros(args.notes) for mode in ('own', 'shuffled', 'none')}
        count = torch.zeros(args.notes)
        for X, program, Y, states in batches:
            for mode in sums:
                ce, valid = per_position_ce(model, [s.to(args.device) for s in X], program.to(args.device),
                                            [y.to(args.device) for y in Y], states[mode].to(args.device))
                sums[mode] += (ce * valid).sum(0).cpu()
            count += valid.sum(0).cpu()
        print(f'\n{path} (it {it}): per band of positions after BOS, CE = the 4 note heads summed')
        print('positions | own CE | gain vs shuffled | gain vs none | targets')
        for lo, hi in BANDS + ((0, 32), (0, args.notes)):
            if lo >= args.notes:
                continue
            hi = min(hi, args.notes)
            n = count[lo:hi].sum()
            own, shuf, none = (sums[m][lo:hi].sum() / n.clamp(min=1) for m in ('own', 'shuffled', 'none'))
            print(f'{lo:4d}-{hi - 1:<4d} | {own:6.3f} | {shuf - own:+.4f} | {none - own:+.4f} | {int(n)}')


if __name__ == '__main__':
    main()
