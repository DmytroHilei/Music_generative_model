"""
Where the notes' attention goes in a caption model: per layer, the share of the note queries' attention on the
front token (window position 0 of a model without a prefix), on the T2 caption prefix slots and on the section slots,
the largest share one head puts on the caption slots, and the value norm of those slots vs the notes'. A slot that
takes much attention with a small value is an attention sink (a no-op target), not a channel for the caption; the
caption mass equal for the true and the null caption says the same. Teacher-forced on held-out labeled windows, CPU.

    python eval/probes/prefix_attention.py checkpoints/f1_local checkpoints/v2_A0
    python eval/probes/prefix_attention.py checkpoints/long_450m_final      # base: the position-0 sink

Reference (2026-10-25, 8 windows x 512 notes, context/claude/caption_450m_2026-10-25.md): long_450m_final puts
15-53% of the attention of layers 9-27 on position 0 (heads up to 99%, value norm 0.13-0.16x), the 107M bases <= 6%;
f1_local (the 450M F1 rehearsal, it 2,399) puts 12-35% on the 16 caption slots in layers 9-27 (heads up to 86%,
value norm 0.06-0.10x) with the true and the null caption alike; v2_A0 / text_t2_v3 (107M) 2-9% (0.23-0.5x).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import musicar.model.gpt as gpt  # noqa: E402
from musicar.data.loader import BOS_PITCH, NoteWindowDataset, collate_windows  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402

_sdpa = F.scaled_dot_product_attention
REC = None  # while set: {'front': n, 'text': (lo, hi), 'sec': (lo, hi), 'notes': n0, 'rows': []}


def recording_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=False, **kw):
    """F.scaled_dot_product_attention that also records the trunk's causal self-attention (not the text adapter's)."""
    out = _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=enable_gqa, **kw)
    if REC is not None and is_causal and q.size(2) == k.size(2) > 1:
        r = q.size(1) // k.size(1)
        k, v = k.repeat_interleave(r, 1).float(), v.repeat_interleave(r, 1).float()
        n = q.size(2)
        s = (q.float() @ k.transpose(-1, -2)) / q.size(-1) ** 0.5
        p = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float('-inf')).softmax(-1)
        p = p[:, :, REC['notes']:]                                    # note queries
        vn = v.norm(dim=-1).mean((0, 1))                              # value norm per position
        notes = vn[REC['notes']:].mean()
        (tl, th), (sl, sh), f = REC['text'], REC['sec'], REC['front']
        REC['rows'].append(dict(front=p[..., :f].sum(-1).mean().item(), text=p[..., tl:th].sum(-1).mean().item(),
                                sec=p[..., sl:sh].sum(-1).mean().item(),
                                head=p[..., tl:th].sum(-1).mean((0, 2)).max().item() if th > tl else 0.0,
                                v_text=(vn[tl:th].mean() / notes).item() if th > tl else float('nan'),
                                v_front=(vn[:f].mean() / notes).item() if f else float('nan')))
    return out


def layout(cfg, front):
    """(front tokens, caption slots, section slots, first note) as sequence ranges; section captions = the null one
    at note 0, so its slots sit right after the caption prefix (musicar.model.gpt.section_layout)."""
    if not cfg.text_prefix:
        return dict(front=front, text=(0, 0), sec=(0, 0), notes=front)
    k = cfg.text_prefix
    n_sec = cfg.section_slots if cfg.section_len and cfg.section_mode == 'prefix' else 0
    return dict(front=0, text=(0, k), sec=(k, k + n_sec), notes=k + n_sec)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--store', default='store:data/cache/gigamidi_heldout#labeled')
    p.add_argument('--windows', type=int, default=8)
    p.add_argument('--notes', type=int, default=512)
    p.add_argument('--front', type=int, default=1, help='models without a prefix: the first N positions = "front"')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    global REC
    torch.set_num_threads(args.threads)
    ds = NoteWindowDataset(args.store, root_dir='.', split='validation', block_size=args.notes,
                           eval_stride=args.notes, programs=True, special_tokens=True, pad_short=True,
                           captions='data/labels')
    idx = torch.linspace(0, len(ds) - 1, args.windows).long().tolist()
    (X, extra), _ = collate_windows([ds[i] for i in idx], extra_names=('program', 'text'))
    encoder, states = None, None
    gpt.F.scaled_dot_product_attention = recording_sdpa
    for path in args.checkpoints:
        model, it = load_model(path)
        model.pad_multiple = 0
        cfg = model.config
        streams = X if cfg.pitch_size > BOS_PITCH else (X[0].clamp(max=BOS_PITCH - 1),) + tuple(X[1:])
        kw = dict(program=extra['program'] if cfg.n_programs else None)
        runs = [('-', None)]
        if cfg.text_prefix:
            if encoder is None:
                from musicar.text.encoder import TextEncoder
                encoder = TextEncoder('sentence-transformers/LaBSE', device='cpu', dtype=torch.float32)
                states = encoder.encode(extra['text'], tokens=True)
            runs = [('true', states), ('null', None)]
        res = {}
        for name, text in runs:
            REC = dict(rows=[], **layout(cfg, args.front))
            with torch.no_grad():
                model.encode(*streams, text=text, **kw)
            res[name], REC = REC['rows'], None
        print(f'\n{path} (it {it}, {cfg.n_layer} layers, caption slots {cfg.text_prefix})')
        if cfg.text_prefix:
            print('layer | caption true: caption slots / section slots / top head | null: caption slots | '
                  'value norm caption slots / notes')
            for i, (a, b) in enumerate(zip(res['true'], res['null'])):
                print(f"{i:5d} | {a['text']:.3f} / {a['sec']:.3f} / {a['head']:.3f} | {b['text']:.3f} | "
                      f"{a['v_text']:.2f}")
        else:
            print(f'layer | first {args.front} position(s): attention / value norm vs the other notes')
            for i, a in enumerate(res['-']):
                print(f"{i:5d} | {a['front']:.3f} / {a['v_front']:.2f}")
    gpt.F.scaled_dot_product_attention = _sdpa


if __name__ == '__main__':
    main()
