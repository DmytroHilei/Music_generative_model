"""
Uniform average of checkpoints of one run (model merging in pretraining, arXiv 2505.12082): theta = mean of the
inputs' float tensors, in float32, saved in bf16. Non-float tensors, model_args and config come from the last input.
The output loads like any checkpoint (generate.py, train.py --init_ckpt).

    python tools/checkpoints/average.py checkpoints/v2_A0/model_bf16_it6000.pt checkpoints/v2_A0/model_bf16_it8000.pt \\
        --out checkpoints/avg_v2_A0/model_bf16_6k_8k.pt
"""

import argparse
from pathlib import Path

import torch

from musicar.model.checkpoint import resolve_checkpoint, strip_compile_prefix


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('checkpoints', nargs='+', help='checkpoint files or run dirs (same architecture)')
    ap.add_argument('--out', required=True, help='output .pt')
    args = ap.parse_args()

    total, ck = None, None
    for path in args.checkpoints:
        ck = torch.load(resolve_checkpoint(path), map_location='cpu', weights_only=False, mmap=True)
        sd = strip_compile_prefix(ck['model'])
        if total is None:
            total = {k: v.float().clone() if v.is_floating_point() else v for k, v in sd.items()}
        else:
            if sd.keys() != total.keys() or any(sd[k].shape != total[k].shape for k in sd):
                raise SystemExit(f'{path}: other tensors or shapes than {args.checkpoints[0]}')
            for k, v in sd.items():
                total[k] = total[k] + v.float() if v.is_floating_point() else v
    n = len(args.checkpoints)
    sd = {k: (v / n).to(torch.bfloat16) if v.is_floating_point() else v for k, v in total.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_args': ck['model_args'], 'iter_num': ck['iter_num'], 'config': ck.get('config', {}),
                'model': sd, 'average': list(args.checkpoints)}, out)
    print(f'average of {n} -> {out}')


if __name__ == '__main__':
    main()
