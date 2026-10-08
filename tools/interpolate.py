"""
Weight interpolation between a pretrained base and a fine-tune of it (WiSE-FT, Wortsman et al. 2022):
theta = (1 - alpha) * theta_base + alpha * theta_finetune, one bf16 checkpoint per alpha. alpha 0 = the base, 1 = the
fine-tune. Tensors the base lacks or has in another shape (the fine-tune's grown style table) are taken from the
fine-tune unscaled, so a learned style label keeps its full strength. The output loads like any checkpoint
(generate.py, eval/seed_compare.py); model_args and config come from the fine-tune.

    python tools/interpolate.py --base checkpoints/long_450m_final --finetune checkpoints/ua_450_s1 \\
        --alphas 0.25 0.5 0.75 --out-dir checkpoints/wise_450_s1
"""

import argparse
from pathlib import Path

import torch

from musicar.checkpoint import resolve_checkpoint


def load(path):
    ck = torch.load(resolve_checkpoint(path), map_location='cpu', weights_only=False, mmap=True)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}
    return ck, sd


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--base', required=True, help='checkpoint file or run dir of the pretrained model')
    ap.add_argument('--finetune', required=True, help='checkpoint file or run dir of its fine-tune')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    ap.add_argument('--out-dir', required=True, help='writes model_bf16_a<alpha>.pt per alpha here')
    args = ap.parse_args()

    _, base = load(args.base)
    ft_ck, ft = load(args.finetune)
    mixed = [k for k, v in ft.items() if v.is_floating_point() and k in base and base[k].shape == v.shape]
    kept = [k for k in ft if k not in mixed]
    print(f"interpolated: {len(mixed)} tensors | taken from the fine-tune as is: {', '.join(kept) or 'none'}")
    unused = [k for k in base if k not in ft]
    if unused:
        print(f"base tensors the fine-tune doesn't have (dropped): {', '.join(unused)}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for a in args.alphas:
        sd = {k: (torch.lerp(base[k].float(), ft[k].float(), a) if k in mixed else ft[k]).to(torch.bfloat16)
              if ft[k].is_floating_point() else ft[k] for k in ft}
        ck = {'model_args': ft_ck['model_args'], 'iter_num': ft_ck['iter_num'], 'config': ft_ck.get('config', {}),
              'model': sd, 'interpolation': {'base': args.base, 'finetune': args.finetune, 'alpha': a}}
        path = out / f'model_bf16_a{a:g}.pt'
        torch.save(ck, path)
        print(f"alpha {a:g} -> {path}")


if __name__ == '__main__':
    main()
