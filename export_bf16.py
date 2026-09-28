"""
Export a training checkpoint for inference/storage: bf16 weights, no optimizer state (~6x smaller).

    python export_bf16.py checkpoints/ladder_L/ckpt.pt            # -> checkpoints/ladder_L/model_bf16.pt

generate.py loads it like any checkpoint (weights are upcast to the fp32 model on load).
Don't resume training from it: fp32 master weights and Adam moments are gone.
"""

import sys
from pathlib import Path

import torch

src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name('model_bf16.pt')
ckpt = torch.load(src, map_location='cpu')
state = {k.removeprefix('_orig_mod.'): (v.to(torch.bfloat16) if v.is_floating_point() else v)
         for k, v in ckpt['model'].items()}
torch.save({'model': state, 'model_args': ckpt['model_args'], 'config': ckpt.get('config', {}),
            'iter_num': ckpt.get('iter_num'), 'best_val_loss': ckpt.get('best_val_loss')}, dst)
print(f"{src} ({src.stat().st_size / 1e6:.0f} MB) -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
