"""
Fake quantization for inference-quality checks (context/claude/inference_pi.md): weights, activations and the KV
cache are rounded to the values int8 / int4 kernels on the Pi would hold, and the model keeps computing in its own
dtype (use float32 so bf16 rounding doesn't blur the int8 comparison).

Spec: '-'-separated parts, e.g. 'w4g32-a8-kv8' (the Pi's planned path), 'w8-a8', 'w4g128', 'none'.
  w<bits>[g<group>]  Linear weights of the transformer blocks, symmetric, one scale per output channel per group of
                     <group> inputs (no g = one scale per output channel). Every other Linear (cascade heads, input
                     projection, text adapter) and every Embedding: int8 per output channel / row, as planned for the
                     Pi.
  a<bits>            dynamic activation quantization at the input of every quantized Linear: asymmetric, one scale
                     and zero point per token (KleidiAI's 8-bit dynamic activations).
  kv<bits>           keys (after QK-norm and RoPE) and values as cached: symmetric, one scale per token and head.
Norm weights and biases stay in float.
"""

import re

import torch
from torch import nn


def parse(spec):
    """'w4g32-a8-kv8' -> dict(w=4, group=32, a=8, kv=8); missing parts are 0; 'none' / '' -> all 0."""
    out = dict(w=0, group=0, a=0, kv=0)
    for part in (spec or '').split('-'):
        if part in ('', 'none'):
            continue
        m = re.fullmatch(r'w(\d+)(?:g(\d+))?|a(\d+)|kv(\d+)', part)
        if not m:
            raise ValueError(f'bad quantization spec part {part!r} in {spec!r}')
        if m.group(1):
            out['w'], out['group'] = int(m.group(1)), int(m.group(2) or 0)
        elif m.group(3):
            out['a'] = int(m.group(3))
        else:
            out['kv'] = int(m.group(4))
    return out


def quantize_symmetric(x, bits, group=0):
    """x rounded to `bits`-bit signed integers times a scale per row (last dim), or per `group` values of each row;
    in x's dtype. int4: -8..7 with scale = max |x| / 7."""
    qmax = 2 ** (bits - 1) - 1
    shape = x.shape
    g = x.float().reshape(*shape[:-1], -1, group) if group else x.float().unsqueeze(-2)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
    return ((g / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(shape).to(x.dtype)


def quantize_dynamic(x, bits=8):
    """Asymmetric per-token (last dim) quantization of activations, in x's dtype."""
    xf = x.float()
    lo = xf.amin(-1, keepdim=True).clamp(max=0)
    hi = xf.amax(-1, keepdim=True).clamp(min=0)
    levels = 2 ** bits - 1
    scale = ((hi - lo) / levels).clamp(min=1e-12)
    zero = (-lo / scale).round()
    q = (xf / scale + zero).round().clamp(0, levels)
    return ((q - zero) * scale).to(x.dtype)


@torch.no_grad()
def fake_quantize(model, spec):
    """Round model's weights in place and install the activation / KV hooks of `spec` (see the module doc).
    Returns a short summary string."""
    s = parse(spec)
    if not any(s.values()):
        return 'none'
    counts = {'trunk': 0, 'other': 0, 'emb': 0, 'per_channel_fallback': 0}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and s['w']:
            trunk = name.startswith('transformer.h.')
            bits, group = (s['w'], s['group']) if trunk else (8, 0)
            if group and mod.weight.shape[1] % group:
                group = 0
                counts['per_channel_fallback'] += 1
            mod.weight.copy_(quantize_symmetric(mod.weight, bits, group))
            counts['trunk' if trunk else 'other'] += 1
        elif isinstance(mod, nn.Embedding) and s['w']:
            mod.weight.copy_(quantize_symmetric(mod.weight, 8))
            counts['emb'] += 1
        if isinstance(mod, nn.Linear) and s['a']:
            mod.register_forward_pre_hook(lambda m, args, bits=s['a']: (quantize_dynamic(args[0], bits),) + args[1:])
        if hasattr(mod, 'kv_bits'):
            mod.kv_bits = s['kv']
    return (f"{spec}: {counts['trunk']} block Linears at w{s['w']}" + (f"g{s['group']}" if s['group'] else '')
            + f", {counts['other']} other Linears + {counts['emb']} Embeddings at int8"
            + (f", {counts['per_channel_fallback']} per-channel (group doesn't divide)" if counts['per_channel_fallback']
               else '') + (f", activations a{s['a']}" if s['a'] else '') + (f", KV kv{s['kv']}" if s['kv'] else ''))
