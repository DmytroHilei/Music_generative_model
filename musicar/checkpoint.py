"""Checkpoint paths and loading."""

from pathlib import Path

import torch
from torch.nn import functional as F


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        for name in ("model_bf16.pt", "best.pt", "ckpt.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"no model_bf16.pt / best.pt / ckpt.pt in {path}")
    return path


def load_expanded(model, state_dict):
    """Load a checkpoint into a model that may have grown: extra rows in dim 0 (pitch vocab 128 -> 130) keep
    their fresh init, keys the checkpoint lacks (style embedding) keep theirs. Returns the grown/new keys.
    Exception: a new program condition embedding is zeroed, so the old pitch/duration/velocity heads see exactly
    the conditioning they were trained with (function-preserving growth to multi-instrument), and so is the
    instrument head's output layer."""
    own = model.state_dict()
    # a learned-position checkpoint loaded into a RoPE model: the position table has no place to go
    dropped = [k for k in state_dict if k.startswith('transformer.wpe.') and k not in own]
    for k in dropped:
        del state_dict[k]
    changed = [k for k in own if k not in state_dict] + [f'{k} (dropped)' for k in dropped]
    for k in [k for k in own if k not in state_dict]:
        # the new instrument head's output also starts at zero = a uniform guess (ln 129), not a random one
        if k.startswith('cascade.cond_emb.program.') or k == 'cascade.heads.program.out.weight':
            state_dict[k] = torch.zeros_like(own[k])
    for k, v in state_dict.items():
        if k == 'transformer.wpe.weight' and own[k].shape[0] > v.shape[0]:
            # longer context with learned positions: stretch the table (position interpolation), so position p
            # gets the old embedding at p * old/new, linearly between neighbouring rows
            state_dict[k] = F.interpolate(v.float().t()[None], size=own[k].shape[0], mode='linear',
                                          align_corners=True)[0].t().to(v.dtype)
            changed.append(f'{k} (stretched {v.shape[0]} -> {own[k].shape[0]})')
        elif k in own and own[k].shape != v.shape:
            assert own[k].shape[1:] == v.shape[1:] and own[k].shape[0] >= v.shape[0], \
                f"{k}: can't grow {tuple(v.shape)} -> {tuple(own[k].shape)}"
            grown = own[k].clone()
            grown[:v.shape[0]] = v
            state_dict[k] = grown
            changed.append(k)
    model.load_state_dict(state_dict, strict=False)
    return changed
