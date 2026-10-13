"""Validation loss: optimized loss and true per-head CE over a fixed set of windows."""

import torch

from musicar.model import STREAM_ORDER


def summed_ce(metrics):
    """ce/total = the 4 note attributes (comparable with the piano runs); ce/total_all also counts the instrument head."""
    metrics['ce/total'] = sum(metrics[f'ce/{name}'] for name in STREAM_ORDER)
    if 'ce/program' in metrics:
        metrics['ce/total_all'] = metrics['ce/total'] + metrics['ce/program']


@torch.no_grad()
def estimate_val_loss(model, batches, ctx):
    """batches: iterable of ((streams, extra inputs), targets) on the model's device; ctx: the autocast context.
    Returns dict: 'loss' (optimized loss, with label smoothing), 'ce/<head>' (true CE, nats) and the summed CEs."""
    model.eval()
    sums, n = {}, 0
    for (streams, extra), Y in batches:
        with ctx:
            parts, loss = model(*streams, **extra, targets=Y)
        sums['loss'] = sums.get('loss', 0.0) + loss.item()
        for name, v in parts.items():
            sums[f'ce/{name}'] = sums.get(f'ce/{name}', 0.0) + v.item()
        n += 1
    model.train()
    out = {k: v / n for k, v in sums.items()}
    summed_ce(out)
    return out
