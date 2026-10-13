"""GPU memory savers for training: selective activation checkpointing policies and spilling saved tensors to RAM."""

import torch

# selective activation checkpointing: ops whose outputs a checkpointed block keeps instead of recomputing
SAC_OPS = {
    'attn': ('_scaled_dot_product_flash_attention', '_scaled_dot_product_cudnn_attention',
             '_scaled_dot_product_efficient_attention'),
    'mm': ('_scaled_mm', 'mm', 'addmm'),
}


def sac_context(save):
    """context_fn for torch.utils.checkpoint: keep the outputs of the SAC_OPS groups in `save` ('attn', 'attn,mm')."""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    ops = {getattr(torch.ops.aten, name).default for group in save.split(',') for name in SAC_OPS[group]
           if hasattr(torch.ops.aten, name)}

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in ops else CheckpointPolicy.PREFER_RECOMPUTE
    return create_selective_checkpoint_contexts(policy)


def spill_to_cpu(limit_bytes):
    """Context (eager only): tensors saved for backward move to pinned RAM while the CUDA allocator holds more than
    limit_bytes, and come back when backward needs them. The check reads the allocator's counter (no sync), so it
    costs nothing below the limit. Only plain, dense tensors >= 1 MB (bf16) move; their strides are kept. Under
    torch.compile it saves nothing: the hooks run after the whole compiled forward, i.e. after its memory peak."""
    def pack(t):
        if (type(t) is torch.Tensor and t.is_cuda and t.numel() >= 1 << 19 and t.storage_offset() == 0
                and t.untyped_storage().nbytes() == t.numel() * t.element_size()
                and torch.cuda.memory_allocated(t.device) > limit_bytes):
            c = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, pin_memory=True)
            c.copy_(t, non_blocking=True)
            return t.device, c
        return t

    def unpack(packed):
        return packed[1].to(packed[0], non_blocking=True) if isinstance(packed, tuple) else packed
    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
