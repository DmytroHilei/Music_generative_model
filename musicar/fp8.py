"""torchao float8 training for the transformer blocks, with an optional per-step cache of the fp8 weights."""

import torch


def convert_fp8(blocks, recipe='tensorwise', skip='', cache_weights=False):
    """torchao float8 training for the Linears in `blocks` (the transformer blocks). recipe: torchao recipe name
    ('tensorwise' | 'rowwise' | 'rowwise_with_gw_hp'); skip: comma-separated module-name suffixes that stay bf16
    (e.g. 'attn.c_proj'). cache_weights (tensorwise only): each weight is cast to fp8 once per optimizer step by
    refresh_fp8_weights() instead of in every forward, recompute and backward of every micro-batch; the caller must
    call refresh_fp8_weights(model) after building the model and after every optimizer step."""
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.float8_linear import Float8Linear
    suffixes = tuple(s for s in skip.split(',') if s)
    convert_to_float8_training(blocks, config=Float8LinearConfig.from_recipe_name(recipe),
                               module_filter_fn=lambda m, name: not (suffixes and name.endswith(suffixes)))
    if cache_weights:
        assert recipe == 'tensorwise', 'fp8 weight caching needs one scale per weight (tensorwise)'
        for m in blocks.modules():
            if isinstance(m, Float8Linear):
                m.__class__ = _cached_float8_linear_class()
                dtype = m.config.cast_config_weight.target_dtype or torch.float8_e4m3fn
                m.register_buffer('w8_data', torch.zeros(m.weight.shape, dtype=dtype, device=m.weight.device),
                                  persistent=False)
                m.register_buffer('w8_scale', torch.ones((), device=m.weight.device), persistent=False)


@torch.no_grad()
def refresh_fp8_weights(model):
    """Cast every cached-fp8 Linear's weight to fp8 (in place, so compiled graphs keep their inputs)."""
    from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
    from torchao.float8.float8_training_tensor import GemmInputRole
    for m in model.modules():
        if hasattr(m, 'w8_data'):
            w8 = hp_tensor_to_float8_dynamic(m.weight, m.w8_data.dtype, m.linear_mm_config,
                                             gemm_input_role=GemmInputRole.WEIGHT,
                                             round_scales_to_power_of_2=m.config.round_scales_to_power_of_2)
            m.w8_data.copy_(w8._data)
            m.w8_scale.copy_(w8._scale)


class _CachedFp8Weight(torch.autograd.Function):
    """The cached fp8 copy of a weight as the matmul operand; the gradient goes straight to the fp32 weight
    (torchao's own weight cast has an identity backward too)."""

    @staticmethod
    def forward(ctx, weight, data, scale, linear_mm_config):
        from torchao.float8.float8_training_tensor import Float8TrainingTensor, GemmInputRole
        return Float8TrainingTensor(data, scale, weight.dtype, linear_mm_config, GemmInputRole.WEIGHT)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None



_CACHED_FP8_LINEAR = None


def _cached_float8_linear_class():
    """Float8Linear whose forward uses the cached fp8 weight (built on first use: this module imports without torchao)."""
    global _CACHED_FP8_LINEAR
    if _CACHED_FP8_LINEAR is None:
        from torchao.float8.float8_linear import Float8Linear, matmul_with_hp_or_float8_args

        class CachedFloat8Linear(Float8Linear):
            def forward(self, input):
                if torch.is_autocast_enabled():  # same autocast handling as Float8Linear.forward
                    input = input.to(torch.get_autocast_gpu_dtype())
                w8 = _CachedFp8Weight.apply(self.weight, self.w8_data, self.w8_scale, self.linear_mm_config)
                output = matmul_with_hp_or_float8_args.apply(input, w8.t(), self.linear_mm_config, self.config)
                if self.bias is not None:
                    output = output + self.bias.to(output.dtype)
                return output
        _CACHED_FP8_LINEAR = CachedFloat8Linear
    return _CACHED_FP8_LINEAR
