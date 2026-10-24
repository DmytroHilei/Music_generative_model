"""Checkpoint paths, model args, loading and saving."""

import os
from pathlib import Path

import torch
from torch.nn import functional as F

# the MusicConfig fields a checkpoint stores as model_args
ARCH_KEYS = ('n_layer', 'n_head', 'n_kv_head', 'n_embd', 'block_size', 'bias', 'pitch_size', 'velocity_size',
             'duration_size', 'delta_time_size', 'n_styles', 'n_programs', 'pos_emb', 'rope_base', 'norm', 'mlp',
             'qk_norm', 'cond_inst', 'n_density', 'n_remain', 'n_progress', 'text_dim', 'text_center', 'text_prefix',
             'section_len', 'section_lead', 'section_slots', 'section_mode', 'section_xattn_every', 'section_repeat',
             'linear_attn', 'heads')
# what checkpoints that predate a key actually used
LEGACY_DEFAULTS = {'n_kv_head': 0, 'n_styles': 0, 'n_programs': 0, 'pos_emb': 'learned', 'rope_base': 10000.0,
                   'norm': 'layernorm', 'mlp': 'gelu', 'qk_norm': False, 'cond_inst': False, 'n_density': 0, 'n_remain': 0,
                   'n_progress': 0, 'text_dim': 0, 'text_center': False, 'text_prefix': 0, 'section_len': 0, 'section_lead': 0,
                   'section_slots': 4, 'section_mode': 'prefix', 'section_xattn_every': 2,
                   'section_repeat': 0, 'linear_attn': '', 'heads': 'cascade'}


def torch_load(path, **kw):
    """torch.load to the CPU (weights_only): torchao.optim imported first (when installed) registers its optimizer-state
    subclasses, so a resumable ckpt.pt with 8-bit AdamW states loads."""
    try:
        import torchao.optim  # noqa: F401
    except ImportError:
        pass
    return torch.load(path, map_location='cpu', **kw)


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
    instrument head's output layer. GQA: a checkpoint with more key / value heads than the model (n_kv_head) has
    each group's K / V projection rows mean-pooled (pool_kv_heads; not function-preserving, the uptraining recovers)."""
    own = model.state_dict()
    cfg, pooled = model.config, []
    for k in [k for k in state_dict if k.endswith(('.attn.c_attn.weight', '.attn.c_attn.bias'))]:
        if k in own and own[k].shape[0] < state_dict[k].shape[0]:
            state_dict[k] = pool_kv_heads(state_dict[k], cfg.n_embd, cfg.n_embd // cfg.n_head, cfg.kv_heads)
            pooled.append(f'{k} (K / V mean-pooled to {cfg.kv_heads} heads)')
    # a learned-position checkpoint loaded into a RoPE model: the position table has no place to go
    dropped = [k for k in state_dict if k.startswith('transformer.wpe.') and k not in own]
    for k in dropped:
        del state_dict[k]
    changed = [k for k in own if k not in state_dict] + [f'{k} (dropped)' for k in dropped] + pooled
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


def pool_kv_heads(w, n_embd, head_size, n_kv):
    """c_attn weight (or bias) rows [q (n_embd) | k | v] with any number of key / value heads -> n_kv of them: each
    group of consecutive heads (the query heads that share a KV head under enable_gqa) averaged."""
    q, kv = w[:n_embd], w[n_embd:]
    k, v = kv.chunk(2, dim=0)
    heads = k.shape[0] // head_size
    assert heads % n_kv == 0, f'{heads} KV heads do not pool into {n_kv}'

    def pool(x):
        return x.reshape(n_kv, heads // n_kv, head_size, *x.shape[1:]).float().mean(1).reshape(
            n_kv * head_size, *x.shape[1:]).to(x.dtype)
    return torch.cat([q, pool(k), pool(v)])


def merge_model_args(saved, wanted, finetune, block_size):
    """Model args for a resume / finetune: the checkpoint's own (`saved`; LEGACY_DEFAULTS, then `wanted`, for keys it
    predates). A finetune may change what load_expanded can map: grow pitch_size / n_styles / n_programs, add the
    text input (text_dim 0 -> the encoder's width), its centering (text_center) or the T2 prefix (text_prefix), convert
    attention layers to linear attention (linear_attn), add the length input (n_remain), pool key / value heads (n_kv_head below the checkpoint's: GQA),
    switch to RoPE, and set block_size (RoPE: any; learned positions:
    larger, the table is stretched)."""
    args = {}
    add_sections = finetune and not saved.get('section_len', 0)
    for k in ARCH_KEYS:
        args[k] = saved[k] if k in saved else LEGACY_DEFAULTS[k] if k in LEGACY_DEFAULTS else wanted[k]
        if finetune and k in ('pitch_size', 'n_styles', 'n_programs') and wanted[k] > args[k]:
            args[k] = wanted[k]  # grow: BOS/EOS rows, a style table, the instrument attribute
        if finetune and k == 'text_dim' and args[k] == 0:
            args[k] = wanted[k]  # add the text input (zero-init text_proj: starts as the old model)
        if finetune and k == 'text_center' and wanted[k]:
            args[k] = True  # text_mean is then set from the training captions (train.py)
        if finetune and k in ('n_remain', 'n_progress') and wanted.get(k) and not args[k]:
            args[k] = wanted[k]  # add the length input (zero-init cond_remain / cond_progress: starts as the old model)
        if finetune and k == 'text_prefix' and args[k] == 0:
            args[k] = wanted[k]  # add the T2 prefix adapter
        if add_sections and k.startswith('section_'):
            args[k] = wanted.get(k, args[k])  # add section captions (fresh adapter; 'xattn' starts as the old model)
        if finetune and k == 'section_repeat' and wanted.get(k):
            args[k] = wanted[k]  # repeat the section caption (no new parameters, only more inserted vectors)
        if finetune and k == 'linear_attn' and wanted.get(k) and not args[k]:
            args[k] = wanted[k]  # convert attention layers to linear attention (c_attn / c_proj carry over)
        if finetune and k == 'n_kv_head' and wanted.get(k) and wanted[k] < (args[k] or args['n_head']):
            args[k] = wanted[k]  # GQA: K / V heads mean-pooled per group (load_expanded)
        if finetune and k == 'pos_emb' and wanted[k] == 'rope':
            args[k] = 'rope'  # switch to RoPE: the position table is dropped (not function-preserving)
    if finetune and args['pos_emb'] == 'rope':
        args['block_size'] = block_size  # nothing in a RoPE model depends on the context length
    elif finetune and block_size > args['block_size']:
        args['block_size'] = block_size  # learned positions: the table is stretched (position interpolation)
    return args


def load_model(path, device='cpu', dtype=torch.float32, quant=None):
    """A trained GPT for inference from a checkpoint file or run dir (resolve_checkpoint): dropout 0, eval mode, on
    `device` in `dtype`. Keys a checkpoint predates take MusicConfig's defaults. quant: a musicar.model.quant spec
    (e.g. 'w4g32-a8-kv8'), applied in float32 before the cast to `dtype`. Returns (model, iter_num or None)."""
    from musicar.model.gpt import GPT, MusicConfig
    ck = torch_load(resolve_checkpoint(path))
    fields = MusicConfig.__dataclass_fields__
    model = GPT(MusicConfig(**{k: v for k, v in ck['model_args'].items() if k in fields}, dropout=0.0))
    state = strip_compile_prefix(ck['model'])
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('aux_')})  # training-only heads
    if quant:
        from musicar.model.quant import fake_quantize
        print(f'fake quantization {fake_quantize(model, quant)}')
    return model.eval().to(device=device, dtype=dtype), ck.get('iter_num')


def strip_compile_prefix(state_dict):
    """Keys saved from a torch.compile'd model start with '_orig_mod.'; drop it (in place)."""
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    return state_dict


def bf16_cpu_state(model):
    """The weights as bf16 on the CPU. A bf16 copy on the GPU (~0.9 GB for the 450M) kept alive by the training
    script's checkpoint dict caused an OOM in every step after the first best-checkpoint save on the 8 GB laptop
    (ua_450_multi)."""
    return {k: v.to('cpu', torch.bfloat16) if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()}


def save_atomic(obj, path):
    """torch.save to path.tmp, then rename: a crash mid-save keeps the old file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)
