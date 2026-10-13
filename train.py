"""
Training script for pretraining and fine-tuning (nanoGPT style).

    python train.py config/pretrain.py
    python train.py config/finetune_skryabin.py
    python train.py config/pretrain.py config/smoke.py       # quick sanity run
    python train.py config/pretrain.py --dropout=0.1 --n_embd=512

init_from:
    'scratch'  - new model
    'resume'   - continue a run from out_dir/ckpt.pt (weights, optimizer, iter, best val)
    'finetune' - weights only from init_ckpt, fresh optimizer / iter / best val loss
"""

import os
# fewer fragmentation OOMs: the caching allocator grows segments instead of carving fixed blocks (set before CUDA starts)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import time
import random
import signal
import sys
from contextlib import nullcontext
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from musicar.checkpoint import (ARCH_KEYS, bf16_cpu_state, load_expanded, merge_model_args, save_atomic,
                                strip_compile_prefix)
from musicar.data_loader import (MaestroDataset, collate_windows, load_styles, source_name, val_loader,
                                 window_conditions)
from musicar.eval.val_loss import estimate_val_loss, summed_ce
from musicar.fp8 import convert_fp8, fp8_supported, refresh_fp8_weights
from musicar.memory import spill_to_cpu
from musicar.model import MusicConfig, GPT, STREAM_ORDER
from musicar.optim import build_optimizer, cooldown_start, freeze_params, lr_at, set_lr

# -----------------------------------------------------------------------------
# I/O
out_dir = 'checkpoints'
eval_interval = 20
eval_iters = 200          # max number of val batches per evaluation (fixed windows, same every time)
style_eval_songs = 0      # > 0: at every evaluation generate from this many fixed songs per style (5 styles) and log
                          # per-style distances to the real continuations (style_eval.py: styles/<mode>/<style>)
style_eval_csv = ''       # a multi-track CSV: style_eval_songs of its validation songs instead (style_eval.csv_set)
style_eval_settings = 'track'  # sampling settings by name (style_eval.SETTINGS): 'track' (base) | 'ua_multi'
style_eval_interval = 0   # > 0: generate only every this many iterations (and at the end), 0 = at every evaluation
style_eval_chunk = 20     # songs generated at once (KV cache ~0.26 GB per song for the 450M at 2,048 notes)
sample_eval_rows = 0      # > 0: at every evaluation also generate from this many fixed multi-instrument prompts and
sample_eval_notes = 1000  # log sample metrics (sample_eval.py: instruments, out-of-band, density, takeover, ...)
eval_only = False
always_save_checkpoint = False
save_iters = ''            # e.g. '25,50,100': also keep bf16 weights at these iterations (out_dir/model_bf16_it<N>.pt),
                           # for judging fine-tune checkpoints by generation, not only by val CE
checkpoint_format = 'full'  # 'full' = fp32 weights + optimizer (resumable), 'bf16' = bf16 weights only (~6x smaller,
                            # not resumable; fine for generation and for init_from='finetune')
init_from = 'scratch'     # 'scratch' | 'resume' | 'finetune'
ckpt_interval_min = 0.0    # >0: also write a full resumable out_dir/ckpt.pt every N minutes and at the end (atomic)
save_pre_cooldown = True   # wsd: full resumable state to out_dir/pre_cooldown.pt just before the cooldown starts, so a
                           # later stage can continue the constant-LR phase on more data and do its own cooldown
init_ckpt = 'checkpoints/ckpt.pt'  # used by 'finetune'

# data
csv_path = 'data/combined.csv'
root_dir = '.'
cache_dir = 'data/cache'
val_csv_path = ''         # main val set (checkpoint selection); '' = same sources as csv_path
val2_csv_path = ''        # optional second val set, only logged as val2/* (e.g. keep the old set comparable)
val_name = ''             # names printed for the val sets ('' = from the source, e.g. 'gigamidi_clean'); wandb keeps val/, val2/
val2_name = ''
val_extra = ''            # more logged-only val sets: 'name=sources;name=sources' -> <name>/* (e.g. 'discover=store:data/cache/discover')
special_tokens = False    # BOS/EOS around every piece (pitch vocab 128 -> 130; a finetune grows a 128 checkpoint)
style_map = ''            # e.g. 'data/styles.json': conditioning on genre/artist (n_styles from the file, 0 = none)
style_lr_mult = 1.0       # Muon runs: LR multiplier for the style table (own AdamW group, no weight decay)
style_dropout = 0.1       # training: probability of dropping the style to 'none' (keeps an unconditional mode)
boundary_frac = 0.1       # with special_tokens: share of train windows placed at a piece's start/end
aug_tempo = 0.0           # training: time stretch up to x(1+aug_tempo) either way per window (0 = off)
aug_velocity = 0          # training: velocity shift up to ±N bins (of 32) per window, clamped (0 = off)
aug_stores = ''           # e.g. '1,0': which train stores get tempo/velocity aug (same order as source_weights); '' = all
source_weights = ''       # e.g. '0.8,0.2': sampling weight per train store (CSVs = one store, then each store:); '' = by size
batch_size = 6
block_size = 512
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
num_workers = 4

# model
n_layer = 6
n_head = 8
n_embd = 256
pitch_size = 128
velocity_size = 32
duration_size = 512    # max_duration_bin + 1
delta_time_size = 512  # max_delta_bin + 1
dropout = 0.3
bias = False
label_smoothing = 0.1
n_styles = 0              # set from style_map
pos_emb = 'learned'       # 'learned' (absolute table, old checkpoints) | 'rope' (rotary; a finetune may switch to it
                          # and then also raise block_size above the checkpoint's). A 'learned' finetune may raise
                          # block_size too: the table is stretched by linear interpolation
rope_base = 10000.0
norm = 'layernorm'        # trunk norm: 'layernorm' | 'rmsnorm'
mlp = 'gelu'              # trunk MLP: 'gelu' (4x) | 'swiglu' (8/3x, same params)
qk_norm = False           # RMSNorm on each head's q and k
cond_inst = False         # condition on the set of instruments in the window (multi-hot, zero-init, needs n_programs)
n_density = 0             # condition on the window's notes/s level (16 = 15 levels + none; 0 = off)
cond_dropout = 0.15       # training: share of windows whose conditions are replaced by 'none'
pad_short = False         # keep files shorter than block_size as one padded whole-file window (masked targets)
act_ckpt = 0              # activation checkpointing: first N transformer blocks recompute in backward (-1 = all)
act_ckpt_save = ''        # selective checkpointing: op outputs the checkpointed blocks keep ('attn' | 'attn,mm')
muon_bf16 = False         # Muon momentum buffer in bf16 (half the Muon optimizer state)
pack_short = False        # train only: short files fill their window with more short files (BOS..EOS each), no padding
min_notes = 64            # with pad_short / pack_short: shorter files are still dropped
n_programs = 0            # multi-instrument: 129 = GM programs + drums (needs a programs.u8 store or reads piano as 0)

# wandb logging
wandb_log = True
wandb_project = 'music-transformer'
wandb_run_name = 'maestro-v1'

# adamw optimizer
optimizer_name = 'adamw'   # 'adamw' | 'muon' (Muon for transformer-block matrices, AdamW for the rest; same lr/wd)
                           # | 'adamw8bit' (torchao 8-bit AdamW states: fits a 450M full fine-tune in 8 GB)
freeze = ''                # comma-separated parameter-name prefixes kept frozen (e.g. 'transformer.music_embeddings')
muon_momentum = 0.95
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate schedule
lr_schedule = 'cosine'    # 'cosine' | 'wsd' (warmup, constant, then 1-sqrt cooldown to min_lr over the last cooldown_frac)
cooldown_frac = 0.2
warmup_iters = 300
lr_decay_iters = 30000
min_lr = 6e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
fp8 = True                # torchao float8 training for the transformer matmuls; only pays off together with compile
fp8_recipe = 'tensorwise' # torchao recipe: 'tensorwise' | 'rowwise' | 'rowwise_with_gw_hp'
fp8_skip = ''             # comma-separated Linear name suffixes kept in bf16, e.g. 'attn.c_proj'
fp8_cache_weights = False # tensorwise: cast each weight to fp8 once per optimizer step, not in every micro-batch pass
oom_retries = 2           # CUDA OOM in a step: 1st retry = free the cache and redo it; 2nd = redo it eager with the
                          # activations spilled to pinned RAM above oom_spill_frac of the GPU (slow, but it gets through)
oom_spill_frac = 0.6
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('musicar/configurator.py').read())  # overrides from command line or config file
styles = load_styles(style_map) if style_map else None
if styles:
    n_styles = len(styles)
if special_tokens:
    pitch_size = max(pitch_size, 130)
config = {k: globals()[k] for k in config_keys}  # logged to wandb and saved in checkpoints
sys.stdout.reconfigure(line_buffering=True)  # eval lines reach a redirected log at once, not when an 8 KB buffer fills


def as_list(value, cast):
    """The configurator turns '--x=1,2' into a tuple and a single value into a number; '' = None."""
    items = [cast(v) for v in str(value).strip('()[] ').split(',') if v.strip()]
    return items or None


# -----------------------------------------------------------------------------
print(f"Device: {device}")
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(seed)
random.seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data
extra_names = (['program'] if n_programs else []) + (['style'] if styles else [])  # optional inputs, in item order
data_kw = dict(root_dir=root_dir, cache_dir=cache_dir, special_tokens=special_tokens, styles=styles,
               programs=n_programs > 0, pad_short=pad_short, min_notes=min_notes)
# train: random windows (uniform over all note positions of all files), with augmentation
train_dataset = MaestroDataset(csv_path, split='train', block_size=block_size, augment=True, **data_kw,
                               style_dropout=style_dropout, boundary_frac=boundary_frac, aug_tempo=aug_tempo,
                               aug_velocity=aug_velocity, pack_short=pack_short,
                               aug_stores=as_list(aug_stores, int), source_weights=as_list(source_weights, float))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                          collate_fn=partial(collate_windows, extra_names=tuple(extra_names)),
                          persistent_workers=num_workers > 0)
make_val_loader = partial(val_loader, batch_size=batch_size, max_batches=eval_iters, num_workers=num_workers,
                          extra_names=extra_names, block_size=block_size, **data_kw)
val_name = val_name or source_name(val_csv_path or csv_path)
val2_name = val2_name or (source_name(val2_csv_path) if val2_csv_path else '')
val_loader_main = make_val_loader(val_csv_path or csv_path)
val2_loader = make_val_loader(val2_csv_path) if val2_csv_path else None
extra_loaders = {name: make_val_loader(src) for name, src in (e.split('=', 1) for e in val_extra.split(';') if e)}
print(f"Validation: {val_name} (main, picks the best checkpoint)" + (f", {val2_name}" if val2_name else "")
      + "".join(f", {name}" for name in extra_loaders))

# -----------------------------------------------------------------------------
# model
iter_num = 0
best_val_loss = float('inf')
wandb_run_id = None
model_args = {k: globals()[k] for k in ARCH_KEYS}
checkpoint = None
if init_from in ('resume', 'finetune'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt') if init_from == 'resume' else init_ckpt
    print(f"{init_from}: loading {ckpt_path}")
    # to CPU: the weights are copied into the model below, and a GPU copy would stay alive through `state_dict`
    # (~0.9 GB for the 450M bf16 base, the margin between fitting and OOM at micro 1 on the 8 GB laptop)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = merge_model_args(checkpoint['model_args'], model_args, init_from == 'finetune', block_size)
else:
    print("Initializing a new model from scratch")

model = GPT(MusicConfig(**model_args, dropout=dropout, label_smoothing=label_smoothing))

if checkpoint is not None:
    state_dict = strip_compile_prefix(checkpoint['model'])
    if init_from == 'finetune':
        grown = load_expanded(model, state_dict)
        if grown:
            print(f"finetune: grown / new parameters (fresh init): {', '.join(grown)}")
    else:
        model.load_state_dict(state_dict)
    state_dict = None
    if init_from == 'resume':
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        wandb_run_id = checkpoint.get('wandb_run_id')
        # different data windows after a restart instead of replaying the ones from the start
        torch.manual_seed(seed + iter_num)
        random.seed(seed + iter_num)

# the window conditions follow the model, not the config: a finetune of a conditioned base (long_450m) left them
# at the config default (off), so it trained and evaluated only the 'none' condition (ua_450_s1, 2026-10-06)
if (cond_inst, n_density) != (model_args['cond_inst'], model_args['n_density']):
    print(f"conditioning from the checkpoint: cond_inst={model_args['cond_inst']}, n_density={model_args['n_density']}")
    cond_inst, n_density = model_args['cond_inst'], model_args['n_density']
    config.update(cond_inst=cond_inst, n_density=n_density)
assert not cond_inst or n_programs == model_args['n_programs'], \
    f"cond_inst: set n_programs={model_args['n_programs']} (the model's instrument vocabulary)"

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)
model.act_ckpt = act_ckpt
model.act_ckpt_save = act_ckpt_save

# before the optimizer is built: fp8 conversion swaps the Linear modules
if fp8 and not fp8_supported(device_type):
    print("fp8 needs an sm_89+ GPU, falling back to bf16")
    fp8 = False
if fp8:
    # only the transformer blocks: embeddings, heads and LayerNorms stay bf16/fp32
    convert_fp8(model.transformer.h, fp8_recipe, fp8_skip, cache_weights=fp8_cache_weights)
    refresh_fp8_weights(model)  # no-op without fp8_cache_weights
    if not compile:
        print("WARNING: fp8 without compile is about 2x SLOWER (unfused scaling kernels)")
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
if freeze:
    freeze_params(model, freeze)
optimizer = build_optimizer(optimizer_name, model, weight_decay, learning_rate, (beta1, beta2), device_type,
                            style_lr_mult=style_lr_mult, muon_momentum=muon_momentum, muon_bf16=muon_bf16,
                            compile_ns=compile)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

raw_model = model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)


def to_device(batch):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def prepare(X, Y, cond_drop_p):
    """A batch on the device; cond_inst / n_density models also get the window conditions of its own notes,
    replaced by 'none' with probability cond_drop_p per window."""
    streams, extra = to_device(X[0]), {k: v.to(device, non_blocking=True) for k, v in X[1].items()}
    if cond_inst or n_density:
        extra['cond'] = window_conditions(streams[0], streams[3], extra.get('program'), n_programs, n_density,
                                          drop_p=cond_drop_p)
    return (streams, extra), to_device(Y)


def infinite_batches(loader):
    while True:
        for X, Y in loader:
            yield prepare(X, Y, cond_dropout)


def val_loss(loader, conditioned=True):
    """conditioned=False (conditioning models): every window gets 'none', i.e. the unconditional model."""
    return estimate_val_loss(model, (prepare(X, Y, 0.0 if conditioned else 1.0) for X, Y in loader), ctx)


if wandb_log:
    import wandb
    if init_from == 'resume' and wandb_run_id:
        wandb.init(project=wandb_project, id=wandb_run_id, resume='allow', config=config)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb_run_id = wandb.run.id
    # x-axis = training iteration (wandb's own 'Step' only counts our log calls, i.e. evaluations)
    wandb.define_metric('iter')
    wandb.define_metric('*', step_metric='iter')
else:
    wandb_run_id = None

save_iter_set = set(as_list(save_iters, int) or ())
cooldown_iter = cooldown_start(lr_decay_iters, cooldown_frac)  # wsd only


def save_resumable(it, name='ckpt.pt'):
    """Full state to out_dir/<name>; `it` = the next iteration to run."""
    path = os.path.join(out_dir, name)
    save_atomic({'model_args': model_args, 'iter_num': it, 'best_val_loss': best_val_loss, 'config': config,
                 'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'wandb_run_id': wandb_run_id}, path)
    tqdm.write(f"saved resumable checkpoint at iter {it} to {path}")


def save_best(it):
    """The best-val checkpoint: bf16 weights (model_bf16.pt) or the full state (ckpt.pt; best.pt when ckpt.pt is the
    periodic resume state, so it isn't rolled back to an older iteration)."""
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {'model_args': model_args, 'iter_num': it, 'best_val_loss': best_val_loss, 'config': config,
            'wandb_run_id': wandb_run_id}
    if checkpoint_format == 'bf16':
        ckpt['model'] = bf16_cpu_state(raw_model)
        name = 'model_bf16.pt'
    else:
        ckpt['model'] = raw_model.state_dict()
        ckpt['optimizer'] = optimizer.state_dict()
        name = 'best.pt' if ckpt_interval_min > 0 else 'ckpt.pt'
    torch.save(ckpt, os.path.join(out_dir, name))
    tqdm.write(f"saved checkpoint to {os.path.join(out_dir, name)}")


n_params = sum(p.numel() for p in raw_model.parameters())
n_train_notes = train_dataset.num_notes()
print(f"Model parameters : {n_params:,} ({n_params / 1e6:.2f}M)")
print(f"Train files      : {train_dataset.n_files:,}  ({n_train_notes:,} notes)")
print(f"Val windows      : {len(val_loader_main.dataset):,}  ({len(val_loader_main):,} batches)")
print(f"Tokens per iter  : {tokens_per_iter:,}")
print(f"Planned epochs   : {tokens_per_iter * (max_iters - iter_num) / n_train_notes:.1f} (notes seen / unique notes)")

# -----------------------------------------------------------------------------
# generation checks at evaluation time
if sample_eval_rows and n_programs:
    from musicar.eval import sample_eval
    sample_prompts = sample_eval.fixed_prompts(sample_eval_rows)
if style_eval_songs and n_programs:
    from musicar.eval import style_eval
    style_songs = style_eval.csv_set(style_eval_csv, style_eval_songs) if style_eval_csv else \
        style_eval.style_set(style_eval_songs)
    print('Style eval: ' + ', '.join(f'{s} {len(v)}' for s, v in style_songs.items()) + ' songs')


def run_style_eval(it):
    """Per-style distances of generated continuations ({} when off or failed: a failure never stops the run)."""
    if not (style_eval_songs and n_programs and device_type == 'cuda' and (
            not style_eval_interval or it % style_eval_interval == 0 or it == max_iters - 1)):
        return {}
    t_s = time.time()
    try:
        res = style_eval.evaluate(raw_model, style_songs, settings=style_eval.SETTINGS[style_eval_settings],
                                  chunk=style_eval_chunk)
        tqdm.write(style_eval.format_line(it, sum(len(v) for v in style_songs.values()), time.time() - t_s, res))
    except Exception as e:
        res = {}
        tqdm.write(f"styles @{it}: FAILED ({type(e).__name__}: {str(e)[:200]}), training continues")
    # give training back the generation's memory (cuBLAS stream workspaces, freed cache blocks)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    return res


def run_sample_eval(it):
    """Sample metrics on fixed multi-instrument prompts ({} when off or failed)."""
    if not (sample_eval_rows and n_programs and device_type == 'cuda'):
        return {}
    t_s = time.time()
    try:  # a failure here (e.g. OOM) must never stop the run: log it and train on
        res = sample_eval.evaluate(raw_model, sample_prompts, sample_eval_notes)
    except Exception as e:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        tqdm.write(f"samples @{it}: FAILED ({type(e).__name__}: {str(e)[:200]}), training continues")
        return {}
    tqdm.write(sample_eval.format_line(it, sample_eval_rows, time.time() - t_s, res))
    return res


# -----------------------------------------------------------------------------
# training loop
batches = infinite_batches(train_loader)
X, Y = next(batches)
train_sums, train_n = {}, 0  # running train metrics since the last eval (with dropout + augmentation)
oom_events = 0  # CUDA OOMs survived by retrying the step
stop_signal = []  # SIGTERM / SIGINT: finish the current step, save a resumable checkpoint, exit 143


def request_stop(signum, frame):
    if not stop_signal:
        tqdm.write(f"got signal {signum}: saving a resumable checkpoint after this step, then exiting")
    stop_signal.append(signum)


def accumulate(X, Y, net):
    """Forward + backward over gradient_accumulation_steps micro-batches; returns the next batch and the metrics."""
    sums = {}
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            parts, loss = net(*X[0], **X[1], targets=Y)
        # kept as GPU tensors to avoid a CPU sync per micro step
        sums['loss'] = sums.get('loss', 0.0) + loss.detach() / gradient_accumulation_steps
        for name, v in parts.items():
            sums[f'ce/{name}'] = sums.get(f'ce/{name}', 0.0) + v / gradient_accumulation_steps
        X, Y = next(batches)
        scaler.scale(loss / gradient_accumulation_steps).backward()
    return X, Y, sums


if ckpt_interval_min > 0:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
t0 = time.time()
last_resumable = time.time()

pbar = tqdm(range(iter_num, max_iters), desc="Training", initial=iter_num, total=max_iters)
for iter_num in pbar:
    lr = lr_at(iter_num, lr_schedule, learning_rate, min_lr, warmup_iters, lr_decay_iters, cooldown_frac)
    set_lr(optimizer, lr)

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        val = val_loss(val_loader_main)
        val2 = val_loss(val2_loader) if val2_loader else {}
        extra = {name: val_loss(loader) for name, loader in extra_loaders.items()}
        if cond_inst or n_density:  # the main val set again with no conditions: the model's unconditional quality
            extra['nocond'] = val_loss(val_loader_main, conditioned=False)
        style_res = run_style_eval(iter_num)
        samples = run_sample_eval(iter_num)
        train = {k: (v / train_n).item() for k, v in train_sums.items()} if train_n else {}
        if train:
            summed_ce(train)
        heads = " ".join(f"{name[:3]} {val[f'ce/{name}']:.3f}" for name in STREAM_ORDER)
        tqdm.write(f"step {iter_num}: train loss {train.get('loss', float('nan')):.4f}, "
                   f"val loss {val['loss']:.4f} | {val_name} CE {val['ce/total']:.3f} ({heads})"
                   + (f" | {val2_name} CE {val2['ce/total']:.3f}" if val2 else "")
                   # instrument head last, so status.py's STEP pattern still matches the line
                   + (f" | prog CE {val['ce/program']:.3f}" if 'ce/program' in val else "")
                   + "".join(f" | {name} CE {e['ce/total']:.3f}" for name, e in extra.items()))
        if wandb_log:
            wandb.log({"iter": iter_num, "lr": lr,
                       **{f"val/{k}": v for k, v in val.items()},
                       **{f"val2/{k}": v for k, v in val2.items()},
                       **{f"{name}/{k}": v for name, e in extra.items() for k, v in e.items()},
                       **{f"samples/{mode}/{k}": v for mode, m in samples.items() for k, v in m.items()},
                       **{f"styles/{mode}/{k}": v for mode, m in style_res.items() for k, v in m.items()},
                       **{f"train/{k}": v for k, v in train.items()}})
        train_sums, train_n = {}, 0

        if val['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = val['loss']
            if iter_num > 0:
                save_best(iter_num)
    if iter_num in save_iter_set:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({'model_args': model_args, 'iter_num': iter_num, 'config': config,
                    'model': bf16_cpu_state(raw_model)},
                   os.path.join(out_dir, f'model_bf16_it{iter_num}.pt'))
        tqdm.write(f"saved weights of iter {iter_num} to {out_dir}/model_bf16_it{iter_num}.pt")
    if iter_num == 0 and eval_only:
        break
    if save_pre_cooldown and lr_schedule == 'wsd' and iter_num == cooldown_iter and iter_num > 0:
        # the weights + optimizer after the last constant-LR step; continue from it with init_from='resume' after
        # copying it to <new out_dir>/ckpt.pt (data, max_iters and lr_decay_iters come from the new config)
        save_resumable(iter_num, 'pre_cooldown.pt')

    # forward backward update, with gradient accumulation; a CUDA OOM retries the step (see oom_retries)
    for attempt in range(oom_retries + 1):
        spill = attempt >= 2 and device_type == 'cuda'
        oom = False
        try:
            with spill_to_cpu(oom_spill_frac * torch.cuda.get_device_properties(device).total_memory) if spill \
                    else nullcontext():
                X, Y, sums = accumulate(X, Y, raw_model if spill else model)
        except torch.OutOfMemoryError:
            if attempt == oom_retries:
                raise
            oom = True
        if not oom:
            break
        # outside the except block, so the traceback no longer holds the failed step's tensors
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        oom_events += 1
        tqdm.write(f"iter {iter_num}: CUDA OOM ({oom_events} so far), retry {attempt + 1}/{oom_retries}"
                   + (" eager with activations spilled to RAM" if attempt + 1 >= 2 else " after freeing the cache"))
        X, Y = next(batches)
    for k, v in sums.items():
        train_sums[k] = train_sums.get(k, 0.0) + v
    train_n += 1
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if fp8 and fp8_cache_weights:
        refresh_fp8_weights(raw_model)  # the cached fp8 weights must follow every update

    if ckpt_interval_min > 0 and (time.time() - last_resumable > 60 * ckpt_interval_min or iter_num == max_iters - 1
                                  or stop_signal):
        save_resumable(iter_num + 1)
        last_resumable = time.time()
        if stop_signal:
            if wandb_log:
                wandb.finish(exit_code=143)
            sys.exit(143)

    t1 = time.time()
    pbar.set_postfix(loss=f"{train_sums['loss'] / train_n:.3f}", lr=f"{lr:.1e}", ms=f"{(t1 - t0) * 1000:.0f}",
                     **({'gb': f"{torch.cuda.max_memory_allocated() / 2**30:.2f}"} if device_type == 'cuda' else {}))
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = t1
