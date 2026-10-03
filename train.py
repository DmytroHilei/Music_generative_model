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
import time
import math
import random
import sys
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import MusicConfig, GPT, STREAM_ORDER
from data_loader import MaestroDataset

# -----------------------------------------------------------------------------
# I/O
out_dir = 'checkpoints'
eval_interval = 20
eval_iters = 200          # max number of val batches per evaluation (fixed windows, same every time)
eval_only = False
always_save_checkpoint = False
checkpoint_format = 'full'  # 'full' = fp32 weights + optimizer (resumable), 'bf16' = bf16 weights only (~6x smaller,
                            # not resumable; fine for generation and for init_from='finetune')
init_from = 'scratch'     # 'scratch' | 'resume' | 'finetune'
ckpt_interval_min = 0.0    # >0: also write a full resumable out_dir/ckpt.pt every N minutes and at the end (atomic)
init_ckpt = 'checkpoints/ckpt.pt'  # used by 'finetune'

# data
csv_path = 'data/combined.csv'
root_dir = '.'
cache_dir = 'data/cache'
val_csv_path = ''         # main val set (checkpoint selection); '' = same sources as csv_path
val2_csv_path = ''        # optional second val set, only logged as val2/* (e.g. keep the old set comparable)
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
cascade_heads = True
cascade_residual = True
pitch_head_blocks = 1
pitch_head_mult = 1
moe_experts = 0
moe_top_k = 2
moe_hidden_frac = 0.5
moe_aux_weight = 0.01
n_styles = 0              # set from style_map
pos_emb = 'learned'       # 'learned' (absolute table, old checkpoints) | 'rope' (rotary; a finetune may switch to it
                          # and then also raise block_size above the checkpoint's). A 'learned' finetune may raise
                          # block_size too: the table is stretched by linear interpolation
rope_base = 10000.0
pad_short = False         # keep files shorter than block_size as one padded whole-file window (masked targets)
act_ckpt = 0              # activation checkpointing: first N transformer blocks recompute in backward (-1 = all)
muon_bf16 = False         # Muon momentum buffer in bf16 (half the Muon optimizer state)
pack_short = False        # train only: short files fill their window with more short files (BOS..EOS each), no padding
min_notes = 64            # with pad_short / pack_short: shorter files are still dropped
future_weight = 0.0       # > 0: auxiliary future-prediction heads (pitch classes / density / register 0-2-4-8 s ahead)
future_horizons = '0-2,2-4,4-8'
n_programs = 0            # multi-instrument: 129 = GM programs + drums (needs a programs.u8 store or reads piano as 0)

# wandb logging
wandb_log = True
wandb_project = 'music-transformer'
wandb_run_name = 'maestro-v1'

# adamw optimizer
optimizer_name = 'adamw'   # 'adamw' | 'muon' (Muon for transformer-block matrices, AdamW for the rest; same lr/wd)
muon_momentum = 0.95
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
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
sdpa_backend = ''         # '' = PyTorch default, or 'flash' | 'cudnn' | 'efficient'
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
from data_loader import load_styles
styles = load_styles(style_map) if style_map else None
if styles:
    n_styles = len(styles)
if special_tokens:
    pitch_size = max(pitch_size, 130)
config = {k: globals()[k] for k in config_keys}  # logged to wandb and saved in checkpoints
sys.stdout.reconfigure(line_buffering=True)  # eval lines reach a redirected log at once, not when an 8 KB buffer fills
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


def collate_fn(batch):
    # X = (4 note streams, dict of the optional model inputs), Y = 4 streams (+ program)
    xs, ys = zip(*batch)
    x = [torch.stack([x[i] for x in xs]) for i in range(len(xs[0]))]
    extra_names = (['program'] if n_programs else []) + (['style'] if styles else [])
    assert len(x) == 4 + len(extra_names), f"{len(x)} input streams, expected 4 + {extra_names}"
    return (tuple(x[:4]), dict(zip(extra_names, x[4:]))), tuple(torch.stack([y[i] for y in ys]) for i in range(len(ys[0])))


# train: random windows (uniform over all note positions of all files), with augmentation
train_dataset = MaestroDataset(csv_path, root_dir=root_dir, split='train', block_size=block_size,
                               augment=True, cache_dir=cache_dir, special_tokens=special_tokens,
                               styles=styles, style_dropout=style_dropout, boundary_frac=boundary_frac,
                               aug_tempo=aug_tempo, aug_velocity=aug_velocity, programs=n_programs > 0,
                               pad_short=pad_short, pack_short=pack_short, min_notes=min_notes,
                               aug_stores=[int(a) for a in str(aug_stores).strip('()[] ').split(',') if a.strip()] if aug_stores else None,
                               source_weights=[float(w) for w in str(source_weights).strip('()[] ').split(',') if w.strip()] if source_weights else None)


def make_val_loader(sources):
    # fixed non-overlapping windows, evenly thinned to at most eval_iters batches -> identical every eval / run
    ds = MaestroDataset(sources, root_dir=root_dir, split='validation', block_size=block_size,
                        cache_dir=cache_dir, eval_stride=block_size, special_tokens=special_tokens, styles=styles,
                        programs=n_programs > 0, pad_short=pad_short, min_notes=min_notes)
    max_val_windows = eval_iters * batch_size
    if len(ds) > max_val_windows:
        step = len(ds) / max_val_windows
        ds = Subset(ds, [int(i * step) for i in range(max_val_windows)])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)


val_loader = make_val_loader(val_csv_path or csv_path)
val2_loader = make_val_loader(val2_csv_path) if val2_csv_path else None
val_dataset = val_loader.dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)

# -----------------------------------------------------------------------------
# model
iter_num = 0
best_val_loss = float('inf')
wandb_run_id = None
arch_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'pitch_size', 'velocity_size',
             'duration_size', 'delta_time_size', 'cascade_heads', 'cascade_residual',
             'pitch_head_blocks', 'pitch_head_mult', 'moe_experts', 'moe_top_k', 'moe_hidden_frac', 'moe_aux_weight',
             'n_styles', 'n_programs', 'pos_emb', 'rope_base', 'future_weight', 'future_horizons']
# what checkpoints that predate a key actually used
legacy_defaults = {'cascade_heads': False, 'cascade_residual': False, 'n_styles': 0, 'n_programs': 0,
                   'pos_emb': 'learned', 'rope_base': 10000.0, 'future_weight': 0.0, 'future_horizons': '0-2,2-4,4-8'}
model_args = {k: globals()[k] for k in arch_keys}
checkpoint = None

if init_from in ('resume', 'finetune'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt') if init_from == 'resume' else init_ckpt
    print(f"{init_from}: loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    for k in arch_keys:
        wanted = model_args[k]
        model_args[k] = checkpoint['model_args'].get(k, legacy_defaults.get(k, model_args[k]))
        if init_from == 'finetune' and k in ('pitch_size', 'n_styles', 'n_programs') and wanted > model_args[k]:
            model_args[k] = wanted  # grow: BOS/EOS rows, a style table, the instrument attribute
        if init_from == 'finetune' and k == 'pos_emb' and wanted == 'rope':
            model_args[k] = wanted  # switch to RoPE: the position table is dropped (not function-preserving)
        if init_from == 'finetune' and k in ('future_weight', 'future_horizons'):
            model_args[k] = wanted  # a training-time choice: the heads are added or dropped per run
    if init_from == 'finetune' and model_args['pos_emb'] == 'rope':
        model_args['block_size'] = block_size  # nothing in a RoPE model depends on the context length
    elif init_from == 'finetune' and block_size > model_args['block_size']:
        model_args['block_size'] = block_size  # learned positions: the table is stretched (position interpolation)
else:
    print("Initializing a new model from scratch")

model = GPT(MusicConfig(**model_args, dropout=dropout, label_smoothing=label_smoothing))

if checkpoint is not None:
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    if init_from == 'finetune':
        grown = model.load_expanded(state_dict)
        if grown:
            print(f"finetune: grown / new parameters (fresh init): {', '.join(grown)}")
    else:
        model.load_state_dict(state_dict)
    if init_from == 'resume':
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        wandb_run_id = checkpoint.get('wandb_run_id')
        # different data windows after a restart instead of replaying the ones from the start
        torch.manual_seed(seed + iter_num)
        random.seed(seed + iter_num)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)
model.act_ckpt = act_ckpt

# before the optimizer is built: fp8 conversion swaps the Linear modules
if fp8 and not (device_type == 'cuda' and torch.cuda.get_device_capability() >= (8, 9)):
    print("fp8 needs an sm_89+ GPU, falling back to bf16")
    fp8 = False
if fp8:
    from torchao.float8 import convert_to_float8_training
    # only the transformer blocks: embeddings, heads and LayerNorms stay bf16/fp32
    convert_to_float8_training(model.transformer.h)
    if not compile:
        print("WARNING: fp8 without compile is about 2x SLOWER (unfused scaling kernels)")
if sdpa_backend:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {'flash': SDPBackend.FLASH_ATTENTION, 'cudnn': SDPBackend.CUDNN_ATTENTION,
               'efficient': SDPBackend.EFFICIENT_ATTENTION}[sdpa_backend]
    sdpa_kernel(backend).__enter__()  # for the whole process
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
if optimizer_name == 'muon':
    from optim import build_muon_optimizer
    optimizer = build_muon_optimizer(model, weight_decay, learning_rate, (beta1, beta2), device_type,
                                     momentum=muon_momentum, style_lr_mult=style_lr_mult,
                                     momentum_dtype=torch.bfloat16 if muon_bf16 else None)
else:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

raw_model = model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)


def to_device(batch):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def to_device_x(X):
    streams, extra = X
    return to_device(streams), {k: v.to(device, non_blocking=True) for k, v in extra.items()}


def infinite_batches(loader):
    while True:
        for X, Y in loader:
            yield to_device_x(X), to_device(Y)


def summed_ce(metrics):
    # ce/total = the 4 note attributes (comparable with the piano runs); ce/total_all also counts the instrument head
    metrics['ce/total'] = sum(metrics[f'ce/{name}'] for name in STREAM_ORDER)
    if 'ce/program' in metrics:
        metrics['ce/total_all'] = metrics['ce/total'] + metrics['ce/program']


@torch.no_grad()
def estimate_val_loss(loader):
    """Returns dict: 'loss' (optimized loss, with label smoothing) and 'ce/<head>' (true CE, nats)."""
    model.eval()
    sums, n = {}, 0
    for X, Y in loader:
        (streams, extra), Y = to_device_x(X), to_device(Y)
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


def get_lr(it):
    # 1) linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if lr_schedule == 'wsd':
        # Hägele et al. 2024: constant LR, then a (1 - sqrt) cooldown; matches or beats cosine at equal compute
        cooldown_start = lr_decay_iters - int(cooldown_frac * lr_decay_iters)
        if it < cooldown_start:
            return learning_rate
        progress = min(1.0, (it - cooldown_start) / max(1, lr_decay_iters - cooldown_start))
        return min_lr + (1 - math.sqrt(progress)) * (learning_rate - min_lr)
    # 2) after decay, min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log:
    import wandb
    if init_from == 'resume' and wandb_run_id:
        wandb.init(project=wandb_project, id=wandb_run_id, resume='allow', config=config)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb_run_id = wandb.run.id
else:
    wandb_run_id = None


def save_resumable(it):
    """Full state to out_dir/ckpt.pt; `it` = the next iteration to run. tmp + rename, so a crash mid-save keeps the old file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'ckpt.pt')
    torch.save({'model_args': model_args, 'iter_num': it, 'best_val_loss': best_val_loss, 'config': config,
                'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'wandb_run_id': wandb_run_id}, path + '.tmp')
    os.replace(path + '.tmp', path)
    tqdm.write(f"saved resumable checkpoint at iter {it} to {path}")

n_params = sum(p.numel() for p in raw_model.parameters())
n_train_notes = train_dataset.num_notes()
print(f"Model parameters : {n_params:,} ({n_params / 1e6:.2f}M)")
print(f"Train files      : {train_dataset.n_files:,}  ({n_train_notes:,} notes)")
print(f"Val windows      : {len(val_dataset):,}  ({len(val_loader):,} batches)")
print(f"Tokens per iter  : {tokens_per_iter:,}")
print(f"Planned epochs   : {tokens_per_iter * (max_iters - iter_num) / n_train_notes:.1f} (notes seen / unique notes)")

# -----------------------------------------------------------------------------
# training loop
batches = infinite_batches(train_loader)
X, Y = next(batches)
train_sums, train_n = {}, 0  # running train metrics since the last eval (with dropout + augmentation)
t0 = time.time()
last_resumable = time.time()

pbar = tqdm(range(iter_num, max_iters), desc="Training", initial=iter_num, total=max_iters)
for iter_num in pbar:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        val = estimate_val_loss(val_loader)
        val2 = estimate_val_loss(val2_loader) if val2_loader else {}
        train = {k: (v / train_n).item() for k, v in train_sums.items()} if train_n else {}
        if train:
            summed_ce(train)
        heads = " ".join(f"{name[:3]} {val[f'ce/{name}']:.3f}" for name in STREAM_ORDER)
        tqdm.write(f"step {iter_num}: train loss {train.get('loss', float('nan')):.4f}, "
                   f"val loss {val['loss']:.4f} | val CE {val['ce/total']:.3f} ({heads})"
                   + (f" | val2 CE {val2['ce/total']:.3f}" if val2 else "")
                   + (f" | future pc CE {val['ce/future_pc']:.3f}" if 'ce/future_pc' in val else "")
                   # instrument head last, so status.py's STEP pattern still matches the line
                   + (f" | prog CE {val['ce/program']:.3f}" if 'ce/program' in val else ""))
        if wandb_log:
            wandb.log({"iter": iter_num, "lr": lr,
                       **{f"val/{k}": v for k, v in val.items()},
                       **{f"val2/{k}": v for k, v in val2.items()},
                       **{f"train/{k}": v for k, v in train.items()}})
        train_sums, train_n = {}, 0

        if val['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = val['loss']
            if iter_num > 0:
                os.makedirs(out_dir, exist_ok=True)
                ckpt = {'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss,
                        'config': config, 'wandb_run_id': wandb_run_id}
                if checkpoint_format == 'bf16':
                    ckpt['model'] = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                                     for k, v in raw_model.state_dict().items()}
                    name = 'model_bf16.pt'
                else:
                    ckpt['model'] = raw_model.state_dict()
                    ckpt['optimizer'] = optimizer.state_dict()
                    name = 'ckpt.pt'
                if name == 'ckpt.pt' and ckpt_interval_min > 0:
                    name = 'best.pt'  # ckpt.pt is the periodic resume state, don't roll it back to an older iter
                torch.save(ckpt, os.path.join(out_dir, name))
                tqdm.write(f"saved checkpoint to {os.path.join(out_dir, name)}")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            parts, loss = model(*X[0], **X[1], targets=Y)
        # kept as GPU tensors to avoid a CPU sync per micro step
        train_sums['loss'] = train_sums.get('loss', 0.0) + loss.detach() / gradient_accumulation_steps
        for name, v in parts.items():
            train_sums[f'ce/{name}'] = train_sums.get(f'ce/{name}', 0.0) + v / gradient_accumulation_steps
        X, Y = next(batches)
        scaler.scale(loss / gradient_accumulation_steps).backward()
    train_n += 1
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if ckpt_interval_min > 0 and (time.time() - last_resumable > 60 * ckpt_interval_min or iter_num == max_iters - 1):
        save_resumable(iter_num + 1)
        last_resumable = time.time()

    t1 = time.time()
    pbar.set_postfix(loss=f"{train_sums['loss'] / train_n:.3f}", lr=f"{lr:.1e}", ms=f"{(t1 - t0) * 1000:.0f}")
    t0 = t1
