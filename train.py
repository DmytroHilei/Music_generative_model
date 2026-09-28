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
init_from = 'scratch'     # 'scratch' | 'resume' | 'finetune'
init_ckpt = 'checkpoints/ckpt.pt'  # used by 'finetune'

# data
csv_path = 'data/combined.csv'
root_dir = '.'
cache_dir = 'data/cache'
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

# wandb logging
wandb_log = True
wandb_project = 'music-transformer'
wandb_run_name = 'maestro-v1'

# adamw optimizer
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 300
lr_decay_iters = 30000
min_lr = 6e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # logged to wandb and saved in checkpoints
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
    xs, ys = zip(*batch)
    return (
        tuple(torch.stack([x[i] for x in xs]) for i in range(4)),
        tuple(torch.stack([y[i] for y in ys]) for i in range(4)),
    )


# train: one random window per file per "epoch", with augmentation
train_dataset = MaestroDataset(csv_path, root_dir=root_dir, split='train', block_size=block_size,
                               augment=True, cache_dir=cache_dir)
# val: fixed non-overlapping windows, evenly thinned to at most eval_iters batches -> identical every eval / run
val_dataset = MaestroDataset(csv_path, root_dir=root_dir, split='validation', block_size=block_size,
                             cache_dir=cache_dir, eval_stride=block_size)
max_val_windows = eval_iters * batch_size
if len(val_dataset) > max_val_windows:
    step = len(val_dataset) / max_val_windows
    val_dataset = Subset(val_dataset, [int(i * step) for i in range(max_val_windows)])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=num_workers)

# -----------------------------------------------------------------------------
# model
iter_num = 0
best_val_loss = float('inf')
arch_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'pitch_size', 'velocity_size',
             'duration_size', 'delta_time_size', 'cascade_heads', 'cascade_residual']
# what checkpoints that predate a key actually used
legacy_defaults = {'cascade_heads': False, 'cascade_residual': False}
model_args = {k: globals()[k] for k in arch_keys}
checkpoint = None

if init_from in ('resume', 'finetune'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt') if init_from == 'resume' else init_ckpt
    print(f"{init_from}: loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    for k in arch_keys:
        model_args[k] = checkpoint['model_args'].get(k, legacy_defaults.get(k, model_args[k]))
else:
    print("Initializing a new model from scratch")

model = GPT(MusicConfig(**model_args, dropout=dropout, label_smoothing=label_smoothing))

if checkpoint is not None:
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if init_from == 'resume':
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
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


def infinite_batches(loader):
    while True:
        for X, Y in loader:
            yield to_device(X), to_device(Y)


@torch.no_grad()
def estimate_val_loss():
    """Returns dict: 'loss' (optimized loss, with label smoothing) and 'ce/<head>' (true CE, nats)."""
    model.eval()
    sums, n = {}, 0
    for X, Y in val_loader:
        X, Y = to_device(X), to_device(Y)
        with ctx:
            parts, loss = model(*X, targets=Y)
        sums['loss'] = sums.get('loss', 0.0) + loss.item()
        for name, v in parts.items():
            sums[f'ce/{name}'] = sums.get(f'ce/{name}', 0.0) + v.item()
        n += 1
    model.train()
    out = {k: v / n for k, v in sums.items()}
    out['ce/total'] = sum(out[f'ce/{name}'] for name in STREAM_ORDER)
    return out


def get_lr(it):
    # 1) linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) after decay, min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

n_params = sum(p.numel() for p in raw_model.parameters())
n_train_notes = train_dataset.num_notes()
print(f"Model parameters : {n_params:,} ({n_params / 1e6:.2f}M)")
print(f"Train files      : {len(train_dataset):,}  ({n_train_notes:,} notes)")
print(f"Val windows      : {len(val_dataset):,}  ({len(val_loader):,} batches)")
print(f"Tokens per iter  : {tokens_per_iter:,}")
print(f"Planned epochs   : {tokens_per_iter * (max_iters - iter_num) / n_train_notes:.1f} (notes seen / unique notes)")

# -----------------------------------------------------------------------------
# training loop
batches = infinite_batches(train_loader)
X, Y = next(batches)
train_sums, train_n = {}, 0  # running train metrics since the last eval (with dropout + augmentation)
t0 = time.time()

pbar = tqdm(range(iter_num, max_iters), desc="Training", initial=iter_num, total=max_iters)
for iter_num in pbar:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        val = estimate_val_loss()
        train = {k: (v / train_n).item() for k, v in train_sums.items()} if train_n else {}
        if train:
            train['ce/total'] = sum(train[f'ce/{name}'] for name in STREAM_ORDER)
        heads = " ".join(f"{name[:3]} {val[f'ce/{name}']:.3f}" for name in STREAM_ORDER)
        tqdm.write(f"step {iter_num}: train loss {train.get('loss', float('nan')):.4f}, "
                   f"val loss {val['loss']:.4f} | val CE {val['ce/total']:.3f} ({heads})")
        if wandb_log:
            wandb.log({"iter": iter_num, "lr": lr,
                       **{f"val/{k}": v for k, v in val.items()},
                       **{f"train/{k}": v for k, v in train.items()}})
        train_sums, train_n = {}, 0

        if val['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = val['loss']
            if iter_num > 0:
                os.makedirs(out_dir, exist_ok=True)
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join(out_dir, 'ckpt.pt'))
                tqdm.write(f"saved checkpoint to {out_dir}")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            parts, loss = model(*X, targets=Y)
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

    t1 = time.time()
    pbar.set_postfix(loss=f"{train_sums['loss'] / train_n:.3f}", lr=f"{lr:.1e}", ms=f"{(t1 - t0) * 1000:.0f}")
    t0 = t1
