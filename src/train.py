import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data import Subset
import random

from model import MusicConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'checkpoints'
eval_interval = 200
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# data
csv_path = '../data/maestro-v3.0.0.csv'
root_dir = '../data'
batch_size = 16
block_size = 512 # або 512 для локальної демки
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
dataset_part=1.0


# model
n_layer = 3
n_embd = 256
n_head = 4
pitch_size = 128
velocity_size = 32
duration_size = 512   # max_duration_bin + 1
position_size = 512   # max_delta_bin + 1

# wandb logging
wandb_log = True # disabled by default
wandb_project = 'music-transformer'
wandb_run_name = 'maestro-v1'

dropout = 0.4  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
label_smoothing = 0.15  # label smoothing factor to prevent overconfidence
early_stopping_patience = 3  # number of eval intervals without improvement before stopping

# adamw optimizer
learning_rate = 3e-4  # max learning rate
max_iters = 10000  # total number of training iterations
weight_decay = 0.5
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0


# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 10000 # should be ~= max_iters per Chinchilla
min_lr = 3e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

"""# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------"""

master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

from data_loader import MaestroDataset


def collate_fn(batch):
    xs, ys = zip(*batch)
    return (
        tuple(torch.stack([x[i] for x in xs]) for i in range(4)),
        tuple(torch.stack([y[i] for y in ys]) for i in range(4)),
    )

def random_subset(dataset, fraction):
    n = int(len(dataset) * fraction)
    indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)

debug = False

train_dataset = MaestroDataset(csv_path, root_dir = root_dir, split='train', block_size=block_size, debug=debug)
#train_dataset = random_subset(train_dataset, fraction=dataset_part)

val_dataset   = MaestroDataset(csv_path, root_dir = root_dir, split='validation', block_size=block_size, debug=debug)
#val_dataset = random_subset(val_dataset, fraction=dataset_part)


train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn, num_workers=12, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=12, pin_memory=True)


iter_num = 0
best_val_loss = float('inf')
patience_counter = 0

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout,
                  pitch_size=pitch_size, velocity_size=velocity_size,
                  duration_size=duration_size, position_size=position_size)


if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = MusicConfig(
        block_size=block_size,
        pitch_size=pitch_size,
        velocity_size=velocity_size,
        duration_size=duration_size,
        position_size=position_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
              'pitch_size', 'velocity_size', 'duration_size', 'position_size']:
        model_args[k] = checkpoint_model_args[k]

    # create the model
    gptconf = MusicConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

train_iter = iter(train_loader)
val_iter = iter(val_loader)


def get_batch(split):
    global train_iter, val_iter
    loader_iter = train_iter if split == 'train' else val_iter
    try:
        X, Y = next(loader_iter)
    except StopIteration:
        # перезапускаємо якщо дійшли до кінця
        if split == 'train':
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
        else:
            val_iter = iter(val_loader)
            X, Y = next(val_iter)

    # X і Y це tuple з 4 тензорів, переносимо на device
    X = tuple(t.to(device) for t in X)
    Y = tuple(t.to(device) for t in Y)
    return X, Y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    print(f"train loader batches: {len(train_loader)}")
    print(f"val loader batches: {len(val_loader)}")
    print(f"eval_iters: {eval_iters}")

    for split, loader in [("train", train_loader), ("val", val_loader)]:

        actual_eval_iters = min(eval_iters, len(loader))
        losses = torch.zeros(actual_eval_iters, device=device)

        loader_iter = iter(loader)
        for k in range(actual_eval_iters):
            X, Y = next(loader_iter)
            X = tuple(x.to(device, non_blocking=True) for x in X)
            Y = tuple(y.to(device, non_blocking=True) for y in Y)

            with ctx:
                p, v, d, pos = X
                logits, loss = model(p, v, d, pos, targets=Y)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=gptconf.__dict__)

# training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model

# замість while True
pbar = tqdm(range(max_iters), desc="Training")
for iter_num in pbar:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                os.makedirs(out_dir, exist_ok=True)
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        elif always_save_checkpoint and iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            os.makedirs(out_dir, exist_ok=True)
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} eval intervals")

        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {patience_counter} eval intervals without improvement")
            break
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    loss_accum = torch.zeros((), device=device)
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            p, v, d, pos = X
            loss_targets = Y
            logits, loss = model(p, v, d, pos, targets=loss_targets)

            # Apply label smoothing by mixing true labels with uniform distribution
            if label_smoothing > 0:
                loss = loss * (1.0 - label_smoothing)

            loss = loss / gradient_accumulation_steps
        loss_accum += loss.detach()
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = 0.0
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss_accum.item()  # true mean over the accumulation window

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss_step": lossf,
                "train/grad_norm": grad_norm,
                "lr": lr,
            })

        """if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")"""

    #iter_num += 1
    #local_iter_num += 1

    """# termination conditions
    if iter_num > max_iters:
        break"""
