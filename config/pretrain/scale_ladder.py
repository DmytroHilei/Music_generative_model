# Scaling ladder on the stores on the laptop (2026-11-17): long_450m's recipe (block, conditions, packing, Muon WSD),
# model size x notes seen. Train = GigaMIDI replay only (134,538 files / 167.9M notes, never seen by long_450m), every
# arm under one pass; val = the GigaMIDI held-out store (same split, 14,296 windows). Sizes and budgets come from
# tools/sweeps/scale_ladder.py, which also scores long_450m_final on the same windows.
csv_path = 'store:data/cache/gigamidi_replay'
val_csv_path = 'store:data/cache/gigamidi_heldout'

n_programs = 129
pos_emb = 'rope'
block_size = 2048
norm = 'rmsnorm'
mlp = 'swiglu'
qk_norm = True
dropout = 0.0
label_smoothing = 0.0
cond_inst = True
n_density = 16
cond_dropout = 0.15
special_tokens = True
pack_short = True

batch_size = 4
gradient_accumulation_steps = 4    # 32,768 notes/step at every size
warmup_iters = 100

optimizer_name = 'muon'
learning_rate = 1e-3
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
muon_bf16 = True
save_pre_cooldown = False

fp8 = True
compile = True
num_workers = 8

eval_iters = 300                   # x 4 rows = 1,200 fixed windows
checkpoint_format = 'bf16'
seed = 1337
wandb_log = False
