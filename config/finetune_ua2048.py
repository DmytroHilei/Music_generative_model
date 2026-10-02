# Ukrainian piano arrangements at 2048 context (2026-10-02): whole songs in one window instead of 512-note slices.
# Base: pilot_mi_grown (RoPE 2048, 129 programs; piano stores read as program 0). The finetune grows the pitch vocab
# 128 -> 130 (BOS/EOS) and the style table (31 styles, zero-init), as B1 did for big_107M.
# Data: real piano covers (transcribed, style 'cover') = the clean target; synthetic reductions (style 'reduction') for
# volume; Aria pop+rock as real replay. Songs are ~1,100 notes (p90 1,550), so pad_short keeps each as one padded
# BOS..EOS window (without it 99% of the songs would be dropped at 2048). Steps: set once the covers are counted.
#   python train.py config/finetune_ua2048.py
init_from = 'finetune'
init_ckpt = 'checkpoints/pilot_mi_grown/model_bf16.pt'
out_dir = 'checkpoints/ua_2048'
wandb_run_name = 'ua-2048'

csv_path = 'data/finetune/ukrainian_covers.csv,data/finetune/ukrainian_reduction_v2.csv,store:data/cache/aria_poprock'
source_weights = '0.5,0.3,0.2'
aug_stores = '1,1,0'               # tempo aug on the Ukrainian sets only, the replay stays real (ua-C3)
aug_tempo = 0.1
val_csv_path = 'data/finetune/ukrainian_covers.csv'           # the target: real arrangements
val2_csv_path = 'data/finetune/ukrainian_reduction_v2.csv'    # comparable with ua-D (at 512: 10.551)
special_tokens = True
style_map = 'data/styles.json'
style_dropout = 0.1
style_lr_mult = 30.0
boundary_frac = 0.0                # whole-song windows already start with BOS and end with EOS

block_size = 2048
pad_short = True
min_notes = 64
n_programs = 129                   # the base has the instrument attribute; piano stores send program 0

batch_size = 2                     # micro 4 OOMs next to ~2.9 GB of desktop GPU memory
gradient_accumulation_steps = 6    # 12 windows ≈ 13k real notes/step (≈ ua-D's 12,288)
max_iters = 1600
lr_decay_iters = 1600
warmup_iters = 30
eval_interval = 100
eval_iters = 100

optimizer_name = 'muon'
learning_rate = 3e-4               # ua-A sweep
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
dropout = 0.1
label_smoothing = 0.0

checkpoint_format = 'bf16'
compile = True
fp8 = True
