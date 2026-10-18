# Scaling ladder: same data, same token budget (~100M tokens = 3300 iters x 30,720), only model size differs.
# Main val = old MAESTRO+GiantMIDI val (comparable with the A/B runs), val2 = Aria val.
#   S: --n_layer=6  --n_embd=256 --n_head=8   (~6M)
#   M: --n_layer=10 --n_embd=384 --n_head=6   (~19M)
#   L: --n_layer=12 --n_embd=512 --n_head=8   (~42M)
#   XL: --n_layer=12 --n_embd=640 --n_head=10 (~65M)
csv_path = 'data/combined.csv,store:data/cache/aria'
val_csv_path = 'data/combined.csv'
val2_csv_path = 'store:data/cache/aria'
batch_size = 6
gradient_accumulation_steps = 10
max_iters = 3300
lr_decay_iters = 3300
warmup_iters = 200
eval_interval = 250
eval_iters = 200
dropout = 0.0
label_smoothing = 0.0
learning_rate = 6e-4
min_lr = 6e-5
seed = 1337

# disk is tight (96 GB partition): save bf16 weights only, ladder/ablation runs are never resumed
checkpoint_format = 'bf16'
