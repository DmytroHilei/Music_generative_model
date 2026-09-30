# Stage A: plain fine-tune of big-107M on the Ukrainian piano reductions (song-level split, deduped:
# data/prepare_finetune.py), 20% replay of Aria so it doesn't forget. ~10 passes over the 452 train songs.
# Architecture comes from the checkpoint. Best checkpoint by Ukrainian val loss (early stopping).
init_from = 'finetune'
init_ckpt = 'checkpoints/big_107M/model_bf16.pt'
out_dir = 'checkpoints/ua_A'
wandb_run_name = 'ua-A'

csv_path = 'data/finetune/ukrainian.csv,store:data/cache/aria'
source_weights = '0.8,0.2'
val_csv_path = 'data/finetune/ukrainian.csv'
val2_csv_path = 'store:data/cache/aria'

batch_size = 6
gradient_accumulation_steps = 4    # 12,288 tokens/step
max_iters = 600                    # ~7.4M tokens: ~5.9M Ukrainian = ~11 passes over 524k notes
lr_decay_iters = 600
warmup_iters = 30
eval_interval = 50
eval_iters = 100

optimizer_name = 'muon'
learning_rate = 3e-4
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
dropout = 0.1
label_smoothing = 0.0

checkpoint_format = 'bf16'         # best Ukrainian-val weights -> model_bf16.pt
compile = True
fp8 = True
