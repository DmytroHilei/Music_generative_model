# Stage B2: fine-tune the pop+rock model (B1, with BOS/EOS + styles) on the Ukrainian reductions, artist = style.
# Same recipe as stage A (song-level split, 20% replay), so A vs B2 compares "plain" vs "pop+rock stage + tokens".
init_from = 'finetune'
init_ckpt = 'checkpoints/poprock_B1/model_bf16.pt'
out_dir = 'checkpoints/ua_B2'
wandb_run_name = 'ua-B2'

csv_path = 'data/finetune/ukrainian.csv,store:data/cache/aria_poprock'
source_weights = '0.8,0.2'
val_csv_path = 'data/finetune/ukrainian.csv'
val2_csv_path = 'store:data/cache/aria_poprock'
special_tokens = True
style_map = 'data/styles.json'
style_dropout = 0.1
boundary_frac = 0.1

batch_size = 6
gradient_accumulation_steps = 4    # 12,288 tokens/step
max_iters = 600
lr_decay_iters = 600
warmup_iters = 30
eval_interval = 50
eval_iters = 100

optimizer_name = 'muon'
learning_rate = 3e-4               # set from stage A's sweep
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
dropout = 0.1
label_smoothing = 0.0

checkpoint_format = 'bf16'
compile = True
fp8 = True
