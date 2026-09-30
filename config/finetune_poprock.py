# Stage B1: continued training of big-107M on Aria pop+rock (data/subset_aria.py) with BOS/EOS and style
# conditioning (genre), 20% replay of all Aria. Grows the checkpoint: pitch vocab 128 -> 130, new style table (zero init).
# Then stage B2 (config/finetune_ua_B.py) fine-tunes this on the Ukrainian reductions with artist styles.
init_from = 'finetune'
init_ckpt = 'checkpoints/big_107M/model_bf16.pt'
out_dir = 'checkpoints/poprock_B1'
wandb_run_name = 'poprock-B1'

csv_path = 'store:data/cache/aria_poprock,store:data/cache/aria'
source_weights = '0.8,0.2'
val_csv_path = 'store:data/cache/aria_poprock'
val2_csv_path = 'data/finetune/ukrainian.csv'   # watch the target domain on the way
special_tokens = True
style_map = 'data/styles.json'
style_dropout = 0.1
boundary_frac = 0.1

batch_size = 6
gradient_accumulation_steps = 20   # 61,440 tokens/step, as in pretraining
max_iters = 3300                   # ~203M tokens: ~162M pop+rock = ~1.75 passes over 92.7M notes, ~70 min
lr_decay_iters = 3300
warmup_iters = 100
eval_interval = 500
eval_iters = 100

optimizer_name = 'muon'
learning_rate = 5e-4               # re-warm below the pretraining peak (1e-3)
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
dropout = 0.0
label_smoothing = 0.0

checkpoint_format = 'bf16'         # best pop+rock-val weights -> model_bf16.pt
ckpt_interval_min = 30.0           # resumable ckpt.pt
compile = True
fp8 = True
