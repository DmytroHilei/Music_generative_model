# The ~14 h run (2026-09-29): compute-optimal size from the Muon iso-FLOP sweep (optimum ~16-20 tokens/param).
# C ~ 1.5e18 FLOPs (measured 50k tok/s x 13.3 h) -> 107M params x 2.33B tokens (22 tok/param, 4.1 passes over 568M notes).
# Scaling fit on the Muon S/M/L runs predicts Aria CE ~5.5-5.7 (iso-M-muon: 6.512).
#   python train.py config/big.py                        # start
#   python train.py config/big.py --init_from=resume     # after a crash/reboot (from out_dir/ckpt.pt)
# WSD: the LR is constant until 80% of lr_decay_iters, so max_iters/lr_decay_iters can still be changed on resume before then.
csv_path = 'data/combined.csv,store:data/cache/aria'
val_csv_path = 'data/combined.csv'
val2_csv_path = 'store:data/cache/aria'
out_dir = 'checkpoints/big_107M'
wandb_run_name = 'big-107M-muon-wsd'

n_layer = 14
n_embd = 768
n_head = 12
dropout = 0.0
label_smoothing = 0.0

batch_size = 6
gradient_accumulation_steps = 20   # 61,440 tokens/step (iso runs: 30,720)
max_iters = 38000                  # 2.33B tokens, 1.235 s/step measured -> ~13.3 h with evals
lr_decay_iters = 38000
warmup_iters = 400

optimizer_name = 'muon'
learning_rate = 1e-3
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2

eval_interval = 1000
eval_iters = 200
checkpoint_format = 'bf16'         # best-val weights -> model_bf16.pt
ckpt_interval_min = 30.0           # full resumable state -> ckpt.pt
compile = True
fp8 = True
seed = 1337
