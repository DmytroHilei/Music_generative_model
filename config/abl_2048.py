# ML checks before the long run (2026-10-03), ~20 min each, run by logs/run_ml_checks.sh.
# Base: ladder-L shape (12L x 512, ~42M) from scratch on the multi-instrument mix at 2048 with RoPE, Muon 1e-3 WSD.
# 1,500 it x 65,536 = 98M notes. Arms: block variants (norm / mlp / qk_norm), a higher LR, and an iso-FLOP triple
# (M / L / 16L x 640 at the L arm's compute, FLOPs per note = 6N + 6 L d T) for the size question.
# Val = GigaMIDI validation without the 54k files whose notes also occur in train (data/dedupe_val.py, 2026-10-03).
csv_path = 'store:data/cache/gigamidi,store:data/cache/aria'
source_weights = '0.7,0.3'
val_csv_path = 'store:data/cache/gigamidi_clean'
val2_csv_path = 'store:data/cache/aria'

n_layer = 12
n_embd = 512
n_head = 8
n_programs = 129
pos_emb = 'rope'
block_size = 2048
dropout = 0.0
label_smoothing = 0.0

batch_size = 8
gradient_accumulation_steps = 4    # 65,536 notes/step
max_iters = 1500
lr_decay_iters = 1500
warmup_iters = 100

optimizer_name = 'muon'
learning_rate = 1e-3
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2

eval_interval = 250
eval_iters = 25                    # 200 windows x 2048 per val set
checkpoint_format = 'bf16'
ckpt_interval_min = 0.0
compile = True
fp8 = True
seed = 1337
