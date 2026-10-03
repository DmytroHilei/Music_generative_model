# The long multi-instrument pretraining run (DRAFT 2026-10-03, not launched): 350M from scratch with the new block.
# Decided: scratch + RMSNorm/SwiGLU/QK-norm (user), 350M (iso-FLOP, agents.md 13b), memory/speed setting from
# optimizations.md (ckpt 24 + 'attn' + fp8 weight cache, ~14.3k notes/s). TODO before launch: source_weights (mix
# tests), learning_rate (2e-3 won at 42M; unverified at 350M).
# Launch through the auto-resume wrapper:
#   logs/run_resumable.sh long_350m 20 config/long_350m.py --out_dir=checkpoints/long_350m --wandb_run_name=long-350m
csv_path = 'store:data/cache/gigamidi,store:data/cache/aria,store:data/cache/discover'
source_weights = '0.4,0.3,0.3'           # TODO from the mix tests
val_csv_path = 'store:data/cache/gigamidi_clean'
val2_csv_path = 'store:data/cache/aria'
val_extra = 'discover=store:data/cache/discover'

n_layer = 28
n_embd = 1024
n_head = 16
n_programs = 129
pos_emb = 'rope'
block_size = 2048
norm = 'rmsnorm'
mlp = 'swiglu'
qk_norm = True
dropout = 0.0
label_smoothing = 0.0
special_tokens = True                    # BOS/EOS (needed by pack_short; the fine-tunes use them)
pack_short = True                        # 1.81B -> 2.50B usable notes for GigaMIDI + Aria at 2048

batch_size = 4
gradient_accumulation_steps = 8          # 65,536 notes/step
max_iters = 110000                       # 7.2B notes ≈ 5.8 days at 14.3k notes/s
lr_decay_iters = 110000
warmup_iters = 1000

optimizer_name = 'muon'
learning_rate = 1e-3                     # TODO: 2e-3 won at 42M
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
muon_bf16 = True

act_ckpt = 24
act_ckpt_save = 'attn'
fp8 = True
fp8_cache_weights = True
compile = True

eval_interval = 1000                     # ~77 min
eval_iters = 50
checkpoint_format = 'bf16'
ckpt_interval_min = 30.0                 # resumable out_dir/ckpt.pt for logs/run_resumable.sh
seed = 1337
