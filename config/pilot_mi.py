# Multi-instrument pilot (to-do 13, 2026-10-01): does growing big_107M pay vs scratch at equal steps?
# Same 107M shape (14L x 768) with RoPE at 2048 context and the instrument attribute, GigaMIDI 0.7 + Aria piano 0.3.
# 3,000 it x 65,536 notes = 197M notes, ~38.8k notes/s (bench.py) -> ~1.5 h with evals.
#   python train.py config/pilot_mi.py --out_dir=checkpoints/pilot_mi_scratch --wandb_run_name=pilot-mi-scratch
#   python train.py config/pilot_mi.py --init_from=finetune --init_ckpt=checkpoints/big_107M/model_bf16.pt \
#       --out_dir=checkpoints/pilot_mi_grown --wandb_run_name=pilot-mi-grown
# Same recipe for both arms (big_107M's Muon lr 1e-3 WSD), so the only difference is the initial weights.
csv_path = 'store:data/cache/gigamidi,store:data/cache/aria'
source_weights = '0.7,0.3'
val_csv_path = 'store:data/cache/gigamidi_clean'   # leak-free (data/dedupe_val.py)
val2_csv_path = 'store:data/cache/aria'    # piano forgetting (big_107M: 5.71)

n_layer = 14
n_embd = 768
n_head = 12
n_programs = 129
pos_emb = 'rope'
block_size = 2048
dropout = 0.0
label_smoothing = 0.0

batch_size = 4
gradient_accumulation_steps = 8    # 65,536 notes/step
max_iters = 3000
lr_decay_iters = 3000
warmup_iters = 200

optimizer_name = 'muon'
learning_rate = 1e-3
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2

eval_interval = 250
eval_iters = 50                    # 200 windows x 2048 per val set
checkpoint_format = 'bf16'
ckpt_interval_min = 30.0
compile = True
fp8 = True
seed = 1337
