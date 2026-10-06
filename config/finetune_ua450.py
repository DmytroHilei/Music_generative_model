# Ukrainian piano arrangements on the 450M base (to-do 15d), stage 1 (2026-10-06): round-1 + round-2 data available
# now (`prepare_finetune.py --round2`: 5,287 train reductions, 652 train covers). Stage 2 continues from
# out_dir/pre_cooldown.pt on the full data (the rest of the round-2 covers) with its own cooldown.
# Recipe = ua_2048_v2 (finetune_ua2048.py) with: covers weight 0.5 -> 0.35 and reductions 0.3 -> 0.45 (the mix grid
# found no difference in 0.3-0.6; the reductions are now 5x the covers and would otherwise get < 1 pass), micro 1
# (the 450M only fits at micro 1 on the laptop), fewer covers passes (~6 vs 8: a bigger model memorises sooner).
#   python train.py config/finetune_ua450.py
init_from = 'finetune'
init_ckpt = 'checkpoints/long_450m_final/model_bf16.pt'
out_dir = 'checkpoints/ua_450_s1'
wandb_run_name = 'ua-450-s1'

csv_path = 'data/finetune/ukrainian_covers_v3.csv,data/finetune/ukrainian_reduction_v3.csv,store:data/cache/aria_poprock'
source_weights = '0.35,0.45,0.2'
aug_stores = '1,1,0'               # tempo aug on the Ukrainian sets only, the replay stays real (ua-C3)
aug_tempo = 0.1
val_csv_path = 'data/finetune/ukrainian_covers_v3.csv'        # the target: real arrangements (47 val covers)
val2_csv_path = 'data/finetune/ukrainian_covers.csv'          # ua_2048_v2's covers val (5.638), for comparison
val_name = 'covers_v3'
val2_name = 'covers_v2'
val_extra = 'reductions=data/finetune/ukrainian_reduction_v3.csv'
special_tokens = True
style_map = 'data/styles.json'
style_dropout = 0.1
style_lr_mult = 30.0
boundary_frac = 0.0                # whole-song windows already start with BOS and end with EOS

block_size = 2048
pad_short = True
min_notes = 64
n_programs = 129

# 12 windows/step like ua_2048_v2; ~0.35 x 12 x ~1,600 notes = ~6.7k covers notes/step -> 950 steps = ~6 passes over
# the 1.06M train covers notes, ~1.3 over the reductions; cooldown from 760 (pre_cooldown.pt, ~4.8 covers passes)
batch_size = 1
gradient_accumulation_steps = 12
max_iters = 950
lr_decay_iters = 950
warmup_iters = 30
eval_interval = 50
eval_iters = 100

optimizer_name = 'muon'
learning_rate = 3e-4               # ua-A sweep (107M); long_450m pretrained at 1e-3, as pilot_mi did before ua_2048
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
dropout = 0.1
label_smoothing = 0.0
muon_bf16 = True

act_ckpt = -1                      # laptop fit (2026-10-03): 450M only at micro 1 with all blocks checkpointed
act_ckpt_save = 'attn'
fp8 = True
fp8_cache_weights = True
compile = True

checkpoint_format = 'bf16'         # best-val weights; the resumable state is ckpt.pt (+ pre_cooldown.pt)
ckpt_interval_min = 30.0
