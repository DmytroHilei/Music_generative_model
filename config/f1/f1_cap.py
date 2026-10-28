# F1 caption phase, after the long run's cooldown (2026-10-27, context/claude/block512_plan_2026-10-26.md "CAP"):
# the full F1 flags at block 512, half the windows at piece starts, no splice / change windows, so ~60 captioned
# openings per step (the long run: 13.8) rebuild the caption reading the long run wore down. CAP on the laptop (600 it
# from f1_stage2: warmup 300 + 300 at 1e-4, no decay): notes 0-31 after BOS +0.076 vs stage 1 +0.071, text_eval above
# stage 1, held-out CE at block 2048 2.073 (no settle needed). Here with a 100-it warmup and its own WSD decay over the
# last 30%, so the phase ends at min_lr; the it-300 weights (constant LR, like CAP's) are kept as well.
# Loaded after f1.py; cloud/run/f1_post.sh runs it with the micro-batch from checkpoints/f1/capphase.env
# (cloud/run/f1_checks.sh) and 128 windows per step (65,536 notes, as in the long run):
#   python train.py config/f1/f1.py config/f1/f1_cap.py --batch_size=M --gradient_accumulation_steps=128/M
init_ckpt = 'checkpoints/f1/model_bf16.pt'
out_dir = 'checkpoints/f1_cap'
wandb_run_name = 'f1-cap'

block_size = 512
boundary_start = 0.5
boundary_end = 0.25              # 2026-10-27, after the 5090 length check of the laptop's caption phase (CAP: 0.05):
                                 # 60 / 120 s requests ended at 0.50 / 0.44 of L (len2 pilot 0.98 / 0.84), so the
                                 # phase keeps the long run's share of piece ends (EOS, length input)
section_splice_frac = 0.0
section_change_frac = 0.0

max_iters = 600
lr_decay_iters = 600
warmup_iters = 100
cooldown_frac = 0.3             # 1e-4 to it 420, then 1 - sqrt down to min_lr at 600
eval_interval = 300
save_every = 300                # model_bf16_it300.pt: the weights before the decay
save_pre_cooldown = False
