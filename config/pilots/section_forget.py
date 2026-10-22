# Section-caption forgetting (context/claude/roadmap.md P0; v2_A0 holds the first section's style and ignores later
# section captions): short arms continuing v2_A0 (change-aware captions, new labels, Aria, A0) on the same data, each
# testing one fix, plus all of them together. An overlay on text_pilot.py + section_pilot.py + section_long.py +
# section_v2.py. 1,500 it x 65,536 notes = 98M notes per arm (WSD: constant LR 1e-4 to 1,200, cooldown after).
#   ctrl     the v2 recipe continued (also: does more training alone help, A6)
#   cf       counterfactual loss with the previous section caption (MuseBarControl) + recognition head
#   splice   25% spliced windows (a section of another file with its caption, A3)
#   repeat   the section caption again every 64 notes (A4)
#   all      cf + recognition head + splice + repeat
# Arms, probe and switch tests: tools/sweeps/section_forget.py
init_from = 'finetune'
init_ckpt = 'checkpoints/v2_A0/model_bf16.pt'
section_lead = 0

max_iters = 1500
lr_decay_iters = 1500
batch_size = 4                     # x 16 = 65,536 notes/step as v2; micro 8 with the second (cf) forward OOMs
gradient_accumulation_steps = 16
section_cf_rows = 4                # every row of a micro-batch (fp8 matmuls need rows * length divisible by 16)
warmup_iters = 50
eval_interval = 500
eval_iters = 50                    # 200 windows per val set (batch 4)
save_iters = ''
ckpt_interval_min = 30
out_dir = 'checkpoints/forget_ctrl'
wandb_run_name = 'sec-forget-ctrl'
