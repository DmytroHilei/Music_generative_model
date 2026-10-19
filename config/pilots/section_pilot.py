# Section captions, short arms (context/claude/text_control.md "Section captions"): an overlay on text_pilot.py that
# continues text_t2_v3_long (T2 prefix, new EN + UK captions) on the same data with a caption per section_len notes
# from that section's own notes. Every arm sees the same notes: 480 it x 65,536 = 31M notes (~0.35 passes over
# tier 1), ~15 min on the laptop; a 'ctrl' arm (section_len 0) gives the continued T2 baseline.
#   python train.py config/pilots/text_pilot.py config/pilots/section_pilot.py --section_mode=xattn \
#       --out_dir=checkpoints/sec_B0 --wandb_run_name=sec-B0
# The arms and their evaluation: tools/sweeps/section_sweep.py
init_ckpt = 'checkpoints/text_t2_v3_long/model_bf16.pt'
text_prefix = 16
boundary_frac = 0.5
pad_short = True

section_len = 256
section_lead = 0
section_slots = 4
section_mode = 'prefix'
section_xattn_every = 2
section_dropout = 0.2

max_iters = 480
lr_decay_iters = 480
warmup_iters = 30
eval_interval = 240                # evals at 0, 240, 479 (6 text val passes each with section captions)
eval_iters = 25                    # 200 windows per val set
val2_csv_path = ''
save_iters = ''
