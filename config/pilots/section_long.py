# Section captions, long arm with change oversampling (context/claude/text_control.md "Section captions"): an overlay on
# text_pilot.py + section_pilot.py. A256 (prefix, caption a whole section early), half the windows around a section
# change of >= 2 points, 8,000 it x 65,536 notes = 524M notes (6.7x a sweep arm), ~4 h on the laptop on AC.
# Resumable in chunks (--max_minutes, full ckpt.pt), driven by tools/sweeps/section_long.py (probe after each chunk,
# early stop):
#   python train.py config/pilots/text_pilot.py config/pilots/section_pilot.py config/pilots/section_long.py \
#       --max_minutes=60 [--init_from=resume]
section_lead = 256
section_mode = 'prefix'
section_change_frac = 0.5
section_change_min = 2

max_iters = 8000
lr_decay_iters = 8000
warmup_iters = 100
eval_interval = 500
eval_iters = 25                    # 200 windows per val set
ckpt_interval_min = 30             # full resumable ckpt.pt (1.4 GB) every 30 min, at the end, on Ctrl+C / SIGTERM
save_iters = '2000,4000,6000'
out_dir = '/var/tmp/musicar_sections/long_A256'   # /home has ~2.5 GB free
wandb_run_name = 'sec-long-A256'
