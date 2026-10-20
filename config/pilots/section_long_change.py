# Change-aware section captions: an overlay on section_long.py (A256, change oversampling, 8,000 it) that changes only
# the section captions: what differs from the previous section is always named (the instrument that entered, 'loud'
# after a soft section, 'without drums' when the drums stop; musicar/text/captions.py section_changed). Same data,
# windows, change scores and probe buckets as long_A256, so the two compare at equal iterations. At the oversampled
# boundaries the plain captions name a change in 61% of the cases (all of them in 33%), these in 99% (96%).
#   .venv/bin/python tools/sweeps/section_long.py --name long_A256_change --out-dir checkpoints/long_A256_change \
#       --config config/pilots/section_long_change.py --minutes 90
section_style = 'change'
out_dir = 'checkpoints/long_A256_change'   # /home: /var/tmp holds long_A256 (1.4 GB ckpt.pt + a copy while saving)
wandb_run_name = 'sec-long-A256-change'
