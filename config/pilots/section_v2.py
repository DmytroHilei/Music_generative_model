# Section captions v2: change-aware section captions on the new label tables (merge of worktree-caption-datasets-survey
# 2026-10-20: GigaMIDI v2 metadata styles, composer / artist captions, Aria genre + composer; the tables were in
# data/labels_next until the swap into data/labels on 2026-10-24). An overlay on text_pilot.py + section_pilot.py + section_long.py
# (init text_t2_v3_long, A256 prefix, change oversampling 0.5, WSD with the cooldown in the last 20%).
# Train data: GigaMIDI replay labeled 33.1k files / 95.9M notes, unlabeled 68.9k / 70.8M, Aria labeled (piano)
# 229.6k / 353.2M. 10,000 it x 65,536 notes = 655M notes: labeled 0.40 (2.7 passes), unlabeled 0.25 (2.3), Aria 0.35
# (0.65); the 0.55 / 0.30 / 0.15 tier weights of text_control.md assume the cloud's ~1B tier-1 notes.
# Validation: gigamidi_heldout#labeled with the new tables (3,648 files, was 2,055: not comparable with earlier runs)
# and Aria labeled as val2. Quick check of A0 / A256 / B0 and the winner's continuation: tools/sweeps/section_v2.py.
captions = 'data/labels'
csv_path = ('store:data/cache/gigamidi_replay#labeled,store:data/cache/gigamidi_replay#unlabeled,'
            'store:data/cache/aria#labeled')
source_weights = '0.40,0.25,0.35'
val2_csv_path = 'store:data/cache/aria#labeled'
val2_name = 'aria_labeled'
section_style = 'change'
max_iters = 10000
lr_decay_iters = 10000
save_iters = '2000,4000,6000,8000'
out_dir = 'checkpoints/v2'
wandb_run_name = 'sec-v2'
