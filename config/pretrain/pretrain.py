# Pretraining on MAESTRO (+ GiantMIDI if present). Baseline = last known run lcqh7fm4, but with cascade heads.
out_dir = 'checkpoints'
init_from = 'scratch'
csv_path = 'data/combined.csv'
root_dir = '.'
wandb_run_name = 'pretrain'
