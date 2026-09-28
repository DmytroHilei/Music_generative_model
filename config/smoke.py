# ~1 minute sanity run: python train.py config/pretrain.py config/smoke.py
out_dir = 'checkpoints_smoke'
wandb_log = False
max_iters = 30
eval_interval = 10
eval_iters = 5
gradient_accumulation_steps = 2
warmup_iters = 5
lr_decay_iters = 30
