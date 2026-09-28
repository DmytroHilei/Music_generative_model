# A/B: cascade heads vs independent heads at a compute-optimal budget for the 5.9M model.
# 4 epochs of 31.3M train notes ~= 125M tokens = 4000 iters x 30,720 tokens.
# Run twice, only cascade_heads differs:
#   python train.py config/pretrain.py config/ab_heads.py --cascade_heads=True  --wandb_run_name=ab-cascade  --out_dir=checkpoints/ab_cascade
#   python train.py config/pretrain.py config/ab_heads.py --cascade_heads=False --wandb_run_name=ab-indep    --out_dir=checkpoints/ab_indep
wandb_project = 'music-transformer'
batch_size = 6
gradient_accumulation_steps = 10
max_iters = 4000
lr_decay_iters = 4000
warmup_iters = 200
eval_interval = 100
eval_iters = 200
dropout = 0.1
label_smoothing = 0.0
learning_rate = 6e-4
min_lr = 6e-5
seed = 1337
