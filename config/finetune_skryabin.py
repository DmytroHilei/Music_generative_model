# Fine-tune a pretrained checkpoint on the Skryabin transcriptions.
out_dir = 'checkpoints_skryabin'
init_from = 'finetune'
init_ckpt = 'checkpoints/ckpt.pt'
csv_path = 'Skryabin/skryabin.csv'
root_dir = 'Skryabin'
wandb_run_name = 'skryabin-finetune'

batch_size = 4
gradient_accumulation_steps = 8
eval_interval = 10
dropout = 0.1
learning_rate = 1e-4
max_iters = 500
warmup_iters = 50
lr_decay_iters = 500
min_lr = 1e-5
