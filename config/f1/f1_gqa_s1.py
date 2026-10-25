# Optional GQA conversion of the finished F1 model for the Pi (F1 itself is full attention since 2026-10-25), stage 1
# of the layer-by-layer conversion (arm 'lw' of tools/sweeps/gqa_convert.py at 107M): an overlay on config/f1/f1.py
# (+ f1_local.py on the laptop). The pca merge of each layer's K / V (16 calibration batches through the teacher
# stream), then only the attention layers train, each on the relative MSE to its own full attention, the rest of the
# model on the original states. Check the held-out CE against F1's own (within ~0.05: eval/probes/gqa_merge.py)
# before a short uptrain on the F1 data from this run's ckpt.pt:
#   python train.py config/f1/f1.py config/f1/f1_gqa_s1.py
#   python train.py config/f1/f1.py --init_ckpt=checkpoints/f1_gqa_s1/ckpt.pt --max_iters=... --lr_decay_iters=...
init_ckpt = 'checkpoints/f1/model_bf16.pt'
n_kv_head = 6                   # groups of 3 query heads
gqa_init = 'pca'                # data-aware K / V merge (musicar/model/gqa.py) instead of the mean-pool
attn_transfer = True
compile = False                 # the converted layers hand their loss over as an attribute
fp8 = False
fp8_cache_weights = False
learning_rate = 1e-3
warmup_iters = 20
max_iters = 300
lr_decay_iters = 300
eval_interval = 50
eval_iters = 25
save_every = 0
save_pre_cooldown = False
ckpt_interval_min = 10.0
out_dir = 'checkpoints/f1_gqa_s1'
wandb_run_name = 'f1-gqa-s1'
