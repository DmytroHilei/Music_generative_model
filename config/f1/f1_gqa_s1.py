# F1 stage 1 of the layer-by-layer GQA conversion (arm 'lw' of tools/sweeps/gqa_convert.py at 107M): an overlay on
# config/f1/f1.py (+ f1_local.py on the laptop). The pca merge of each layer's K / V (16 calibration batches through
# the teacher stream), then only the attention layers train, each on the relative MSE to its own full attention, the
# rest of the model on the original states. F1 then starts from this run's ckpt.pt:
#   python train.py config/f1/f1.py config/f1/f1_gqa_s1.py
#   python train.py config/f1/f1.py --init_ckpt=checkpoints/f1_gqa_s1/ckpt.pt
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
