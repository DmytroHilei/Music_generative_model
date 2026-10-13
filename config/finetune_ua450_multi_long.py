# Longer Ukrainian multi-track fine-tune: ua_450_multi's constant-LR weights at it750 (no optimizer state) + 2,250 it
# with a fresh adamw8bit (10 warmup), WSD 20 % = cooldown over the last 450 it; 750 + 2,250 = 3,000 it in all
# (~4 passes over the multi-track train set). Stacked on the two configs of ua_450_multi:
#   python train.py config/finetune_ua450.py config/finetune_ua450_multi.py config/finetune_ua450_multi_long.py
init_ckpt = 'checkpoints/ua_450_multi/model_bf16_it750.pt'
out_dir = 'checkpoints/ua_450_multi_long'
wandb_run_name = 'ua-450-multi-long'

max_iters = 2250
lr_decay_iters = 2250
save_iters = '750,1350,2249'       # = 1,500 / 2,100 / 3,000 it in all
save_pre_cooldown = True           # out_dir/pre_cooldown.pt at it 1,800 (2,550 in all), to extend again
