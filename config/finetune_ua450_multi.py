# Ukrainian multi-track fine-tune of the 450M base (to-do 16). Stacked on finetune_ua450.py (450M laptop settings):
#   python train.py config/finetune_ua450.py config/finetune_ua450_multi.py
# Data: data/finetune/ukrainian_multi_v1.csv (4,811 train / 547 val songs, 13.77M / 1.52M notes, drums bass guitar
# vocals piano other as GM programs) + replay from store:data/cache/gigamidi_replay (167.9M notes).
# Optimizer: the piano AdamW recipe (tools/adamw_sweep.py finalist adamw_lr2e-4: adamw8bit, LR 2e-4, dropout 0,
# warmup 10, WSD 20 %, ~2 passes; 2x the steps and dropout 0.1 were no better).
# Length: 12 windows x ~2,048 notes x 0.75 multi share ~ 18.4k multi notes/step -> 1,500 it ~ 2 passes over the
# multi-track train set (the piano winner = ~2 passes over the covers); ~75 min + evals on the laptop.
init_ckpt = 'checkpoints/long_450m_final/model_bf16.pt'
out_dir = 'checkpoints/ua_450_multi'
wandb_run_name = 'ua-450-multi'

csv_path = 'data/finetune/ukrainian_multi_v1.csv,store:data/cache/gigamidi_replay'
source_weights = '0.75,0.25'
aug_stores = '1,0'                 # tempo aug on the Ukrainian set only, the replay stays real
aug_tempo = 0.1
val_csv_path = 'data/finetune/ukrainian_multi_v1.csv'         # 547 val songs, checkpoint selection
val_name = 'multi_v1'
val2_csv_path = 'store:data/cache/gigamidi_heldout'           # forgetting: 10 % of the clean GigaMIDI test
val2_name = 'gigamidi_heldout'
val_extra = ''
style_map = ''                     # the base has no style table; no style label for this set

optimizer_name = 'adamw8bit'
learning_rate = 2e-4
dropout = 0.0
max_iters = 1500
lr_decay_iters = 1500
warmup_iters = 10
eval_interval = 100
eval_iters = 100
save_iters = '750,1499'            # 1 pass (before the cooldown) and the end
save_pre_cooldown = False
