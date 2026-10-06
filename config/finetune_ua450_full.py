# Ukrainian piano arrangements on the 450M base, direct run on the complete data (2026-10-06): the round-2 downloads
# are finished, `*_v4.csv` = v3 + the last 9 round-2 covers (659 / 49 covers, 5,287 / 603 reductions). Same recipe
# as stage 1, stacked on it; the first run with the base's window conditions (train.py fix 5845a82).
#   python train.py config/finetune_ua450.py config/finetune_ua450_full.py
out_dir = 'checkpoints/ua_450_full'
wandb_run_name = 'ua-450-full'

csv_path = 'data/finetune/ukrainian_covers_v4.csv,data/finetune/ukrainian_reduction_v4.csv,store:data/cache/aria_poprock'
val_csv_path = 'data/finetune/ukrainian_covers_v4.csv'        # 49 val covers
val2_csv_path = 'data/finetune/ukrainian_covers.csv'          # ua_2048_v2's covers val (5.638), for comparison
val_name = 'covers_v4'
val_extra = 'reductions=data/finetune/ukrainian_reduction_v4.csv'
