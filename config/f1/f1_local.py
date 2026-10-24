# F1 on the laptop (8 GB): an overlay on config/f1/f1.py for the rehearsal and the step profile. The local stores
# in place of the cloud ones (gigamidi_train is not on the laptop: gigamidi_replay is its local sample, as in
# section_v2.py), no gigamidi_clean (not on the laptop); the 450M fits only at micro 1 with every block checkpointed.
#   python train.py config/f1/f1.py config/f1/f1_local.py --max_iters=300 --lr_decay_iters=300
csv_path = ('store:data/cache/gigamidi_replay#labeled,store:data/cache/aria#labeled,'
            'store:data/cache/gigamidi_replay#unlabeled')
val_extra = ''
out_dir = 'checkpoints/f1_local'
wandb_run_name = 'f1-local'

batch_size = 1
gradient_accumulation_steps = 32
act_ckpt = -1
num_workers = 4
eval_iters = 100                # 100 windows per val set at micro 1
