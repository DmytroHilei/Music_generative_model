# Text control pilot P1 (context/claude/text_control.md "Runs"): does the 107M follow captions, and does the trunk need
# training? Base pilot_mi_grown (107M, 129 instruments, RoPE; grown to BOS/EOS here). Captions from the provisional
# sampler (musicar/text/captions.py: MidiCaps, XMIDI, features, gated pseudo-labels; English only), LaBSE frozen.
# Data: local tier 1 (labeled GigaMIDI replay files) 0.7 + the unlabeled rest 0.3; 2,400 it x 65,536 notes = 157M
# notes, ~1.7 passes over tier 1 (65M notes in files > 1,024 notes).
#   python train.py config/pilots/text_pilot.py --out_dir=checkpoints/text_p1_full --wandb_run_name=text-p1-full
#   python train.py config/pilots/text_pilot.py --freeze_trunk=True --out_dir=checkpoints/text_p1_frozen \
#       --wandb_run_name=text-p1-frozen
# Frozen arm: text adapter + the pitch embedding and pitch output (their new BOS/EOS rows can't stay random).
# P2 (after the gradient check, text_control.md "Results"): full arm, centered caption vectors, 25% of windows at a
# piece start, adapter at the base LR (centered vectors have norm sqrt(768) instead of 1):
#   python train.py config/pilots/text_pilot.py --text_center=True --boundary_frac=0.5 --text_lr_mult=1.0 \
#       --out_dir=checkpoints/text_p2 --wandb_run_name=text-p2
# T2 (prefix, same compute as P2; adapter at the default 10x LR: fresh MLP behind a zero-init gate):
#   python train.py config/pilots/text_pilot.py --text_prefix=16 --boundary_frac=0.5 --out_dir=checkpoints/text_t2 \
#       --wandb_run_name=text-t2
# T2 v3 (new EN + UK captions, complete instrument lists / solo / without drums, short files padded):
#   python train.py config/pilots/text_pilot.py --text_prefix=16 --boundary_frac=0.5 --pad_short=True \
#       --out_dir=checkpoints/text_t2_v3 --wandb_run_name=text-t2-v3
# T2 v3 at double length: + --max_iters=4800 --lr_decay_iters=4800 --save_iters=2400 (out_dir text_t2_v3_long)
init_from = 'finetune'
init_ckpt = 'checkpoints/pilot_mi_grown/model_bf16.pt'
csv_path = 'store:data/cache/gigamidi_replay#labeled,store:data/cache/gigamidi_replay#unlabeled'
source_weights = '0.7,0.3'
val_csv_path = 'store:data/cache/gigamidi_heldout#labeled'    # captioned held-out files: val CE + text gain
val_name = 'heldout_labeled'
val2_csv_path = 'store:data/cache/gigamidi_heldout#unlabeled'  # the rest (feature / pseudo-label captions)
val2_name = 'heldout_unlabeled'

text_encoder = 'sentence-transformers/LaBSE'
captions = 'data/labels'
caption_dropout = 0.15
text_lr_mult = 10.0                # adapter 1e-3, the rest 1e-4
freeze_trunk_keep = 'transformer.text_proj,transformer.music_embeddings.pitch,cascade.heads.pitch.out'

n_layer = 14
n_embd = 768
n_head = 12
n_programs = 129
pos_emb = 'rope'
special_tokens = True
boundary_frac = 0.1
block_size = 1024
dropout = 0.0
label_smoothing = 0.0

batch_size = 8
gradient_accumulation_steps = 8    # 65,536 notes/step
max_iters = 2400
lr_decay_iters = 2400
warmup_iters = 50
optimizer_name = 'adamw'
learning_rate = 1e-4
min_lr = 1e-5
weight_decay = 0.1
lr_schedule = 'wsd'
cooldown_frac = 0.2
save_pre_cooldown = False

eval_interval = 200
eval_iters = 50                    # 400 windows x 1,024 per val set (x3 for the text gain)
save_iters = '1200'
checkpoint_format = 'bf16'
always_save_checkpoint = True      # keep the last weights (val CE alone doesn't pick a text model)
compile = True
fp8 = True
wandb_project = 'music-transformer'
seed = 1337
