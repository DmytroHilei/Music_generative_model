# F1: the caption run of the 450M on a rented RTX 5090 (context/claude/plan_5h_2026-10-23.md "Path to F1",
# text_control.md "Runs"). Continues long_450m_final with everything the 107M pilots settled, in one run:
#   - T2 text prefix (16 vectors from frozen LaBSE), EN + UK captions, adapter at 10x LR (text_pilot.py)
#   - section captions: prefix slots at section starts, lead 0, change-aware, change oversampling 0.5, section
#     attributes v2 (section_pilot.py, section_long.py, section_v2.py, section_forget.py)
#   - section recipe s50_v2 (2026-10-23, logs/sections/forget.md): splice 0.5 + caption repeat every 64 notes, no
#     counterfactual loss (switch mean 0.98 vs 0.94 for all_v2 with cf, at ~half the step cost)
#   - length control v2 (len2): seconds-to-the-end + progress levels, +-10% target jitter, length tables at 10x LR
#   - full attention (18 KV heads): GQA dropped 2026-10-25. The pca merge broke the QK-norm 450M (held-out CE 2.01 ->
#     6.57 at the merge; the 2,400-it rehearsal still 2.115 vs 2.058 for full attention after 600 it) and it was not
#     what kept captions out; context/claude/caption_450m_2026-10-25.md. A GQA conversion for the Pi, if any, is its
#     own step after F1 (config/f1/f1_gqa_s1.py)
# Open until their pilots report (set here to the planned choice): remain_jitter vs remain_slack and length_words
# (len2 / len2_slack / len2_words).
# Data = the cloud stores (cloud/sync/hub.py) in the text_control.md tiers: GigaMIDI labeled (MidiCaps, XMIDI,
# GigaMIDI metadata, artists) 0.55, Aria labeled (genre / composer) 0.15, GigaMIDI unlabeled (feature captions) 0.30.
# Needs on the instance: data/labels/gigamidi_train.features.npz (data_prep/labels/song_features.py; not on the Hub
# yet: without it the GigaMIDI captions have no instruments / pace / length words and the unlabeled tier none) and
# checkpoints/long_450m_final/model_bf16.pt.
# Length: ~3B notes (~46k steps; F1 length rule: extend the constant phase only for captions, at the cooldown start
# from pre_cooldown.pt). On the instance (cloud/README.md), the LR stays (no LR probe: the probe is for pretraining):
#   python cloud/setup/preflight.py --run f1 --config config/f1/f1.py --hours H --skip-lr-probe
#   POST_SWEEP=0 cloud/run/launch.sh f1
# Laptop (rehearsal, profile): + config/f1/f1_local.py
init_from = 'finetune'
init_ckpt = 'checkpoints/long_450m_final/model_bf16.pt'
out_dir = 'checkpoints/f1'
wandb_run_name = 'f1'

csv_path = ('store:data/cache/gigamidi#labeled,store:data/cache/aria#labeled,'
            'store:data/cache/gigamidi#unlabeled')
source_weights = '0.55,0.15,0.30'
val_csv_path = 'store:data/cache/gigamidi_heldout#labeled'    # captioned held-out files: CE, text and section gains
val_name = 'heldout_labeled'
val2_csv_path = 'store:data/cache/aria#labeled'
val2_name = 'aria_labeled'
val_extra = 'clean=store:data/cache/gigamidi_clean'           # long_450m's val set: forgetting of the base

# the base's shape (long_450m.py; also read by preflight's benchmark); the finetune adds the rest
n_layer = 28
n_embd = 1152
n_head = 18
n_kv_head = 0                   # full attention (GQA dropped 2026-10-25, see the header)
n_programs = 129
pos_emb = 'rope'
block_size = 2048
norm = 'rmsnorm'
mlp = 'swiglu'
qk_norm = True
cond_inst = True
n_density = 16
cond_dropout = 0.15
special_tokens = True
boundary_frac = 0.5             # 25% of the windows at a piece start, 25% at its end (BOS / EOS, length input)
pad_short = True                # short files as one padded window (no packing: one caption per window)
dropout = 0.0
label_smoothing = 0.0

text_encoder = 'sentence-transformers/LaBSE'
captions = 'data/labels'
caption_dropout = 0.15
text_prefix = 16
text_lr_mult = 10.0

section_len = 256
section_lead = 0
section_slots = 4
section_mode = 'prefix'
section_dropout = 0.2
section_style = 'change'
section_attrs = 'v2'
section_change_frac = 0.5
section_change_min = 2
section_splice_frac = 0.5
section_repeat = 64

n_remain = 33
n_progress = 17
remain_jitter = 0.1
remain_lr_mult = 10.0
length_words = 'v2'

batch_size = 4                  # overridden per machine (run.env); batch_size x accumulation x 2048 = 65,536 notes
gradient_accumulation_steps = 8
max_iters = 46000               # overridden from the budget by preflight
lr_decay_iters = 46000
warmup_iters = 300
optimizer_name = 'adamw8bit'    # the 450M piano fine-tune's optimizer; the same on the laptop rehearsal (8 GB)
learning_rate = 1e-4            # the 107M pilots' fine-tune LR
min_lr = 1e-5
weight_decay = 0.1
lr_schedule = 'wsd'
cooldown_frac = 0.2

act_ckpt = 0                    # overridden per machine
act_ckpt_save = 'attn'
fp8 = True
fp8_cache_weights = True
compile = True
num_workers = 8

eval_interval = 1000
eval_iters = 50
style_eval_songs = 0
checkpoint_format = 'bf16'
always_save_checkpoint = True   # the last weights (val CE alone doesn't pick a text model)
save_every = 2500               # bf16 snapshots: averaging, and the per-pass guard (text_control.md "Label tiers")
ckpt_interval_min = 30.0
seed = 1337
wandb_project = 'music-transformer'
