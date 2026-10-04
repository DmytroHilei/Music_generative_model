# The long multi-instrument pretraining run: ~450M from scratch with the new block, on a rented RTX 5090 (vast.ai).
# Decisions (2026-10-03, agents.md): scratch + RMSNorm/SwiGLU/QK-norm (−0.75 val CE over 3 seeds), 450M (iso-FLOP fit
# for a ~EUR 20 budget, still trainable on the laptop at micro 1), mix GigaMIDI .4 / Aria .3 / Discover(27%) .3.
# Machine-specific settings (micro-batch, checkpointing, max_iters from the budget) come from cloud/preflight.py
# through out_dir/run.env; never launch this file directly on the instance, use cloud/launch.sh (see cloud/README.md).
csv_path = 'store:data/cache/gigamidi,store:data/cache/aria,store:data/cache/discover'
source_weights = '0.4,0.3,0.3'
val_csv_path = 'store:data/cache/gigamidi_clean'
val2_csv_path = 'store:data/cache/aria'
val_extra = 'discover=store:data/cache/discover'

n_layer = 28
n_embd = 1152
n_head = 18                     # head size 64
n_programs = 129
pos_emb = 'rope'
block_size = 2048
norm = 'rmsnorm'
mlp = 'swiglu'                  # hidden 3072
qk_norm = True
dropout = 0.0
label_smoothing = 0.0
cond_inst = True                # condition on the window's instrument set (A/B 2026-10-03: no CE cost, out-of-band
n_density = 16                  # notes 50% -> 25-30%, density follows the request in the right direction)
cond_dropout = 0.15
special_tokens = True           # BOS/EOS: needed by pack_short and by the fine-tunes
pack_short = True               # short files share windows: GigaMIDI + Aria usable notes 1.81B -> 2.50B

batch_size = 4                  # overridden per machine; batch_size x accumulation x 2048 = 65,536 notes/step
gradient_accumulation_steps = 8
max_iters = 100000              # overridden from the budget by cloud/preflight.py
lr_decay_iters = 100000
warmup_iters = 1000

optimizer_name = 'muon'
learning_rate = 1e-3            # see the LR probe in agents.md before changing
min_lr = 0.0
lr_schedule = 'wsd'
cooldown_frac = 0.2
muon_bf16 = True                # also keeps the checkpoint loadable on the 8 GB laptop

act_ckpt = 0                    # overridden per machine (5090: none; laptop: all blocks at micro 1)
act_ckpt_save = 'attn'
fp8 = True
fp8_cache_weights = True
compile = True
num_workers = 8

eval_interval = 2000
eval_iters = 50
sample_eval_rows = 8            # at every eval also sample 8 fixed multi-instrument prompts x 1,000 notes (~1 min on
sample_eval_notes = 1000        # the 5090) and log samples/<none|band>/<metric> (sample_eval.py)
checkpoint_format = 'bf16'      # best-val weights for sampling; the resumable state is out_dir/ckpt.pt
ckpt_interval_min = 30.0
seed = 1337
wandb_project = 'music-transformer'
