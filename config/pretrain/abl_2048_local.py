# abl_2048.py on the stores still on the laptop (2026-10-23): gigamidi_train and gigamidi_clean are only on the Hub,
# so GigaMIDI replay (134,538 files / 167.9M notes; 98M notes per arm = 0.4 passes at 0.7) + Aria, validated on the
# GigaMIDI held-out store. Not comparable with the abl_* logs of 2026-10-03. New block (RMSNorm + SwiGLU + QK-norm).
# Used by tools/sweeps/muon_ab.py.
csv_path = 'store:data/cache/gigamidi_replay,store:data/cache/aria'
val_csv_path = 'store:data/cache/gigamidi_heldout'
val2_csv_path = 'store:data/cache/aria'
norm = 'rmsnorm'
mlp = 'swiglu'
qk_norm = True
