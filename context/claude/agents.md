# MusicAutoregressiveModel: agent context

Read this before touching the code. It records what the project is, what state it's in, what we know is
wrong, and the ranked list of experiments. Update the **Experiment log** at the bottom after every run.
Last full update: **2026-09-28 22:41**.

## Goal

1. Pretrain an autoregressive symbolic-music model on solo piano MIDI (MAESTRO v3 + GiantMIDI-Piano + **Aria-MIDI**).
2. Fine-tune it to generate **piano arrangements in the style of Skryabin** (Скрябін, the Ukrainian
   pop-rock band, not the composer) and similar Ukrainian, mostly minor-key, pop-rock: melody + bass + chords on one piano.
3. Later (agreed order): **multi-track** band arrangements (instrument attribute in the cascade, Lakh MIDI pretraining,
   fine-tune on per-stem transcriptions), and a **Mamba/SSM** comparison.

Constraints: **laptop only** for now. RTX 5060 Laptop (8 GB VRAM, Blackwell sm_120, 24 CPU threads), 30 GB RAM,
runs of a few hours, overnight at most. A rented B200 was discussed as a possible later step (see "Scaling outlook").

## Repository layout

| Path | What it is |
|---|---|
| `model.py` | nanoGPT-style decoder. `MusicEmbeddings` concatenates 4 embeddings (pitch/velocity/duration/delta_time, each `n_embd/4`), learned absolute positions, flash SDPA. **Compound tokens: 1 position = 1 note with all 4 attributes.** Output: `CascadeHeads` predicts **dt → pitch → duration → velocity**, each head conditioned on the earlier attributes (chain rule). v2 = `ResidualHead` `out(LN(z + MLP(LN(z))))` with z = h + cond embeddings (default, `cascade_residual=True`). v1 = plain MLP heads (`cascade_residual=False`, the `ab_cascade` checkpoint). Legacy = 4 independent heads (`cascade_heads=False`, pre-2026-09-28 checkpoints). Options: `pitch_head_blocks`/`pitch_head_mult` (asymmetric bigger pitch head), `moe_experts`/`moe_top_k`/`moe_hidden_frac`/`moe_aux_weight` (dropless top-k MoE FFN, `MoEMLP`). `forward(..., targets)` returns `(parts, loss)`: loss = optimized sum of CEs (+ MoE aux), parts = per-head CE without label smoothing |
| `data_loader.py` | `MaestroDataset`: `csv_path` can combine CSVs and prebuilt stores (`'a.csv,store:data/cache/aria'`). Tokens live in a **memory-mapped flat store** per split (`data/cache/<name>_<split>_<hash>/tokens.u16`, shape (notes, 4) uint16, + `offsets.npy` + `meta.json`), built in parallel (24 procs). Train windows are sampled **uniformly over all note positions** (long files proportionally more often). `eval_stride` gives deterministic val windows. Broken or missing files are skipped. Transposition ±5 is limited so notes stay in 0..127. Optional `aug_tempo` (log-uniform time stretch of duration + delta_time, stochastic rounding) and `aug_velocity` (symmetric ±N-bin shift, clamped), both set from `train.py`. `tokenize_midi()` accepts paths or file-like objects. Time is quantized at 20 ms, clamped to 511 bins |
| `train.py` | Single training script. nanoGPT `configurator.py`: `python train.py config/<x>.py --key=value`. `init_from` = scratch, resume or finetune. `val_csv_path` = main val (checkpoint selection), `val2_csv_path` = extra val logged as `val2/*`. Flags: `compile`, `fp8` (both **on by default** since 2026-09-29; fp8 falls back to bf16 below sm_89; torchao float8 on the transformer blocks, converted **before** the optimizer is built), `sdpa_backend`, `lr_schedule` ('cosine' | 'wsd': constant then 1−√ cooldown over the last `cooldown_frac`, Hägele et al. 2024), `ckpt_interval_min` (>0: full resumable `out_dir/ckpt.pt` every N min + at the end, atomic tmp+rename; the best-val save then goes to `best.pt`/`model_bf16.pt`; resume reseeds data with seed+iter and continues the same wandb run), `source_weights` ('0.8,0.2': sampling weight per train store; CSVs = one store, then each `store:`), `special_tokens` (BOS/EOS around every piece: pitch 128/129, pitch vocab 130; a finetune grows a 128 checkpoint via `GPT.load_expanded`), `style_map` (conditioning: style embedding added at every position, zero-init; CSV column style/artist, store metadata genre, combined.csv = classical), `style_dropout` (0.1), `style_lr_mult` (Muon runs: style table in its own AdamW group, no wd, LR × mult), `boundary_frac` (0.1: windows placed at piece start/end, else only ~0.2% see BOS/EOS). Prints "Planned epochs" at startup, so check it |
| `config/` | `pretrain.py`, `finetune_skryabin.py`, `smoke.py` (stack after another config), `ab_heads.py` (4-epoch A/B of the head designs, 6M), `ladder.py` (scaling ladder, 100M tokens, MAESTRO+GiantMIDI+Aria), `big.py` (the 14 h run: 107M, WSD, resumable) |
| `generate.py` | Samples from a checkpoint, writes MIDI, renders MP3 (fluidsynth + ffmpeg). `--prompt file.mid --prompt-notes 64`, or a validation piece: `--val-prompt combined|aria [--val-index i | --val-genre pop] [--prompt-start n]` (`--list-val` lists them; Aria labels show file_id/genre/composer). `--checkpoint` takes a file or a run dir (model_bf16.pt → best.pt → ckpt.pt), `--num-samples N`, `--device cpu --threads 6` to sample next to a training run (slows training ≈ 9%). **KV cache** in `GPT.generate` (always on, preallocated `KVCache`; matches a full re-run to 1e-5). **Batched**: `--num-samples N [--batch-size B]`, one prompt per row (random val piece per row), per-row density control; a batch costs about one sample on GPU (B=32: 3,323 notes/s vs 283 at B=1, shared GPU). **CUDA graph** for the one-note step on CUDA (`--no-cuda-graph` to disable, identical notes), bf16 on CUDA (`--dtype`). Remaining speed ideas: `context/claude/optimizations.md`. Learned absolute positions → past 512 notes the cache is rebuilt from a shorter window every `--slide` notes (default 128). Optional `--anchor N` (off by default) pins the first N notes (the prompt's theme) in front of the recent window. Density control without retraining (off by default): `--dt-bias b` adds b·log(max(bin,1)) to the delta_time logits (chord bin 0 keeps its odds vs bin 1). Final model: **no bias** (density +15% vs real, best key-sim) or **0.1** (exact density); 0.3 was right only for the mid-training checkpoint. `--density N|prompt` = controller tracking a notes/s target. Post-processing is off by default. Builds the config from every checkpoint field (legacy defaults for old checkpoints) |
| `sample_sweep.py` | Compares sampling settings without listening: the same real Aria val prompts under `--configs "T=1.0" "T=1.0,dt_bias=0.3" ...` (keys T, top_k, dt_bias, density, anchor); reports notes/s, same-onset %, velocity, duration and key-sim vs the real continuation. `--device cpu` next to training |
| `jobstatus.py` | Progress files `logs/jobs/<name>_<pid>.json` written by generate.py and sample_sweep.py; `status.py` shows them as "Sampling jobs" |
| `eval_samples.py` | Generated vs. real MIDI: per-feature histogram overlap plus notes/s and pitch-class entropy |
| `export_bf16.py` | Checkpoint → bf16 weights without optimizer state (74 → 12 MB for 6M). Loadable by `generate.py`, not resumable |
| `optim.py` | `Muon` (Newton-Schulz orthogonalized momentum, Moonlight 0.2·√max(m,n) scaling so it shares AdamW's lr/wd/schedule), `CombinedOptimizer`, `build_muon_optimizer` (Muon for 2D matrices in `transformer.h`, AdamW for the rest). Enabled with `train.py --optimizer_name=muon` |
| `bench.py` | GPU-side throughput benchmark on synthetic data: micro-batch × compile × SDPA backend × fp8, `--profile` for a kernel breakdown |
| `status.py` | **Live dashboard** (`.venv/bin/python status.py`, `--once`): GPU/disk, every training run (state, progress, speed, ETA, val/Aria CE, per head), download and reduction progress. Read-only |
| `prepare_giantmidi.py` | Builds `data/combined.csv` (MAESTRO + GiantMIDI, GiantMIDI split by composer). Fixed 2026-09-28: ignores the extra MAESTRO CSV columns |
| `data/prepare_finetune.py` | Ukrainian reductions → `data/finetune/ukrainian.csv` (split, midi_filename, artist, title): titles from `data/audio/songs.csv` via the YouTube id or the file name, transliterated; **dedupe** per (artist, title) incl. `skryabin_local` = Скрябін (18 dropped); **song-level split** by title hash (10%): 502 songs = 452 train / 50 val, ~501k / 60k notes |
| `data/subset_aria.py` | Genre subset of the Aria token store without the archive: `--genres pop,rock --name aria_poprock` → `data/cache/aria_poprock_*` (62,108 train files, 92.7M notes) + `data/aria/aria_poprock.csv` |
| `data/styles.json` | Style vocabulary (index = id, 0 = none): 10 Aria genres + 18 Ukrainian artists + `reduction` (id 29, domain label). **Tied to trained checkpoints: only append, never reorder** (a fine-tune grows the table) |
| `data/prepare_aria.py` | Tokenizes **Aria-MIDI straight from the .tar.gz** (no extraction) into stores `data/cache/<name>_{train,validation}`, 1% of recordings to val (by file_id), `--genres pop,rock` filter, metadata CSV `data/aria/<name>.csv` (file_id, segment, split, genre, composer, audio_score, n_notes) |
| `data/fetch_songs.py`, `data/artists.txt` | YouTube search per artist (aliases with `|`, e.g. `Танок на Майдані Конго|ТНМК`), filters (duration 90–480 s, blocklist of live/cover/compilation/хіти words, title must contain the artist), dedupe by song, yt-dlp mp3 download, retry with backoff on network errors. Output `data/audio/<artist>/` + `data/audio/songs.csv`. Re-runs skip what's already downloaded. `--quality` sets the mp3 kbps |
| `data/audio_to_piano.py` | Band mp3 → **piano reduction**: Demucs `htdemucs_6s` → basic-pitch per part: vocals → monophonic melody (merged re-triggers, legato), bass → monophonic bass (legato), piano+guitar+other → harmony (onset 0.6, frame 0.4, min 100 ms, merge re-triggers, drop the quietest 30%, max 3 notes at once). Writes `midi/`, `debug/*.parts.mid` (one track per part), `songs.csv` with a key estimate. **Stems are deleted after each song** (~350 MB/song) unless `--keep-stems`. `--delete-audio` also deletes each newly reduced mp3. Runs in `.venv-audio` |
| `data/preprocess.py`, `data/mp3_to_midi.py` | MAESTRO download; the old full-mix piano transcription (superseded by `audio_to_piano.py`) |
| `train_finetune.py` | **Deprecated**, replaced by `train.py config/finetune_skryabin.py` |
| `logs/` (git-ignored) | Run logs plus runner scripts: `run_ab.sh`, `run_ladder.sh`, `run_fp8_L.sh`, `run_L_variants.sh`, `run_reduce.sh` |

**Environments.**
- `.venv/`: training. Python 3.12, torch 2.14.0+cu130, torchao 0.18, rich, wandb. The system `python3` has no torch.
- `.venv-audio/`: audio. torch/torchaudio cu130, demucs, **basic-pitch 0.4 on onnxruntime** (installed `--no-deps`, because its
  TensorFlow pin has no Python 3.12 wheels), yt-dlp, librosa, piano_transcription_inference.
- Verified on sm_120: bf16 matmuls, flash SDPA, cuDNN attention, fused AdamW, torch.compile, torchao float8.
- wandb project `music-transformer` (logged in via `~/.netrc`).

## Data (2026-09-28)

| Source | Where | Train | Val | Notes |
|---|---|---|---|---|
| MAESTRO v3 | `data/20xx/` | 962 files | 137 | `combined.csv`. The 177 test rows are ignored |
| GiantMIDI **curated** (surname-checked, Google Drive release) | `data/giantmidi/midi/surname_checked_midis/` | 6,761 | 475 | split by composer, absolute paths in the CSV |
| → `data/combined.csv` store | `data/cache/combined_*` | 7,301 files > 512 notes, **31.3M notes** | 2.11M notes | tokenizes in about 1 min |
| **Aria-MIDI deduped** (CC-BY-NC-SA 4.0, user accepted the disclaimer) | archive `data/aria/aria-midi-v1-deduped-ext.tar.gz` (2 GB, kept) → `data/cache/aria_{train,validation}` | 367,318 files, **550.8M notes** | 3,735 files, 5.7M notes | 0 failed files. Genres: classical 113k, pop 70k, soundtrack 55k, jazz 19k, **rock 6.7k**, none 94k |
| Ladder training mix (`combined.csv,store:aria`) | | 338,982 files > 512 notes, **568M notes** | | Aria is about 95% of the notes |

**Validation sets.** `val` = the old MAESTRO+GiantMIDI val windows. They're **identical** to the A/B runs: the v2 checkpoint
re-evaluates to exactly 9.412 after the storage rewrite. `val2` = Aria val. **Caution:** once training includes Aria, the old
val is off-distribution (train ≈ Aria val, while old val is about 0.6–0.8 higher). Compare old-val numbers only within the same
training mix.

**Fine-tune audio (2026-09-28 22:41).**
- 18 artists in `data/artists.txt`, **493 songs** downloaded, about 25 per artist. Motor'rolla found only 14.
- ТНМК needed the alias and now has 25.
- About 17 videos failed: HTTP 403 or age-restricted. Age-restricted videos need browser cookies, which we deliberately don't use.
- 27 hand-picked mp3s in `Skryabin/`.
- **Piano reduction: DONE 2026-09-29 01:50.** **520 songs** (+3 test songs) in `data/finetune/<artist>/midi/`, 0 failures,
  51 MB total. Key estimate: about 349 minor / 133 major (unverified estimator). Some artists have >25 songs because re-downloads of
  failed songs picked other videos: **dedupe by song title** before building the fine-tune split. Also dedupe `skryabin_local`
  against `Скрябін`.
- Test songs with version history: `data/finetune/skryabin_test/{v1 (cluttered), v2 (decluttered), current (legato)}/`.
- **User verdict:** "can recognize tracks by the piano version alone". The cluttered v1 sounded busy and v2 "too fast"
  (choppy), which the legato fix addressed ("sounds fine").
- Key estimates came out "major" for all 3 test songs, **unverified**. The minor-key filter will need a better key detector.

## Known bugs / defects

- **B1 FIXED**: independent heads → CascadeHeads v2 (−0.41 nats/note vs independent at 4 epochs, see log).
- **B2 FIXED**: per-head CE without label smoothing is logged (`val/ce/<head>`, `train/ce/<head>`).
- **B3 FIXED**: `init_from='finetune'` resets optimizer, iter and best_val_loss.
- **B4 FIXED**: deterministic strided val windows, evenly thinned to `eval_iters` batches.
- **B5 worked around** with `--prompt` (BOS token still TODO).
- **B6 FIXED**: post-processing in `generate.py` is off by default.
- **B7 partly fixed**: the transposition no longer clamps. Tempo and velocity augmentation are still TODO.
- **B8 in progress**: band-mix transcription replaced by the Demucs + basic-pitch piano reduction.
- **B9 FIXED**: memory-mapped token stores.
- Old data losses: the original `combined.csv` was overwritten by a crashed `prepare_giantmidi.py` run. It was rebuilt, so nothing depends on it.

## Experiment findings (2026-09-28)

**1. Head design, 6M model, 4 epochs on MAESTRO+GiantMIDI (`config/ab_heads.py`).**
- Cascade v2 (residual heads) 9.412 < cascade v1 9.793 < independent 9.824.
- Shuffling the conditioning showed v1 *does* use it: +0.28 on pitch|dt, +0.47 on duration, +0.85 on velocity.
- v1 lost on pitch because the fresh MLP head had no direct path and the condition embeddings stayed tiny (rms 0.03–0.15 vs about 1).
  v2 fixed both: residual form, cond_emb init std 0.5, no weight decay on cond_emb.
- In v2, pitch equals independent: knowing dt doesn't help pitch beyond h.

**2. Scaling ladder**: 100M tokens (0.2 epoch of 568M notes), dropout 0, LS 0, lr 6e-4 cosine, block 512, eager bf16.

| run | params | old val CE | Aria val CE | pitch / vel / dur / dt (old val) |
|---|---|---|---|---|
| S | 6.1M | 9.438 | 8.613 | 2.735 / 2.082 / 2.919 / 1.703 |
| M | 20.2M | **8.891** | **7.931** | 2.428 / 2.030 / 2.826 / 1.607 |
| L | 41.7M | **8.623** | **7.601** | 2.265 / 2.006 / 2.782 / 1.569 |
| XL | 64.7M (12L×640) | **8.472** | **7.399** | 2.185 / 1.990 / 2.755 / 1.542 |

- S → M: −0.55 old val / −0.68 Aria. More than half of it is **pitch** (−0.31), while velocity is nearly saturated (−0.05).
  Clearly capacity-bound, so scale.
- S on the Aria mix reaches 9.438 on the old val. The A/B v2 run, trained only on that data at 125M tokens, got 9.412. So Aria transfers well.

**3. Performance on sm_120 (`bench.py`, L model, synthetic data, ladder paused with SIGSTOP).**
- Kernel time: **GEMM 77%** (`cutlass_80_tensorop_bf16_s16816gemm`, 64×64 tiles: Ampere-style mma.sync, expected for bf16 on
  consumer Blackwell, which has no wgmma/tcgen05), flash attention about 9%, elementwise 17%, LayerNorm 5%, CE + optimizer < 1%.
- The 6M model is launch/CPU-bound: GPU at 54 W, Python at 91% CPU. The 42M model is compute-bound at about 16 TFLOP/s useful.

| config (L, tokens/s) | micro 12 | micro 30 |
|---|---|---|
| eager bf16 (micro 6 = 62.6k) | 61.0k | 55.5k |
| + compile | 64.6k | 62.3k |
| + compile + cuDNN attention | 66.1k | 62.8k |
| fp8 eager | 31.0k | 28.5k (unfused casts, don't) |
| **fp8 + compile** | 69.9k | **75.1k (+20%, −25% memory)** |

- Attention isn't worth optimizing at context 512. It will matter at 1024+.
- FP8 gains should grow with width. Its quality check is queued (ladder-L-fp8).
- Checkpoints: fp32 master weights + fp32 Adam, with bf16 autocast compute. Pure bf16 weights would lose updates, so don't.
  Use `export_bf16.py` for storage and inference.

## Scaling outlook (rules of thumb, computed 2026-09-28)

- Laptop: about 1.6–1.9·10¹³ useful FLOP/s. Chinchilla-optimal N = √(C/120): 4 h → about 47M, 8 h → about 66M, 24 h → about 115M,
  1 week → about 300M.
- **Data ceiling:** 582M unique notes × about 4 useful passes → about 115M params. More data (Aria pruned, about 1.2B notes) raises it to about 250M.
- VRAM ceiling: 16 bytes/param → about 250–300M realistic with activations.
- **Practical laptop sweet spot: 60–115M.**
- One B200 for a week: about 5·10²⁰ (bf16) to 1·10²¹ (fp8) FLOPs → about 2–3B params compute-optimal. But all public symbolic music
  (order-of-magnitude estimate 5–10B notes) supports only about 1–2B. GPT-3 175B is about 300–600× more compute, needs 2.8 TB of training
  state, and needs about 100–500× more music data than exists, so it's not feasible. Published SOTA symbolic models are 0.6–1B
  (Aria, Anticipatory Music Transformer). Suggested path: laptop ladder → a 1–2 day B200 run at about 300–500M → a week-long run only if
  scaling holds. Missing for that: FSDP (multi-GPU only), gradient checkpointing, WSD schedule.

## Queue (2026-09-28 22:41; `status.py` shows it live)

| # | Run | Log | Purpose |
|---|---|---|---|
| 1 | ladder-L 42M | `logs/ladder_L.log` | scaling (running, slowed to about 1–2 it/s by the reduction sharing the GPU) |
| 2 | ladder-XL 65M | `logs/ladder_XL.log` | scaling |
| 3 | ladder-L-fp8 (`--fp8=True --compile=True`) | `logs/ladder_L_fp8.log` | fp8 quality vs. bf16 L. The speed comparison is noisy because of contention; trust `bench.py` for speed |
| 4 | ladder-L-pitchhead (`--pitch_head_blocks=2 --pitch_head_mult=2`, 43.3M) | `logs/ladder_L_pitchhead.log` | capacity where the ladder says it pays (pitch) |
| 5 | ladder-L-moe8 (`--moe_experts=8 --moe_top_k=2`, 117M total / about 42M active) | `logs/ladder_L_moe8.log` | learned MoE. Compare per step **and** per wall-clock hour (eager, python loop over experts, expect about neutral per hour on this GPU) |
| 6 | iso-FLOP at C ≈ 4.85e16: iso-M (20M, 400M tok, 13,020 it), iso-L (42M, 194M tok, 6,315 it), iso-XL (65M, 125M tok, 4,070 it), compile on | `logs/iso_*.log`, runner `logs/run_isoflop.sh` | which size wins at equal compute → overnight model size |
| 7 | ladder-L-lr3e-4, ladder-L-lr1e-3, ladder-L-muon (`optimizer_name=muon`, Moonlight-scaled Muon for block matrices + AdamW rest, same lr/wd) | `logs/ladder_L_{lr3e-4,lr1e-3,muon}.log`, runner `logs/run_opt.sh` | best peak LR, and Muon vs AdamW (baseline ladder-L = AdamW 6e-4). Compare Muon per step **and** per hour: smoke on 6M was 12.85 vs 13.28 val after 60 steps but about 2× slower per step |
| 8 | Done: ladder-L-muon-lr1e-3 and iso-{S,M,L}-muon. With Muon the iso-FLOP optimum at C≈4.85e16 is **M (~20M, parabola fit 22M, ~16 tok/param)**. S +0.173, L +0.040 Aria vs M. iso-XL-muon was skipped at the user's call, since L was already worse | `logs/iso_{S,M,L}_muon.log` | does Muon shift the iso-FLOP size optimum → **no, it stays at M** |
| 9 | **big-107M done** 2026-09-30 06:50 (12 h 27 min + evals, one resume at 18:23 worked). Final **Aria 5.706**, val 6.987 (in the predicted 5.47–5.72). Checkpoints: `checkpoints/big_107M/model_bf16.pt` (final = best), `ckpt.pt` (resumable). Samples: `samples/big_final/` | `logs/big_107M.log` | the model for fine-tuning |
| 10 | **Fine-tune stages done (2026-09-30)**: A (LR sweep, old split), B1 (pop+rock + BOS/EOS + genre styles, shortened by resume to a 300-step cooldown at 2,412), A2 (plain, fixed split, LR 3e-4), B2 (from B1, artist styles). Best: `checkpoints/ua_A2/model_bf16.pt` (iter 400), `checkpoints/ua_B2/model_bf16.pt` (iter 350, has BOS/EOS + styles). Samples: `samples/ua_A2/`, `samples/ua_B2/` | `logs/ua_A*.log`, `logs/poprock_B1.log`, `logs/ua_B2.log` | see the log rows |
| 11 | **Bigger fine-tune set + augmentation (2026-10-01)**. (a) `logs/run_fetch_reduce2.sh`: up to 50 songs/artist, 15 new artists appended to `data/artists.txt` (33 total), mp3 at 128 kbps, reduced while downloading with `--delete-audio` (new mp3s are deleted after their MIDI is written; re-downloadable via `data/audio/songs.csv`). Disk freed first with the user's OK: `test.pt` (May baseline), `ladder_L_moe8`, `ua_A_lr1e-3`, `ua_A_lr1e-4`. (b) `logs/run_C.sh`: augmentation ablation on the current 448 songs, B3 ×30 recipe: `C_tempo` (`aug_tempo=0.1`), `C_tempovel` (+ `aug_velocity=2`), `C_seed2` (baseline, seed 2 = noise estimate). Then: rebuild the split with `prepare_finetune.py --style reduction` (the hash split keeps old songs on their side), fine-tune with the winning augmentation, and still report the old 48-song val | `logs/fetch2.log`, `logs/reduce2.log`, `logs/ua_C_*.log` | does augmentation delay overfitting; how much does 3× data help |
| — | piano reduction of all downloaded songs | `logs/reduce.log` | ETA about 01:00–01:30. ТНМК songs may need one more `logs/run_reduce.sh` pass afterwards |

## Plan for the 24 h run (agreed direction, 2026-09-28)

WSD schedule (constant LR, then decay to about 0 over the last about 15%) instead of cosine. Resumable checkpoints every about 30 min
(`checkpoint_format='full'`). Grad-norm logging plus loss-spike skipping. No weight decay on embeddings and norms. Keep lr·wd with an
EMA timescale of about 20–25% of the run. 64k–128k tokens/step (optionally ramped). **Muon** (Moonlight-scaled) + AdamW for embeddings/heads, lr 6e-4 (maybe 1e-3, to check), fp8 + compile, dense trunk. Optional: weight EMA / checkpoint averaging, QK-LayerNorm. Model size from the iso-FLOP result, expected 60–115M.

## Hypotheses / next steps, ranked

1. ✅ Ladder + iso-FLOP done. **At equal compute, smaller + more tokens wins** (iso-M 6.770 < iso-L 6.954 < iso-XL 7.175): music
   needs ≥ 20 tokens/param. Size the 24 h run by tokens: about 2–3e18 FLOPs with boost → **about 70–100M params** at ≥ 20 tok/param
   (lower if an iso-S run shows the optimum is below M). The fixed-token ladder alone overstated the value of size.
2. ✅ fp8 + compile adopted for the long run (+0.02 nats, −14% to −20% time).
2b. ✅ **Muon adopted** (−0.54 Aria at L, about 2× token efficiency, +6% time/step). LR: AdamW flat at 6e-4 to 1e-3, 3e-4 too low.
    Optional: check a higher LR for Muon (Moonlight scaling, e.g. 1e-3) before the 24 h run.
3. ❌ Asymmetric pitch head rejected (+0.04). ❌ MoE rejected on this laptop (+0.10 per step, 2.5× slower). Scale the dense trunk.
4. **Fine-tune stages:** Aria **pop+rock** subset (`prepare_aria.py --genres pop,rock` or filter `aria.csv`, about 77k files) as the middle
   domain → the Ukrainian reductions (song-level split, artist/style token, replay 10–30% pretraining data, LoRA or low LR).
5. Better key detection (audio-based) for the minor-key filter. Dedupe covers of the same song across `Skryabin/` and `data/audio/Скрябін`.
6. Context 1024 + RoPE (attention becomes relevant, test cuDNN attention again). Tempo and velocity augmentation (B7). BOS token (B5).
7. Beat-based tokens (REMI-like) for the pop target, compared against the performance-timing tokens.
8. Stage 2 multi-track: an instrument attribute in the cascade (dt → instrument → pitch → dur → vel), instrument-type experts,
   Lakh MIDI pretraining, Demucs stems transcribed per instrument (drums need a drum transcriber).
9. Diffusion for arrangement/infilling: see "Diffusion idea". Start with masked discrete diffusion on our tokens.
10. Mamba/SSM: controlled comparison (same tokens, params and budget), mainly for long context. `mamba-ssm` on sm_120 may need a source build.

## Architecture options (2026-09-30)

Full table with effects, costs, papers and **the user's verdict/notes columns**: `context/claude/architecture.md`.
Don't overwrite the user's columns there; record test results in the experiment log and update the status column.
Summary of the agent's suggested order:
1. Next pretraining run: RoPE + RMSNorm + SwiGLU + QK-norm, start/end tokens, conditioning tokens (genre/artist/density), context 1024.
2. Fine-tune: context extension to 2048+ (PI/YaRN), style token, Anticipatory-Music-Transformer-style melody → accompaniment.
3. Only if long-form structure still fails: learned memory tokens (RMT/AutoCompressors) or Museformer bar attention.
4. Research: hybrid SSM / linear attention (flash-linear-attention is Triton, likely easier on sm_120 than mamba-ssm), masked diffusion.
Tested and rejected: MoE FFN, bigger pitch head. Speed work (not architecture) lives in `context/claude/optimizations.md`.

## Multi-instrument pretraining: to-do (2026-10-01)

Goal: one model over all instruments (GigaMIDI + Aria + reductions), grown from `big_107M` so the piano knowledge is kept
(function-preserving init: new parts start at zero, step-0 loss on piano val must equal the old model's), then fine-tuned on
Ukrainian multi-track songs. Generation gets `--instrument` (mask the instrument head to an allowed set).

**Data**
1. ☑ Disk: /home resized to 115 GB (54 GB free on 2026-10-01); / is 97 GB (67 GB free, root-owned). Probe: ~21 KB/file,
   validation = 213,621 files → ~4.7 GB, so all ~2.1M files ≈ 47 GB. Tight on /home.
2. ◐ Tokenize GigaMIDI (`data/prepare_gigamidi.py`, log `logs/prepare_gigamidi.log`, `--min-free-gb 5`, on the dashboard).
   Validation 209,500 files / 245M notes, test 209,492 / 244M (≈2% failed). **Train nests one level deeper** (category
   .zips inside the training zip) and first came out empty; fixed (each category zip decompressed in memory), restarted
   2026-10-01 23:00 (~1.7M files). Share kept by the loader (files > block_size): 92% of notes at 512, 80% at 2048
   (the dropped files are mostly short drums-only loops). Then the 5.5 GB zip can go (ask the user).
3. ☐ Expressive vs quantized label per file (GigaMIDI is ~70% machine-timed, constant velocity): compute an onset-on-grid
   fraction ourselves (the zip has no NOMML) → style label, so piano expressiveness isn't diluted. Optional: the HF metadata CSV
   (1.25 GB, gated) for genres.
4. ☐ Existing piano stores (Aria, MAESTRO, GiantMIDI, reductions) = program 0 implicitly (no re-tokenizing).
5. ☐ Ukrainian multi-track data: the mp3s were deleted after reduction → re-download from `data/audio/songs.csv` (minus
   `exclude.txt`), Demucs stems → basic-pitch per stem (bass → 33, vocals → melody program, other → guitar/keys), drums need a
   drum transcriber (ADTOF / Omnizart). Process + delete per song, like `--delete-audio`.
6. ☐ Mix weights (GigaMIDI / Aria piano / reductions) and a filter for pathological files (1-note, >30 min, drums-only share).

**Tokens / model**
7. ☑ (2026-10-01, `n_programs=129`; smoke-tested, see log) Instrument attribute: 129 values (128 GM programs + drums) with an embedding added like the style table; head in the cascade
   dt → instrument → pitch → dur → vel (pitch/dur/vel heads see the instrument). Drums: duration loss masked.
8. ☑ Context 2048 with **RoPE** (user's call 2026-10-01): `pos_emb='rope'` (q/k rotated in every layer, `rope_base` 10000,
   no parameters; `'learned'` stays the default for old checkpoints). A finetune may switch to it and raise `block_size`;
   `wpe` is dropped. **Not function-preserving:** big_107M with RoPE instead of wpe = 12.59 vs 5.82 on Aria at step 0
   (random ≈ 21), so the pilot (13) must measure the recovery. Bench (107M, fp8+compile): 512 learned 50.7k notes/s,
   512 RoPE 47.2k (−7%, cos/sin recomputed per forward, could be cached), 2048 RoPE 38.8k at 4.1 GB peak (micro 4).
9. ☐ Growth path in `load_grown`: zero-init instrument embedding + new input columns of the heads (function-preserving),
   optional depth/width up-scaling (duplicate layers with zero-init output projections) for 250–400M.
10. ☐ Memory check on 8 GB: 400M + Muon at 2048 context likely needs activation checkpointing; fall back to ~250M.

**Infra**
11. ◐ Loader: `programs=True` reads `programs.u8` (piano stores = 0), drums not transposed, drum duration 0 in x / -1 in
   y. train.py passes program/style as keyword inputs, logs `ce/program`, `ce/total_all` (`ce/total` stays the 4 note
   heads). Windows at 2048 work with RoPE (smoke-tested).
12. ☐ Eval: per-head CE incl. instrument; val sets GigaMIDI val, Aria val (piano forgetting), Ukrainian reductions.
13. ☐ Pilots (1–2 h each) before the long run: grown vs scratch at equal steps (does preserving weights pay?), RoPE vs interpolated
   wpe, 129 programs vs 17 families. Then size the multi-day run (compute-optimal ≈ 400M / 8B tokens per week on this laptop).
14. ☐ Long run with `ckpt_interval_min` (test crash-resume first), dashboard rows.

**Generation**
15. ☐ `generate.py`: sample the instrument head, `--instrument` allowed set (mask logits), write one MIDI track per program
   (drums on channel 10), per-instrument polyphony limits.
16. ☐ Ukrainian multi-track fine-tune (data from 5), same recipe as ua-D (replay, tempo aug, steps ∝ songs).

## Diffusion idea (discussed 2026-09-28, not started)

Diffusion would **add** a capability (arrangement and editing) rather than replace the AR model. The natural time is after the 24 h
pretraining and the Skryabin fine-tune, so there's a strong AR baseline to compare against.

- **Audio diffusion** (Stable Audio, AudioLDM, MusicLDM): generates waveforms and skips MIDI. Out of scope: large models and data,
  copyright and voice imitation issues, a different pipeline.
- **Symbolic diffusion, the relevant family:** Polyffusion (piano-roll U-Net on POP909, 2023), whole-song hierarchical cascaded diffusion
  (ICLR 2024), masked/discrete diffusion on tokens (MaskGIT / MDLM / SEDD style).
- **Gains vs. AR:** native infilling and arrangement ("melody given → accompaniment", "regenerate bars 5–8"), better global structure
  (the whole segment at once), control via conditioning + classifier-free guidance (a "Skryabin-ness" strength knob, chords, density).
- **Costs:** grid/piano-roll representations lose our 20 ms performance timing and velocity nuance. No comparable likelihood, so
  judgment by ear plus `eval_samples.py` stats. 20–1000 denoising steps per segment.
- **Target use:** melody-conditioned **accompaniment generation**, the inverse of `audio_to_piano.py`: take a Skryabin-style vocal line
  (already extracted by Demucs + basic-pitch) and generate the piano part under it.
- **Options, cheapest first:**
  1. **Masked discrete diffusion on our own compound tokens:** same transformer without the causal mask, trained to un-mask randomly
     masked notes/attributes. It reuses the data stores, the heads idea and most of `train.py`. **Recommended first experiment.**
  2. **Piano-roll diffusion (Polyffusion-style)** on beat-quantized data (Aria pop + reductions + POP909). Needs the beat-grid
     representation (hypothesis 7).
  3. **Hybrid / hierarchical:** diffusion plans the structure (bars, chords, melody outline) and the AR model renders the expressive notes.

## Working conventions for agents

- **Git remote:** `origin` = `github.com/DmytroHilei/Music_generative_model` (public). This project is branch **`transformer-v2`**
  (local branch has the same name, so a plain `git push` works). `transformer-v2` is the **GitHub default branch** and, since 2026-09-29, holds
  everything: the old projects were merged in (history kept, unrelated histories) as `legacy/lstm/` (from `main`: the user's
  LSTM chord model + earlier transformer comparison) and `legacy/transformer/` (from `legacy-transformer`: the old
  `../MusicAutoregresiveTransformer` folder). The old branches (`main`, `legacy-transformer`, `master`) were
  deleted on 2026-09-29 at the user's request: `transformer-v2` is the only branch, and their commits live on in its history. **Never force-push.
  Don't create new GitHub repos**: the user wants everything in this one.
- The README of `transformer-v2` will be rewritten later by the user. The local, git-ignored `README_TODO.md` lists what it should cover.

- Make one change per experiment. Log the wandb run id, git commit, config diff and per-head val CE below.
- **Never kill processes by a `pgrep -f`/`pkill -f` pattern in the same shell command that contains that pattern**: it matches its own shell and kills it (happened twice on 2026-09-29/30). Kill by PID or process group from a separate command.
- **Don't delete checkpoints, data, archives or wandb runs without asking.** Before any `rm -rf`, check what's inside.
  Once, the user moved the Aria archive into `data/aria/` right after a partial extraction had been deleted there.
- **GPU power (2026-09-29): FIXED at 00:37.** The user enabled `nvidia-powerd` → 55 W to ~88 W, 1.88 to 2.6 GHz, 69 to 84 °C, roughly
  1.3–1.4× faster training. Runs from ladder-L-pitchhead on have boost, so wall-clock times before and after aren't comparable.
  Before the fix the RTX 5060 was stuck at the **55 W default limit** (max 105 W, throttle reason `0x4` SW power cap)
  because `nvidia-powerd` (Dynamic Boost) had no systemd unit: Ubuntu ships it only in `/usr/share/doc/nvidia-kernel-common-595/`.
  The fix was handed to the user (copy it to `/etc/systemd/system`, then `enable --now`). `logs/gpu.csv` logs power/clocks/throttle every minute.
  No auto-suspend on AC; lid close = suspend (keep it open overnight); platform profile `performance`.
- Watch the disk: 96 GB partition, often under 10 GB free. **2026-09-28:** deleted the Aria archive (fully tokenized, 2 GB) and
  replaced finished `checkpoints/{ab_*,ladder_S,M,L}/ckpt.pt` with verified bf16 `model_bf16.pt` (max rel err ≤ 3.8e-3 = bf16
  precision). `config/ladder.py` sets `checkpoint_format='bf16'`, so ladder, ablation and iso runs are **not resumable**. `ladder_XL`
  started before that and wrote a full `ckpt.pt`, which was converted to bf16 after it finished (verified). Demucs stems are 350 MB/song. Checkpoints with Adam state are about 12 bytes/param.
- Launch at most one training job on the GPU at a time. Benchmark with the training paused (`kill -STOP` / `-CONT`).
- Never edit a bash runner script while it is executing.
- **`pgrep -f` self-match trap (hit twice on 2026-09-28):** a loop like `while pgrep -f X` also matches (a) a trigger whose own
  command line contains X and (b) the *launching shell* if the runner script was written via a heredoc in the same command, because
  the script text is in that shell's argv. Wait on a **PID** (`kill -0 $PID`) or a file marker instead. Queue with a PID-wait trigger instead, and don't use `pgrep -f` with a
  pattern that also appears in the trigger's own command line: that bug blocked the ladder for about 10 min.
- Use `tqdm.write` for eval lines so logs stay parseable by `status.py`.
- sudo needs a password, so hand such commands to the user (`! cmd`).
- Don't commit data, checkpoints, audio or wandb dirs.

## Experiment log

| Date | Run id | Commit | Change | Val CE per head (p/v/d/dt) = total | Notes |
|---|---|---|---|---|---|
| 2026-05-17 | lcqh7fm4 | none | old baseline 6L×256, dropout 0.3, LS 0.1, 30k iters | total with LS only: 11.06 | plateau, train ≈ val |
| 2026-09-28 | smoke | — | Phase 0 pipeline check | — | init CE = ln(vocab) |
| 2026-09-28 | a52uageu (ab-indep) | d5b38ae | independent heads, 6M, 4k × 30,720 tok, dropout 0.1, LS 0 | **2.751** / 2.279 / 3.083 / 1.712 = 9.824 | |
| 2026-09-28 | bcuakmc7 (ab-cascade) | d5b38ae | cascade v1 (MLP heads) | 3.056 / 2.054 / 2.964 / 1.719 = 9.793 | pitch head worse despite dt |
| 2026-09-28 | 9dv8hnzd (ab-cascade-v2) | f4ab505 | cascade v2 residual heads | 2.759 / **2.040** / **2.915** / **1.699** = **9.412** | now the default |
| 2026-09-28 | ladder-S | bc8a5e0 | 6.1M, 100M tok, combined + Aria, dropout 0 | 2.735 / 2.082 / 2.919 / 1.703 = 9.438; Aria 8.613 | |
| 2026-09-28 | ladder-M | bc8a5e0 | 20.2M (10L×384) | 2.428 / 2.030 / 2.826 / 1.607 = **8.891**; Aria **7.931** | −0.55 / −0.68 vs S |
| 2026-09-29 | iso-S-muon | 48ecda2 | 6.1M, 1.33B tokens (43,300 it), Muon lr 1e-3, compile, C≈4.85e16 | val **7.894** (pit 1.850 vel 1.942 dur 2.662 del 1.440); Aria **6.685** | −0.085 Aria vs iso-M AdamW (6.770), but the optimizer differs. iso-M-muon is the fair comparison. First attempt was killed by a reboot at step 4,458 |
| 2026-09-29 | iso-M-muon | 6fd57e0 | 20M, 400M tokens (13,020 it), Muon lr 1e-3, compile + **fp8**, C≈4.85e16, ~40 min | val **7.725** (pit 1.767 vel 1.925 dur 2.626 del 1.407); Aria **6.512** | **Best so far.** −0.173 Aria vs iso-S-muon (6.685) despite the ≈+0.02 fp8 penalty. −0.258 vs iso-M AdamW (6.770), so Muon's gain holds at full horizon. All four heads improve over S, pitch the most (−0.083). The iso-FLOP optimum with Muon is ≥ M |
| 2026-09-29 | iso-L-muon | 7c1f811 | 42M, 194M tokens (6,315 it), Muon lr 1e-3, compile + fp8, C≈4.85e16, ~33 min | val **7.759** (pit 1.785 vel 1.928 dur 2.631 del 1.416); Aria **6.552** | +0.040 Aria vs iso-M-muon (6.512): past the optimum. Muon makes the curve flatter on the large side (AdamW: L +0.184 vs M). Parabola in log N over S/M/L gives an optimum of ≈22M (≈16 tok/param), predicted loss 6.510, so Chinchilla's ~20 tok/param holds. iso-XL-muon not run |
| 2026-09-29 | anchor test | — | big-107M @ iter 8000, CPU, 5 Aria val prompts (128 notes) → 1,200 new notes, seed 1, `anchor` 0 vs 128; stats on the last 400 notes vs the prompt | key-sim (1 − ½·L1 of pitch-class hist) **0.626 → 0.687**, register shift **7.2 → 5.0** semitones | Anchor helps a bit (key better in 4/5, register in 3/5), n=5, noisy. Keep it as an opt-in flag. Samples @ iter 5000 via `eval_samples.py` vs 100 MAESTRO+GiantMIDI val pieces: mean OA **0.72** (5 real val excerpts: 0.82). Generated is denser (14.6 vs ~9–10.5 notes/s), softer (vel 60 vs 69), more polyphonic, partly the Aria style (95% of training) vs a classical reference |
| 2026-09-29 | density sweep | — | big-107M @ iter 9000, CPU, 5 Aria val prompts (64) → 448 notes, `sample_sweep.py` (`samples/sweeps/density*.txt`) | real: 8.2 notes/s, 30.1% same-onset, vel 70. T=1.0: 13.4 / 37.3% / 78, key-sim 0.684. **dt_bias=0.3: 8.8 / 30.1% / 77, key-sim 0.726** | The model is ~1.6× too dense even at T=1 (its own distribution). Lower T makes it worse (T=0.7 collapses into chord stacks). **dt_bias 0.3 matches real density and chord rate and improves key-sim** → recommended. 0.5+ breaks chords into arpeggios (a first version that also biased bin 0 did this even at 0.5: 9% same-onset). `density=prompt` undershoots (intros are calmer than what follows) and has the worst key-sim. Re-check on the final model |
| 2026-09-30 | big-107M final | 2ee9742 | 14L×768×12h, 106.8M, 2.33B tokens (38,000 it × 61,440), Muon lr 1e-3, WSD 20% cooldown to 0, compile + fp8, 12 h 27 min | val **6.987** (pit 1.434 vel 1.821 dur 2.473 del 1.259); Aria **5.706** | **Best model.** Scaling-fit prediction 5.47–5.72 (central 5.6) held. Cooldown (last 20%) gave 5.979 → 5.706 = −0.27. vs iso-M-muon: −0.81 Aria. vs May baseline true CE ≈ 9.8 → 6.99 on the old val |
| 2026-09-30 | final sample check | — | `sample_sweep.py` (5 and 20 Aria val prompts, 448 notes) + `eval_samples.py` vs 100 MAESTRO/GiantMIDI val pieces | **mean OA 0.812** (real excerpts 0.82; step 5000: 0.72). 20 prompts, T=1.0 no bias: 7.1 vs real 6.2 notes/s, 28.7 vs 31.0% same-onset, vel 65.0 vs 65.3, key-sim 0.764 | **The density problem mostly trained away** (+15% vs real, was +63% at iter 9000). New recommendation: **no bias** (best key-sim), or `--dt-bias 0.1` for exact density (fewer chords, 25%). 0.3 now overshoots (6.96 notes/s, 20% chords). Tiersen prompt (1 sample): modulated from C minor toward A♭ major/F minor (key-sim to prompt 0.585), 5.4 vs real 4.2 notes/s |
| 2026-09-30 | decode benchmark | edacd57 | big-107M bf16 on the idle GPU, 64-note prompt → 448 notes | B=1: eager 514, **CUDA graph 630 notes/s (1.22×)**; B=8: 3,830; B=32: 7,437; B=128: 9,691 (graph = eager in batches) | Roofline (weights + KV reads at 384 GB/s): 39% at B=1, 60% at B=128. At B=1 the eager cascade heads + sampling are the next cost (optimizations I3); in batches, full-window KV reads (I6/I7) |
| 2026-09-30 | ua-A (LR sweep) | a50cb83 | big-107M → Ukrainian reductions (old split, 1 leaked song), 20% Aria replay, 600 it × 12,288 tok (~11 passes), Muon WSD, dropout 0.1 | best Ukrainian val: lr 1e-4 **10.564** (still falling), **3e-4 10.524 @400**, 1e-3 10.690 @200 then 11.71 (overfits). Start 11.98 | 3e-4 best; overfits after ~7 passes. Aria val +0.3–0.4 (forgetting despite replay). Duration head barely moves (3.90 → 3.71): legato reductions. Samples vs real reductions: OA 0.68 (big-107M) → 0.81, polyphony 6.1 → 2.6 (real 3.0), density 9.6 → 4.7 (real 5.1) |
| 2026-09-30 | poprock-B1 | a50cb83 | big-107M grown (pitch 130 = BOS/EOS, 29 styles zero-init) on Aria pop+rock (80%) + all Aria (20%), 61,440 tok/step, Muon 5e-4 WSD, boundary_frac 0.1; stopped at 2,412 and resumed with a 300-step cooldown | pop+rock val **5.622 → 5.531** (cooldown −0.034). Ukrainian val 11.91 → **14.57**, all of it **velocity** (2.71 → 4.68; pitch/dur/dt unchanged) | Pop+rock gains little over big-107M. The velocity convention of the reductions (basic-pitch amplitude, mean ~76) differs from Aria; B1 got confidently wrong on it. Not a BOS/EOS/style bug (plain windows, style none/artist all equal) |
| 2026-09-30 | ua-A2 vs ua-B2 | a50cb83 | same fixed split (496 songs, 448/48), same recipe (LR 3e-4, 600 it); B2 starts from B1 with BOS/EOS + artist styles | best Ukrainian val: **A2 10.471 @400**, **B2 10.499 @350** | **The pop+rock stage doesn't help the target loss** (B2 = A2 within noise). B2 fixed B1's velocity drift in 100 steps (5.31 → 2.43) |
| 2026-09-30 | B2 style + EOS check | — | 155 Ukrainian val windows scored with the correct artist style, none, or a wrong artist; unprompted samples up to 1,024 notes (`--min-notes 150`) | correct 10.4753, none +0.0053, wrong +0.0016. Style norms: artists 0.21, genres 0.38. EOS: 1 of 8 unprompted samples ended by itself (769 notes, 160 s) | **Artist style has no measurable effect yet**: ~25 songs per artist, 350 steps from zero init, and the reductions make artists alike (real Скрябін vs Океан Ельзи stats overlap 0.91). EOS rarely fires: past 512 notes the model can't tell how long the piece is. Next: a domain label ('reduction') instead of artists, a higher LR for the style table, or longer context |
| 2026-09-30 | ua-B3 (domain label) | — | like B2 (from B1) but one style 'reduction' for all 448 songs (`data/finetune/ukrainian_reduction.csv`), style table in its own AdamW group, no wd, LR ×10 / ×30 (`style_lr_mult`) | best Ukrainian val **×10 10.492 @400, ×30 10.489 @400** (A2 10.471, B2 10.499). Label vs none: whole windows +0.002 / +0.007; **first 8 notes of a piece +0.040 / +0.043** (B2 artists: +0.012). Style norms 1.7 / 2.7 (B2: 0.2) | Label ≈ free on the loss: after a few notes the context already identifies the domain, and with 80% reductions the whole model is a reduction model. **But it steers generation** (B3 ×30, 8 unprompted samples each): `reduction` 5.50 notes/s, 1.09 notes/onset, pitch 55.3 = real reductions (5.11 / 1.06 / 55.3); `pop` 7.65 / 1.42 / 60.8, halfway to Aria style (9.76 / 1.51 / 62.0); `classical` barely moves (not in the fine-tune mix). **Recommended fine-tuned model: `checkpoints/ua_B3_x30`** (≈ A2 loss + BOS/EOS + a working domain knob) |
| 2026-10-01 | ua-C (augmentation) | 7396237 | B3 ×30 recipe on the 448 songs, 600 it: `C_tempo` (`aug_tempo=0.1`, all stores), `C_tempovel` (+ `aug_velocity=2`), `C_seed2` (no aug, seed 2) | best Ukrainian val: seed2 **10.500 @400** then up to 10.515 (B3 ×30 seed 1: 10.489). tempo **10.479 @599**, tempovel 10.497 @599. val2 (Aria pop+rock) 5.90 → **6.07 / 6.27** | **Tempo aug stops the overfitting** (still falling at 600; no-aug peaks at 400 and rises). Seed noise ≈ ±0.01. Velocity aug hurts: drop it. Augmenting the replay store too inflates val2 (the replay should stay real) → `aug_stores` |
| 2026-10-01 | ua-C3 (tempo on UA only) | c3950fb | as C_tempo with `aug_stores=1,0` (replay real): ±10% vs ±20%, then ±20% at 1,200 it (WSD over 1,200) | ±10% **10.481**, ±20% **10.480** @599 (tie), val2 5.87 (= no-aug level). 1,200 it: minimum **10.499 @600** (LR still high), then rises to 10.59 at the end while train falls 9.78 → 9.35; val2 5.89 | Keeping the replay real fixes val2 at no cost on the target. **±10% = ±20%**. On 448 songs **600 steps is the budget**: at 1,200 it overfits even with tempo aug, and the final decay doesn't recover it. For the bigger set, scale steps with the number of Ukrainian notes (~600 per 448 songs) |
| 2026-10-01 | ua-D (full set) | 7024793 | from B1, 1,280 train / 165 val songs (`ukrainian_reduction_v2.csv`, 101 wrong hits excluded via `data/finetune/exclude.txt`), tempo ±10% on UA only, 1,700 it (600 × 1280/448) | **same-val comparison** (eval_only): old 48-song val C_tempo_ft 10.481 → **D 10.238 (−0.243)**; new 165-song val 10.787 → **10.551 (−0.236)**; val2 5.871 → 5.864 | **2.9× the songs = −0.24 CE, ~24× seed noise**, the biggest fine-tune gain so far (domain label / pop+rock stage / aug: ≤0.03 each). All heads improve, pitch most (−0.09). The dashboard's 10.551 is on the new, harder val (new artists) so it is not comparable to C's 10.481. Still falling at 1,700 (−0.008/100 it), no overfitting, no extra forgetting → more data / steps still pay. **New recommended fine-tuned model: `checkpoints/ua_D_big`** |
| 2026-10-01 | multi-instrument smoke | (this commit) | big_107M grown to `n_programs=129` (`init_from=finetune`), Muon lr 3e-4, 5 warmup, 40 it × 4k notes, GigaMIDI test as train 0.7 + Aria 0.3 | step 0: Aria val 5.728 (4 heads identical to big_107M in a direct check), prog CE 4.860 = ln 129. Step 39: GigaMIDI val 6.676 → 5.299, prog 1.315, **Aria 5.728 → 6.667** | Growth is function-preserving. Piano forgetting is immediate at this LR/warmup: pilots need a long warmup, lower LR and/or more Aria replay |
| 2026-10-01 | RoPE 2048 smoke | (this commit) | big_107M → `pos_emb=rope`, `block_size=2048`, `n_programs=129`, Muon 3e-4, 20 warmup, 120 it × 16k notes, GigaMIDI test 0.7 + Aria 0.3 | Aria val 11.99 → 9.97 @30 → 7.78 @119 (big_107M: 5.73); GigaMIDI val 11.72 → 5.97; prog 0.13 | Recovers fast but far from done after 2M notes; the pilot needs ≥ 100M notes to judge grown-vs-scratch |
| 2026-10-02 | pilot-mi grown vs scratch | b77daac | `config/pilot_mi.py`: 108M, RoPE 2048, 129 programs, GigaMIDI 0.7 + Aria 0.3, 3,000 it × 65,536 = 197M notes, Muon 1e-3 WSD (200 warmup, 20% cooldown); grown = big_107M (wpe dropped), scratch = fresh init; 83 min each | GigaMIDI **grown 2.232** (0.526 / 0.616 / 0.848 / 0.242) vs **scratch 2.597** (0.667 / 0.689 / 0.949 / 0.291); Aria **5.819 vs 6.931**; prog 0.218 vs 0.258 | **Growing pays despite the RoPE switch.** Scratch's final GigaMIDI = grown @ ~1,150 it; end-of-stable (2,250) scratch 2.918 = grown @ ~600 it → ~2.5–4× step efficiency. Aria: scratch never reaches grown @250 (6.35). Gap still closing on GigaMIDI (3.96 @250 → 0.91 @1500 → 0.37 @3000), slower on Aria (1.11). Grown: RoPE step-0 hit 11.80 → 6.35 @250, then flat ~6.07–6.11 through the stable phase, cooldown → 5.82 (big_107M 5.71; at 2048 ctx with 30% replay). Scratch shows a phase transition @750–1250 (pitch 1.95 → 1.09) that grown skips |
| 2026-10-02 | pilot-mi wpe (stretched positions) | 1201c6e | as pilot-mi grown but `pos_emb=learned`, big_107M's 512-row wpe linearly stretched to 2048; 77 min (RoPE 83) | GigaMIDI **2.211** (0.521 / 0.610 / 0.840 / 0.240) vs RoPE 2.232; Aria **5.796** vs 5.819; prog 0.217 vs 0.218 | **Stretched wpe wins, but the margin shrinks with training**: step 0 Aria 10.17 vs 11.80, @250 GigaMIDI −0.39, @500 −0.18, @1000 −0.07, @2000 −0.04, final −0.021 (Aria −0.023). One seed, same val windows. 8% faster. Aria still plateaus ~6.05 in the stable phase like RoPE, so the remaining +0.09 vs big_107M is forgetting/replay, not positions. Trade-off kept in mind: learned positions cap the context at 2048 and block rolling KV-cache generation (RoPE allows both) |
| 2026-10-02 | 2048 data check + ua-2048 smoke | (this commit) | store lengths at 2048; `pad_short` (short files = one padded BOS..EOS window, masked targets, sampled per note like long files); smoke: pilot_mi_grown → reductions v2 + Aria pop, 40 it, micro 2 × 6 | Kept at 2048 without padding: GigaMIDI train 80% of notes, **Aria train 43%**, **Ukrainian reductions 8 of 1,280 songs** (median 1,089 notes). Smoke: val (whole songs) 11.95 → 11.47, Aria pop 5.43, 2.1 s/it | The pilots' Aria val at 2048 = only the 791 longest files, so their 5.82 is **not comparable** to big_107M's 5.71 (512, all files). Micro 4 OOMs with ~2.9 GB of desktop GPU memory in use. User: reductions are the weak point (messy notes, no structure) → real piano covers (`fetch_songs.py --mode covers`, `transcribe_covers.py`, style `cover`), `config/finetune_ua2048.py` |
| 2026-09-29 | ladder-L-muon-lr1e-3 | d6c63c5 | L, Muon, lr 1e-3 (min 1e-4) | 1.913 / 1.961 / 2.684 / 1.473 = **8.032**; Aria **6.898** | **−0.16 vs Muon 6e-4**: Muon wants a higher LR (AdamW was flat 6e-4..1e-3). **New default: Muon lr 1e-3** |
| 2026-09-29 | ladder-L-muon | 7c38542 | L with `optimizer_name=muon` (Moonlight-scaled, lr 6e-4, wd 0.1) | 1.974 / 1.978 / 2.713 / 1.497 = **8.162**; Aria **7.061** | **−0.54 Aria vs best AdamW**, only about 6% slower/step (25:41 vs 24:18). About 2× token efficiency (≈ AdamW iso-L at 194M tokens). **Adopt** |
| 2026-09-29 | ladder-L-lr1e-3 | 7c38542 | L, AdamW lr 1e-3 (min 1e-4) | 2.286 / 1.999 / 2.777 / 1.564 = 8.626; Aria 7.598 | = lr 6e-4: flat optimum at ≥ 6e-4 |
| 2026-09-29 | ladder-L-lr3e-4 | 7c38542 | L, AdamW lr 3e-4 (min 3e-5) | 2.376 / 2.042 / 2.838 / 1.615 = 8.871; Aria 7.920 | +0.32: too low |
| 2026-09-29 | iso-M | 6497c76 | 20.2M, 400M tokens (13,020 it), compile, C≈4.85e16 | 1.905 / 1.940 / 2.667 / 1.438 = **7.949**; Aria **6.770** | **best model so far**. Beats ladder-XL (7.399) at similar compute |
| 2026-09-29 | iso-L | 6497c76 | 41.7M, 194M tokens (6,315 it) | 1.988 / 1.956 / 2.691 / 1.468 = 8.104; Aria 6.954 | +0.18 vs iso-M |
| 2026-09-29 | iso-XL | 6497c76 | 64.7M, 125M tokens (4,070 it) | 2.082 / 1.974 / 2.724 / 1.505 = 8.286; Aria 7.175 | +0.41 vs iso-M. Loss rises with size at fixed compute, so the optimum is ≤ 20M at C≈4.85e16 (≥ 20 tokens/param) |
| 2026-09-29 | ladder-L-moe8 | 84a33ed | L with MoE FFN: 8 experts, top-2, hidden 2C (117M total / 42M active) | 2.349 / 2.010 / 2.785 / 1.581 = 8.724; Aria 7.695 | **negative**: +0.10 vs dense L *per step* and about 2.5× slower per step (about 62 min with boost). Ahead only at step 1000. Too few tokens per expert at 100M; unfused expert loop. Revisit only with far more tokens and fused kernels |
| 2026-09-29 | ladder-L-pitchhead | 84a33ed | L + pitch head 2 blocks × 2 width (+1.6M) | 2.301 / 2.009 / 2.784 / 1.570 = 8.664; Aria 7.648 | **negative**: +0.04 vs L, and pitch itself got worse. Pitch gains come from trunk capacity (context), not the head. Keep the default heads |
| 2026-09-29 | ladder-L-fp8 | 71bdf16 | L with `--fp8=True --compile=True` | 2.280 / 2.008 / 2.783 / 1.571 = 8.643; Aria 7.622 | **+0.02 (+0.27%) vs bf16 L, 14% less wall-clock** (34:08 vs 39:52, both contended). Use fp8+compile for the long run |
| 2026-09-28 | ladder-XL | bc8a5e0 | 64.7M (12L×640) | 2.185 / 1.990 / 2.755 / 1.542 = **8.472**; Aria **7.399** | the power-law fit (α≈0.22) predicted 7.42. −0.32/doubling, same as M→L, so no extra flattening yet |
| 2026-09-28 | ladder-L | bc8a5e0 | 41.7M (12L×512) | 2.265 / 2.006 / 2.782 / 1.569 = **8.623**; Aria **7.601** | −0.27 / −0.33 vs M. Per doubling (Aria): S→M −0.39, M→L −0.32. Fixed 100M tokens undertrain the bigger models, so the gains are underestimated |
