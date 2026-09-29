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
| `data_loader.py` | `MaestroDataset`: `csv_path` can combine CSVs and prebuilt stores (`'a.csv,store:data/cache/aria'`). Tokens live in a **memory-mapped flat store** per split (`data/cache/<name>_<split>_<hash>/tokens.u16`, shape (notes, 4) uint16, + `offsets.npy` + `meta.json`), built in parallel (24 procs). Train windows are sampled **uniformly over all note positions** (long files proportionally more often). `eval_stride` gives deterministic val windows. Broken or missing files are skipped. Transposition ±5 is limited so notes stay in 0..127. `tokenize_midi()` accepts paths or file-like objects. Time is quantized at 20 ms, clamped to 511 bins |
| `train.py` | Single training script. nanoGPT `configurator.py`: `python train.py config/<x>.py --key=value`. `init_from` = scratch, resume or finetune. `val_csv_path` = main val (checkpoint selection), `val2_csv_path` = extra val logged as `val2/*`. Flags: `compile`, `fp8` (torchao float8 on the transformer blocks, converted **before** the optimizer is built), `sdpa_backend`. Prints "Planned epochs" at startup, so check it |
| `config/` | `pretrain.py`, `finetune_skryabin.py`, `smoke.py` (stack after another config), `ab_heads.py` (4-epoch A/B of the head designs, 6M), `ladder.py` (scaling ladder, 100M tokens, MAESTRO+GiantMIDI+Aria) |
| `generate.py` | Samples from a checkpoint, writes MIDI, renders MP3 (fluidsynth + ffmpeg). `--prompt file.mid --prompt-notes 64`. Post-processing is off by default. Builds the config from every checkpoint field (legacy defaults for old checkpoints) |
| `eval_samples.py` | Generated vs. real MIDI: per-feature histogram overlap plus notes/s and pitch-class entropy |
| `export_bf16.py` | Checkpoint → bf16 weights without optimizer state (74 → 12 MB for 6M). Loadable by `generate.py`, not resumable |
| `optim.py` | `Muon` (Newton-Schulz orthogonalized momentum, Moonlight 0.2·√max(m,n) scaling so it shares AdamW's lr/wd/schedule), `CombinedOptimizer`, `build_muon_optimizer` (Muon for 2D matrices in `transformer.h`, AdamW for the rest). Enabled with `train.py --optimizer_name=muon` |
| `bench.py` | GPU-side throughput benchmark on synthetic data: micro-batch × compile × SDPA backend × fp8, `--profile` for a kernel breakdown |
| `status.py` | **Live dashboard** (`.venv/bin/python status.py`, `--once`): GPU/disk, every training run (state, progress, speed, ETA, val/Aria CE, per head), download and reduction progress. Read-only |
| `prepare_giantmidi.py` | Builds `data/combined.csv` (MAESTRO + GiantMIDI, GiantMIDI split by composer). Fixed 2026-09-28: ignores the extra MAESTRO CSV columns |
| `data/prepare_aria.py` | Tokenizes **Aria-MIDI straight from the .tar.gz** (no extraction) into stores `data/cache/<name>_{train,validation}`, 1% of recordings to val (by file_id), `--genres pop,rock` filter, metadata CSV `data/aria/<name>.csv` (file_id, segment, split, genre, composer, audio_score, n_notes) |
| `data/fetch_songs.py`, `data/artists.txt` | YouTube search per artist (aliases with `|`, e.g. `Танок на Майдані Конго|ТНМК`), filters (duration 90–480 s, blocklist of live/cover/compilation/хіти words, title must contain the artist), dedupe by song, yt-dlp mp3 download, retry with backoff on network errors. Output `data/audio/<artist>/` + `data/audio/songs.csv`. Re-runs skip what's already downloaded |
| `data/audio_to_piano.py` | Band mp3 → **piano reduction**: Demucs `htdemucs_6s` → basic-pitch per part: vocals → monophonic melody (merged re-triggers, legato), bass → monophonic bass (legato), piano+guitar+other → harmony (onset 0.6, frame 0.4, min 100 ms, merge re-triggers, drop the quietest 30%, max 3 notes at once). Writes `midi/`, `debug/*.parts.mid` (one track per part), `songs.csv` with a key estimate. **Stems are deleted after each song** (~350 MB/song) unless `--keep-stems`. Runs in `.venv-audio` |
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
| 8 | ladder-L-muon-lr1e-3 and iso-S-muon done. **iso-M-muon** (20M, 400M tok, 13,020 it, Muon lr 1e-3, compile) running (started 2026-09-29 16:04) | `logs/iso_{S,M}_muon.log` | does Muon shift the iso-FLOP size optimum (S vs M, both Muon) |
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
  (local branch has the same name, so a plain `git push` works). `main` = the user's LSTM chord model + earlier transformer comparison,
  `legacy-transformer` = history of the old `../MusicAutoregresiveTransformer` folder. **Never push to `main`, never force-push.
  Don't create new GitHub repos**: the user wants everything in this one.
- The README of `transformer-v2` will be rewritten later by the user. The local, git-ignored `README_TODO.md` lists what it should cover.

- Make one change per experiment. Log the wandb run id, git commit, config diff and per-head val CE below.
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
