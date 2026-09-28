# MusicAutoregressiveModel: agent context

Read this before touching the code. It records what the project is, what state it's in, what we know is
wrong, and the ranked list of experiments. Update the **Experiment log** at the bottom after every run.

## Goal

1. Pretrain a small autoregressive symbolic-music model on solo piano MIDI (MAESTRO v3 + GiantMIDI-Piano).
2. Fine-tune it to generate **piano arrangements in the style of Skryabin** (Скрябін, the Ukrainian
   pop-rock band, not the composer): pop song structure, melody + chord accompaniment, on one piano.

Constraints: **laptop only**. RTX 5060 Laptop (8 GB VRAM, Blackwell sm_120), 30 GB RAM, runs of at most a few
hours. Prefer models of roughly 5–30M params and bf16.

## Repository layout

| Path | What it is |
|---|---|
| `model.py` | nanoGPT-style decoder. `MusicEmbeddings` concatenates 4 embeddings (pitch/velocity/duration/delta_time, each `n_embd/4`), learned absolute positions. Output: `CascadeHeads` (default, `cascade_heads=True`), which predicts dt → pitch → duration → velocity with each head conditioned on the earlier ones. The legacy path is 4 independent linear heads (`cascade_heads=False`, needed for pre-2026-09-28 checkpoints). `forward(..., targets)` returns `(parts, loss)`: loss = sum of CEs with label smoothing (optimized), parts = per-head CE without LS (for logging) |
| `data_loader.py` | `MaestroDataset` plus the module-level `tokenize_midi()`. Tokens are cached in `data/cache/*.pkl` (keyed by file list + tokenizer params). Missing files are skipped with a warning. `eval_stride=N` gives deterministic windows for validation. Time is quantized at 20 ms, clamped to 511 bins (~10.2 s). The train `__getitem__` returns **one random 512-note window per file** |
| `train.py` | **Single** training script (pretrain + finetune). nanoGPT `configurator.py`: `python train.py config/<x>.py --key=value`. `init_from` = `scratch`, `resume` or `finetune` (weights only from `init_ckpt`; fresh optimizer, iter and best val). Logs `val/ce/<head>` + `train/ce/<head>` (CE without LS) and the planned number of epochs |
| `config/` | `pretrain.py`, `finetune_skryabin.py`, `smoke.py` (about 1 min sanity run, stack it after another config) |
| `train_finetune.py` | **Deprecated**, replaced by `train.py config/finetune_skryabin.py`. Delete it once the new path is confirmed |
| `data/audio_to_piano.py` | Band mp3 to **piano reduction**: Demucs `htdemucs_6s` stems, then basic-pitch per part (vocals to monophonic melody, bass to monophonic bass, piano+guitar+other to capped harmony), merged into one piano track + `debug/*.parts.mid` + key estimate in `songs.csv`. Runs in **`.venv-audio`** |
| `data/fetch_songs.py`, `data/artists.txt` | YouTube search per artist, then filter (duration, blocklist of live/cover/compilation words, title must contain artist), dedupe by song, download mp3 with yt-dlp. `--dry-run` only lists. Output `data/audio/<artist>/` + `songs.csv` |
| `eval_samples.py` | Compares generated MIDI with real MIDI: per-feature histogram overlap (pitch, pitch class, velocity, duration, IOI, polyphony, interval) plus notes/s and pitch-class entropy |
| `generate.py` | Samples from a checkpoint, writes MIDI, renders MP3 through fluidsynth + ffmpeg. `--prompt file.mid --prompt-notes 64` seeds with real notes. Polyphony and gap post-processing is now off by default (`--max-polyphony`, `--max-delta` to enable) |
| `prepare_giantmidi.py` | Builds `data/combined.csv` (MAESTRO + GiantMIDI, GiantMIDI split by composer) |
| `data/preprocess.py` | Downloads MAESTRO |
| `data/mp3_to_midi.py` | Turns MP3s into MIDI with `piano_transcription_inference` (optionally Demucs first), then writes `skryabin.csv` |
| `Skryabin/` | 27 MP3s of the band plus `midi/` transcriptions plus `skryabin.csv` |
| `checkpoints/ckpt.pt` | Last pretrain (about 5.3M params, weights + Adam state is about 63 MB). `test.pt` (317 MB) is an older, larger experiment |
| `checkpoints_skryabin/ckpt.pt` | Fine-tuned checkpoint, **possibly stale** (see bug B3) |
| `=2.7.0` | Junk: pip output from an unquoted `pip install torch>=2.7.0`. Safe to delete |

Environment: **`.venv/`** in the project root (created 2026-09-28 with `uv venv`, Python 3.12, torch 2.14.0+cu130).
Verified on the RTX 5060 (sm_120): bf16 matmul, flash SDPA and fused AdamW all work.
Use `.venv/bin/python` or `source .venv/bin/activate`. The system `python3` has no torch.
Audio tools live in a **separate env `.venv-audio/`** (torch 2.14 cu130, torchaudio, demucs, basic-pitch 0.4 on
**onnxruntime** installed with `--no-deps` because its TensorFlow pin has no Python 3.12 wheels, yt-dlp, librosa,
piano_transcription_inference). Demucs + basic-pitch take about 20 s per song on the GPU.

## Current state (verified 2026-09-28)

- The project wasn't under git until 2026-09-28 (`git init` done, `.gitignore` excludes data, checkpoints,
  wandb and audio). Nothing is committed yet.
- **Data (2026-09-28):** `data/combined.csv` = MAESTRO v3 (962 train / 137 val) + **GiantMIDI curated subset**
  (`data/giantmidi/midi/surname_checked_midis/`, 7,236 files from the Google Drive release, only titles checked against
  composer surnames; split by composer: 6,761 train / 475 val). Total **7,723 train / 612 val**. The 177 MAESTRO
  `test` rows are ignored by the loader. GiantMIDI paths in the CSV are absolute. The old `giant_midis/` folder
  is obsolete. After tokenization: **7,301 train files longer than 512 notes, 31.3M train notes**, 1,200 fixed val
  windows (200 batches). Tokenizing in parallel (24 procs) takes about 1 min, and cached loading takes seconds. The old CSV was lost (overwritten by the crashed prepare run), but nothing depends on it.
- Token budget: the old default of 122,880 tokens/iter × 30k iters is about **118 epochs** of the combined set (about 650 of MAESTRO alone).
  4 epochs ≈ 1k iters and 10 epochs ≈ 2.5k iters at the default batch. The old 5M model ran at about 0.47 s/iter, so about 20 min for 10 epochs. `train.py` prints
  "Planned epochs" at startup, so always check it.
- Last pretrain run (wandb `lcqh7fm4`, 30k iters, about 4 h): loss 21.06 → 12.0 at 2k → 11.4 at 4.6k →
  **plateau at about 11.05 from about 20k onward, train ≈ val**. This is **underfitting / capacity-bound, not
  overfitting**. The earlier overfitting was from older configs.
- Several older runs (`izuwb57w`, `ecb075ie`, `a7jux28v`, …) show val loss far below train loss (for example 1.27 vs 8.4).
  That's impossible for a healthy setup, so the val split or loss code was different or broken back then. Don't trust
  those numbers.
- Budget of the last run: 6 × 512 × 40 accumulation steps = 122,880 tokens/iter, so 30k iters ≈ 3.7B note-tokens. MAESTRO +
  GiantMIDI hold about 45M notes (estimate, measure it), so that's roughly 80 passes. The data-constrained scaling result
  (Muennighoff et al., 2023: up to about 4 epochs ≈ fresh data, returns fade quickly after that) says **most of that compute
  was wasted**. Train on fewer tokens with a bigger model, or keep this budget with a much bigger model.

## Known bugs / defects (fix these before any sweep)

- **B1 (FIXED 2026-09-28: CascadeHeads v2 = residual heads, the default. −0.41 nats/note vs independent, see log): next-note attributes are predicted independently.** All 4 heads read the same hidden state `h_t`, and
  `generate()` samples each one independently. P(pitch, dur, dt, vel | ctx) is modeled as a product of marginals,
  so the loss can't drop below the conditional-dependence gap, and samples mix incompatible attributes.
  This is the most likely reason the loss plateaus and generations sound incoherent.
- **B2 (fixed 2026-09-28, per-head CE logged): a total loss with label smoothing hides what's going on.** Only the sum of 4 CEs with LS = 0.1 is logged.
  LS over 512 classes adds a large constant floor. Log **per-head CE without label smoothing** (and bits per note).
- **B3 (fixed 2026-09-28, `init_from='finetune'`): fine-tune keeps the pretraining `best_val_loss`.** `train_finetune.py` loads `best_val_loss` from the MAESTRO
  checkpoint, so a Skryabin checkpoint is saved only if Skryabin val loss beats MAESTRO val loss. It also loads the
  pretraining optimizer state. Reset both.
- **B4 (fixed 2026-09-28, `eval_stride`): validation is mostly noise.** The val set has one random window per file. For Skryabin (about 4 val songs, batch 4),
  `estimate_loss` evaluates about **one batch**. Use fixed, deterministic, strided windows that cover every val file.
- **B5 (worked around 2026-09-28 with `--prompt`, BOS token still TODO): out-of-distribution generation seed.** Generation starts from `(pitch 0, vel 0, dur 10, dt 0)`. Pitch 0 never
  appears in the data. Add a BOS token, or prompt with a real snippet.
- **B6 (fixed 2026-09-28): `generate.py` post-processing hides the model's real output.** `max_polyphony=4` and `max_delta=0.5` alter the
  output. Judge raw samples first. Also, `n_embd` falls back to 512 while the model default is 256.
- **B7: augmentation.** Pitch shift uses `clamp`, which bends out-of-range notes instead of dropping them or
  limiting the shift. There's no tempo augmentation (scale dt/dur ±10%) and no velocity augmentation.
- **B8 (in progress: `data/audio_to_piano.py`, first 3 songs rendered for listening in `data/finetune/skryabin_test/listen/`): fine-tune data is out of domain.** `Skryabin/midi` is *piano transcription of full band mixes* (vocals,
  guitars, drums, synths), which gives noisy MIDI full of ghost notes. That's a different distribution from both MAESTRO and the target.
- **B9 (fixed 2026-09-28, `data/cache`): tokenization is re-parsed on every start.** It takes minutes, and every experiment pays for it again. Cache
  tokenized arrays (`.npy`/pickle keyed by the tokenizer config).
- Minor: `wandb.init(config=MusicConfig)` logs the class, not the run config. `train.py` and `train_finetune.py` are
  about 95% duplicated, so merge them into one script with a config file or CLI.

## Hypotheses, ranked

Expected gains are guesses. Verify each one with one change per run, against the same fixed val windows and
per-head CE without label smoothing.

### Phase 0: infrastructure (no model change, needed for everything else)
- One `train.py` with config via YAML or argparse, `--init-from scratch|resume|finetune`.
- Cached tokenization, deterministic val windows, per-head CE logs, `git commit` before each run, and the run id
  recorded here.
- Fix B3–B6.
- A small objective-eval script for generated samples: pitch-class histogram entropy, note density, polyphony,
  IOI distribution, share of notes on the beat grid. Compare against the same stats on the val set.

### Phase 1: cheap wins and hyperparameter search (expect a few %, as you said)
1. **Dropout 0.3 → 0.0–0.1** and label smoothing → 0. The model underfits (train ≈ val), so regularization only hurts it now.
2. **Scale the model** within 8 GB: for example 8 layers × 512 wide (about 25M params), block 1024, batch tokens around 32–64k.
   With about 45M notes, Chinchilla-style scaling points to 20–50M params.
3. **Token budget of about 4 epochs** instead of about 80 passes. Use shorter runs and spend the compute on model size.
4. LR sweep {3e-4, 6e-4, 1e-3} × warmup {300, 1000}. Try cosine with a 10% floor versus WSD (warmup-stable-decay).
5. Tempo and velocity augmentation (B7).
6. `torch.compile` and flash SDPA. Check that compile works on sm_120 with the installed torch.

### Phase 2: representation and architecture (where the real gains probably are)
1. **Fix B1, factorized sequential heads.** Predict `dt → pitch | dt → dur | dt,pitch → vel | …`, feeding
   the embedding of the (teacher-forced or sampled) earlier sub-token into the next head, as in the
   Compound-Word Transformer (Hsiao et al., 2021). This is a small code change with probably the largest effect.
2. **Flattened event tokens** as an alternative: REMI / MIDI-Like / Structured through `miditok`. Sequences get 3–4×
   longer, but there's exact autoregressive factorization for free. Compare against (1) at equal compute.
3. **Beat-based time grid for the pop target.** MAESTRO is performance timing with no beat info, but pop piano arrangements
   are grid-based. REMI (bar/position tokens) fits the Skryabin goal far better. One option is to pretrain on
   beat-quantized data (GiantMIDI + POP909 + Lakh-piano) instead of raw MAESTRO timing.
4. **Relative positions**: RoPE (cheap) or ALiBi. Music Transformer (Huang et al., 2018) showed relative attention
   matters for music. Also allows extrapolation to longer generations.
5. **Tied input/output embeddings** per attribute, plus a BOS token and song-start conditioning.
6. **SSM / Mamba.** An honest take: at 512–1024 context and about 25M params a transformer is fine, and Mamba mostly pays
   off at **long context** (whole songs, 4k–16k events, especially with REMI tokens). Try it as a controlled ablation
   *after* 1–3: same tokenizer, same param count, same token budget. The risk is that `mamba-ssm` CUDA kernels on Blackwell
   sm_120 may need a source build. Fallbacks: a pure-PyTorch Mamba-2 (slow), or a hybrid (for example 1 attention layer per
   4 Mamba layers). Worth doing mainly as a learning exercise and for long-form structure (verse/chorus repetition).

### Phase 3: fine-tuning on small data
- Fine-tune on **clean pop piano** data first (POP909, Pop1K7), then on Skryabin-like data.
- Use LoRA, or freeze the lower layers, plus low LR (1e-5–5e-5), plus **replay** (mix 10–30% pretraining data) to avoid
  collapse or memorization on about 30 songs.
- Early-stop on a song-level held-out split. With under 50 songs, use k-fold, because a single split is noise.
- Style conditioning: an artist/genre token (for example `<ukr-pop>`, `<skryabin>`) so the whole pool of similar songs can train
  together and the target style is chosen at sampling time.

## Fine-tune data acquisition plan (approved: public MIDI + audio + karaoke MIDI)

Private research use only. Don't redistribute downloaded audio or MIDI, and keep it out of git (see `.gitignore`).

1. **Public MIDI (clean, do this first)**
   - **POP909**: 909 pop songs as piano arrangements (melody/bridge/piano tracks). The closest public match to the "pop
     piano arrangement" target.
   - **Pop1K7**: about 1.7k piano covers of pop songs transcribed from YouTube (from the CP-Transformer paper).
   - **Lakh MIDI (LMD-matched)**: multi-track. Collapse the non-drum tracks into a piano reduction. Filter by the Million
     Song Dataset artist tags for rock/pop, and Eastern-European artists where they exist.
   - **Aria-MIDI** (2025, about 1M transcribed piano recordings with metadata) if it's reachable: filter for pop covers.
2. **Audio → MIDI, fixing B8**: transcribe **piano covers** of songs instead of band mixes.
   `piano_transcription_inference` is accurate on solo piano and poor on full mixes.
   - `yt-dlp` search queries: `"<song title> скрябін piano cover"`, `"<song> фортепіано"`, `"<song> piano tutorial"`.
   - Artist list for similar style: Скрябін, Океан Ельзи, Бумбокс, Друга Ріка, Мертвий Півень, Воплі Відоплясова,
     Танок на Майдані Конго, Плач Єремії, С.К.А.Й., Антитіла, Kozak System, Тартак, The Hardkiss, Kalush (check fit).
   - Quality filters: duration 1–8 min. Reject audio that isn't piano, using a CLAP zero-shot "solo piano" score or the
     transcription's note density and pitch-range heuristics. Deduplicate by song.
   - Similarity ranking: embed candidates with CLAP or MERT (audio) or a MIDI embedding, and rank by distance to the
     centroid of the real Skryabin songs. Keep the top-N and log the scores in a CSV so the choice can be reviewed.
3. **Karaoke / "мінусовки" MIDI**: real multi-track arrangements of the actual Ukrainian songs. Scrape politely
   (rate limit, respect robots.txt), then reduce to piano: melody track, chord tracks folded into the piano range,
   bass kept, drums removed.
4. Output: `data/finetune/<source>/*.mid` plus one CSV with columns `split,midi_filename,source,artist,song,score`. Split by
   **song**, never by window, so covers of the same song never end up in both train and val.

## Working conventions for agents

- Make one change per experiment. Log the wandb run id, git commit, config diff and the per-head val CE in the log below.
- Don't delete checkpoints, data or wandb runs without asking.
- Long runs belong to the user. Launch at most one background training at a time (8 GB VRAM) and write a short
  smoke run (`max_iters` about 50) before any long run.
- Keep the nanoGPT style of the existing code. Comments may be in Ukrainian or English.
- Don't commit data, checkpoints, audio or wandb dirs.

## Open questions

- Where is the Python env with torch? (Record it here.)
- MAESTRO pretraining versus beat-based pop pretraining: decide after the Phase 2.3 comparison.
- Is the Pop1K7 / Aria-MIDI download still available? Check before relying on it.

## Experiment log

| Date | Run id | Commit | Change | Val CE per head (p/v/d/dt) | Notes |
|---|---|---|---|---|---|
| 2026-05-17 | lcqh7fm4 | none | baseline 6L×256, dropout 0.3, LS 0.1, 30k iters | only the total is logged: 11.06 | plateau from about 20k, train ≈ val. Unknown whether GiantMIDI was present |
| 2026-09-28 | smoke | none | Phase 0 pipeline check, cascade heads, 30 iters, MAESTRO only | 4.14 / 2.98 / 3.26 / 2.81 | only verifies that pretrain → finetune → generate → eval_samples all run. Uniform-init CE = ln(vocab) as expected |
| 2026-09-28 | bcuakmc7 (ab-cascade) | 1bc701b + config/ab_heads.py | cascade v1 (MLP heads), 4k iters × 30,720 tok (3.9 epochs), dropout 0.1, LS 0 | 3.056 / 2.054 / 2.964 / 1.719 = **9.793** | |
| 2026-09-28 | 9dv8hnzd (ab-cascade-v2) | d5b38ae + v2 heads | cascade v2 residual heads, otherwise identical | 2.759 / **2.040** / **2.915** / **1.699** = **9.412** | best. Ahead from step 500 on. About 10% slower/iter, +0.27M params |
| 2026-09-28 | a52uageu (ab-indep) | same | independent heads, otherwise identical | **2.751** / 2.279 / 3.083 / **1.712** = 9.824 | indep ahead until about step 1.2k. Pitch much better than cascade v1 |

**A/B diagnosis (2026-09-28):** shuffling the conditioning inputs of ab-cascade raises val CE by 0.28 (pitch|dt),
0.47 (dur) and 0.85 (vel), so the cascade *does* use the conditioning. But its pitch head *with* dt (3.07) is worse
than the independent pitch head *without* it (2.75). Causes: (1) the head is a fresh MLP with no direct linear path, and
(2) the condition embeddings stayed tiny (rms 0.03–0.15 vs about 1 for h: 0.02 init + weight decay + LayerNorm).
→ v2 `ResidualHead`: `out(LN(z + MLP(LN(z))))`, z = h + cond, cond_emb init std 0.5, no weight decay on cond_emb,
residual-style small init of `c_proj`. At init it equals independent heads + conditioning.
**Result:** v2 = 9.412 (best). Pitch equals independent (dt doesn't help pitch beyond h), and vel/dur/dt keep the cascade
gains. Train CE > val CE in all runs (dropout + augmentation), so still capacity-bound: scale the model next.

**Skryabin piano reduction, first look (2026-09-28):** 3 songs, 13–15 notes/s vs 5.5–8.7 for the old full-mix
transcription and about 7.7 for MAESTRO val. The harmony part is probably over-dense (pads and strums split into
repeated notes). The key estimate said "major" for all 3, unverified. Waiting for the user's listening verdict.
