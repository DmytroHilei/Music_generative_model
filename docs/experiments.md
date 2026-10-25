# Experiments

A summary of the experiment log behind this repository (September–October 2026). Losses are validation
cross-entropy in nats per note, summed over the note attributes (pitch + velocity + duration + delta-time, plus the
instrument where stated), unless a row says otherwise. Numbers from different validation sets are not comparable with
each other. Every comparison below changed one thing at a time; single-seed results are marked as such.

Hardware: a laptop RTX 5060 (8 GB, Blackwell sm_120) for almost everything, and one rented RTX 5090 for the 450M
pretraining run (~41 h).

## 1. Output heads

Each position is one note, so the model has to predict 4–5 attributes per step. 6M model, 4 epochs of
MAESTRO + GiantMIDI.

| Heads | pitch / vel / dur / dt | Total |
|---|---|---|
| 4 independent linear heads | 2.751 / 2.279 / 3.083 / 1.712 | 9.824 |
| Cascade v1 (dt → pitch → dur → vel, MLP heads) | 3.056 / 2.054 / 2.964 / 1.719 | 9.793 |
| **Cascade v2 (residual heads)** | 2.759 / 2.040 / 2.915 / 1.699 | **9.412** |

v1 used the conditioning (shuffling it cost +0.28 to +0.85 per head) but its pitch head lost the direct path from the
trunk and its condition embeddings stayed tiny. v2 fixed both (residual form, larger init, no weight decay on the
condition embeddings): **−0.41 nats/note**. The cascade later gained an instrument head (dt → instrument → pitch → …).

## 2. Scaling

**Fixed-token ladder** (100M tokens, MAESTRO + GiantMIDI + Aria-MIDI): S 6M 8.613 → M 20M 7.931 → L 42M 7.601 →
XL 65M 7.399 (Aria val). A power-law fit (α ≈ 0.22) on S–L predicted XL at 7.42 (measured 7.399). More than half of
the S → M gain was pitch; velocity was nearly saturated.

**Iso-FLOP** (equal compute, C ≈ 4.85·10¹⁶): AdamW M 6.770 < L 6.954 < XL 7.175; with Muon S 6.685, **M 6.512**,
L 6.552. The optimum sits at ≈ 20M params (≈ 16–20 tokens/param), i.e. Chinchilla's ratio holds for music. A second
iso-FLOP curve with the new transformer block (C = 3.2·10¹⁶, 4 sizes) put the optimum at ≈ 9M (≈ 45 notes/param);
extrapolated to a week of compute, the optimum is ≈ 200–400M. The final run used 450M on a rented GPU.

**Forecasts that held:** the 107M run was predicted at 5.47–5.72 Aria CE and finished at 5.706; the 450M run's
cooldown forecasts landed in range.

## 3. Optimizer and precision

| Change (42M model) | Result |
|---|---|
| AdamW LR 3e-4 / 6e-4 / 1e-3 | 7.920 / 7.601 / 7.598 Aria: flat optimum at ≥ 6e-4 |
| **Muon** (Newton–Schulz orthogonalized momentum, Moonlight scaling) at 6e-4 | 7.061: **−0.54 vs the best AdamW**, ≈ 2× token efficiency, +6% time per step |
| Muon at 1e-3 | 6.898: another −0.16; became the pretraining default |
| **fp8 training + torch.compile** | +0.02 CE (+0.27%) for −14% wall-clock; +20% throughput, −25% memory in the benchmark |
| New block, Muon vs AdamW at the same LR 1e-3 (3 seeds each) | Muon −1.38 (AdamW not LR-tuned for this block) |

**Fine-tuning is different:** for the 450M model on a few hundred piano covers, Muon overfit; 8-bit AdamW at LR 2e-4,
dropout 0, ~2 passes gave the best generations (Section 7).

## 4. Architecture changes

| Change | Setting | Result |
|---|---|---|
| **RMSNorm + SwiGLU + QK-norm** ("new block") | 42M, context 2048, 3 seeds | clean GigaMIDI 3.886 ± 0.075 → **3.137 ± 0.013** (−0.75, 10 sd, lower seed spread) |
| **Growing a trained model** vs training from scratch | 108M, 197M notes; the 107M piano model grown to 129 instruments and RoPE, new parts zero-initialised (BOS/EOS came later, in the caption pilots) | GigaMIDI 2.232 vs 2.597, Aria 5.819 vs 6.931: **≈ 2.5–4× step efficiency**. Adding the instrument attribute alone is function-preserving (identical loss at step 0) |
| RoPE vs the learned position table stretched 512 → 2048 | same pilot | stretched table −0.02 at the end (−0.39 early); RoPE kept for longer context and rolling KV-cache decoding |
| Window conditioning (instrument set + notes/s level, condition dropout) | 42M | no CE cost (3.125 vs 3.137 ± 0.013); off-band instruments in generation 50% → 25% |
| Domain / artist style label (fine-tune) | 107M | artist labels no measurable effect; a single "reduction" domain label steers generation density and polyphony to the target style |

## 5. Tested and rejected

Kept in [`musicar/model/rejected.py`](../musicar/rejected.py) with their results and the wiring needed to reuse them.

| Idea | Result | Why it lost |
|---|---|---|
| Mixture-of-experts FFN (8 experts, top-2) | +0.10 CE per step, 2.5× slower | too few tokens per expert at this scale, unfused expert loop |
| Bigger pitch head (2 blocks × 2 width) | +0.04, pitch itself worse | pitch gains come from trunk capacity, not the head |
| Future-prediction auxiliary heads (pitch classes / density / register 0–8 s ahead) | −0.002 at λ 0.2 (¼ of the seed spread), +0.01 at λ 1.0; no structure metric moved | the heads learned the key, not foresight |
| Velocity augmentation (±2 bins) | 10.497 vs 10.479 (tempo-only), forgetting 6.07 → 6.27 | hurts; tempo augmentation kept |
| Separate pop/rock pretraining stage before the fine-tune | no gain on the target (10.499 vs 10.471) | the general model already covers it |

## 6. Data

- **Corpora:** MAESTRO v3 + GiantMIDI-Piano (31M notes), Aria-MIDI (551M notes), GigaMIDI and Discover MIDI
  (2.0B notes kept after deduplication): ≈ 4.5B unique notes usable at context 2048, tokenized straight from the
  archives into memory-mapped stores.
- **Leakage found and fixed:** 8.3% of GigaMIDI files are exact duplicates and 54k of its validation files also occur
  in its train split, so its validation CE was optimistic. A deduplicated validation set (153k files) is used since.
  Discover was deduplicated against GigaMIDI by content hash while tokenizing (≈ 21% of the sampled subset were
  duplicates of GigaMIDI files).
- **Packing** short files into full windows (BOS/EOS-separated) raised the usable notes at context 2048 from 1.81B to
  2.50B.
- **Ukrainian target data** built from audio: about 5,900 songs transcribed to piano reductions and 5,400 (310
  artists) to multi-track MIDI (Demucs source separation → per-stem transcription), plus ~700 transcribed real piano
  covers. Song-level split by title hash, so no validation song appears in any training set.
- **Data quality mattered more than mixing:** reductions carry almost no repetition (2% melody 4-gram repeats vs 28% in
  real covers: transcription noise makes every chorus different), so models trained on them could not learn song
  structure. Adding real covers made the samples closer to real songs than the reductions themselves are (distance
  0.47 vs 0.68). Mixing proportions within 0.3–0.6 made no difference beyond seed noise.
- **More data beat every trick:** 2.9× the songs gave −0.24 CE (≈ 24× seed noise); domain labels, an extra stage and
  augmentation each gave ≤ 0.03. Tempo augmentation (±10%) stopped overfitting on a small set.

## 7. Main runs

| Model | Data | Compute | Result |
|---|---|---|---|
| 107M piano (14 layers × 768) | 2.33B tokens, MAESTRO + GiantMIDI + Aria | 12.5 h, laptop, Muon, WSD schedule, fp8 | Aria 5.706 (predicted 5.47–5.72); the 20% cooldown gave −0.27 |
| **450M multi-instrument** (new block, RoPE 2048, 129 instruments, conditioning) | 8.3B notes, GigaMIDI + Aria + Discover | ~41 h, rented RTX 5090, resumable with Hub sync | clean GigaMIDI 1.679, Aria 5.091; −0.56 vs the 108M pilot for ≈ 180× compute |
| Piano fine-tune of the 450M | Ukrainian piano covers + reductions + 20% Aria pop/rock replay | 350 steps, 8-bit AdamW | distance to the real continuations of 27 held-out covers 0.12–0.13 (4 seeds) vs 0.24 for the earlier best model and for the base; clearly preferred in a 5-song listening comparison |
| Multi-track fine-tune of the 450M | ~4,800 Ukrainian multi-track songs + 25% GigaMIDI replay | 1,500 steps | target CE 9.30 → 8.53; distance to the real continuation 0.69 vs 1.14 for the base (12 songs, one seed), no density runaways (base: 3 of 12) |

## 8. Generation and evaluation

- **Validation CE alone misled:** the base 450M had lower CE on piano covers than the fine-tuned 107M (5.45 vs 5.64)
  but generated worse whole songs (too dense, less repetition). Model choices were made on generation metrics.
- **Generation metrics:** distance of each sample to its own song's real continuation (density, chord rate, key
  similarity, melody repetition, recurrence, register drift, instrument set, takeover, consistency over 4 s windows),
  run during training and in seeded comparisons; multiple sampling seeds, held-out songs for final choices (a winner
  on 10 songs dropped from 0.23 to 1.3–2.1 on 60), and listening tests for the final pick.
- **Sampling:** temperature 0.9 halved the distance to real songs vs 1.0 (melody repetition 9% → 42%, real 37%);
  top-p 0.9 was in every good multi-instrument setting; per-style settings found by a successive-halving search.
- **Inference speed:** KV cache with a CUDA-graph one-note step: 630 notes/s at batch 1 (1.22× over eager), 9,700
  notes/s at batch 128 on the laptop GPU.

## 9. Systems work

- Throughput profiling (kernel breakdown with nsys / ncu) on sm_120; selective activation checkpointing (+6% at 350M),
  per-step fp8 weight cache (+2%), batched Muon Newton–Schulz, CUDA OOM recovery (retry, then spill activations to
  RAM). Result: a 450M model trains and fine-tunes on an 8 GB laptop GPU.
- Two OOM bugs traced to their cause and fixed: a bf16 checkpoint copy kept alive on the GPU, and a cuBLAS workspace
  per new CUDA stream in the in-training generation check.
- A cloud pipeline for rented GPUs: data and checkpoints through private Hugging Face repos with checksums, pinned
  environment, preflight (benchmark, LR probe, resume and SIGTERM tests), crash-restart wrapper, hourly checkpoint sync;
  rehearsed end to end on the laptop before any paid hour.

## 10. Text control

Prompts in English or Ukrainian, with no network or API at generation time: a frozen multilingual sentence encoder
(LaBSE) reads the prompt, and the music model learns to follow it from captions built offline. Pilots on the 107M
model (grown to 129 instruments, BOS/EOS, RoPE), each 2,400 steps × 65,536 notes (~1 h on the laptop), same data.

**Labels.** MidiCaps captions (168k) and XMIDI emotion / genre labels (108k) were matched to the token stores by an
exact note hash; key, density, loudness, register, length and instruments were computed for every file. A linear
probe on the 450M's hidden states beat note statistics alone by 11–18 points (XMIDI emotion 0.52 vs 0.40, XMIDI genre
0.56 vs 0.45, key mode 0.84 vs 0.66). Pseudo-labels from those embeddings (layers 7 + 14, linear) gave 4 mood groups
at ~80% precision for 76% of files. Drum loops and very short files got confident nonsense, so pseudo-labels are used
only for files ≥ 20 s that are not drum-only.

**Text encoder.** In a 10-pair EN / UK test, LaBSE put every prompt nearest to its own translation (20 / 20);
multilingual-e5-small 12 / 20, e5-base 10 / 20.

| Run | Conditioning | Caption gain, first 8 notes¹ | Generation from BOS |
|---|---|---|---|
| P1, frozen trunk | pooled vector → linear, added at every position | — | no consistent effect |
| P1, trained trunk | same | +0.222 | tempo / energy words, weak |
| P2 | caption vectors centred on the mean and rescaled; 25% of windows at a piece start | +0.323 | instruments, tempo, loudness with guidance (piano share 0.81 vs 0.29) |
| **T2** | 16 prefix vectors from LaBSE token states (learned queries, MLP, gate) | **+0.480** | loudness, guitar, strings; several without guidance; Ukrainian worse |
| T2 + new captions | 30% Ukrainian captions | — ² | Ukrainian piano share +0.75 and tempo follow; English unchanged |

¹ CE with another file's caption minus with the own caption, first window of 256 held-out files (same files for
these rows). ² validation captions changed.

- **The caption only matters at the start.** The gain is +0.2–0.5 over the first 8 notes and +0.001–0.002 after 128;
  afterwards the music itself tells the model what the caption says. A 1,024-position average hides the effect.
- **Weak following was a signal problem, not a wiring bug.** Gradients reached the adapter (its weights grew from zero
  to norm 10.9), but 39% of the caption vectors' energy lay in their shared direction, so the adapter mostly learned
  "a caption is present". Centring the vectors raised the opening gain by 45%.
- **A trainable trunk is needed:** the frozen trunk never learned to start a piece from the new BOS token.
- **T2 at the same compute** beat P2 on every validation measure (+0.0043 vs +0.0030 average caption gain, val CE
  2.687 vs 2.691). Its first launch went to NaN: an all-zero prefix (zero gate and null vectors) gave NaN gradients in
  the trunk; a small nonzero init fixed it.
- **LaBSE aligns languages in its sentence vector, not in its token states.** T2 needed Ukrainian captions in training;
  with them, Ukrainian prompts follow.
- **Captions from the local LLM (Qwen3 8B) were not usable:** broken English templates ("music for {instruments} make
  me"), invented content in about half of the Ukrainian label lines, adjective gender errors in ~29% of its MidiCaps
  translations. Replaced by a hand-written English + Ukrainian vocabulary with gender agreement; the translations were
  repaired by rules and filtered with a Hunspell dictionary (86% kept).
- **Drums and key mode did not follow in any run.** Drum-only files are short (below the 1,024-note window), so they
  made up 0.1–0.3% of training windows, while 93% of labeled windows contained drums and only a third of their captions
  said so.
- **Evaluation:** pooled notes/s (all notes over all seconds) reversed one tempo pair because a few slow pieces
  dominate it; per-piece medians do not. 16 pieces per prompt only resolve large effects.
