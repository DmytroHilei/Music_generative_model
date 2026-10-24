# MusicAutoregresiveModel

An autoregressive symbolic-music transformer. It is pretrained on piano and multi-instrument MIDI (MAESTRO,
GiantMIDI, Aria-MIDI, GigaMIDI, Discover MIDI) and fine-tuned on Ukrainian pop-rock songs, as piano arrangements and as
multi-track band arrangements.

Each position is one note made of compound tokens (pitch, velocity, duration and delta-time on a 20 ms grid). Cascade
output heads predict these, plus the instrument. The window can be conditioned on the instrument set, density and
style.

## Overview

- **Data built:**
  - public MIDI corpora (MAESTRO, GiantMIDI-Piano, Aria-MIDI, GigaMIDI, Discover MIDI) tokenized straight from their
    archives into memory-mapped token stores, ≈ 4.5B unique notes, deduplicated across corpora;
  - a Ukrainian target set built from audio: about 5,900 songs separated into stems (Demucs) and transcribed to piano
    reductions and multi-track MIDI, plus ~700 transcribed piano covers.
- **Models:** from 6M to 450M parameters. The 450M multi-instrument model (RoPE, context 2,048 notes, 129 instruments,
  instrument-set and density conditioning) was trained on 8.3B notes on a rented RTX 5090 (~41 h). Everything else,
  including fine-tuning the 450M, ran on an 8 GB laptop GPU (fp8, activation checkpointing, 8-bit optimizer states).
- **Method:** one change per experiment, validation loss per note attribute, generation metrics against the real
  continuation of each song, seed checks and listening tests for final choices. Scaling ladders and iso-FLOP curves
  sized the models, and the forecasts held (107M: predicted 5.47–5.72, measured 5.706).
- **Selected results:**
  - cascade output heads −0.41 nats/note vs independent heads;
  - Muon −0.54 vs AdamW in pretraining, but AdamW for small fine-tunes;
  - RMSNorm + SwiGLU + QK-norm −0.75 over 3 seeds;
  - growing a trained model 2.5–4× faster than training from scratch;
  - the fine-tuned 450M halves the distance to real piano covers vs the earlier best model.
- **Rejected after testing:** mixture of experts, a bigger pitch head, future-prediction heads, velocity augmentation
  (kept in [`musicar/model/rejected.py`](musicar/model/rejected.py) with their results).

Full results: [`docs/experiments.md`](docs/experiments.md).

## Architecture

![Captioned music model](docs/figures/architecture.png)

The figure shows the captioned layout of the 450M model as F1 will train it: 427M trainable parameters with the two
caption adapters and grouped-query attention (6 key / value heads for 18 query heads; 464M with full attention).
The sections are numbered in reading order:

- **1–4: the model.** Captions go through the frozen LaBSE encoder and a small learned-query adapter, and become a
  few vectors inside the note sequence. The trunk is 28 pre-norm blocks over compound note tokens. The cascade heads
  pick Δt, instrument, pitch, duration and velocity one after another, each seeing the values already chosen.
- **5: generation**, with classifier-free guidance over the captions.
- **6: the section-caption recipe.** Without it the model follows the first section's caption and ignores later ones.
- **7: training curves** behind the design choices: trunk block, base pretraining, section-caption measures, and
  the conversion to grouped-query attention.
- **8–9: training stages and the audio → MIDI pipeline** that builds the Ukrainian fine-tune set.
- **10–11: glossary and references.**

Status: the music model without captions is trained. The caption stage at 450M (the F1 run) is still planned, so the
caption results in the figure come from 107M pilots. F1 will also add a length-control input (roadmap item 11), not
drawn yet; infilling (item 12) comes in a later fine-tune. The cascade-vs-independent-heads curve will be added after that
A/B is re-run on the current setup; the only existing one is from an early 6M piano model.

The figure is TikZ / pgfplots: [`docs/figures/architecture.tex`](docs/figures/architecture.tex) (PDF and SVG next to
it). [`docs/figures/curves_data.py`](docs/figures/curves_data.py) rebuilds the curve data from the run logs, then
`pdflatex architecture.tex` in `docs/figures/`.

## Layout

```
train.py              training (nanoGPT-style configs: python train.py config/<group>/<run>.py --key=value)
generate.py           sampling: MIDI + MP3, prompts from files or validation sets
musicar/              core package
  model/              gpt.py (transformer, embeddings, cascade heads, KV cache), checkpoint.py (paths, loading into
                      a grown model, saving), rejected.py (tested-and-rejected options: MoE, future heads, ...)
  training/           optim.py (AdamW, AdamW8bit, Muon, LR schedule), fp8.py (torchao float8), memory.py (selective
                      activation checkpointing, spilling to RAM), configurator.py (overrides for train.py)
  data/               loader.py (tokenization, memory-mapped token stores, the dataset, window conditions),
                      midi_io.py (tokens -> MIDI, MIDI -> MP3)
  text/               encoder.py (frozen LaBSE), vocab.py (EN + UK caption vocabulary), captions.py (per-file captions)
  generation/         sampling.py (KV-cache decoding, CUDA graph, density / instrument / EOS / guidance controls),
                      settings_router.py (generation settings from the seed song)
  eval/               evaluations used during training and by the eval/ scripts
  jobstatus.py        progress files for the dashboard
config/               run configs: pretrain/, finetune/, pilots/, smoke.py (stacks on any of them)
data/                 datasets, token stores and their metadata (see data/README.md); skryabin/ = the hand-picked songs
data_prep/            datasets/ (downloads, tokenization into stores, store splits), songs/ (song search, download,
                      audio -> MIDI, fine-tune CSVs), labels/ (features, labels, pseudo-labels, caption text)
docs/                 experiment results (experiments.md), figures/ (architecture figure and its curve data)
eval/                 settings/ (sampling sweeps, settings search), compare/ (seeded comparisons, text control,
                      sample statistics), probes/ (hidden-state probes, label coverage)
tools/                status.py (dashboard), sweeps/ (fine-tune sweeps), bench/ (GPU / CPU throughput),
                      checkpoints/ (bf16 export, weight interpolation)
cloud/                rented-GPU pipeline: setup/ (instance setup, preflight), run/ (launch, auto-resume, stop),
                      sync/ (Hugging Face Hub checkpoints and data), status/
tests/                CPU tests per package area: python tests/run.py [filter]  (or pytest tests/)
legacy/               earlier LSTM and transformer versions
```

`checkpoints/`, `logs/`, `samples/`, `results/` and the data caches are not tracked by git.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .     # the musicar package
```

MP3 rendering needs `fluidsynth`, `ffmpeg` and a General MIDI soundfont (`fluid-soundfont-gm`).

## Usage

```bash
.venv/bin/python train.py config/finetune/finetune_ua450.py                       # train / fine-tune
.venv/bin/python generate.py --checkpoint checkpoints/ua_2048_v2 --style cover --temperature 0.9 \
    --max-new-tokens 2000 --output samples/out.mid                       # sample
.venv/bin/python tools/status.py                                         # live dashboard of runs and jobs
.venv/bin/python eval/compare/seed_compare.py --checkpoints checkpoints/a checkpoints/b \
    --out-dir samples/compare                                            # continue Ukrainian songs, compare
.venv/bin/python -m musicar.eval.structure_eval --checkpoints checkpoints/a   # whole-song structure metrics
```

## Data and licenses

The training corpora keep their own licenses (MAESTRO, GiantMIDI and Aria-MIDI are non-commercial). Song audio is
downloaded for private research only and is not redistributed. The repository contains only the scripts.
