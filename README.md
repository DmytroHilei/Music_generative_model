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

## Layout

```
train.py              training (nanoGPT-style configs: python train.py config/<run>.py --key=value)
generate.py           sampling: MIDI + MP3, prompts from files or validation sets
musicar/              core package
  model.py            the architecture: transformer, embeddings, cascade heads, KV cache
  sampling.py         generation: KV-cache decoding, CUDA graph, density / instrument / EOS controls
  optim.py            AdamW, AdamW8bit, Muon
  fp8.py              torchao float8 training of the transformer blocks
  memory.py           selective activation checkpointing, spilling saved tensors to RAM
  rejected.py         archive of tested-and-rejected architecture options (MoE, future heads, ...)
  data_loader.py      tokenization, memory-mapped token stores, the training dataset, window conditions
  midi_io.py          tokens -> MIDI, MIDI -> MP3
  checkpoint.py       checkpoint path resolution, loading into a grown model
  configurator.py     command-line / config-file overrides for train.py
  jobstatus.py        progress files for the dashboard
  eval/               evaluations used during training and by the eval/ scripts
config/               run configs
data/                 datasets, token stores and their metadata (see data/README.md)
data_prep/            dataset preparation: downloads, tokenization into stores, audio -> MIDI
docs/                 experiment results (experiments.md)
eval/                 analysis scripts: seeded comparisons, sampling sweeps, settings search
tools/                status dashboard, throughput benchmark, fine-tune sweep, bf16 export
cloud/                rented-GPU pipeline (setup, preflight, launch, checkpoint sync)
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
