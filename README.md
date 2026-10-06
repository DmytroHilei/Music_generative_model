# MusicAutoregresiveModel

An autoregressive symbolic-music transformer. It is pretrained on piano and multi-instrument MIDI (MAESTRO,
GiantMIDI, Aria-MIDI, GigaMIDI) and fine-tuned to make piano arrangements of Ukrainian pop-rock songs.

Each position is one note made of compound tokens (pitch, velocity, duration and delta-time on a 20 ms grid). Cascade
output heads predict these, plus the instrument. The window can be conditioned on the instrument set, density and
style.

## Layout

```
train.py              training (nanoGPT-style configs: python train.py config/<run>.py --key=value)
generate.py           sampling: MIDI + MP3, prompts from files or validation sets
musicar/              core package
  model.py            the transformer, embeddings, cascade heads, KV-cache generation
  optim.py            Muon + AdamW
  data_loader.py      tokenization, memory-mapped token stores, the training dataset
  midi_io.py          tokens -> MIDI, MIDI -> MP3
  checkpoint.py       checkpoint path resolution
  configurator.py     command-line / config-file overrides for train.py
  jobstatus.py        progress files for the dashboard
  eval/               evaluations used during training and by the eval/ scripts
config/               run configs
data/                 dataset preparation, song download, audio -> piano reduction
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
.venv/bin/python train.py config/finetune_ua450.py                       # train / fine-tune
.venv/bin/python generate.py --checkpoint checkpoints/ua_2048_v2 --style cover --temperature 0.9 \
    --max-new-tokens 2000 --output samples/out.mid                       # sample
.venv/bin/python tools/status.py                                         # live dashboard of runs and jobs
.venv/bin/python eval/seed_compare.py --checkpoints checkpoints/a checkpoints/b \
    --out-dir samples/compare                                            # continue Ukrainian songs, compare
.venv/bin/python -m musicar.eval.structure_eval --checkpoints checkpoints/a   # whole-song structure metrics
```

## Data and licenses

The training corpora keep their own licenses (MAESTRO, GiantMIDI and Aria-MIDI are non-commercial). Song audio is
downloaded for private research only and is not redistributed. The repository contains only the scripts.
