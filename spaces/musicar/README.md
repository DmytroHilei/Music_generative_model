---
title: musicar
emoji: 🎹
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
license: cc-by-nc-sa-4.0
short_description: Text-to-MIDI and MIDI continuation with musicar models
models:
  - Dimitri-206265/musicar-L-caption
  - Dimitri-206265/musicar-L
  - Dimitri-206265/musicar-S
  - Dimitri-206265/musicar-S-piano
  - sentence-transformers/LaBSE
---

# musicar

A demo of the [musicar](https://github.com/DmytroHilei/Music_generative_model) models: autoregressive transformers
that write MIDI one note at a time. It runs on ZeroGPU: each request borrows a GPU for its generation.

| Model | Parameters | Plays | Per request | Speed here |
|---|---|---|---|---|
| [musicar-L-caption](https://huggingface.co/Dimitri-206265/musicar-L-caption) (default) | 477M | from a description in English or Ukrainian | 10-60 s of music, at most 2,048 notes | 96 notes/s (1,173 notes in 12.2 s) |
| [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) | 464M | 128 General MIDI instruments + drums | up to 2,048 notes | 156 notes/s (1,968 in 12.6 s) |
| [musicar-S](https://huggingface.co/Dimitri-206265/musicar-S) | 108M | 128 General MIDI instruments + drums | up to 2,048 notes | 326 notes/s (2,048 in 6.3 s) |
| [musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar-S-piano) | 107M | solo piano | up to 2,048 notes | 717 notes/s (2,048 in 2.9 s) |

Speeds: one request on this Space's ZeroGPU (half an RTX Pro 6000 Blackwell, bf16, CUDA graphs), plus 1-6 s to
get a GPU (the most right after the Space starts). Each request generates from the start, or continues an uploaded MIDI file, and returns the MIDI file,
an audio rendering (fluidsynth with the FluidR3 GM soundfont) and a piano roll. musicar-L-caption uses the code
repository's `f1` preset (guidance 2.5, DRY on pitch-class chords, the length input) with temperature 0.9 and top-p
0.9, the settings of its model card's caption evaluation: at the preset's own temperature 1.1 without top-p, pieces
spread over many instruments. musicar-S has no start-of-piece token and gets very dense when it starts from nothing,
so without an upload it continues a built-in 4-bar start (I-V-vi-IV, piano, bass, drums) that is left out of the
result.

Hugging Face gives every visitor a daily ZeroGPU allowance (2 minutes without an account, 5 with a free one, 40 with
PRO). A request reserves its expected GPU time up front (start-up plus its notes at about 75% of the speeds above,
with a margin: at most 45 s) and is charged the time it uses. For longer sessions use the
[Colab notebook](https://colab.research.google.com/github/DmytroHilei/Music_generative_model/blob/transformer-v2/notebooks/musicar_colab.ipynb).

## Run it locally

```bash
sudo apt install fluidsynth fluid-soundfont-gm   # for the audio; without them the app returns MIDI and the piano roll
pip install gradio==6.31.0 -r requirements.txt
python app.py
```

It uses a CUDA GPU when there is one, else the CPU (ask for fewer notes there).

## License

The models are released under CC BY-NC-SA 4.0: their training data (GigaMIDI, Aria-MIDI, Discover MIDI and others,
see the model cards) are under non-commercial terms.
