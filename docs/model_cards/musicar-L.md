---
license: cc-by-nc-sa-4.0
tags:
- music
- midi
- symbolic-music
- music-generation
- pytorch
---

# musicar-L

A 464M-parameter autoregressive transformer that writes multi-instrument MIDI one note at a time, pretrained on 8.3B
notes of public MIDI data.

| Parameters | Context | Instruments | Training notes | License |
|---|---|---|---|---|
| 464.24M | 2,048 notes | 129 (128 General MIDI programs + drums) | 8.3B (GigaMIDI, Aria-MIDI, Discover MIDI) | CC BY-NC-SA 4.0 |

![Architecture of musicar-L](https://huggingface.co/Dimitri-206265/musicar/resolve/main/L/diagram.png)

## Model summary

musicar-L is the project's pretrained multi-instrument model. Each sequence position is one whole note, and
five cascade output heads choose its time shift, instrument, pitch, duration and velocity one after another. It is a
pretrained model only: it has no text captions, and besides a MIDI prompt its only controls are two optional window
conditions (the set of instruments and the notes per second). It is the starting point of the project's planned
captioned model (see Future work).

Smaller bases of the same family: [musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar/tree/main/S-piano) and [musicar-S](https://huggingface.co/Dimitri-206265/musicar/tree/main/S).

## Model details

| | |
|---|---|
| Developed by | Dmytro Hilei |
| Model type | Decoder-only transformer, autoregressive over notes (symbolic music, MIDI) |
| Parameters | 464.24M: trunk 445.98M, cascade heads 16.28M, embeddings 1.99M |
| Architecture | 28 pre-norm blocks, width 1152: RMSNorm → causal self-attention (18 heads × 64, full attention; Q, K, V from one Linear 1152 → 3456; QK-norm = RMSNorm over each head's queries and keys; RoPE; output Linear 1152 → 1152) → residual add; RMSNorm → SwiGLU feed-forward (gate and up 1152 → 3072, down 3072 → 1152) → residual add; final RMSNorm. Five cascade output heads: Δt → instrument → pitch → duration → velocity |
| Cascade heads | Head *i* reads a running state z₀ = h_t (the trunk state), z_i = z_{i−1} + E_i(a_i), where E_i embeds the value chosen by head *i* (tables of 512, 129, 130 and 512 rows × 1152; the last head needs none); each head is LayerNorm → Linear 1152 → 1152 → GELU → Linear 1152 → 1152 (+ residual) → LayerNorm → Linear 1152 → V. a_i is the true value in training (teacher forcing) and the sampled value in generation |
| Tokenization | One compound token per note on a 20 ms grid: delta-time = time since the previous note's onset (512 steps of 20 ms), instrument (129), pitch (130: 128 MIDI pitches + BOS + EOS), duration (512 steps of 20 ms), velocity (32 levels); delta-time and duration are clipped at 10.22 s. The notes of all tracks are merged into one sequence in onset order; a drum note's pitch is the drum sound. No bar, beat or tempo tokens |
| Input embedding | Pitch, velocity, duration and delta-time tables of width 288 (130, 32, 512 and 512 rows), concatenated (1152) → Linear 1152 → 1152; an instrument table (129 × 1152) added per note; the optional window conditions added at every position: the window's instrument set (129-dim multi-hot → Linear 129 → 1152) and its notes per second (a table of 16 rows: 15 levels + "none") |
| Instruments | 129: the 128 General MIDI programs + drums |
| Context | 2,048 notes (about 1–4 minutes of music) |
| Positions | RoPE (rotary position embedding, base 10,000) in every attention layer; no position table |
| Special tokens, conditions | BOS / EOS (pitch values 128 / 129) mark the start and end of a piece; window conditions as above, "none" = unconditional |
| Precision of the weights | bfloat16 |
| License | CC BY-NC-SA 4.0 |
| Code | https://github.com/DmytroHilei/Music_generative_model |

## How to use

From a clone of the code repository, with its environment installed (README, "Setup"):

```bash
hf download Dimitri-206265/musicar --include "L/*" --local-dir checkpoints/musicar
.venv/bin/python generate.py --checkpoint checkpoints/musicar/L/model_bf16.pt --sample-instruments \
    --max-new-tokens 1000 --output samples/musicar-L.mid
```

One token is one note, so this writes up to 1,000 new notes, starting from the start-of-piece token (BOS); if the
model ends the piece earlier, the output stops at its EOS (`--min-notes` forbids an early end). `--sample-instruments`
lets the instrument head choose each note's instrument; without it every note is piano. `--instruments` limits the
choice (a list of General MIDI programs such as `0,33,25,128`, where 128 = drums, or `prompt` = the prompt's own
instruments) and `--instrument-temperature` sets a separate temperature for it. `--prompt file.mid` continues the first
`--prompt-notes` notes (default 64) of a MIDI file. The window conditions are set with `--cond-instruments` (a program
list, `prompt` or `none`; by default the prompt's instruments when the prompt has them, else none) and `--cond-density`
(notes per second, `prompt` or `none`, the default). Sampling: `--temperature` (default 0.85), `--top-k`, `--top-p`.
Past 2,048 notes the KV cache is rebuilt from the most recent notes every `--slide` notes (default 512). `generate.py`
also renders an MP3 unless `--no-mp3` is given (needs fluidsynth, ffmpeg and a General MIDI soundfont).

Loading the pickle-free files directly:

```python
import json
from safetensors.torch import load_file
from musicar.model.gpt import GPT, MusicConfig

cfg = json.load(open("checkpoints/musicar/L/config.json"))
model = GPT(MusicConfig(**cfg["model_args"]))
model.load_state_dict(load_file("checkpoints/musicar/L/model.safetensors"))
model.eval()
```

`MusicConfig` defaults to training settings (dropout 0.2), so call `model.eval()` before inference;
`musicar.model.checkpoint.load_model` does this for `model_bf16.pt`. Sampling with the KV cache is
`musicar.generation.sampling.generate`.

## Intended use

- Research on symbolic music generation: multi-instrument modelling, sampling, conditioning.
- Continuing a MIDI prompt (piano or multi-instrument), or generating from the start token.
- A base for fine-tuning: the project's fine-tunes on Ukrainian songs (piano arrangements and multi-track) start from
  it, and so does the planned captioned model.

Out of scope:

- Commercial use (see License).
- Text-conditioned generation: this checkpoint has no caption input.
- Audio: the model reads and writes MIDI notes, not sound.

## Training data

| Dataset | Sampling weight | What it is | License note |
|---|---|---|---|
| GigaMIDI ([arXiv:2502.17726](https://arxiv.org/abs/2502.17726)) | 0.4 | Large web collection of multi-track MIDI (its own training split) | Non-commercial (the project's notes list CC BY-NC 4.0); check GigaMIDI's own terms (Hugging Face `Metacreation/GigaMIDI`) |
| Aria-MIDI, deduplicated ([arXiv:2504.15071](https://arxiv.org/abs/2504.15071)) | 0.3 | Piano transcriptions from recordings; its notes enter as piano | CC BY-NC-SA 4.0 (non-commercial) |
| Discover MIDI ([Hugging Face](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)) | 0.3 | Large MIDI collection, deduplicated against GigaMIDI and within itself | CC BY-NC-SA (non-commercial) |

Each training window of 2,048 notes is drawn from one dataset by these weights. BOS / EOS mark where pieces start and
end, and pieces shorter than a window share windows. Pitches are transposed at random by up to ±5 semitones (drums are
not transposed). The run saw 8.3B notes, about 1.8 passes over the data on average (about 4.5 over Aria-MIDI). None of
the project's audio-derived transcriptions of Ukrainian songs were used.

## Training procedure

| | |
|---|---|
| Initialization | From scratch |
| Steps | 126,582 |
| Batch | 65,536 notes per step (32 windows × 2,048 notes) |
| Notes seen | 8.3B (≈ 1.8 epochs on average; Aria-MIDI ≈ 4.5) |
| Data mix | GigaMIDI 0.4, Aria-MIDI 0.3, Discover MIDI 0.3 |
| Objective | Sum of the five cross-entropies (Δt, instrument, pitch, duration, velocity), teacher forcing |
| Optimizer | Muon (Moonlight scaling, momentum 0.95) for the weight matrices of the transformer blocks, AdamW (β 0.9, 0.95) for the rest; one learning rate for both; weight decay 0.1; gradient clipping 1.0 |
| Learning rate | 10⁻³, WSD schedule: 1,000 warmup steps, constant, then 1 − √ decay to 0 over the last 20% of the steps |
| Regularization | Dropout 0, label smoothing 0; both window conditions replaced by "none" in 15% of the windows, so "none" stays usable |
| Precision | bf16 autocast with fp8 (torchao float8) matmuls in the transformer blocks, torch.compile; the checkpoint holds bf16 weights |
| Hardware, time | One rented RTX 5090, about 41 h |
| Config | `config/pretrain/long_450m.py` (launched through `cloud/run/launch.sh`) |

## Evaluation

Metric: teacher-forced next-note cross-entropy on held-out validation windows of 2,048 notes, in nats per note, summed
over the four note attributes (Δt, pitch, duration, velocity); the instrument head's cross-entropy is reported
separately. Lower is better. It measures how well the model predicts real music, not how good its samples sound; it is
not a listening test. One run.

Validation set: clean GigaMIDI validation, i.e. GigaMIDI's validation split without the files whose notes also occur in
its training split. The window conditions are measured on each validation window.

| Clean GigaMIDI validation | Cross-entropy (nats / note) |
|---|---|
| First evaluation (step 2,000, 0.13B notes) | 2.807 |
| Step 100,000, just before the cooldown | 1.876 |
| End of training (8.3B notes) | **1.679** (Δt 0.170, pitch 0.367, duration 0.644, velocity 0.497) |
| End of training, both window conditions "none" | 1.681 |
| Instrument head, end of training | 0.117 |

The diagram (section 6) shows every evaluation of the run, one every 2,000 steps.

## Limitations and biases

- The data lean toward Western popular and classical music in General MIDI instrumentation; other traditions, and
  instruments outside General MIDI, are thin or missing.
- GigaMIDI is collected from the web, so its quality varies, and the model learns from all of it.
- Aria-MIDI is transcribed from recordings, so it contains transcription errors (missed, extra or mistimed notes),
  which the model learns as if they were music.
- Control is limited to the prompt and the two window conditions (instrument set, notes per second): no control of
  genre, mood or form, and no text input.
- Long pieces lose structure: past 2,048 notes the window slides and earlier material leaves the context, and time is
  absolute (no bars or beats).
- Sampling can run away into very dense passages: when continuing 12 Ukrainian songs (one seed; temperature 0.9,
  top-p 0.9, Δt bias 0.1, conditioned on the prompt's instruments), 3 of the 12 continuations reached 45 to 179 notes
  per second.
- Only notes are modelled: no tempo, pedal or other controller events; velocity has 32 levels and time a 20 ms grid.
- Generated output may reproduce passages from the training data.

## Future work

Planned, not done:

- A captioned fine-tune (F1), planned on a rented RTX 5090: captions from a frozen multilingual LaBSE encoder in English
  and Ukrainian, section captions that change during a piece, and length control. The recipe is being settled in
  laptop pilots.
- Then F2: a captioned fine-tune on about 5,400 Ukrainian songs transcribed from audio.
- Later: regenerating parts of a piece (infilling), and inference on small devices (int8 / int4 on a Raspberry Pi 5,
  possibly with grouped-query attention converted after the caption training).

## Files

| File | What it is |
|---|---|
| `model_bf16.pt` | Training checkpoint: bf16 weights + `model_args` + the training config. This is what the repository's scripts load (`generate.py --checkpoint`, `train.py` with `init_from='finetune'`) |
| `model.safetensors` + `config.json` | The same weights without pickle; `config.json` holds `model_args` |
| `diagram.png` | The architecture diagram above |

All in the `L/` folder of the repository; the other two models are the sibling folders.

## License

CC BY-NC-SA 4.0. The training data are under non-commercial terms (Aria-MIDI CC BY-NC-SA 4.0, Discover MIDI
CC BY-NC-SA, GigaMIDI non-commercial), so the model is released under the same non-commercial, share-alike terms.
Check GigaMIDI's own terms before any reuse.

## Citation

```bibtex
@misc{hilei2026musicar,
  author       = {Dmytro Hilei},
  title        = {musicar: autoregressive symbolic music models},
  year         = {2026},
  howpublished = {\url{https://github.com/DmytroHilei/Music_generative_model}},
  note         = {musicar-L}
}
```

Please also cite the training datasets: GigaMIDI ([arXiv:2502.17726](https://arxiv.org/abs/2502.17726)), Aria-MIDI
([arXiv:2504.15071](https://arxiv.org/abs/2504.15071)) and Discover MIDI
([Hugging Face](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)).
