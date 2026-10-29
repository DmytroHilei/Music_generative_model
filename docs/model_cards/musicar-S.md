---
license: cc-by-nc-sa-4.0
base_model: Dimitri-206265/musicar-S-piano
base_model_relation: finetune
datasets:
- Metacreation/GigaMIDI
- loubb/aria-midi
tags:
- music
- midi
- symbolic-music
- music-generation
- midi-generation
- symbolic-music-generation
- music-transformer
- generative-music
- ai-music
- composition
- transformer
- decoder-only
- autoregressive
- rope
- swiglu
- rmsnorm
- qk-norm
- pytorch
- multi-instrument
- general-midi
- drums
- model-growth
---

# musicar-S

A 108M-parameter autoregressive transformer that writes multi-instrument MIDI one note at a time, grown from [musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar-S-piano).

| Parameters | Context | Instruments | Training notes | License |
|---|---|---|---|---|
| 107.93M | 2,048 notes | 129 (128 General MIDI programs + drums) | 197M (band stage), after the piano model's 2.33B | CC BY-NC-SA 4.0 |

![Architecture of musicar-S](https://huggingface.co/Dimitri-206265/musicar-S/resolve/main/diagram.png)

## Model summary

musicar-S is musicar-S-piano, the project's earlier piano base, grown to 129 instruments: each note gained
an instrument attribute with its own embedding and output head, the learned positions were replaced by RoPE, and the
context grew from 512 to 2,048 notes. It was then trained for 3,000 steps (197M notes) on GigaMIDI and Aria-MIDI. It is
the base of the project's 107M caption pilots. For new work it is superseded by the [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L)
and, for generation from text, by [musicar-L-caption](https://huggingface.co/Dimitri-206265/musicar-L-caption).

## Model details

| | |
|---|---|
| Developed by | Dmytro Hilei |
| Model type | Decoder-only transformer, autoregressive over notes (symbolic music, MIDI) |
| Parameters | 107.93M: trunk 99.11M, cascade heads 7.90M, embeddings 0.92M |
| Architecture | 14 pre-norm blocks, width 768: LayerNorm → causal self-attention (12 heads × 64, RoPE on queries and keys; Q, K, V from one Linear 768 → 2304; output Linear 768 → 768) → residual add; LayerNorm → GELU MLP 768 → 3072 → 768 → residual add; final LayerNorm. Five cascade output heads: Δt → instrument → pitch → duration → velocity |
| Cascade heads | Head *i* reads a running state z₀ = h_t (the trunk state), z_i = z_{i−1} + E_i(a_i), where E_i embeds the value chosen by head *i* (a table V_i × 768); each head is LayerNorm → Linear 768 → 768 → GELU → Linear 768 → 768 (+ residual) → LayerNorm → Linear 768 → V. a_i is the true value in training (teacher forcing) and the sampled value in generation |
| Tokenization | One compound token per note on a 20 ms grid: pitch (128 MIDI pitches), velocity (32 levels), duration (512 steps of 20 ms), delta-time = time since the previous note's onset (512 steps of 20 ms; both clipped at 10.22 s), and the instrument (drum notes carry no duration). Pitch, velocity, duration and delta-time: four embedding tables of width 192, concatenated (768) → Linear 768 → 768; the instrument: a table of 129 × 768, added to that note vector. No bar, beat or tempo tokens |
| Instruments | 129: the 128 General MIDI programs + drums |
| Context | 2,048 notes |
| Positions | RoPE (rotary position embedding, base 10,000) in every attention layer; no position table |
| Special tokens, conditions | None: no BOS / EOS tokens, no window conditions |
| Precision of the weights | bfloat16 |
| License | CC BY-NC-SA 4.0 |
| Code | https://github.com/DmytroHilei/Music_generative_model |

## How to use

From a clone of the code repository, with its environment installed (README, "Setup"):

```bash
hf download Dimitri-206265/musicar-S --local-dir checkpoints/musicar-S
.venv/bin/python generate.py --checkpoint checkpoints/musicar-S/model_bf16.pt --sample-instruments \
    --max-new-tokens 1000 --output samples/musicar-S.mid
```

One token is one note, so this writes 1,000 new notes, one MIDI track per instrument. `--sample-instruments` lets the
instrument head choose each note's instrument; without it every note is piano. `--instruments` limits the choice (a
list of General MIDI programs such as `0,33,25,128`, where 128 = drums, or `prompt` = the prompt's own instruments) and
`--instrument-temperature` sets a separate temperature for the instrument choice. Without `--prompt` the generation
starts from a single artificial seed note, which is out of distribution; `--prompt file.mid` continues the first
`--prompt-notes` notes (default 64) of a MIDI file. Past 2,048 notes the KV cache is rebuilt from the most recent notes
every `--slide` notes (default 512). `generate.py` also renders an MP3 unless `--no-mp3` is given (needs fluidsynth,
ffmpeg and a General MIDI soundfont).

Loading the pickle-free files directly:

```python
import json
from safetensors.torch import load_file
from musicar.model.gpt import GPT, MusicConfig

cfg = json.load(open("checkpoints/musicar-S/config.json"))
model = GPT(MusicConfig(**cfg["model_args"]))
model.load_state_dict(load_file("checkpoints/musicar-S/model.safetensors"))
model.eval()
```

`MusicConfig` defaults to training settings (dropout 0.2), so call `model.eval()` before inference;
`musicar.model.checkpoint.load_model` does this for `model_bf16.pt`. Sampling with the KV cache is in
`musicar/generation/sampling.py`.

## Intended use

- Research on symbolic music generation: multi-instrument modelling, model growth, sampling.
- Continuing a MIDI prompt, piano or multi-instrument.
- A small, cheap base for fine-tuning experiments and pilots (it is the base of the project's 107M caption pilots).

Out of scope:

- Commercial use (see License).
- Text-conditioned generation: the model has no caption input (the caption pilots add one in a fine-tune).
- Audio: the model reads and writes MIDI notes, not sound.

## Training data

| Dataset | Share | What it is | License note |
|---|---|---|---|
| GigaMIDI | 0.7 of the sampled windows | Multi-instrument MIDI (128 General MIDI programs + drums); the training split holds 1.96B notes in 1.68M files | Check GigaMIDI's own terms (Hugging Face `Metacreation/GigaMIDI`) |
| Aria-MIDI (deduplicated) | 0.3 | Piano MIDI transcribed from recordings (551M notes in 367k files), kept in the mix against forgetting the piano | CC BY-NC-SA 4.0 (non-commercial) |
| Inherited: MAESTRO v3, GiantMIDI-Piano, Aria-MIDI | the starting weights | The pretraining of musicar-S-piano (2.33B notes of solo piano) | Non-commercial |

Windows of 2,048 notes are sampled per dataset by these weights, then uniformly over the note positions of files longer
than 2,048 notes (shorter files were not used in this run). Pitches are transposed by up to ±5 semitones (drums are not
transposed). None of the project's audio-derived transcriptions of Ukrainian songs were used.

## Training procedure

| | |
|---|---|
| Initialization | Grown from musicar-S-piano: the trunk, the note embedding, the four note heads and their condition embeddings copied; new: the instrument embedding (129 × 768), the instrument head and its condition embedding (129 × 768), with both tables and the head's output layer starting at zero, so the instrument changes no prediction at step 0; the learned position table dropped for RoPE, the one change that is not function-preserving (Aria-MIDI cross-entropy 11.8 at step 0, 6.35 after 250 steps) |
| Steps | 3,000 |
| Batch | 65,536 notes per step (32 windows × 2,048 notes) |
| Notes seen | 197M (≈ 0.1 pass over the training notes) |
| Data mix | GigaMIDI 0.7, Aria-MIDI 0.3 |
| Optimizer | Muon (Moonlight scaling, momentum 0.95) for the 56 weight matrices of the blocks (99.09M parameters), AdamW (β 0.9, 0.95) for the rest; weight decay 0.1; gradient clipping 1.0; all parameters trained |
| Learning rate | 10⁻³, WSD schedule: 200 warmup steps, constant, then 1 − √ decay to 0 over the last 20% of the steps |
| Regularization | Dropout 0, label smoothing 0 |
| Precision | bf16 autocast with fp8 (torchao float8) matmuls in the transformer blocks, torch.compile; released weights bf16 |
| Hardware, time | One laptop RTX 5060 (8 GB), 83 min; finished 2026-10-02 |
| Config | `config/pilots/pilot_mi.py` with `--init_from=finetune --init_ckpt=<musicar-S-piano>` |

## Evaluation

Metric: teacher-forced next-note cross-entropy on held-out validation windows of 2,048 notes (200 windows per set), in
nats per note, summed over the four note attributes (Δt, pitch, duration, velocity); the instrument head's
cross-entropy is reported separately. Lower is better. It measures how well the model predicts real music, not how good
its samples sound; it is not a listening test. One seed.

| Validation set | musicar-S | Same recipe from scratch |
|---|---|---|
| GigaMIDI's validation split (not deduplicated) | **2.232** (pitch 0.526, velocity 0.616, duration 0.848, Δt 0.242) | 2.597 |
| Aria-MIDI validation (2,048-note windows) | 5.819 | 6.931 |
| Instrument head, GigaMIDI validation | 0.218 | 0.258 |

Growing the trained piano model was worth 0.37 nats per note on GigaMIDI at equal steps: the run from scratch ended
where the grown run was at about step 1,150.

Two caveats. GigaMIDI's validation split shares files with its training split (by a loose content hash, 54,209 of
its 209,500 validation files also occur in training; found after this run), so 2.232 is optimistic and not comparable
with the deduplicated ("clean") GigaMIDI validation that musicar-L reports (1.679). At 2,048 notes the Aria-MIDI
validation holds only its 791 longest files, so 5.819 is not comparable with musicar-S-piano's 5.706 (512-note
windows, all files).

## Limitations and biases

- Short band training: only 3,000 steps (197M notes) on multi-instrument data, so it is weaker than musicar-L
  (8.3B notes).
- Only files longer than 2,048 notes were used in this stage, so short pieces and loops are under-represented.
- Instrument choice can drift when sampled freely: in tests of a sibling pilot (the same recipe with stretched
  learned positions), instrument sampling at temperature 0.9 wandered through dozens of instruments, while the
  prompt's instrument set or an instrument temperature of 0.5 kept the set stable.
- No BOS / EOS tokens: the model has no notion of where a piece starts or ends. It does not end pieces by itself, and
  starting without a prompt is out of distribution.
- Time is absolute (20 ms steps): there are no bars or beats, so metre has to be inferred from the timing.
- The data lean toward Western classical and popular music.
- Aria-MIDI is transcribed from recordings, so it contains transcription errors (missed, extra or mistimed notes),
  which the model learns as if they were music.
- Generated output may reproduce passages from the training data; this was not measured for this model.

## Future work

The S models will not be trained further; they stay as small, cheap bases for pilots (this model is the base of the
caption pilots that came before musicar-L's captioned run). That run, F1 (captions from a frozen multilingual LaBSE
encoder, in English and Ukrainian, section captions and length control), is done and released as
[musicar-L-caption](https://huggingface.co/Dimitri-206265/musicar-L-caption). Planned, not done: a captioned fine-tune
of it on about 5,400 Ukrainian songs transcribed from audio.

## Files

| File | What it is |
|---|---|
| `model_bf16.pt` | Training checkpoint: bf16 weights + `model_args` + the training config. This is what the repository's scripts load (`generate.py --checkpoint`, `train.py` with `init_from='finetune'`) |
| `model.safetensors` + `config.json` | The same weights without pickle; `config.json` holds `model_args` |
| `diagram.png` | The architecture diagram above |

## License

CC BY-NC-SA 4.0. The model starts from weights trained on non-commercial data (MAESTRO and Aria-MIDI under CC
BY-NC-SA 4.0, GiantMIDI-Piano non-commercial as well) and is trained further on Aria-MIDI and GigaMIDI, so it is
released under the same non-commercial, share-alike terms. Check GigaMIDI's own terms before any reuse.

## Citation

```bibtex
@misc{hilei2026musicar,
  author       = {Dmytro Hilei},
  title        = {musicar: autoregressive symbolic music models},
  year         = {2026},
  howpublished = {\url{https://github.com/DmytroHilei/Music_generative_model}},
  note         = {musicar-S}
}
```
