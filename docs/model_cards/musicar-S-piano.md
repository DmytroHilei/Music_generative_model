---
license: cc-by-nc-sa-4.0
datasets:
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
- piano
- solo-piano
- classical-music
- maestro
- giantmidi-piano
---

# musicar-S-piano

A 107M-parameter autoregressive transformer that writes solo-piano MIDI one note at a time.

| Parameters | Context | Instruments | Training notes | License |
|---|---|---|---|---|
| 106.84M | 512 notes | piano only | 2.33B | CC BY-NC-SA 4.0 |

![Architecture of musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar-S-piano/resolve/main/diagram.png)

## Model summary

musicar-S-piano is the project's earlier piano base. It is a 14-layer decoder-only transformer over compound note
tokens (one sequence position = one whole note), pretrained on 2.33B notes of solo-piano MIDI (MAESTRO,
GiantMIDI-Piano, Aria-MIDI) on one 8 GB laptop GPU in 12.5 hours. It was later grown into [musicar-S](https://huggingface.co/Dimitri-206265/musicar-S) (129
instruments, 2,048-note context). Both S models are superseded by the [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) for new work.

## Model details

| | |
|---|---|
| Developed by | Dmytro Hilei |
| Model type | Decoder-only transformer, autoregressive over notes (symbolic music, MIDI) |
| Parameters | 106.84M: trunk 99.11M, cascade heads 6.52M, embeddings 1.21M (with the 512 × 768 position table) |
| Architecture | 14 pre-norm blocks, width 768: LayerNorm → causal self-attention (12 heads × 64; Q, K, V from one Linear 768 → 2304; output Linear 768 → 768) → residual add; LayerNorm → GELU MLP 768 → 3072 → 768 → residual add; final LayerNorm. Four cascade output heads: Δt → pitch → duration → velocity |
| Cascade heads | Head *i* reads a running state z₀ = h_t (the trunk state), z_i = z_{i−1} + E_i(a_i), where E_i embeds the value chosen by head *i* (a table V_i × 768); each head is LayerNorm → Linear 768 → 768 → GELU → Linear 768 → 768 (+ residual) → LayerNorm → Linear 768 → V. a_i is the true value in training (teacher forcing) and the sampled value in generation |
| Tokenization | One compound token per note on a 20 ms grid: pitch (128 MIDI pitches), velocity (32 levels), duration (512 steps of 20 ms) and delta-time = time since the previous note's onset (512 steps of 20 ms; both clipped at 10.22 s). Four embedding tables of width 192, concatenated (768) → Linear 768 → 768. No bar, beat or tempo tokens |
| Instruments | Piano only: no instrument attribute |
| Context | 512 notes |
| Positions | Learned absolute position embedding (512 × 768), added to the note vector |
| Special tokens, conditions | None: no BOS / EOS tokens, no window conditions |
| Precision of the weights | bfloat16 |
| License | CC BY-NC-SA 4.0 |
| Code | https://github.com/DmytroHilei/Music_generative_model |

## How to use

From a clone of the code repository, with its environment installed (README, "Setup"):

```bash
hf download Dimitri-206265/musicar-S-piano --local-dir checkpoints/musicar-S-piano
.venv/bin/python generate.py --checkpoint checkpoints/musicar-S-piano/model_bf16.pt \
    --max-new-tokens 1000 --output samples/musicar-S-piano.mid
```

One token is one note, so this writes 1,000 new notes. Without `--prompt` the generation starts from a single artificial
seed note, which is out of distribution; `--prompt file.mid` continues the first `--prompt-notes` notes (default 64) of
a MIDI file. Past 512 notes the KV cache is rebuilt from the most recent notes every `--slide` notes (default 128).
`generate.py` also renders an MP3 unless `--no-mp3` is given (needs fluidsynth, ffmpeg and a General MIDI soundfont).

Loading the pickle-free files directly:

```python
import json
from safetensors.torch import load_file
from musicar.model.gpt import GPT, MusicConfig

cfg = json.load(open("checkpoints/musicar-S-piano/config.json"))
model = GPT(MusicConfig(**cfg["model_args"]))
model.load_state_dict(load_file("checkpoints/musicar-S-piano/model.safetensors"))
model.eval()
```

`MusicConfig` defaults to training settings (dropout 0.2), so call `model.eval()` before inference;
`musicar.model.checkpoint.load_model` does this for `model_bf16.pt`. Sampling with the KV cache is in
`musicar/generation/sampling.py`.

## Intended use

- Research on symbolic music generation: tokenization, output heads, scaling, sampling.
- Continuing a solo-piano MIDI prompt.
- A small, cheap base for fine-tuning experiments and pilots.

Out of scope:

- Commercial use (see License).
- Text-conditioned generation: the model has no caption input.
- Audio: the model reads and writes MIDI notes, not sound.
- Instruments other than piano.

## Training data

| Dataset | Share of training notes | What it is | License note |
|---|---|---|---|
| MAESTRO v3 and GiantMIDI-Piano (curated subset) | ≈ 5% together (31.4M notes, 7,723 files) | Classical piano MIDI | Non-commercial (MAESTRO: CC BY-NC-SA 4.0) |
| Aria-MIDI (deduplicated) | ≈ 95% (550.8M notes, 367,318 files) | Piano MIDI transcribed from recordings; classical is its largest genre, then pop, soundtrack, jazz and rock | CC BY-NC-SA 4.0 (non-commercial) |

Windows of 512 notes are sampled uniformly over the note positions of files longer than 512 notes, so each dataset's
share follows its note count. Pitches are transposed by up to ±5 semitones. None of the project's audio-derived
transcriptions of Ukrainian songs were used.

## Training procedure

| | |
|---|---|
| Initialization | From scratch |
| Steps | 38,000 |
| Batch | 61,440 notes per step (120 windows × 512 notes) |
| Notes seen | 2.33B (≈ 4 passes over the training notes) |
| Optimizer | Muon (Moonlight scaling, momentum 0.95) for the 56 weight matrices of the blocks (99.09M parameters), AdamW (β 0.9, 0.95) for the rest; weight decay 0.1; gradient clipping 1.0 |
| Learning rate | 10⁻³, WSD schedule: 400 warmup steps, constant, then 1 − √ decay to 0 over the last 20% of the steps |
| Regularization | Dropout 0, label smoothing 0 |
| Precision | bf16 autocast with fp8 (torchao float8) matmuls in the transformer blocks, torch.compile; master weights fp32, released weights bf16 |
| Hardware, time | One laptop RTX 5060 (8 GB), 12 h 27 min; finished 2026-09-30 |
| Config | `config/pretrain/big.py` |

## Evaluation

Metric: teacher-forced next-note cross-entropy on held-out validation windows of 512 notes, in nats per note, summed
over the four attributes (Δt, pitch, duration, velocity); lower is better. It measures how well the model predicts
real music, not how good its samples sound; it is not a listening test.

| Validation set | Cross-entropy (nats / note) |
|---|---|
| Aria-MIDI validation (3,735 files) | **5.706** |
| MAESTRO + GiantMIDI-Piano validation (612 files) | 6.987 (pitch 1.434, velocity 1.821, duration 2.473, Δt 1.259) |

The Aria-MIDI result is inside the 5.47–5.72 forecast that the project's scaling ladder (smaller runs) gave before
this run. The final 20% of the schedule (the decay) lowered it from 5.979 to 5.706. This is the evaluation of that
time (512-note windows, the validation sets of that time): it is not comparable with the numbers of musicar-S or
musicar-L, which use 2,048-note windows and other validation sets.

## Limitations and biases

- Classical-heavy data: MAESTRO and GiantMIDI-Piano are classical, and classical is Aria-MIDI's largest genre.
- Short memory: the model sees at most 512 notes; longer pieces are generated with a sliding window, so material from
  further back is forgotten.
- No BOS / EOS tokens: the model has no notion of where a piece starts or ends. It does not end pieces by itself, and
  starting without a prompt is out of distribution.
- Time is absolute (20 ms steps): there are no bars or beats, so metre has to be inferred from the timing.
- The data lean toward Western classical and popular music.
- Aria-MIDI is transcribed from recordings, so it contains transcription errors (missed, extra or mistimed notes),
  which the model learns as if they were music.
- Generated output may reproduce passages from the training data; this was not measured for this model.

## Future work

Planned, not done. The S models will not be trained further; they stay as small, cheap bases for pilots (musicar-S is the base of the caption pilots that come before musicar-L's captioned run). The project's next models build
on musicar-L: first a captioned fine-tune (F1, planned: captions from a frozen multilingual LaBSE encoder, in
English and Ukrainian, section captions and length control; the recipe is being settled in laptop pilots), then a
captioned fine-tune on about 5,400 Ukrainian songs transcribed from audio.

## Files

| File | What it is |
|---|---|
| `model_bf16.pt` | Training checkpoint: bf16 weights + `model_args` + the training config. This is what the repository's scripts load (`generate.py --checkpoint`, `train.py` with `init_from='finetune'`) |
| `model.safetensors` + `config.json` | The same weights without pickle; `config.json` holds `model_args` |
| `diagram.png` | The architecture diagram above |

## License

CC BY-NC-SA 4.0. The training data are licensed for non-commercial use only (MAESTRO and Aria-MIDI under CC BY-NC-SA
4.0, GiantMIDI-Piano non-commercial as well), so the weights are released under the same non-commercial, share-alike
terms.

## Citation

```bibtex
@misc{hilei2026musicar,
  author       = {Dmytro Hilei},
  title        = {musicar: autoregressive symbolic music models},
  year         = {2026},
  howpublished = {\url{https://github.com/DmytroHilei/Music_generative_model}},
  note         = {musicar-S-piano}
}
```
