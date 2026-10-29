---
license: cc-by-nc-sa-4.0
base_model: Dimitri-206265/musicar-L
datasets:
- Metacreation/GigaMIDI
- loubb/aria-midi
- projectlosangeles/Discover-MIDI-Dataset
- amaai-lab/MidiCaps
language:
- en
- uk
tags:
- music
- midi
- symbolic-music
- music-generation
- midi-generation
- symbolic-music-generation
- text-to-midi
- text-to-music
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
- labse
- classifier-free-guidance
- pytorch
- multi-instrument
- general-midi
- drums
- fp8-training
---

# musicar-L-caption

[musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) fine-tuned to follow a text caption in English or
Ukrainian and a requested length in seconds: a 477M-parameter autoregressive transformer that writes multi-instrument
MIDI one note at a time.

| Parameters | Context | Inputs | Fine-tuning notes | License |
|---|---|---|---|---|
| 476.94M (+ the frozen LaBSE text encoder, 471M, not included) | 2,048 notes | caption (EN / UK), length in seconds, optional MIDI prompt | 2.18B on top of musicar-L's 8.3B | CC BY-NC-SA 4.0 |

## Model summary

The trunk, heads and note tokens are musicar-L's (each sequence position is one whole note; five cascade output heads
choose its time shift, instrument, pitch, duration and velocity one after another). The fine-tune (F1) adds three
inputs:

- **Caption**: a frozen multilingual [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) encoder reads the
  caption; an adapter turns its token states into 16 vectors placed in front of the music, which every note attends
  to. Classifier-free guidance works because 15% of the training windows had no caption.
- **Length**: every note also gets the seconds left to the requested end (33 log levels from 0.5 s to 20 min) and
  the share of the piece already played (17 levels), so the piece ends near the requested length with its EOS.
- **Section captions**: 4 more prefix vectors that describe the current 256-note section (instruments, pace,
  dynamics, register, rhythm, texture), so a caption can change during a piece. `generate.py` does not take
  user-written section captions yet; `--keep-style` (on in `config/generate/f1.toml`) uses this input to hold the
  opening's own description over the rest of the piece.

It follows broad attributes: pace, loudness, register, piano / no drums / strings / guitar, sad vs happy, in both
languages. It does not reliably follow genre, key or specific instruments beyond these (see Evaluation and
Limitations). The project's web page serves this model ("musicar", invite or GitHub sign-in, see the code
repository's README).

Pretrained bases of the same family: [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) (this model's
base), [musicar-S](https://huggingface.co/Dimitri-206265/musicar-S),
[musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar-S-piano).

## Model details

| | |
|---|---|
| Developed by | Dmytro Hilei |
| Model type | Decoder-only transformer, autoregressive over notes (symbolic music, MIDI), text-conditioned through a prefix |
| Fine-tuned from | [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) |
| Parameters | 476.94M: trunk 445.98M, cascade heads 16.28M, note embeddings and window conditions 1.99M, caption adapter 6.33M, section adapter 6.31M, length tables 0.06M |
| Architecture | As musicar-L: 28 pre-norm blocks, width 1152, 18 heads × 64 with QK-norm and RoPE, full attention, SwiGLU 3072, RMSNorm; five cascade heads Δt → instrument → pitch → duration → velocity |
| Text encoder | `sentence-transformers/LaBSE` (frozen, 768-dim token states, full precision; downloaded on first use, not in this repository) |
| Caption adapter | 16 learned queries attend once over the caption's LaBSE token states (16 heads, width 768) → LayerNorm → Linear 768 → 2048 → GELU → Linear 2048 → 1152 → a per-channel gate; a caption-less input gets 16 learned "null" vectors instead (the unconditional half of guidance). The 16 vectors sit before the notes as a prefix |
| Section adapter | The same shape with 4 queries; its vectors describe the current 256-note section and are repeated every 64 notes |
| Length input | Two tables added at every note: seconds to the end (33 levels: none, "at the end" < 0.5 s, then 31 log steps to 1,200 s) and progress = elapsed / total (17 levels) |
| Window conditions | musicar-L's instrument-set and notes-per-second inputs are still there, but captioned windows were trained without them (always "none"); leave them off |
| Tokenization | As musicar-L: one compound token per note on a 20 ms grid; BOS / EOS = pitch values 128 / 129 |
| Context | 2,048 notes (the long fine-tuning run); the last 600 steps were at 512 |
| Precision of the weights | bfloat16 |
| License | CC BY-NC-SA 4.0 |
| Code | https://github.com/DmytroHilei/Music_generative_model |

## How to use

From a clone of the code repository, with its environment installed (README, "Setup"):

```bash
hf download Dimitri-206265/musicar-L-caption --local-dir checkpoints/musicar-L-caption
.venv/bin/python generate.py --config f1 --checkpoint checkpoints/musicar-L-caption/model_bf16.pt \
    --text "a calm jazz piano trio" --length-seconds 60 --output samples/calm_jazz.mid
```

`--config f1` loads the decoding settings the project tuned for this model (`config/generate/f1.toml`): guidance 2.5,
temperature 1.1, the DRY repetition penalty on pitch-class chords (multiplier 3), instruments chosen by the model, no
end before 85% of the requested length and a hard stop at 1.2×. Captions can be Ukrainian (`--text "сумна повільна
балада для фортепіано"`). Without `--length-seconds` the model chooses the length itself. `--best-of N` generates N
candidates and keeps the one whose notes best match the caption; `--prompt file.mid` continues a MIDI file instead of
starting from BOS.

Starting from a real opening (the project's best setting, see Evaluation) needs a captioned token store of GigaMIDI,
which is not part of this repository: `--retrieve gigamidi_replay_train --retrieve-strict genre,mood` picks the first
128 notes of a training song whose caption matches, and the model continues it.

Loading the pickle-free files directly:

```python
import json
from safetensors.torch import load_file
from musicar.model.gpt import GPT, MusicConfig

cfg = json.load(open("checkpoints/musicar-L-caption/config.json"))
model = GPT(MusicConfig(**cfg["model_args"]))
model.load_state_dict(load_file("checkpoints/musicar-L-caption/model.safetensors"))
model.eval()
```

`MusicConfig` defaults to training settings (dropout 0.2), so call `model.eval()` before inference;
`musicar.model.checkpoint.load_model` does this for `model_bf16.pt`. The caption vectors come from
`musicar.text.encoder.text_encoder(device).encode([caption], tokens=True)`; sampling with the KV cache and guidance is
`musicar.generation.sampling.generate`, the whole path (caption, length, best-of, retrieval) is
`musicar.generation.piece.Generator`.

## Intended use

- Research on text-conditioned symbolic music generation: caption following, guidance, length control.
- Sketching multi-instrument MIDI from a short description (mood, pace, loudness, register, instruments) in English or
  Ukrainian, to edit further in a DAW.
- A base for further captioned fine-tunes (the project's planned F2 on Ukrainian songs).

Out of scope:

- Commercial use (see License).
- Detailed control: exact instruments, key, chord progression, tempo in BPM, lyrics or song structure.
- Audio: the model reads and writes MIDI notes, not sound.

## Training data

Notes: the same sources as musicar-L. Captions are built per training window from labels of the files:

| Source | Sampling weight | Captions from |
|---|---|---|
| GigaMIDI, files with labels | 0.55 | MidiCaps captions ([arXiv:2406.02255](https://arxiv.org/abs/2406.02255)), XMIDI emotion / genre labels, GigaMIDI v2 metadata (curated styles, artist), plus features measured on the notes |
| Aria-MIDI, files with labels | 0.15 | Aria-MIDI's genre and composer metadata |
| GigaMIDI, files without labels | 0.30 | Features measured on the notes only (instruments, pace, register, loudness, key, length) |

A caption is drawn per window: Ukrainian with probability 0.3, else English. For a MidiCaps file the whole MidiCaps
caption (or its Ukrainian translation) is used 40% of the time; otherwise a random subset of the file's attributes is
rendered with a hand-written bilingual vocabulary as a noun phrase ("a sad slow piano ballad"), a request ("Play a slow piece for strings",
"Зіграй ...") or tags. Known artists and composers (415 hand-written entries) can be named
("a piece by Chopin", "у стилі гурту Queen"). 15% of the windows have no caption. None of the project's
audio-derived transcriptions of Ukrainian songs were used.

## Training procedure

Three stages after musicar-L, 65,536 notes per step in each:

| | Caption pickup | F1 long run | Caption phase |
|---|---|---|---|
| Steps | 1,500 | 31,098 | 600 |
| Notes | 0.10B | 2.04B | 0.04B |
| Window | 512 notes | 2,048 notes | 512 notes |
| What trains | Captions only | Captions, section captions, length | Captions, section captions, length |
| Windows at a piece start / end | 50% / 5% | 50% / 25% | 50% / 25% |
| Hardware, time | Laptop RTX 5060 (8 GB) | One rented RTX 5090, about 14 h | Same RTX 5090, 20 min |
| Config | `logs/queue_20261026_tail.sh` (job 6'') | `config/f1/f1.py` | `config/f1/f1_cap.py` |

The first stage exists because a strong unconditional trunk learns to ignore a new prefix at 2,048 notes: in short
windows that start at a piece's beginning, the caption is worth more per loss position. The long run was then
dominated by notes far from the caption, which wore the reading down, and the short caption phase at the end rebuilt
it (half the windows at piece starts, about 60 captioned openings per step instead of 14).

| | |
|---|---|
| Objective | Sum of the five cross-entropies (Δt, instrument, pitch, duration, velocity), teacher forcing |
| Optimizer | 8-bit AdamW, learning rate 10⁻⁴ (caption / section adapters and length tables 10×), weight decay 0.1, gradient clipping 1.0 |
| Schedule | Long run: WSD with 300 warmup steps and a 1 − √ decay to 10⁻⁵ over the last 20%; caption phase: 100 warmup steps, decay over the last 30% |
| Section windows | Half the windows of the long run join two pieces at a section boundary (no file caption, the second piece's section captions), so the section caption changes inside the window; section captions dropped in 20% |
| Length input | The target length jittered by ±10% in training |
| Regularization | Dropout 0, label smoothing 0; caption dropout 0.15 |
| Precision | bf16 autocast with fp8 (torchao float8) matmuls in the transformer blocks, torch.compile; the checkpoint holds bf16 weights |

## Evaluation

One run, one seed per measurement. These measure what can be counted on the notes, not how good the music sounds; no
listening test was run.

**Cross-entropy** (teacher-forced, nats per note, the four note attributes summed, 2,048-note windows, window
conditions "none"):

| | musicar-L | F1 long run | musicar-L-caption |
|---|---|---|---|
| Captioned GigaMIDI held-out (3,648 files) | - | 1.882 | 1.898 (Δt 0.197, pitch 0.379, duration 0.759, velocity 0.563) |
| Clean GigaMIDI validation (musicar-L's set) | 1.681 | 1.750 | 1.763 |
| Instrument head, captioned held-out | - | 0.126 | 0.128 |

The fine-tune costs about 0.08 nats on musicar-L's validation set.

**How much the caption helps right after BOS** (200 held-out openings of 128 notes, cross-entropy with the true
caption vs with none, in nats per note): notes 0-3 −0.35, notes 0-31 −0.068, notes 0-127 −0.020. The caption mostly
shapes how a piece starts; later notes depend mostly on the notes before them.

**Does it follow the caption?** 17 caption pairs (8 attributes in English, the same in Ukrainian, plus sad / happy),
16 pieces of 512 notes per caption from BOS, temperature 0.9, top-p 0.9. Selected contrasts (caption + vs its
opposite or no caption):

| Pair | Measured | Guidance 1 | Guidance 2 |
|---|---|---|---|
| fast vs slow | notes per second | 20.0 vs 5.5 | 28.7 vs 3.9 |
| loud vs quiet | mean velocity (of 32) | 26.0 vs 16.2 | 31.0 vs 16.2 |
| high vs low | mean pitch | 70.7 vs 39.9 | 76.2 vs 40.6 |
| piano vs none | piano share of notes | 1.00 vs 0.33 | 1.00 vs 0.33 |
| no drums vs none | pieces with drums | 0.00 vs 0.56 | 0.00 vs 0.56 |
| strings vs none | string share | 0.10 vs 0.02 | 0.40 vs 0.02 |
| guitar vs none | guitar share | 0.54 vs 0.27 | 0.63 vs 0.27 |
| drums vs none | drum share | 0.42 vs 0.26 | 0.29 vs 0.26 |
| minor vs major | minor-key share | 0.27 vs 0.33 | 0.39 vs 0.20 |
| fast vs slow, Ukrainian | notes per second | 19.7 vs 18.1 | 28.2 vs 12.6 |

Follow score F (the mean over the pairs of each contrast over a fixed "clearly audible" step, clipped; 1 = every
attribute moves audibly): 0.82 at guidance 1, 1.03 at guidance 2 (the F1 long run before the caption phase: 0.77 /
0.82). With the tuned settings of `config/generate/f1.toml` (guidance 2.5, temperature 1.1, DRY): F 1.06; starting
from a retrieved real opening: F 1.27, with fewer verbatim repeats (13% vs 17% of the notes inside exact copies of
16+ chords).

**Length** (32 pieces per length from BOS, ended by the model's own EOS, before the end-floor rule of
`config/generate/f1.toml`):

| Requested | Median end | Within 10% | Within 20% |
|---|---|---|---|
| 30 s | 31.2 s | 50% | 75% |
| 60 s | 57.7 s | 59% | 81% |
| 120 s | 116.8 s | 31% | 56% |

## Limitations and biases

- Caption following is broad, not precise: the attributes above move, but genre words, key and named instruments
  other than piano / strings / guitar / drums are weak or ignored; "drums" barely moves at guidance 2, and minor vs
  major only moves with guidance.
- The caption mostly sets the opening. Over a long piece the music follows its own notes, and past 2,048 notes the
  window slides and the opening leaves the context.
- The model ends some pieces early on its own: in a test of 120-s requests, 1 in 6 ended around 45 s. The tuned
  settings forbid an end before 85% of the length.
- Sampling can run into very dense passages and into verbatim loops: with the tuned settings, 11% of 10-s windows
  were over 40 notes per second and 17% of the notes sat inside exact repeats of 16+ chords (real songs: about 10% and
  13%).
- Captions are partly synthetic (built from labels and measured features) and their labels lean toward Western
  popular and classical music in General MIDI instrumentation; the data biases of musicar-L apply.
- User-written section captions are not reachable from `generate.py` yet (only `--keep-style` uses the section
  input), and section switching has no separate evaluation at this size.
- Generated output may reproduce passages from the training data, more so when continuing a retrieved real opening.

## Future work

Planned, not done:

- Section-by-section generation in the app (the trained section captions).
- F2: a captioned fine-tune on about 5,400 Ukrainian songs transcribed from audio.
- Inference on small devices (int8 / int4 on a Raspberry Pi 5).

## Files

| File | What it is |
|---|---|
| `model_bf16.pt` | Training checkpoint: bf16 weights + `model_args` + the training config. This is what the repository's scripts load (`generate.py --checkpoint`, `train.py` with `init_from='finetune'`) |
| `model.safetensors` + `config.json` | The same weights without pickle; `config.json` holds `model_args` |

## License

CC BY-NC-SA 4.0. The training notes are under non-commercial terms (Aria-MIDI CC BY-NC-SA 4.0, Discover MIDI
CC BY-NC-SA, GigaMIDI non-commercial), and the captions use MidiCaps (CC BY-SA 4.0) and XMIDI labels (no license
stated; research use), so the model is released under non-commercial, share-alike terms. LaBSE (Apache 2.0) is not
redistributed here. Check GigaMIDI's and XMIDI's own terms before any reuse.

## Citation

```bibtex
@misc{hilei2026musicar,
  author       = {Dmytro Hilei},
  title        = {musicar: autoregressive symbolic music models},
  year         = {2026},
  howpublished = {\url{https://github.com/DmytroHilei/Music_generative_model}},
  note         = {musicar-L-caption}
}
```

Please also cite the training datasets: GigaMIDI ([arXiv:2502.17726](https://arxiv.org/abs/2502.17726)), Aria-MIDI
([arXiv:2504.15071](https://arxiv.org/abs/2504.15071)), Discover MIDI
([Hugging Face](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)), MidiCaps
([arXiv:2406.02255](https://arxiv.org/abs/2406.02255)) and XMIDI, and the LaBSE encoder
([arXiv:2007.01852](https://arxiv.org/abs/2007.01852)).
