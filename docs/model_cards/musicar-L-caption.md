---
license: cc-by-nc-sa-4.0
language:
- en
- uk
base_model: Dimitri-206265/musicar-L
datasets:
- Metacreation/GigaMIDI
- loubb/aria-midi
- amaai-lab/MidiCaps
tags:
- music
- midi
- symbolic-music
- music-generation
- midi-generation
- symbolic-music-generation
- text-to-music
- text-to-midi
- music-transformer
- generative-music
- ai-music
- composition
- transformer
- decoder-only
- autoregressive
- classifier-free-guidance
- labse
- multilingual
- ukrainian
- rope
- swiglu
- rmsnorm
- qk-norm
- pytorch
- multi-instrument
- general-midi
- fp8-training
---

# musicar-L-caption

A 477M-parameter autoregressive transformer that writes multi-instrument MIDI from a text prompt in English or
Ukrainian, one note at a time. It is [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) fine-tuned with
captions, section captions and a length input (the project's F1 run).

| Parameters | Context | Instruments | Prompt languages | Fine-tuning notes | License |
|---|---|---|---|---|---|
| 477M trainable (+ frozen LaBSE text encoder, 471M, downloaded separately) | 2,048 notes | 129 (128 General MIDI programs + drums) | English, Ukrainian | ≈ 2.1B on top of musicar-L's 8.3B | CC BY-NC-SA 4.0 |

![Architecture of musicar-L-caption](https://huggingface.co/Dimitri-206265/musicar-L-caption/resolve/main/diagram.png)

## Model summary

The prompt goes through a frozen multilingual sentence encoder (LaBSE). A small adapter with 16 learned queries turns
its token states into 16 vectors placed in front of the notes. The music model is musicar-L's: 28 pre-norm blocks
over one compound token per note, and five cascade heads choosing time shift, instrument, pitch, duration and
velocity one after another. Generation uses classifier-free guidance over the caption. No network or API is needed at
generation time.

Two more inputs were trained:

- **Section captions:** a caption per 256-note section, inserted where the section starts and repeated every 64
  notes, so the music can change during a piece. The command line takes one caption per piece; section captions are
  available through the Python API (`musicar.generation.sampling.generate(sections=...)`).
- **Length:** the requested length in seconds, given to every note as the time left and the share of the piece
  played. `--length-seconds` sets it.

Listen first: [text-to-MIDI samples](https://github.com/DmytroHilei/Music_generative_model/tree/transformer-v2/docs/listen)
(6 prompts, English and Ukrainian, MP3 and MIDI). They come from an earlier checkpoint of the same run (after the
first caption stage), not from these exact weights.

Other models of the family: [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) (the base, no captions),
[musicar-S](https://huggingface.co/Dimitri-206265/musicar-S),
[musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar-S-piano).

## Model details

| | |
|---|---|
| Developed by | Dmytro Hilei |
| Model type | Decoder-only transformer, autoregressive over notes (symbolic music, MIDI), conditioned on text |
| Parameters | 477M trainable: musicar-L's 464.24M (trunk 445.98M, cascade heads 16.28M, embeddings 1.99M) + the two caption adapters (≈ 13M) + the length tables (0.06M). LaBSE (471M) is frozen and not part of these weights |
| Trunk | As musicar-L: 28 pre-norm blocks, width 1152, RMSNorm, causal self-attention with 18 heads × 64 (full attention), QK-norm, RoPE, SwiGLU (hidden 3072), final RMSNorm |
| Text encoder | [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) (multilingual BERT-base, frozen), token states of up to 128 tokens |
| Caption adapter | 16 learned queries attend once over the LaBSE token states, then LayerNorm → Linear 768 → 2048 → GELU → Linear 2048 → 1152 and a per-channel gate: 16 vectors. An empty prompt gets 16 learned null vectors (the "no caption" of dropout and guidance). The vectors sit in front of the notes and carry no loss |
| Section captions | A second adapter of the same form, 4 vectors per section caption, inserted into the sequence at each 256-note section start and again every 64 notes inside the section |
| Length input | Two zero-initialised tables added at every note: the seconds left until the requested end (33 log-scaled levels, 0 = none) and the share of the piece played (17 levels) |
| Tokenization | As musicar-L: one compound token per note on a 20 ms grid (Δt 512, instrument 129, pitch 130 incl. BOS / EOS, duration 512, velocity 32); all tracks merged in onset order |
| Window conditions | musicar-L's instrument-set and notes-per-second conditions are still in the weights, but were switched off on every captioned window; use the caption instead |
| Precision of the weights | bfloat16 |
| License | CC BY-NC-SA 4.0 |
| Code | https://github.com/DmytroHilei/Music_generative_model |

## How to use

From a clone of the code repository, with its environment installed (README, "Setup"). LaBSE is downloaded from the
Hugging Face Hub on first use (about 1.9 GB).

```bash
hf download Dimitri-206265/musicar-L-caption --local-dir checkpoints/f1_cap
.venv/bin/python generate.py --config f1 --text "A melancholic pop rock song with drums, guitar and bass" \
    --length-seconds 60 --output samples/f1.mid
.venv/bin/python generate.py --config f1 --text "Повільна сумна мелодія для фортепіано" \
    --length-seconds 90 --output samples/f1_uk.mid
```

`--config f1` loads `config/generate/f1.toml`, the settings chosen by a decoding search: guidance 2.5, temperature 1.1,
instruments sampled by the model, a repetition penalty on pitch-class chords (DRY, multiplier 3), no end before 85% of
the requested length, at most 2,048 notes. Any flag on the command line overrides the preset, e.g. `--cfg-scale 2`,
`--best-of 4` (keep the candidate that best matches the caption) or `--num-samples 4`. `--retrieve
gigamidi_replay_train` starts from a real 128-note opening that matches the caption (needs the project's GigaMIDI
store); this scored higher in the decoding search (see Evaluation). `generate.py` also renders an MP3 unless `--no-mp3`
is given (needs fluidsynth, ffmpeg and a General MIDI soundfont).

What the captions were trained on: instruments, tempo / pace, loudness, register, key, length, genre and mood words,
and artist names from GigaMIDI metadata. Short, concrete prompts like the ones above work best.

Loading in Python:

```python
import torch
from musicar.model.checkpoint import load_model
model, _ = load_model("checkpoints/f1_cap/model_bf16.pt", device="cuda", dtype=torch.bfloat16)   # eval mode
```

Sampling with the KV cache, captions, section captions and guidance is `musicar.generation.sampling.generate`; the
caption's token states come from `musicar.text.encoder`.

## Intended use

- Research on text-conditioned symbolic music generation: caption following, guidance, multilingual prompts, section
  captions, length control.
- Sketching multi-instrument MIDI from a short description, to be edited in a DAW.
- A base for further captioned fine-tunes (the project's next one, F2, is on about 5,400 Ukrainian songs transcribed
  from audio).

Out of scope:

- Commercial use (see License).
- Audio: the model writes MIDI notes, not sound; the sound depends on the synthesizer used.
- Lyrics and vocals: the model has no notion of words or singing.
- Imitating a specific living artist for release.

## Training data

| Tier | Sampling weight | What it is | Caption source |
|---|---|---|---|
| GigaMIDI, labeled | 0.55 | GigaMIDI training files with a human or dataset label | [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) captions (168k), [XMIDI](https://github.com/xmusic-project/XMIDI_Dataset) emotion / genre labels (108k), GigaMIDI metadata (genre, artist) |
| Aria-MIDI, labeled | 0.15 | Piano transcriptions with genre / composer metadata | Aria-MIDI metadata |
| GigaMIDI, unlabeled | 0.30 | The other GigaMIDI training files | Words computed from the notes (instruments, pace, loudness, register, key, length) |

Labels were matched to the files by an exact note hash. Captions are built per training window from hand-written
English and Ukrainian vocabularies (with gender agreement in Ukrainian) and the computed features; 30% of the captions
are Ukrainian (templates and MidiCaps translations repaired by rules and a Hunspell filter). 15% of the windows get no
caption, 20% of the sections no section caption, so the unconditional model stays usable for guidance. Half of the
section-captioned windows are spliced from two files, so the section caption has to be read to predict the change.
Discover MIDI, which musicar-L was pretrained on, is not in this stage. None of the project's Ukrainian songs were used.

## Training procedure

| | |
|---|---|
| Initialization | musicar-L, then 1,500 iterations of plain captions (no sections, no length) at 512 notes with half of the windows at piece starts |
| Long run | 31,098 iterations × 65,536 notes (32 windows × 2,048) ≈ 2.04B notes; half of the windows start at a piece start, a quarter end at a piece end |
| Caption phase | 600 iterations × 65,536 notes (128 windows × 512) at piece openings, to restore the caption reading the long run wore down |
| Objective | Sum of the five cross-entropies, teacher forcing; no loss on the caption and section vectors |
| Optimizer | 8-bit AdamW, weight decay 0.1, gradient clipping 1.0; the caption adapters and length tables at 10× the learning rate |
| Learning rate | 10⁻⁴, WSD: 300 warmup steps, constant, 1 − √ decay to 10⁻⁵ over the last 20% (caption phase: 100 warmup, decay over the last 30%) |
| Precision | bf16 autocast with fp8 (torchao float8) matmuls in the transformer blocks, torch.compile |
| Hardware, time | One rented RTX 5090, TODO h (long run + caption phase) |
| Config | `config/f1/f1.py`, then `config/f1/f1_cap.py` (launched through `cloud/run/launch.sh f1`) |

The design choices (prefix adapter, centring, section recipe, length input) were settled in 107M pilots; see
[docs/experiments.md](https://github.com/DmytroHilei/Music_generative_model/blob/transformer-v2/docs/experiments.md),
section 10.

## Evaluation

One run; no listening study yet.

| Measure | Before the caption phase | Released weights |
|---|---|---|
| Held-out captioned GigaMIDI, cross-entropy (nats / note, 2,048-note windows) | 1.882 | 1.898 |
| Clean GigaMIDI validation (musicar-L's set; musicar-L: 1.679) | TODO | TODO |
| Caption following F (`text_eval`, guidance 2; higher is better) | 0.82 | **1.03** |
| Caption gain over the first notes of a piece (CE with another file's caption − with its own) | TODO | TODO |
| Length following (`eval/compare/length_follow.py`, 60 s / 120 s requests) | TODO | TODO |

The held-out CE is measured with captions, so it is not comparable to musicar-L's number on its own validation set.

Decoding search (`tools/sweeps/gen_search.py`), score J = F − 0.25 Q, where F is caption following and Q the distance
to real songs plus penalties for runaway density and copying: the `f1` preset (guidance 2.5, temperature 1.1, DRY 3)
scored J 0.786, the next settings 0.709 and 0.560. The weights before the caption phase scored 0.295 and failed the
search's gates. Starting from a retrieved real opening scored 0.992 vs 0.805 from the start token.

## Limitations and biases

- **The caption acts mostly at the start.** In the pilots the caption's effect on prediction was large over the first
  notes and almost gone after ~128: afterwards the model follows the music it has written. The long run weakened the
  caption reading, which is why the caption phase exists.
- **Not every word is followed.** In the 107M pilots drums and key mode (major / minor) were not followed reliably;
  instruments, tempo and loudness were. TODO: per-attribute results for these weights.
- **Section captions** were tested in the 107M pilots (switch score 0.98), not measured on these weights yet, and the
  command line does not expose them.
- **Length:** without the end floor, about 1 in 6 pieces asked for 120 s ended at about 45 s; the `f1` preset forbids
  an end before 85% of the requested length.
- **Noisy captions:** MidiCaps genre and mood come from automatic audio taggers; GigaMIDI is web-collected and varies in
  quality; Aria-MIDI is transcribed from recordings and contains transcription errors.
- The data lean toward Western popular and classical music in General MIDI instrumentation; Ukrainian prompts are
  understood, but the music is not specifically Ukrainian.
- Only notes are modelled: no tempo map, bars, pedal or controllers; time on a 20 ms grid, 32 velocity levels.
- Generated output may reproduce passages from the training data, more so when an artist is named.

## Files

| File | What it is |
|---|---|
| `model_bf16.pt` | bf16 weights + `model_args` + the training config; what the repository's scripts load (`generate.py --checkpoint`, `--config f1`) |
| `model.safetensors` + `config.json` | The same weights without pickle; `config.json` holds `model_args` |
| `diagram.png` | The architecture diagram above |

## License

CC BY-NC-SA 4.0. The training data are under non-commercial terms (Aria-MIDI CC BY-NC-SA 4.0, GigaMIDI
non-commercial; MidiCaps captions CC BY-SA 4.0). XMIDI states no license and is meant for research use; only its
emotion and genre labels were used, as caption words, and none of its files are redistributed. LaBSE (Apache 2.0) is
not included in this repository. Check the datasets' own terms before any reuse.

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

Please also cite GigaMIDI ([arXiv:2502.17726](https://arxiv.org/abs/2502.17726)), Aria-MIDI
([arXiv:2504.15071](https://arxiv.org/abs/2504.15071)), MidiCaps ([arXiv:2406.02255](https://arxiv.org/abs/2406.02255)),
XMIDI ([GitHub](https://github.com/xmusic-project/XMIDI_Dataset)) and LaBSE ([arXiv:2007.01852](https://arxiv.org/abs/2007.01852)).
