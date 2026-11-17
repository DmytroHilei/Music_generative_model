# musicar: compute request

**An open multi-instrument MIDI model, trained by one student on one rented consumer GPU, that matches open models of
up to twice its size. The request: compute to train it at 1B and 1.8B parameters and publish how it scales.**

Dmytro Hilei, bachelor student · [code](https://github.com/DmytroHilei/Music_generative_model) ·
[weights](https://huggingface.co/Dimitri-206265) · [demo](https://musicar-ivory.vercel.app) ·
[samples](https://musicar-ivory.vercel.app/samples) · [listen side by side](https://musicar-ivory.vercel.app/compare)

## What exists

- **musicar-L**, 464M parameters: one token per note, 129 instruments, context 2,048 notes. Trained from scratch on
  8.3B notes in 41 hours on one rented RTX 5090 (about €30 of rental).
- **musicar-L-caption**: the same model with text input, English and Ukrainian.
- Weights, code, model cards and every result below are public, including the negative ones.

## Evidence

| Claim | Number | Against |
|---|---|---|
| Continues real multi-instrument music more faithfully (CLaMP 3 retrieval of the true continuation, 96 prompts, 1,000 new notes) | MRR **0.811** | AMT large (780M) 0.640, SkyTNT medium (234M) 0.666; ahead of both in all 10,000 paired draws |
| Follows short captions, in two languages (caption following F) | English **1.07**, Ukrainian **0.95** | MIDI-LLM (1.7B) 0.60 / 0.09, Text2midi 0.54 / 0.06 |
| Training outcome was forecast before the run | predicted 5.47–5.72, measured **5.706** (107M model) | |
| Runs on small hardware | 127 notes/s on an 8 GB laptop GPU, 1.1 GiB of memory, about 5× real time | |

All comparisons use the same prompts for every system, each with its authors' sampling settings
([tables and method](../../README.md#comparison-with-other-open-models)). Two limits of these results: on our own
density statistic AMT large and musicar-L are too close to rank, and on solo piano Aria medium is level or ahead on
CLaMP 3.

## What it cannot do yet

On 495 long MidiCaps test captions musicar-L-caption is last of three systems: exact key 0.042 (the shuffled floor is
0.051) against 0.317 for MIDI-LLM. Its frozen LaBSE encoder and 13M adapter read short captions and miss key and tempo
in long ones. The larger runs unfreeze the top encoder layers and train the caption stage on more varied
captions; this table is the target they have to move.

## Scaling

<!-- scale-forecast -->
- Two earlier forecasts held on piano data: a power-law fit on 6M–42M models predicted the 65M one at 7.42
  (measured 7.399), and the 107M run was predicted at 5.47–5.72 (measured 5.706).
- Equal-compute runs put the best size at 16–20 notes per parameter, the ratio the XL and XXL rows below use.
- No loss is promised for 1B. A ladder of short runs (6M–108M parameters, 40M–160M notes, musicar-L's recipe) was
  fitted to forecast it and failed its own check: it predicts 2.23 nats per note for musicar-L, which measures 1.641
  on the same held-out GigaMIDI windows ([table and fit](scaling/fit.md)). Runs this short do not extrapolate to a
  full one, so the 1B run is the measurement.
<!-- /scale-forecast -->

## The request

| Run | Parameters | Notes seen | H100 hours (estimate) |
|---|---|---|---|
| XL | 1B | 20B | 150–200 |
| XXL | 1.8B | 35B | 500–650 |
| Ablations at XL (text encoder, context 8,192) | | | 150–200 |

The hours are scaled from the measured 464M run (2.3·10¹⁹ FLOP in 41 h on an RTX 5090, FP8 + `torch.compile`) and
assume an H100 is 1.0–1.3× that GPU; a one-hour benchmark on the granted hardware replaces the assumption. 1.8B and
not 2.5B because the open MIDI corpora hold about 8–9B unique notes.

## What comes back

- Weights, code and training logs of every run under the current licence (CC BY-NC-SA 4.0), on Hugging Face.
- The forecast against the measured loss, published whether it holds or not.
- The head-to-head tables rerun with the new models, and a blind listening test
  ([the test page](https://musicar-ivory.vercel.app/compare/index.html?test)).
- The provider named in the model cards and the write-up.
