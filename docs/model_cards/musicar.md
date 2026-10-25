---
license: cc-by-nc-sa-4.0
tags:
- music
- midi
- symbolic-music
- music-generation
- pytorch
---

# musicar: MIDI generation transformers

Autoregressive transformers over symbolic music (MIDI), written one note at a time: each sequence position is one
whole note (time shift, instrument, pitch, duration, velocity on a 20 ms grid), predicted by cascade output heads
that each see the values already chosen. Three pretrained models, one per folder; none of them has text captions yet
(a captioned fine-tune is planned, see Future work). The code is at
https://github.com/DmytroHilei/Music_generative_model.

| Folder | Parameters | What it is | Context | Validation (nats / note) | Card |
|---|---|---|---|---|---|
| `L/` | 464M | multi-instrument (129 instruments), window conditions (instrument set, notes per second), BOS / EOS; the base of the planned captioned model | 2,048 notes | clean GigaMIDI 1.679 | [card](https://huggingface.co/Dimitri-206265/musicar/blob/main/L/README.md) |
| `S/` | 108M | the piano model grown to 129 instruments and RoPE; the base of the 107M caption pilots | 2,048 notes | GigaMIDI 2.232 (original split) | [card](https://huggingface.co/Dimitri-206265/musicar/blob/main/S/README.md) |
| `S-piano/` | 107M | solo piano, learned positions; the earlier base | 512 notes | Aria-MIDI 5.706 | [card](https://huggingface.co/Dimitri-206265/musicar/blob/main/S-piano/README.md) |

The three validation sets differ, so the numbers are not comparable across rows; each card states its metric.

![The three pretrained models](https://huggingface.co/Dimitri-206265/musicar/resolve/main/diagram.png)

## How to use

From a clone of the code repository, with its environment installed (its README, "Setup"). Download one folder:

```bash
hf download Dimitri-206265/musicar --include "L/*" --local-dir checkpoints/musicar
.venv/bin/python generate.py --checkpoint checkpoints/musicar/L/model_bf16.pt --sample-instruments \
    --max-new-tokens 1000 --output samples/musicar-450m.mid
```

`--sample-instruments` lets the instrument head choose each note's instrument (multi-instrument models; without it every
note is piano); `--prompt file.mid` continues a MIDI file; `--temperature`, `--top-k`, `--top-p` set the sampling. The
pickle-free files load without the scripts:

```python
import json
from safetensors.torch import load_file
from musicar.model.gpt import GPT, MusicConfig

cfg = json.load(open("checkpoints/musicar/L/config.json"))
model = GPT(MusicConfig(**cfg["model_args"]))
model.load_state_dict(load_file("checkpoints/musicar/L/model.safetensors"))
model.eval()
```

Each folder holds `model_bf16.pt` (the training checkpoint: bf16 weights + `model_args` + training config, what the
repository's scripts load), `model.safetensors` + `config.json` (the same weights without pickle), `diagram.png` (the
model's architecture) and `README.md` (its full card: details, training data and procedure, evaluation, limitations).

## Intended use

- Research on symbolic music generation: multi-instrument modelling, sampling, conditioning, model growth.
- Continuing a MIDI prompt, or generating from the start token (450M).
- Bases for fine-tuning: the project's fine-tunes on Ukrainian songs start from the 450M, the caption pilots from
  musicar-S.

Out of scope: commercial use (see License), text-conditioned generation (no caption input in these checkpoints), audio.

## Training data

| Model | Data | Notes seen |
|---|---|---|
| L | GigaMIDI 0.4, Aria-MIDI 0.3, Discover MIDI 0.3 (sampling weights) | 8.3B |
| S | GigaMIDI 0.7, Aria-MIDI 0.3, from the piano model's weights | 197M |
| S-piano | MAESTRO, GiantMIDI-Piano, Aria-MIDI | 2.33B |

All of it is under non-commercial terms (Aria-MIDI and Discover MIDI CC BY-NC-SA, MAESTRO CC BY-NC-SA, GiantMIDI-Piano
non-commercial, GigaMIDI CC BY-NC per the project's notes; check GigaMIDI's own terms). None of the project's
audio-derived transcriptions of Ukrainian songs were used.

## Limitations and biases

- The data lean toward Western popular and classical music in General MIDI instrumentation.
- Aria-MIDI is transcribed from recordings and carries transcription errors; GigaMIDI is web MIDI of varying quality.
- Control is limited to the prompt (and, for the 450M, the two window conditions): no text input, no bars or beats.
- Sampling can run away into very dense passages; generated output may reproduce passages from the training data.

Each card lists the model's own limitations.

## Future work

Planned, not done: a captioned fine-tune of the 450M (F1: captions from a frozen multilingual LaBSE encoder in English
and Ukrainian, section captions that change during a piece, length control; the recipe is being settled in laptop
pilots), then F2, a captioned fine-tune on about 5,400 Ukrainian songs transcribed from audio; later infilling and
small-device inference (int8 / int4 on a Raspberry Pi 5). The S models are not trained further.

## License

CC BY-NC-SA 4.0 for all three: the training data are under non-commercial terms, so the weights are released under
the same non-commercial, share-alike terms.

## Citation

```bibtex
@misc{hilei2026musicar,
  author       = {Dmytro Hilei},
  title        = {musicar: autoregressive symbolic music models},
  year         = {2026},
  howpublished = {\url{https://github.com/DmytroHilei/Music_generative_model}}
}
```

Please also cite the training datasets: GigaMIDI ([arXiv:2502.17726](https://arxiv.org/abs/2502.17726)), Aria-MIDI
([arXiv:2504.15071](https://arxiv.org/abs/2504.15071)), Discover MIDI
([Hugging Face](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)), MAESTRO and GiantMIDI-Piano.
