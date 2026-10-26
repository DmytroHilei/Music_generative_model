# Listen: musicar text-to-MIDI samples

Pieces written by musicar from a text prompt alone. Each one starts from an empty sequence: no notes are borrowed
from existing music. Audio (MP3), video and MIDI for each prompt are added here as they are rendered.

| prompt | language | what it contrasts |
|---|---|---|
| A slow, quiet solo piano piece in a minor key | English | slow, solo piano |
| A fast, energetic rock song with drums, electric guitar and bass | English | fast, full band |
| A calm acoustic guitar piece without drums, in a high register | English | no drums, high register |
| A lively jazz piece with piano, double bass and drums | English | jazz trio |
| Швидка весела пісня з барабанами та електрогітарою | Ukrainian | fast, drums and guitar |
| Повільна сумна мелодія для фортепіано | Ukrainian | slow, sad piano |

## How they were made

- Model: the 477M caption checkpoint after the first caption stage (block 512 on piece openings), a research
  checkpoint on the way to the captioned musicar-L. The prompt goes through a frozen LaBSE encoder into 16 prefix
  vectors.
- Sampling: classifier-free guidance 2, temperature and top-p as in `generate.py`, a repetition penalty on chords,
  1,000 notes. Each file is the best of 4 candidates by an automatic caption-match score; no hand editing.
- Rendering: FluidSynth with the FluidR3 General MIDI soundfont.

Licence: CC BY-NC-SA 4.0, as the models. Code and model cards: the repository root and
[huggingface.co/Dimitri-206265](https://huggingface.co/Dimitri-206265).
