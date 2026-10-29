# Listen: musicar text-to-MIDI samples

Pieces written by musicar from a text prompt alone. Each one starts from an empty sequence: no notes are borrowed
from existing music. The MP3s are cut at 75 seconds with a fade; the MIDI files are complete.

| prompt | language | listen | MIDI | instruments in the result |
|---|---|---|---|---|
| A slow, quiet solo piano piece in a minor key | English | [1](slow_piano_1.mp3) · [2](slow_piano_2.mp3) | [1](slow_piano_1.mid) · [2](slow_piano_2.mid) | piano |
| A fast, energetic rock song with drums, electric guitar and bass | English | [1](fast_rock_1.mp3) · [2](fast_rock_2.mp3) | [1](fast_rock_1.mid) · [2](fast_rock_2.mid) | guitars, bass, drums (+ sax in 1) |
| A calm acoustic guitar piece without drums, in a high register | English | [1](guitar_nodrums_1.mp3) · [2](guitar_nodrums_2.mp3) | [1](guitar_nodrums_1.mid) · [2](guitar_nodrums_2.mid) | guitar, no drums |
| A lively jazz piece with piano, double bass and drums | English | [1](jazz_band_1.mp3) · [2](jazz_band_2.mp3) | [1](jazz_band_1.mid) · [2](jazz_band_2.mid) | piano, acoustic bass, drums |
| Швидка весела пісня з барабанами та електрогітарою | Ukrainian | [1](uk_fast_drums_1.mp3) | [1](uk_fast_drums_1.mid) | drums, bass, guitar and a crowded mix |
| Повільна сумна мелодія для фортепіано | Ukrainian | [1](uk_slow_piano_1.mp3) · [2](uk_slow_piano_2.mp3) | [1](uk_slow_piano_1.mid) · [2](uk_slow_piano_2.mid) | piano (+ flute in 2) |

## How they were made

- Model: the 477M caption checkpoint after the first caption stage (block 512 on piece openings,
  `f1_plain512_long`), the starting point of the F1 run. It is not the released captioned model, which is
  [musicar-L-caption](https://huggingface.co/Dimitri-206265/musicar-L-caption) (F1 and its caption phase,
  `f1_cap`). The prompt goes through a frozen LaBSE encoder into 16 prefix vectors.
- Sampling: classifier-free guidance 2, temperature and top-p as in `generate.py`, a repetition penalty on chords,
  up to 1,000 notes (a piece may end earlier). Each file is one of the best 2 of 8 candidates by an automatic
  caption-match score; one sample was left out after listening; no hand editing.
- Rendering: FluidSynth with the FluidR3 General MIDI soundfont.

Licence: CC BY-NC-SA 4.0, as the models. Code and model cards: the repository root and
[huggingface.co/Dimitri-206265](https://huggingface.co/Dimitri-206265).
