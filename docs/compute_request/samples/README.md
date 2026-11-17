# Samples

Fourteen clips, one or two per style, rendered with the FluidR3 General MIDI soundfont.

**How they were picked.** CLaMP 3 (a music–text retrieval model) chose them, not listening. For each style the clip
is the top-scoring file among the candidates: the closest to the real continuation for `continuation/`, the closest to
the caption for `captions/`. So these are the model's best cases per style, not a random draw. A random draw of
continuations next to other systems is on the [side-by-side page](https://musicar-ivory.vercel.app/compare), with a
blind test at [`?test`](https://musicar-ivory.vercel.app/compare/index.html?test).

## Continuation: musicar-L

Each clip is the last few seconds of a real opening followed by 30 s that musicar-L wrote. The prompts come from the
head-to-head sets (96 multi-instrument and 96 solo-piano openings). "Rank" is where CLaMP 3 places musicar-L's
continuation among the 96 real continuations, when asked which one is closest to the true one. "Lead" is how much
closer it is than the best other system on the same prompt: AMT large or SkyTNT medium for multi-instrument, Aria
medium for piano.

| Clip | Style | Instruments of the prompt | Opening + continuation | Rank of 96 | Lead |
|---|---|---|---|---|---|
| [rock_cont96_p64.mp3](continuation/rock_cont96_p64.mp3) | rock | Drawbar Organ, Overdriven and Distortion Guitar, bass, strings | 3 s + 30 s | 1 | +0.12 |
| [jazz_cont96_p76.mp3](continuation/jazz_cont96_p76.mp3) | jazz | piano, bass, trumpet, alto sax, drums | 8 s + 30 s | 1 | +0.13 |
| [acoustic_cont96_p89.mp3](continuation/acoustic_cont96_p89.mp3) | acoustic | piano, steel-string guitar, bass, flute, drums | 7 s + 30 s | 1 | +0.09 |
| [cinematic_cont96_p55.mp3](continuation/cinematic_cont96_p55.mp3) | cinematic | bells, harp, timpani, synth strings, choir | 8 s + 30 s | 2 | +0.41 |
| [chamber_cont96_p58.mp3](continuation/chamber_cont96_p58.mp3) | chamber | viola, cello, flute | 8 s + 30 s | 1 | +0.26 |
| [piano_piano96_p36.mp3](continuation/piano_piano96_p36.mp3) | piano | solo piano | 8 s + 30 s | 1 | +0.20 |
| [piano_piano96_p19.mp3](continuation/piano_piano96_p19.mp3) | piano | solo piano | 8 s + 30 s | 1 | +0.20 |

The openings are short excerpts (synthesised from MIDI) of existing pieces from the GigaMIDI and Aria-MIDI research
datasets, here only to show the model; the rights to the music stay with its authors. To have an excerpt removed, open
an issue in the [repository](https://github.com/DmytroHilei/Music_generative_model/issues).

## Captions: musicar-L-caption

Music from the text alone, with no opening. The settings are the released ones (`generate.py --config f1`): 8
candidates per caption, 60 s each, and the first 45 s are kept here. "Score" is the CLaMP 3 cosine between the music
and its caption.

| Clip | Caption | Language | Instruments the model chose | Score |
|---|---|---|---|---|
| [piano_en.mp3](captions/piano_en.mp3) | a gentle romantic solo piano piece | English | piano | 0.28 |
| [acoustic_guitar_en.mp3](captions/acoustic_guitar_en.mp3) | a calm acoustic guitar piece without drums | English | steel-string guitar, flute, bass | 0.27 |
| [jazz_en.mp3](captions/jazz_en.mp3) | a lively jazz piece with piano, double bass and drums | English | piano, double bass, drums | 0.25 |
| [jazz_uk.mp3](captions/jazz_uk.mp3) | спокійний джаз для фортепіано, контрабаса і барабанів (calm jazz for piano, double bass and drums) | Ukrainian | piano, drums, double bass, steel-string guitar | 0.26 |
| [electronic_en.mp3](captions/electronic_en.mp3) | an upbeat electronic dance track with synthesizers and drums | English | synth pads, square lead, drums | 0.22 |
| [rock_en.mp3](captions/rock_en.mp3) | a fast, energetic rock song with drums, electric guitar and bass | English | overdriven guitar, drums, electric bass | 0.12 |
| [strings_en.mp3](captions/strings_en.mp3) | a slow, sad string quartet | English | string ensemble, flute, synth brass | 0.17 |

The string quartet clip is the best of 32 candidates, not 8. It is one of four captions that came out weak in the
first 8 and were given 24 more. Three of them are still left out, because none of their 32 candidates matched the
caption clearly:

- an epic orchestral film score (best score 0.12, with strings but no brass or timpani)
- a Ukrainian folk song with accordion and violin (best score below 0; the model never produced an accordion)
- a slow, sad Ukrainian piano melody (best score 0.07, with drums and bass added)

The weakness on long, detailed captions is described in the [pitch](../pitch.md).
