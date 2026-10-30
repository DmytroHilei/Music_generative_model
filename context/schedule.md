# Schedule

What comes next, in order, with time estimates. Written for a human reader first; agents find the technical
details in `context/claude/agents.md` (item 15h for text control, "Inference + private demo", "Audio rendering").
Update the status column when a step is done.

Times: **work** = focused working days (writing code, checking results); **machine** = unattended compute.
Calendar weeks are targets, not commitments. Last update: 2026-10-30 (F1 + musicar-L-caption).
Directions after F1: `context/claude/future_directions_2026-10-30.md`.

## Overview

| # | Project | Work | Machine | Target |
|---|---|---|---|---|
| 1 | **Text control**: generate from a text prompt ("sad slow piano ballad"), fully offline | ~12-14 days (~11 left after Oct 15) | ~1 laptop night + ~16-25 h rented 5090 (~€7-10; +€2-3 if step 8) | weeks of Oct 15 - Nov 4; step 4 done 2026-10-28 |
| 2 | **Private demo** on the Raspberry Pi: chat-like page (browser → Pi → MIDI/MP3) | see `context/claude/demo.md` + ~3 days for the chat features below | — | after 1; the chat page runs today on a Vercel page + laptop engine (invite / GitHub sign-in), the Pi later |
| 3 | **Audio rendering** (stem render → codec-token renderer, ladder in `context/claude/future_directions_2026-10-30.md`) | not estimated yet | laptop nights for the first tests | after 2 |

## 1. Text control

Goal: type a prompt in English or Ukrainian and get music that matches it, with no internet and no paid API.
How: a small frozen text model turns the prompt into a vector that is added to the music model's input at every note;
the music model learns to follow it. The labels it learns from come from public captioned datasets plus labels we make
ourselves for the rest of the pretraining data.

| Phase | Step | Work | Machine | Status |
|---|---|---|---|---|
| 0 | Check disk space, check the local LLM | 0.5 day | — | ✅ 2026-10-15 |
| 1.1 | Download MidiCaps (captions) and XMIDI (emotion + genre labels), check licenses | 0.5 day | ~5 min | ✅ 2026-10-15 |
| 1.2 | Match the labels to songs already in our token stores (by file checksum) | 0.5 day | minutes | ✅ 2026-10-15 (local stores; full stores on the 5090) |
| 1.3 | Compute simple per-song features (key, tempo, density, register, instruments) | 0.5 day | minutes | ✅ 2026-10-15 (local stores; ~1.5M songs take a few minutes on the laptop CPU) |
| 1.4 | **Probe**: does our model's internal representation know genre / emotion / artist? | 0.5 day | minutes | ✅ 2026-10-15 passed: +11-18 points over plain statistics |
| 1.5 | Embed all songs (on the rented 5090), label the rest with classifiers trained on the good labels, keep only confident labels | 0.5 day | ~2-3 h on the 5090 (~€1) | ✅ 2026-10-16 local stores (Aria overnight); full stores on the 5090 |
| 1.6 | Caption builder: templates + Qwen paraphrases, one caption per training window | 1-1.5 days | ~1 h | ◐ 2026-10-18: Qwen paraphrases + Ukrainian labels replaced by hand-written EN + UK templates; MidiCaps UK translations fixed by rules (86% kept); waits for the user's review |
| 2 | Model code: text encoder + adapter, training flags, text prompts in generation, tests | 2.5 days | — | ✅ 2026-10-17 (F1 trained on the EN + UK captions, 30% Ukrainian) |
| 2b | Model code for the chat demo: pieces that end at a requested length, and regenerating a part ("change the ending") | 1.5 days | — | ◐ length trained in F1 (step 4); part regeneration open (a later fine-tune) |
| 3 | Pilot on the 107M model: does it follow captions? frozen vs trainable model; then length + part-regeneration check | 1.5 days | one evening + ~1 h | ✅ pilots finished; P1 2026-10-17: trainable model wins, following weak; P2 2026-10-18 (centered caption vectors, more piece starts): instruments + tempo + loudness follow with guidance; T2 prefix 2026-10-18 chosen; with the new EN + UK captions Ukrainian prompts follow too (piano, tempo); key, drums not yet |
| 4 | Train the 450M with text (+ length/ending + part regeneration) on the rented RTX 5090, 2-3 passes over the good labels | 0.5 day | 12-20 h (+2 h setup), ~€5-8 | ✅ 2026-10-28: F1 (31,098 steps, ~14 h) + caption phase f1_cap (600 steps), released as musicar-L-caption; length trained, part regeneration not |
| 5 | Redo the Ukrainian fine-tunes (piano, multi-track) on top, with captions | 1.5 days | ~2-3 h | ☐ |
| 6 | Evaluation and listening (English + Ukrainian prompts) | 1 day | ~1 h | ☐ |
| 7 | Text model on the Pi (joins the demo) | 0.5 day | — | ☐ |
| 8 | Only if the Pi is too slow: a smaller, faster copy of the model trained to imitate the 450M | 1 day | 3-6 h rented 5090 (~€2-3) | ☐ optional |

Extra checks done 2026-10-15: training, fine-tuning and generation still work after the model.py incident; the text
model was chosen (LaBSE: Ukrainian prompts match their English translations 20/20, the smaller candidates 10-12/20).
The exact technical pipeline is in `context/claude/text_control.md`.

Rough calendar: week of Oct 15 = phases 0-1; week of Oct 22 = phases 2-3; week of Oct 29 - Nov 5 = phases 4-6.

### Chat-style demo (decided 2026-10-15)

The demo is a chat: type "a sad slow piano ballad", get a piece; then "make it sadder", "add drums", "change the
ending". What that needs:

| What | Where | Effort | Status |
|---|---|---|---|
| Pieces end at a requested length (today the model rarely ends a piece by itself) | trained in step 4 | in step 2b | ✅ 2026-10-28 (F1) |
| Regenerate a part ("new ending", "redo bars 9-16") | trained in step 4 | in step 2b | ☐ a later fine-tune |
| 3-4 versions generated at once (nearly free on the Pi), the best one kept by a small classifier, or shown to pick from | demo app | 0.5 day | ✅ best-of N in the app (kept by caption match) |
| Instant edits without the model: faster / slower, higher / lower, louder / softer, swap or remove an instrument, loop / cut | demo app | 1 day | ☐ |
| Understanding messages in English and Ukrainian with the same text model (no chatbot LLM needed) | demo app | 1-1.5 days | ☐ |
| Smaller, faster model if the Pi is too slow | step 8 | 1 day + €2-3 | ☐ optional |
| Add or regenerate a single instrument; change the mood halfway through a piece | later | — | ☐ later |

### More passes on the good labels

The ~270k songs with real labels (MidiCaps captions, XMIDI human labels, Ukrainian songs) are shown 2-3 times in the
450M run; the songs with only computed or predicted labels less than once. Each pass uses a different wording of the
caption, so repeats teach more than they overfit. The laptop pilot already does ~1.6 passes and shows whether more
passes still help; the 450M run keeps a snapshot after every pass. This makes step 4 longer (12-20 h instead of
2.5-5 h), still roughly €5-8.

### Decision points

- **1.4 probe.** If the model's embeddings don't separate genre / emotion / artist, skip the cluster labels; captions
  then come only from MidiCaps, XMIDI, metadata and the computed features. The plan still works, with fewer labeled songs.
- **3 pilot.** Decides whether the music model stays frozen (cheaper, keeps quality) or is trained along with the
  adapter, and how much data the 450M run needs.

### Datasets (checked 2026-10-15)

| Dataset | What | Size | License | Already in our data? |
|---|---|---|---|---|
| MidiCaps | 168k full-sentence captions (key, tempo, chords, instruments, genre, mood) | 2 GB download | CC-BY-SA 4.0 | 99.98% are GigaMIDI files |
| XMIDI | 108k files, 11 emotions + 6 genres, labeled by 10 professional annotators | 450 MB download | **none stated**: research use only, don't share, cite the paper | 96% are GigaMIDI files |

Both mostly label music the model has already seen, which is fine: what it learns now is the link between words and
music, not new music. Caveats: MidiCaps genre/mood come from automatic audio taggers (noisy; key, tempo, chords and
instruments are reliable); 1,699 XMIDI files appear twice with different labels.

### Resources

- **Disk:** ~10 GB free on /home, ~6.6 GB on /. Enough for phases 1-3. The full training data for phases 1.5 and 4 is
  on the private Hugging Face repo and is processed on the rented machine. If space runs out: `checkpoints/` holds
  42 GB, much of it superseded runs (decide what to delete), and there is an unused 711 GB Windows partition.
- **Local LLM:** Qwen3 8B through Ollama, fits the GPU (5.5 GB), ~65 tokens/s. Used once to write cluster
  descriptions and caption paraphrases, never at generation time.
- **No Anthropic API calls** in the pipeline or at runtime (works offline, no latency, no outside dependency).

### Risks

- The probe fails → fewer labels (see decision points).
- Captions too uniform → the model only understands template wording; Qwen paraphrases and the MidiCaps captions are
  there to prevent that, and the pilot tests it with free-form prompts.
- Ukrainian prompts: the text encoder is multilingual; 30% of the F1 training captions are Ukrainian. text_eval at
  guidance 2 on f1_cap: EN 1.07, UK 0.97.
