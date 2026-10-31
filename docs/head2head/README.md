# Head-to-head scores

Copies of the report files of [`eval/compare/head2head.py`](../../eval/compare/head2head.py),
[`h2h_midicaps.py`](../../eval/compare/h2h_midicaps.py) and [`h2h_clamp3.py`](../../eval/compare/h2h_clamp3.py), which
write to `results/head2head/` (not tracked, with the generated MIDI files). Each `.md` has a `.json` of the same name.

| Files | Run | What |
|---|---|---|
| `scores`, `scores_first200` | 2026-10-30 | the 32-prompt tables: continuation, piano, text, over 1,000 and 200 new notes |
| `bootstrap_cont`, `bootstrap_piano` | 2026-10-30 | paired bootstrap of D on the 32 prompts |
| `scores_96`, `scores_96_first200` | 2026-10-31 | 96 prompts: `cont96` with its groups (`no_drums`, `with_drums`, `published`, `new`), `piano96` |
| `bootstrap_cont96`, `bootstrap_piano96` | 2026-10-31 | paired bootstrap of D on the 96 prompts and the groups |
| `midicaps_scores`, `midicaps_clamp3` | 2026-10-31 | MidiCaps test captions: TB, TBT, CK, CKD, instrument scores, CLaMP 3 |
| `text_clamp3` | 2026-10-31 | CLaMP 3 on the 32-prompt run's text outputs |

Other checks of the MidiCaps run, on the same note-based estimators (not in these files): musicar-L-caption's tonic is right
in 8.1% of the files (chance 8.3%) and major / minor in 54%; MIDI-LLM 36% and 69%, Text2midi 38% and 67%, real files 62% and 81%.
Rank correlation of tempo with the asked tempo: 0.21, 0.07, 0.06, real files 0.34. The same preset on short prompts:
minor 54% of pieces for "a piece in a minor key" vs 12% for "a piece in a major key"; 72 vs 11 notes per second for fast vs slow
(`results/gen_search/f1`, b19 rows).
