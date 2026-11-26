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
| `cont96_clamp3_n1000`, `cont96_clamp3_n200`, `piano96_clamp3_n1000`, `piano96_clamp3_n200` | 2026-11-04 | CLaMP 3 continuation scores on the 96 prompts: closeness, contrast and MRR to the real continuation, per group, Fréchet |
| `midicaps_spare_scores`, `midicaps_spare_clamp3` | 2026-11-04 | F1 settings on 100 spare MidiCaps test captions after the 500: preset, top-p 0.9, keep-style + lock off, guidance 4.0, `--retrieve` |

Other checks of the MidiCaps run, on the same note-based estimators (not in these files): musicar-L-caption's tonic is right
in 8.1% of the files (chance 8.3%) and major / minor in 54%; MIDI-LLM 36% and 69%, Text2midi 38% and 67%, real files 62% and 81%.
Rank correlation of tempo with the asked tempo: 0.21, 0.07, 0.06, real files 0.34. The same preset on short prompts:
minor 54% of pieces for "a piece in a minor key" vs 12% for "a piece in a major key"; 72 vs 11 notes per second for fast vs slow
(`results/gen_search/f1`, b19 rows).

## Other benchmarks and metrics (survey 2026-11-25)

There is no standard benchmark suite or public leaderboard / arena for symbolic music generation: each paper picks its
own metrics. What exists, and where it stands here:

| Benchmark / metric | What it measures | Status here |
|---|---|---|
| Fréchet Music Distance, FMD ([arXiv 2412.07948](https://arxiv.org/abs/2412.07948), [pip](https://pypi.org/project/frechet-music-distance/)) | Fréchet distance between generated and real sets in CLaMP / CLaMP 2 embedding space (FID for symbolic music); used e.g. by [BEAT](https://arxiv.org/abs/2604.19532) | Covered in spirit: `h2h_clamp3.py` reports a Fréchet distance on CLaMP 3 features (CLaMP 2 weights skipped to save disk). The package with CLaMP 2 would give the number other papers quote |
| MusPy / mgeval descriptors (Yang & Lerch 2020) | Overlap / divergence of pitch-class, interval, rhythm, density histograms vs real data | Close to D (density + instrument statistics); not run as the published feature set |
| [smg-metrics](https://pypi.org/project/smg-metrics) | 25 objective metrics: harmony, rhythm, polyphony, distribution, structure | Not checked beyond the PyPI page |
| Text2midi / MidiCaps metric set | Text-to-MIDI: CLAP score, tempo bin (TB / TBT), key (CK / CKD), instrument match, compression ratio | Run: `midicaps_scores`, `midicaps_clamp3` |
| Groove consistency, scale consistency (MusPy; used in BEAT) | Rhythmic regularity, staying in key | Not run; would show a beat-position effect |
| [MusicConstraintBench](https://arxiv.org/abs/2609.23665) (2026 preprint) | 2,180 automatically verified multi-constraint generation prompts | Not run; possible caption-control check (seen at abstract level only) |
| Blind human preference (A/B listening) | Preference | Not run; thesis human-eval idea |

Not applicable (notation understanding / QA benchmarks for LLMs, not generators): [ABC-Eval](https://arxiv.org/abs/2509.23350),
[LilyBench](https://arxiv.org/abs/2606.08722), the Indian classical benchmark ([arXiv 2606.05522](https://arxiv.org/abs/2606.05522)),
[MIDI-LLaMA](https://arxiv.org/abs/2601.21740) captioning. [MuSpike](https://arxiv.org/abs/2508.19251) benchmarks spiking networks.

Held-out CE / perplexity does not compare across tokenizers (Aria, AMT, Text2midi): only between musicar models.

A release-style table would take the head-to-head rows (musicar-S / L / L-caption, Aria, AMT, Text2midi, SkyTNT) with
columns FMD (CLaMP 2) on GigaMIDI and Aria, CLaMP 3 continuation scores, D / descriptor overlap, MidiCaps metrics,
groove / scale consistency and later human preference, against a cost axis (parameters, notes/s on GPU / CPU / Pi,
training FLOPs).
