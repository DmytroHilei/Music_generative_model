# Pre-registered predictions: data-constrained scaling runs (RTX 5090, from 2026-11-21)

Committed 2026-11-21 ~02:11 UTC, before the results of the rows below were known, except the rows under "Already
visible". The source is the scaling plan §8.1. The decision rules are in plan §8.2–8.6. Units: GigaMIDI held-out CE,
nats/note, 4 heads summed. P = repeat penalty = control − fresh at 160M notes seen (U = 41.48M for the replay quarter,
3.86 passes).

## Already visible at commit time (exploratory, not confirmatory)
These ran in P0 before this file was committed. Their predictions are listed for the record only.

| run | prediction | measured (2 seeds unless noted) |
|---|---|---|
| P_6M at q (A-S1) | R* 2.5/4.1/8/15 → 0.207/0.135/0.073/0.040; (a0) 0.13, (P2/b) 0.071, (a1) ≲ 0.05 | control 2.934/2.943, fresh 2.845/2.868 → P = 0.082 |
| 42M f0.5 (A-U6) | Chai-tied / Muennighoff R_D* 4.32 / power ω 0.279: 0.073 / 0.030 / 0.025 | control 2.464/2.470 vs fresh 2.433 → P = 0.034 |
| 42M f0.125 (A-U6) | 0.221 / 0.336 / 0.391 | control 2.942/2.940 → P = 0.508 (above every form) |
| P_20M at q (A-S3) | (a0 / P2-b / a1) 0.13 / 0.104 / ~0.07 | control 2.656/2.658, fresh 2.531 (seed 1337 only) → P ≈ 0.126 |

## Confirmatory predictions (results not yet run)
- **Penalty by size at q** (a0 / P2-b / a1):
  - 3M: 0.13 / 0.054 / ≲ 0.03;
  - 217M: ~0.16 / 0.226 / control_217 ≈ control_108.
- **6M repetition axis vs census fresh** (fresh slope 0.40 per doubling):

  | notes (R) | a0 R_D* 4.6 / 8 / 15 | P4 δ=1 | P4 δ=1.675 | Chai & Xiong |
  |---|---|---|---|---|
  | 80M (0.93) | .074 / .045 / .025 | .023 | .011 | .049 |
  | 160M (2.86) | .162 / .101 / .056 | .071 | .071 | .071 |
  | 320M (6.7) | .332 / .212 / .121 | .167 | .297 | .094 |
  | 640M (14.4) | .614 / .414 / .246 | .358 | 1.07 | .121 |

- **Iso-FLOP:** t < 0 at 3e16 (μP direction); raw r* biased up by +4–36% at 3e16 and +1–11% at 3e17 before the t0
  correction.
- **Items 1–2 (regularization):**
  - fresh is best near λ 0.2–0.4 (gain 0–0.03); control is best at λ ≥ 1;
  - Lovelace's P factors: ×0.61–0.94 at λ 0.3, ×0.19–0.61 at λ 1.0, ≤ ×0.2 at λ 3.0;
  - embwd0 loses ~0.02;
  - the fresh gain at lr 2e-3 is positive but < 0.08.
- **Synthetic data:** value per synthetic note v stays ~0.57 while the student is smaller than the teacher (musicar-L, 464M).
- **Discover / Aria:** no prediction.

## Primary families (plan §8.0)
- frontier: P at 3M/6M/20M/217M, the shape q, ω;
- iso: r* per budget and the drift;
- items 1–2: ΔP_BEST at 42M (f0.25, f0.125) and 108M, and the N-gain under BEST;
- items 3–4: g4, φ, sbp25 − synth25, synth_clean25 − control;
- Discover: ρ.

Everything else is exploratory.
Effect threshold: |Δ| ≥ max(0.012, 2·SE_pair) at 42M, 0.017 at 108M.
