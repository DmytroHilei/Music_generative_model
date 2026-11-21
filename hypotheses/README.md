# Hypotheses

Experimental training changes, one file each, kept out of the main code until one earns its place. `train.py
--hypothesis=<name>[:key=value...]` imports `hypotheses/<name>.py` and calls its `Hypothesis` at five fixed points
(`hypotheses/__init__.py`: `dataset`, `model`, `optimizer`, `batch`, `forward`); the model, the loader and the rest
of train.py don't change for it, and the checkpoints stay those of the plain model (a hypothesis's own modules live on
the hypothesis object: no resume of such an arm).

Rules:
- The file's docstring says the idea, where it comes from, what result would confirm it and the command of the arm.
- Tests in `tests/hypotheses/` (CPU, tiny models) check the hooks' mechanics, as for the regular code.
- **Works** (the arm beats its control on the metric the docstring names): the change moves into the model / loader /
  train.py as a regular setting with its tests, the file is deleted, the log row in `context/claude/agents.md` says so.
- **Null or harmful**: the file is deleted; the result stays in the log row and the notes.
- Config-only arms (block size, boundary shares, caption dropout, open_weight) need no hypothesis file.
- The spec is split on ':' (so no ':' inside a value); values parse as Python literals, else strings.
- The CPU smoke of all four through train.py (synthetic store, 6 steps): the session's scratchpad script
  `smoke_hyp.py` of 2026-10-26; the CI's `train-cpu` case is the template for a permanent one.

| name | idea | hooks | metric | status |
|---|---|---|---|---|
| `open_corrupt` | the opening's input notes of captioned windows replaced at random (word dropout against a decoder ignoring its conditioning) | batch | early-notes text gain vs plain512 (+0.015 @600) | written 2026-10-26, untested |
| `open_mi` | InfoNCE: the opening's mean state must identify its caption among the batch's (pays attention to the prefix slots directly) | model, optimizer, forward | early-notes text gain; caption-slot attention (prefix_attention.py) | written 2026-10-26, untested |
| `open_cf` | counterfactual hinge: the own caption must beat a rolled one by eta over the opening (two cropped forwards) | forward | early-notes text gain; ocf_margin | written 2026-10-26, untested |
| `open_proxy` | proxy distillation: the opening target = softmax(the 450M's null-caption logits + the 107M caption model's caption shift), KL per head (three no-grad cropped forwards) | model, optimizer, forward | early-notes text gain; proxy_kl | written 2026-10-26, untested |
| `aug_dyn` | whole-window dynamics shift: every velocity of a window moves by one random offset in ±shift bins (no clamping), captioned windows skipped; for training that repeats the unique data | batch | late-run val CE on a ~4-epoch repeated-data run vs control (scale_ladder store) | written 2026-11-19, untested |
| `aug_octave` | single-track octave shift: one random pitched program of a multi-instrument window moves ±12 (input and target, in range), captioned windows skipped | batch | late-run val CE on the ~4-epoch repeated-data run vs control | written 2026-11-19, untested |
| `aug_family` | instrument swap within its GM family: per window, a random permutation of each family of 8 applied to the program streams and the instrument-set condition; piano-only and captioned windows skipped | batch | late-run val CE (+ instrument CE on real data) on the ~4-epoch run vs control | written 2026-11-19, untested |
| `drop_late` | late dropout: every dropout module and the attention's dropout_p switch from 0 to p once, at iteration `start` (one recompile) | model, batch | late-run val CE vs drop10 and control on the ~3.9-pass 42M run (scaling plan D-R2) | written 2026-11-21, untested |
| `wd_split` | split weight decay: Muon's matrices (`muon`), the input embedding tables (`emb`, 292,096 params at 42M) and the rest of AdamW's decayed group (`adam`, 3,556,864) each with their own decay | model, optimizer | val CE vs wd30x / control (wd30x_trunk, embwd0; scaling plan D-R3, D-O4) | written 2026-11-21, untested |
| `stack` | several hypotheses in one arm (`parts=name/k=v;name/k=v`): dataset and batch hooks chained in order, every model / optimizer hook, at most one forward | all | as its parts (BEST = REG + aug_dyn + aug_family) | written 2026-11-21, untested |
| `distill` | token-level KD: alpha x CE + (1 - alpha) x the sum over the 5 heads of KL(mean of the members' softmaxes \|\| student), members frozen in bf16, loaded under fork_rng; K + 1 compiled training graphs | model, forward | val CE and retention vs control / L_4 on the ~3.9-pass 42M run (scaling plan E-E5 distill4) | written 2026-11-21, untested |
