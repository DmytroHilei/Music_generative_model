# Architecture options

Model changes worth considering, with expected effect, cost and fit for the goal: Skryabin-style pop-rock piano
arrangements, later multi-track. Compiled 2026-09-30 from agents.md, the long-context discussion and standard
literature. Effects are **expectations to test, not results**, unless marked tested.

**For the user's judgment:** fill in the last two columns (verdict: yes / later / no / ?; notes: your reading,
better papers, your own version of the implementation). Add rows for new ideas. Agents: don't overwrite these two
columns, and when an option gets tested, record the result in the agents.md experiment log and set its status here.

Current model (for reference): nanoGPT decoder block (pre-LayerNorm, GELU MLP 4×, full multi-head attention, no bias),
learned absolute positions (512), compound tokens (1 position = 1 note with pitch/velocity/duration/delta-time),
cascade heads dt → pitch → duration → velocity, Muon + AdamW.

Status: **tested** (result in the experiment log) · **done** (in the code) · **planned** (in agents.md hypotheses) ·
**new** (not in the plan yet).

## A. Transformer block (need a new pretraining run; cheap to implement)

| # | option | what changes | expected effect | cost | fit / notes | key refs | status | **your verdict** | **your notes** |
|---|---|---|---|---|---|---|---|---|---|
| A1 | **RoPE** | rotary position encoding on q/k instead of learned absolute positions | ≈ neutral loss at 512; **unlocks A-context options**: context extension (C1), rolling KV cache, attention sinks | ~30 lines in `model.py`; retrain | prerequisite for most long-context work; do it in the next pretraining run | Su et al. 2021 (RoFormer) | planned (hyp. 6) | | |
| A2 | RMSNorm | LayerNorm without mean/bias | ≈ neutral loss, slightly faster (LN is ~5% of time) | trivial | free with A1 | Zhang & Sennrich 2019 | new | | |
| A3 | SwiGLU MLP | gated MLP, hidden 8/3·d instead of 4·d GELU | small loss gain at equal params (standard in LLaMA-class models) | small | low risk | Shazeer 2020 | new | | |
| A4 | QK-LayerNorm | normalise q and k before attention | training stability at high LR; may let Muon use a higher LR | small | cheap insurance for long runs | Dehghani et al. 2023; Wortsman et al. 2023 | planned (optional) | | |
| A5 | GQA / MQA | fewer K/V heads than query heads | **KV cache 3–12× smaller** → large-batch decode much faster (the KV cache is the bottleneck past B≈8); small quality cost | small; retrain or uptrain | pairs with batched generation | Ainslie et al. 2023; Shazeer 2019 | new | | |
| A6 | Speedrun tricks: ReLU², logit soft-capping, value embeddings, U-net skips | small block changes tuned together with Muon in the modded-nanoGPT speedrun | each a few % in token efficiency on text; unproven on music | small each; ablate one by one on the ladder (L) | cheap ladder experiments | So et al. 2021 (Primer); Gemma 2 2024; Jordan et al. 2024 (modded-nanogpt) | new | | |
| A7 | Multi-token prediction | extra head(s) predicting note t+2 (t+3) from the same hidden state | better sample efficiency and planning; **the heads can draft for self-speculative decoding** (2–3× decode) | medium (the cascade heads make it a 2-note cascade) | synergy with inference speed | Gloeckle et al. 2024; DeepSeek-V3 2024 (MTP) | new | | |

## B. Representation and conditioning

| # | option | what changes | expected effect | cost | fit / notes | key refs | status | **your verdict** | **your notes** |
|---|---|---|---|---|---|---|---|---|---|
| B1 | **Start/end tokens** | BOS at piece start, EOS at the end | natural openings and **endings** (currently: no stop, windows start mid-piece) | small; retrain or fine-tune | needed for whole songs | — | planned (B5) | | |
| B2 | **Conditioning tokens** (genre, composer/artist, key, tempo, density) | a prefix token (or embedding) per window from metadata | **control without prompts**: "pop, minor, calm". Aria already has genre/composer labels for ~74% of files; density conditioning would replace the `--dt-bias` workaround | small; retrain (condition dropout ~10% to keep unconditional mode) | high value for the fine-tune (artist/style token already planned there) | CTRL (Keskar et al. 2019); MuseNet 2019 | partly planned (fine-tune style token) | | |
| B3 | Beat-based tokens (REMI-like bar/beat grid) | quantised timing with bar/position tokens instead of 20 ms deltas | better rhythm and bar structure for pop, **enables bar-level methods** (C5, E2); loses expressive timing | large: new tokenizer + beat tracking + retrain | pop target yes, classical/performance no. Compare against current tokens | Huang & Yang 2020 (REMI); Hsiao et al. 2021 (Compound Word) | planned (hyp. 7) | | |
| B4 | Tempo / velocity augmentation | time-stretch and velocity shift during training | robustness, less overfitting at 4 epochs | small | cheap | — | planned (B7) | | |
| B5 | Instrument attribute (multi-track) | cascade dt → instrument → pitch → dur → vel | band arrangements | large (data: Lakh + stem transcription) | stage 2 of the project | — | planned (hyp. 8) | | |

## C. Long context and memory

The model sees 512 notes (~30–60 s). Songs are ~1,000–2,500 notes.

| # | option | what changes | expected effect | cost | fit / notes | key refs | status | **your verdict** | **your notes** |
|---|---|---|---|---|---|---|---|---|---|
| C0 | Pinned anchor (inference only) | keep the first N notes in front of the recent window | small: key-sim 0.63 → 0.69, register drift 7.2 → 5.0 semitones after 1,200 notes | none | `generate.py --anchor N` | StreamingLLM-like idea | **done** (opt-in) | | |
| C1 | **Longer context: 1024 pretrain → 2048–4096 via fine-tune** | train longer windows; extend RoPE with position interpolation / YaRN | **likely covers whole pop songs** (2,048 notes ≈ 2–4 min) | needs A1; attention cost grows (~30% of time at 2k) | most reliable long-form fix for the target | Chen et al. 2023 (PI); Peng et al. 2023 (YaRN) | planned (hyp. 6) | | |
| C2 | Attention sinks / StreamingLLM | keep a few first tokens + rolling window (inference) | stable endless generation, no drift collapse; no real long memory | tiny with RoPE | cheap add-on to C1 | Xiao et al. 2023 | new | | |
| C3 | Transformer-XL segment recurrence | cache previous segment's hidden states as extra context | longer effective memory; used by Pop Music Transformer | medium; retrain | a bit dated vs C1 | Dai et al. 2019; Huang & Yang 2020 | new | | |
| C4 | **Learned summary / memory tokens** ("compact in embedding space") | each segment ends with k memory vectors fed into the next (RMT) or accumulated summary vectors (AutoCompressors) | memory of theme/key/texture across segments, unbounded length | medium; **fine-tune on top of the pretrained model** on long pieces | best "summarize" option; if C1 isn't enough | Bulatov et al. 2022 (RMT); Chevalier et al. 2023 (AutoCompressors); Mu et al. 2023 (gist tokens); Ge et al. 2023 (ICAE) | new | | |
| C5 | **Museformer-style bar attention** | full attention to structurally related bars (1, 2, 4, 8, 12, 16 back) + one summary token per other bar | long-form **musical** structure (repeats, sections) at low cost | large; needs bar info (B3) | music-specific, strong fit for song form | Yu et al. 2022 (Museformer) | new | | |
| C6 | Memorizing Transformers (kNN memory) | retrieve past keys/values by similarity | recall of distant motifs | medium; light fine-tune | good for exact repeats | Wu et al. 2022 | new | | |
| C7 | Infini-attention | compressive linear-attention memory inside each attention layer | unbounded context in fixed memory | medium–large; continued pretraining | research-grade | Munkhdalai et al. 2024 | new | | |

## D. Alternative sequence models (SSM and linear attention)

| # | option | what changes | expected effect | cost | fit / notes | key refs | status | **your verdict** | **your notes** |
|---|---|---|---|---|---|---|---|---|---|
| D1 | Pure Mamba / Mamba-2 | selective state-space layers instead of attention | linear-time training, **constant-memory decode** (no KV cache); known weakness at exact copying/recall, and music repeats a lot | large; `mamba-ssm` on sm_120 may need a source build | mainly a long-context play; compare at equal tokens/params/compute | Gu & Dao 2023; Dao & Gu 2024; Jelassi et al. 2024 (copying weakness) | planned (hyp. 10) | | |
| D2 | **Hybrid SSM + attention** | mostly Mamba layers with a few attention layers (or sliding-window attention) | keeps attention's recall and SSM's long-context efficiency: the usual choice now over pure SSMs | large (same tooling as D1) | the version worth testing if SSM is tested | Lieber et al. 2024 (Jamba); Ren et al. 2024 (Samba); Glorioso et al. 2024 (Zamba) | new | | |
| D3 | Linear attention: RWKV / RetNet / GLA / (Gated) DeltaNet | attention replaced by a recurrent linear form | like D1; the `flash-linear-attention` library is Triton, so **likely easier on sm_120 than mamba-ssm** | medium–large | practical route to the SSM question on this laptop | Peng et al. 2023 (RWKV); Sun et al. 2023 (RetNet); Yang et al. 2023 (GLA); Yang et al. 2024 (DeltaNet, Gated DeltaNet) | new | | |
| D4 | xLSTM | modernised LSTM blocks (sLSTM/mLSTM) | competitive with transformers at small scale | large | curiosity; the repo has an old LSTM project in `legacy/lstm/` | Beck et al. 2024 | new | | |

## E. Other generation paradigms (add capabilities)

| # | option | what changes | expected effect | cost | fit / notes | key refs | status | **your verdict** | **your notes** |
|---|---|---|---|---|---|---|---|---|---|
| E1 | **Anticipatory Music Transformer** | interleave "anticipated" control events (e.g. the melody) into the AR sequence ahead of time | **melody → accompaniment and infilling with an AR model**, no diffusion | medium: data interleaving + fine-tune | direct fit for "vocal line in, piano part out" | Thickstun et al. 2023 | new | | |
| E2 | Masked discrete diffusion on our compound tokens | same transformer without causal mask, trained to unmask notes/attributes | infilling, regenerate bars, whole-segment structure, classifier-free guidance | medium–large | recommended first diffusion experiment | Sahoo et al. 2024 (MDLM); Lou et al. 2024 (SEDD); Chang et al. 2022 (MaskGIT) | planned (hyp. 9) | | |
| E3 | Piano-roll diffusion | U-Net on a beat-grid piano roll | strong for pop accompaniment; loses expressive timing | large; needs B3 | | Min et al. 2023 (Polyffusion) | planned (option 2) | | |
| E4 | Hierarchical whole-song generation | form → phrases → chords/lead sheet → notes, each level its own model | whole-song structure by construction | large | the "summarize like text" route: explicit, editable plans | Wang et al. 2024 (ICLR) | new | | |

## F. Capacity

| # | option | what changes | expected effect | cost | fit / notes | key refs | status | **your verdict** | **your notes** |
|---|---|---|---|---|---|---|---|---|---|
| F1 | Mixture of experts FFN | top-k routed experts | **tested: +0.10 per step and 2.5× slower on this laptop** (8 experts, top-2) | — | revisit only on a big GPU with grouped-GEMM kernels | Fedus et al. 2021 (Switch) | tested, rejected | | |
| F2 | Asymmetric (bigger) pitch head | more capacity in the pitch head | **tested: +0.04, worse** | — | scale the trunk instead | — | tested, rejected | | |

## Suggested order (agent's view, for comparison with yours)

1. **Next pretraining run:** A1 RoPE + A2 RMSNorm + A3 SwiGLU + A4 QK-norm together (cheap, low risk), plus B1 start/end
   tokens and B2 conditioning tokens. Context 1024.
2. **Fine-tune stage:** C1 context extension to 2048+, the Skryabin style token (B2), E1 anticipation for melody → accompaniment.
3. **If long-form structure still fails:** C4 memory tokens, or C5 bar attention (after B3).
4. **Research branch:** D2/D3 hybrid SSM vs the transformer at equal budget; E2 masked diffusion.
