# HPC optimizations backlog

Speed work that isn't done yet, from the HPC review on 2026-09-29. Estimates are rough until profiled; measure
first (item T1) before building anything. Hardware: RTX 5060 Laptop (sm_120, 8 GB, 105 W cap).

## Where the time goes (training, 107M big run)

- About 50k tokens/s ≈ 3.5·10¹³ useful FLOP/s including attention. Estimated peak (RTX 5090 specs scaled by SM
  count): ~30–35 TFLOPS bf16, ~60–70 fp8 → **~50% of the fp8 peak**. Realistic headroom in training: 1.2–1.5×.
- Kernel time (42M profile): GEMM 77% (`cutlass_80` Ampere-style 64×64 tiles), elementwise 17% (includes fp8
  scaling), LayerNorm 5%, attention 9%, CE + optimizer < 1%.
- The laptop runs at 101 W of 105 W, 82 °C.

## T1 result: 350M at 2048 (2026-10-03, `logs/prof/t1_350M.*`)

Config: 28L×1024 (365M), RoPE 2048, 129 programs, fp8 + compile, Muon bf16, checkpointing on all blocks, micro 4 × 8,
65,536 notes/step: 4.95 s/step, 13.2k notes/s, 5.5 GB. Profiled with `nsys` (NVTX phases) + `ncu` (speed of light).
- **GPU 100% busy** (4,955 ms of kernels per 4,952 ms step) and at its **~100 W power cap** (2.5 GHz, 80 °C, reason 0x4).
- Phases: forward 24%, backward 68% (**including the checkpoint recompute ≈ one forward, ~24% of the step**), clip 0.3%,
  optimizer 7%.
- Kernels: fp8 GEMM (cuBLASLt `nvjet_sm120`, native Blackwell, TMA, 128×128 tiles) 32%; **flash attention 27%**
  (FA2 kernels; bwd alone 14%); **Triton fused elementwise carrying the fp8 amax + casts 25%**; bf16 GEMM
  (`cutlass_80`) 8.5%, of which 5.3% is Muon Newton-Schulz in the optimizer; other 7%.
- `ncu`: fp8 GEMM 78–82% and flash bwd 90% of SM throughput, so the kernels themselves are near their limit. Speed can
  only come from doing less work.
- fp8 vs bf16: **13.2k vs 8.1k notes/s (fp8 = 1.63×)** at equal memory. cuDNN / efficient attention = flash (12.8k).

Verdicts: T2 keep (optimizer 7%, NS on small Ampere-style tiles). **T3 drop** (the fp8 GEMMs already run on native
sm_120 cuBLASLt kernels; the `cutlass_80` finding was bf16 at 42M). T4 done (micro 8 +6% but 7.2 GB). **T5 keep, top**
(amax/cast kernels ~25%). **T6 drop** (eval ≈ 1% of wall time). **T7/T8/T9 drop** (GPU never idle). T10 at the power cap
already; cooling can only hold clocks. **T11 drop** (~2 GB save every 30 min ≈ 0.4%). **T13 no backend gain**
(attention is 27% but cuDNN/efficient = flash, flash is compute-bound): attention cost is now an architecture question
(sliding-window layers, see the ML list). T14 low (2–3%). **New H1 keep, top**: selective checkpointing (keep the
attention outputs, ~0.5 GB, to skip the flash forward in the recompute ≈ 6%; leave a few blocks un-checkpointed with the
remaining 2.5 GB). New H2 low (offloading Muon momentum frees only 0.7 GB).

## Done

| item | result |
|---|---|
| fp8 + torch.compile (training) | +20% tokens/s at L, −25% memory |
| KV cache (inference) | 3.8× on CPU at 200 notes; preallocated static cache since 2026-09-29 |
| Batched generation (inference) | `generate.py --num-samples N [--batch-size B]`. On GPU (shared with training): 283 notes/s at B=1 → 1,753 at B=8 → 3,323 at B=32 |
| CUDA-graph decode (inference) | one-note transformer step captured once and replayed (`--no-cuda-graph` to disable), same notes as eager. **Measured on the idle GPU (2026-09-30): 1.22× at B=1 (514 → 630 notes/s), none in batches** (not launch-bound there). Roofline share: 39% at B=1, 60% at B=128 |
| bf16 inference on CUDA | `generate.py --dtype auto` (bf16 on CUDA, fp32 on CPU) |

## Training backlog

| # | optimisation | expected speedup | bottleneck it attacks | how to implement |
|---|---|---|---|---|
| T1 | Profile the 107M config | 0 (tells which of the rest is real) | unknown share of fp8 scaling, Muon and eval at 107M | `bench.py --profile --n_layer 14 --n_embd 768 --n_head 12 --micro 6` on an idle GPU; time one eval pass separately |
| T2 | Batched, compiled Muon Newton-Schulz | 3–6% of each step | 5 NS iterations × 56 matrices ≈ 3·10¹² FLOPs/step (~8% of step compute) as many small launches | group same-shaped matrices (qkv, fc, proj), stack, run NS as `torch.bmm` in bf16, wrap in `torch.compile` |
| T3 | Force max-autotune GEMM | 0–15% (GEMM = 77% of time) | generic `cutlass_80` 64×64 tiles; Inductor disabled autotune ("Not enough SMs for max_autotune") | `torch._inductor.config.max_autotune_gemm = True` and override the `is_big_gpu` check, or `compile(mode="max-autotune")`; enable Triton/CUTLASS backends; long one-off compile |
| T4 | Micro-batch sweep at 107M | 0–10% | GEMM shape vs cache/power; at 42M micro 6 beat 12 (112k vs 96k tok/s) | `bench.py --micro 4 6 8 12` for 14/768; also micro 8 × accumulation 15 |
| T5 | fp8 recipe / scaling overhead | 3–8% | amax reductions and casts inside the 17% elementwise | compare torchao tensorwise vs rowwise; check compile fuses the casts; skip fp8 for the smallest Linears |
| T6 | Cheaper evaluation | ~2–4% of wall time (estimate) | 2 val sets × 200 batches every 1,000 steps | time it first; Aria val only every N evals, fewer batches, bigger no-grad eval batches |
| T7 | Remove per-step CPU/GPU syncs | < 1% at 107M, 5–20% for small models | the tqdm postfix formats a GPU tensor → a sync every step | update the postfix from an async-copied value every N steps |
| T8 | CUDA graphs for training (`mode="reduce-overhead"`) | 1–3% at 107M, 1.5–2× at 6–20M | launch overhead; the 6M model was CPU/launch-bound (Python 91% CPU) | compile with `reduce-overhead`; batch shapes are already static |
| T9 | `pin_memory=True` in the train DataLoader | < 1% | synchronous H2D copies from pageable memory | one flag; hygiene |
| T10 | Power and cooling | 5–15% (sustained clocks) | 105 W cap, 82 °C | cooling pad, raised rear, performance power profile; `nvidia-smi -lgc` to lock clocks and smooth throttling |
| T11 | Async checkpoint save | negligible | ~1 GB `torch.save` every 30 min | background thread with CPU copies; only matters for frequent saves |
| T12 | fp8 attention / fp8 everywhere | ~0 on this GPU | FlashAttention-3 fp8 is Hopper-only | skip on sm_120 |
| T13 | Long context (1024–2048, with RoPE) | — (attention cost grows) | attention 9% → ~30% of time at 2k | cuDNN attention or FlexAttention then; recheck `sdpa_backend` |
| T14 | Short-to-long sequence warmup | ~5–10%, only at long context | attention cost early in training | only once context ≥ 2k |

## Inference backlog

| # | optimisation | expected speedup | bottleneck it attacks | how to implement |
|---|---|---|---|---|
| I1 | Speculative decoding | 2–3× | one full 107M forward per note | **draft models exist**: iso-M-muon (20M) trained on the same data. Draft proposes k notes, the big model verifies them in one pass; acceptance = rejection sampling over the 4 cascaded heads (dt → pitch → dur → vel) |
| I2 | Weight-only int8 quantization | 1.5–2× GPU decode, 2–3× CPU | reading weights from memory during decode | torchao `int8_weight_only` (GPU); `torch.ao` dynamic quant or ONNX Runtime / OpenVINO (CPU); verify with `sample_sweep.py` |
| I3 | Put the cascade heads + sampling in the graph | 5–15% of decode | 4 sequential small head steps and multinomial kernels run eagerly after the graph | capture heads + sampling too (CUDA-graph-safe RNG), or `torch.compile` the whole one-note step |
| I4 | Rolling KV cache without refills (needs RoPE) | removes the refill prefill every 128 notes past the window (~30% of long-generation cost) | learned absolute positions force a refill | comes with the planned RoPE retrain (plus attention sinks) |
| I6 | Attend only over filled cache slots | up to ~2× at large batch | the static cache attends over all 512 slots every step (masked), so on average twice the KV reads needed for 512-note generation; past B≈8 the KV cache, not the weights, bounds decode | round the fill up to a multiple of 64 and capture one CUDA graph per bucket (8 graphs), slicing k/v to the bucket |
| I7 | fp8 (or int8) KV cache | ~1.8× at large batch | KV bytes: 22 MB per row at 512 notes vs 214 MB of weights shared by the batch; theoretical ceiling ~17.5k notes/s from KV reads | store K/V in fp8 with a per-head scale, dequantise inside attention; check sample stats with `sample_sweep.py`. GQA (architecture.md A5) shrinks it 3–12× instead, but needs retraining |
| I5 | CPU bf16 | ~2× memory / bandwidth on CPU | fp32 weights on CPU | only if the CPU has AMX / AVX-512 BF16; `--dtype bf16 --device cpu` is already possible, measure it |
