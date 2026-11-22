#!/bin/bash
# H100 benchmark for the compute request (context/claude/musicar_eurohpc_dev_outline.tex, appendix B items 1-3):
# notes/s, FP8 vs BF16 and peak memory under a 64 GB cap (MareNostrum 5 ACC's H100s) for musicar-L and the proposed
# 1B / 1.8B shapes, plus one profile of the 1.8B step. Synthetic batches: no data, tokens or secrets needed.
#   git clone https://github.com/DmytroHilei/Music_generative_model.git music && cd music
#   bash cloud/run/h100_bench.sh              # setup (ENV_ONLY) + all configs: ~45-60 min on an H100 SXM
#   SMOKE=1 bash cloud/run/h100_bench.sh      # tiny shapes, a few minutes on any GPU: checks the script
#   SKIP_SETUP=1 ...                          # the venv is already there; REF=<5090 jsonl> adds the speed factor
# Results: logs/h100/{gpu.txt, bench.log, bench.jsonl, profile_1p8b.log, bench.md}.
# Copy back from the laptop: rsync -az -e "ssh -p <port>" root@<host>:music/logs/h100/ logs/h100/   then destroy the
# instance in the vast.ai console.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=logs/h100
mkdir -p $OUT
if [ "${SKIP_SETUP:-0}" != 1 ]; then
    ENV_ONLY=1 bash cloud/setup/setup_instance.sh || { echo "setup failed"; exit 1; }
fi
PY=.venv/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv \
    > $OUT/gpu.txt 2>&1
cat $OUT/gpu.txt

COMMON=(--block 2048 --pos_emb rope --norm rmsnorm --mlp swiglu --qk_norm 1 --n_programs 129 --optim muon_bf16
        --compile 1 --steps 6 --mem_cap_gb 64 --out $OUT/bench.jsonl)
LIMIT=20m
if [ "${SMOKE:-0}" = 1 ]; then
    COMMON=(--block 512 --pos_emb rope --norm rmsnorm --mlp swiglu --qk_norm 1 --n_programs 129 --optim muon_bf16
            --compile 0 --steps 2 --out $OUT/bench.jsonl)
    LIMIT=5m
fi
run() {
    echo "=== $(date +%T) $*" | tee -a $OUT/bench.log
    timeout -k 1m $LIMIT $PY tools/bench/bench.py "${COMMON[@]}" "$@" 2>&1 | grep -v -i warning | tee -a $OUT/bench.log
}
profile() {
    echo "=== $(date +%T) profile $*" | tee -a $OUT/bench.log
    timeout -k 1m $LIMIT $PY tools/bench/bench.py "${COMMON[@]}" "$@" --profile > $OUT/profile_1p8b.log 2>&1
    tail -5 $OUT/profile_1p8b.log
}

if [ "${SMOKE:-0}" = 1 ]; then
    run --n_layer 4 --n_embd 256 --n_head 4 --tokens_per_step 4096 --micro 2 4 --fp8 1 0
    run --n_layer 4 --n_embd 256 --n_head 4 --tokens_per_step 1024 --micro 2 --fp8 1
    profile --n_layer 4 --n_embd 256 --n_head 4 --tokens_per_step 4096 --micro 2 --fp8 1
else
    # musicar-L, the reference: micro 4 is the 5090 run's setting
    run --n_layer 28 --n_embd 1152 --n_head 18 --tokens_per_step 65536 --micro 4 8 16 --fp8 1 0
    # 1B: primary (musicar-L's depth, wider) and a wider, shallower alternative
    run --n_layer 28 --n_embd 1664 --n_head 26 --tokens_per_step 65536 --micro 4 8 16 --fp8 1 0
    run --n_layer 22 --n_embd 1920 --n_head 30 --tokens_per_step 65536 --micro 8 16 --fp8 1
    # 1.8B: primary and wide; the one-node plan needs it under 64 GB without sharding
    run --n_layer 36 --n_embd 2048 --n_head 32 --tokens_per_step 65536 --micro 4 8 --fp8 1 0
    run --n_layer 36 --n_embd 2048 --n_head 32 --tokens_per_step 65536 --micro 16 --fp8 1 --act_ckpt -1
    run --n_layer 28 --n_embd 2304 --n_head 36 --tokens_per_step 65536 --micro 4 8 --fp8 1
    # per-GPU share of a 65,536-note step on 4 data-parallel GPUs (each GPU still runs the whole Muon update)
    run --n_layer 28 --n_embd 1664 --n_head 26 --tokens_per_step 16384 --micro 8 --fp8 1
    run --n_layer 36 --n_embd 2048 --n_head 32 --tokens_per_step 16384 --micro 8 --fp8 1
    # where the 1.8B step's time goes: forward / backward / clip / optimizer and the top CUDA kernels
    profile --n_layer 36 --n_embd 2048 --n_head 32 --tokens_per_step 65536 --micro 8 --fp8 1
fi

$PY tools/bench/bench_report.py $OUT/bench.jsonl ${REF:+--ref "$REF"} | tee $OUT/bench.md
echo "done $(date +%T): results in $OUT/"
