#!/bin/bash
# F1 checks on the instance, after cloud/setup/preflight.py (run.env) and before cloud/run/launch.sh (~12 min, 5090).
# preflight benchmarks the base model with Muon; F1 adds LaBSE on the GPU, the adapters and 8-bit AdamW, so:
#   1. the init weights are there (checkpoints/f1_plain512_long, from the laptop: cloud/run/f1_ship.sh)
#   2. the real step time: 60 steps of config/f1/f1.py at run.env's micro-batch (half of it after a CUDA OOM); MICRO /
#      ACCUM / MAX_ITERS in run.env rewritten so the run fits HOURS at that speed (x 0.92 for evals and saves), capped
#      at MAX_ITERS_CAP (46,000 = ~3B notes, the F1 plan; a faster run ends sooner instead of training longer)
#   3. length control after the laptop's caption phase (checkpoints/f1_capphase): 30 / 60 / 120 s + none, 32 rows each
#      -> results/f1_checks/length_capphase (eval/compare/length_follow.py)
#   4. the caption phase (config/f1/f1_cap.py) for 20 steps at the largest micro-batch that runs without an OOM
#      (<= 4x the long run's) -> checkpoints/f1/capphase.env, read by cloud/run/f1_post.sh after the run
# Ends with CHECKS OK.  Logs: logs/f1_checks_*.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ENV=checkpoints/f1/run.env
[ -f $ENV ] || { echo "$ENV missing: run cloud/setup/preflight.py --run f1 --config config/f1/f1.py first"; exit 1; }
set -a; source $ENV; set +a
# checkpoints/f1.hours (one number) overrides run.env's HOURS (the budget changed after preflight)
if [ -s checkpoints/f1.hours ]; then HOURS=$(cat checkpoints/f1.hours); sed -i "s/^HOURS=.*/HOURS=$HOURS/" $ENV; fi
MAX_ITERS_CAP=${MAX_ITERS_CAP:-46000}
mkdir -p logs results/f1_checks
fail() { echo "CHECKS FAILED: $*"; exit 1; }
step() { echo; echo "=== [f1_checks $(date +%T)] $*"; }
# median ms per step over steps 30..n-3 and the peak GB of a train.py log (tqdm postfix ms= / gb=)
timing() {
    $PY - "$1" <<'PYEOF'
import re, statistics, sys
text = open(sys.argv[1], errors='ignore').read().replace('\r', '\n')
rows = [(int(n), int(total), float(gb), int(ms)) for n, total, gb, ms in
        re.findall(r'^Training:.*?\| *(\d+)/(\d+) \[[^\]]*?gb=([\d.]+)[^\]]*?ms=(\d+)\]', text, re.M)]
steady = [ms for n, total, gb, ms in rows if 30 <= n <= total - 3]
print(int(statistics.median(steady)) if steady else 0, max((gb for *_, gb, _ in rows), default=0.0))
PYEOF
}

step "1/4 init weights"
[ -s checkpoints/f1_plain512_long/model_bf16.pt ] || fail "checkpoints/f1_plain512_long/model_bf16.pt missing (cloud/run/f1_ship.sh weights)"
ls -la checkpoints/f1_plain512_long/model_bf16.pt
grep -q "CUDA OOM" logs/preflight_train.log 2>/dev/null && echo "note: the preflight smoke run hit a CUDA OOM at micro $MICRO"

step "2/4 the real F1 step time (60 steps)"
MS=0
for m in $MICRO $((MICRO / 2)); do
    [ $m -ge 1 ] || continue
    log=logs/f1_checks_timing_m$m.log
    rm -rf checkpoints/f1_timing
    $PY train.py config/f1/f1.py --out_dir=checkpoints/f1_timing --batch_size=$m --gradient_accumulation_steps=$((32 / m)) \
        --act_ckpt=$ACT_CKPT --max_iters=60 --lr_decay_iters=$MAX_ITERS --eval_interval=100000 --eval_iters=1 \
        --save_every=0 --ckpt_interval_min=0 --save_pre_cooldown=False --always_save_checkpoint=False \
        --wandb_log=False > $log 2>&1
    code=$?
    rm -rf checkpoints/f1_timing
    if [ $code = 0 ] && ! grep -q "CUDA OOM" $log; then
        read MS GB < <(timing $log); break
    fi
    echo "micro $m: exit $code$(grep -q 'CUDA OOM' $log && echo ', CUDA OOM') ($log)"
done
[ "$MS" -gt 0 ] || fail "no clean timing run (logs/f1_checks_timing_m*.log)"
FIT=$($PY -c "print(min($MAX_ITERS_CAP, int($HOURS * 3600 * 0.92 / ($MS / 1000))))")
echo "micro $m x $((32 / m)): $MS ms/step, peak $GB GB (preflight's base-model bench gave $MAX_ITERS steps)"
sed -i "s/^MICRO=.*/MICRO=$m/; s/^ACCUM=.*/ACCUM=$((32 / m))/; s/^MAX_ITERS=.*/MAX_ITERS=$FIT/" $ENV
MICRO=$m
grep -q "^F1_STEP_MS=" $ENV || echo "F1_STEP_MS=$MS" >> $ENV
grep -q "^MAX_ITERS_PREFLIGHT=" $ENV || echo "MAX_ITERS_PREFLIGHT=$MAX_ITERS" >> $ENV
$PY -c "print(f'run: {$FIT:,} steps = {$FIT * 65536 / 1e9:.2f}B notes, x {$MS / 1000:.2f} s / 0.92 = {$FIT * $MS / 1000 / 0.92 / 3600:.1f} h')"

step "3/4 length control after the laptop's caption phase"
if [ -s checkpoints/f1_capphase/model_bf16.pt ]; then
    if $PY eval/compare/length_follow.py --checkpoints checkpoints/f1_capphase --lengths 30,60,120 --rows 32 \
            --out results/f1_checks/length_capphase > logs/f1_checks_length.log 2>&1; then
        cat results/f1_checks/length_capphase/summary.md
    else
        echo "length check failed (logs/f1_checks_length.log); not a blocker for the launch"
    fi
else
    echo "skipped: no checkpoints/f1_capphase"
fi

step "4/4 the caption phase, 20 steps"
CAP_MICRO=""
for m in 64 32 16 8 4 2 1; do
    [ $m -le $((MICRO * 4)) ] || continue
    log=logs/f1_checks_cap_m$m.log
    rm -rf checkpoints/f1_cap_dry
    $PY train.py config/f1/f1.py config/f1/f1_cap.py --init_ckpt=checkpoints/f1_plain512_long/model_bf16.pt \
        --out_dir=checkpoints/f1_cap_dry --batch_size=$m --gradient_accumulation_steps=$((128 / m)) --act_ckpt=$ACT_CKPT \
        --max_iters=20 --eval_interval=100000 --eval_iters=1 --save_every=0 --ckpt_interval_min=0 \
        --always_save_checkpoint=False --wandb_log=False > $log 2>&1
    code=$?
    rm -rf checkpoints/f1_cap_dry
    if [ $code = 0 ] && ! grep -q "CUDA OOM" $log; then CAP_MICRO=$m; break; fi
    echo "micro $m: exit $code$(grep -q 'CUDA OOM' $log && echo ', CUDA OOM') ($log)"
done
[ -n "$CAP_MICRO" ] || fail "the caption phase ran at no micro-batch (logs/f1_checks_cap_m*.log)"
echo "CAP_MICRO=$CAP_MICRO" > checkpoints/f1/capphase.env
tr '\r' '\n' < logs/f1_checks_cap_m$CAP_MICRO.log | grep -E "^Training:" | tail -1
echo
echo "CHECKS OK: long run micro $(grep ^MICRO= $ENV | cut -d= -f2), $(grep ^MAX_ITERS= $ENV | cut -d= -f2) steps;" \
     "caption phase micro $CAP_MICRO x $((128 / CAP_MICRO))"
echo "launch: POST_SWEEP=0 POST_RUN=\"bash cloud/run/f1_post.sh\" POST_RUN_DIR=results/f1_post POST_RUN_LIMIT_MIN=120 cloud/run/launch.sh f1"
