#!/bin/bash
# Quick look at the long run: processes, progress, last evals, GPU, sync, disk.
#   cloud/status.sh [run name]
cd "$(dirname "$0")/.."
RUN=${1:-long_450m}
OUT=checkpoints/$RUN
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null && echo running || echo "not running"; }
echo "run $RUN | wrapper: $(alive logs/$RUN.wrapper.pid) | sync: $(alive logs/$RUN.sync.pid)" \
     "$( [ -f $OUT/DONE ] && echo '| DONE')$( [ -f $OUT/STOP ] && echo '| STOP requested')"
[ -f "$OUT/run.env" ] && grep -E "^(MAX_ITERS|MICRO|ACT_CKPT|HOURS|BENCH_NOTES_PER_S)=" "$OUT/run.env" | tr '\n' ' ' && echo
echo "--- progress"
tr '\r' '\n' < "logs/$RUN.log" 2>/dev/null | grep "Training:" | tail -1
echo "--- last evals"
tr '\r' '\n' < "logs/$RUN.log" 2>/dev/null | grep -E "^step [0-9]+:" | tail -3
echo "--- wrapper / errors"
grep -E "^\[wrapper\]" "logs/$RUN.log" 2>/dev/null | tail -3
tr '\r' '\n' < "logs/$RUN.log" 2>/dev/null | grep -E "Traceback|Error|CUDA OOM" | tail -3
echo "--- GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm --format=csv,noheader
echo "--- checkpoint sync"
tail -2 "logs/$RUN.sync.log" 2>/dev/null
[ -f "$OUT/ckpt.pt" ] && echo "local ckpt.pt: $(date -r "$OUT/ckpt.pt" '+%F %T')"
echo "--- disk: $(df -h . | awk 'NR==2 {print $4 " free"}')"
