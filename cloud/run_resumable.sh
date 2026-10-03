#!/bin/bash
# Auto-resume wrapper around train.py: after a crash (non-zero exit) it restarts from out_dir/ckpt.pt.
#   cloud/run_resumable.sh <log file> <max restarts> <train.py args...>   (args must include --out_dir=...)
# A deliberate stop (cloud/stop.sh) creates out_dir/STOP first, so the wrapper exits instead of restarting.
# Exit codes: 0 = training finished, 1 = gave up after max restarts, 2 = stopped on request.
cd "$(dirname "$0")/.."
log=$1; max_restarts=$2; shift 2
out_dir=$(printf '%s\n' "$@" | sed -n 's/^--out_dir=//p' | tail -1)
[ -n "$out_dir" ] || { echo "run_resumable.sh: --out_dir=... is required"; exit 1; }
mkdir -p "$out_dir" "$(dirname "$log")"
for attempt in $(seq 0 "$max_restarts"); do
    extra=()
    [ -f "$out_dir/ckpt.pt" ] && extra=(--init_from=resume)
    echo "[wrapper] attempt $attempt $(date '+%F %T') ${extra[*]}" | tee -a "$log"
    .venv/bin/python train.py "$@" "${extra[@]}" >> "$log" 2>&1
    code=$?
    echo "[wrapper] exit $code $(date '+%F %T')" | tee -a "$log"
    if [ $code -eq 0 ]; then
        touch "$out_dir/DONE"
        exit 0
    fi
    if [ -f "$out_dir/STOP" ]; then
        echo "[wrapper] STOP file present: not restarting" | tee -a "$log"
        exit 2
    fi
    sleep 30
done
echo "[wrapper] giving up after $max_restarts restarts" | tee -a "$log"
exit 1
