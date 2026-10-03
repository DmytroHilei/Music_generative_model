#!/bin/bash
# Quick look at the long run: processes, progress, speed, finish time, last evals, GPU, sync, disk.
#   cloud/status.sh [run name]            # once
#   cloud/status.sh [run name] -w [secs]  # live: refreshes every secs (default 10) until Ctrl-C
cd "$(dirname "$0")/.."
RUN=long_450m; WATCH=0; EVERY=10
for a in "$@"; do
    case $a in
        -w|--watch) WATCH=1 ;;
        [0-9]*) EVERY=$a ;;
        *) RUN=$a ;;
    esac
done
OUT=checkpoints/$RUN
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null && echo running || echo "NOT RUNNING"; }

show() {
    echo "run $RUN | wrapper: $(alive logs/$RUN.wrapper.pid) | sync: $(alive logs/$RUN.sync.pid)" \
         "$( [ -f $OUT/DONE ] && echo '| DONE')$( [ -f $OUT/STOP ] && echo '| STOP requested')   [$(date '+%F %T')]"
    [ -f "$OUT/run.env" ] && grep -E "^(MAX_ITERS|MICRO|ACT_CKPT|HOURS|LR|BENCH_NOTES_PER_S)=" "$OUT/run.env" | tr '\n' ' ' && echo
    echo "--- progress"
    bar=$(tr '\r' '\n' < "logs/$RUN.log" 2>/dev/null | grep "Training:" | tail -1)
    echo "$bar"
    # speed and finish time from the tqdm line: "[elapsed<remaining, X s/it | X it/s"
    python3 - "$bar" <<'EOF'
import re, sys, datetime
bar = sys.argv[1]
m = re.search(r'(\d+)/(\d+) \[[\d:]+<([\d:]+),\s*([\d.]+)(s/it|it/s)', bar)
if m:
    cur, tot, eta, v, unit = int(m[1]), int(m[2]), m[3], float(m[4]), m[5]
    s_per_it = v if unit == 's/it' else 1 / v
    parts = [int(x) for x in eta.split(':')]
    secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    finish = datetime.datetime.now() + datetime.timedelta(seconds=secs)
    print(f'{cur / tot:6.2%} done | {65536 / s_per_it:,.0f} notes/s | {s_per_it:.2f} s/step | '
          f'{cur * 65536 / 1e9:.2f}B of {tot * 65536 / 1e9:.2f}B notes | training ends ~{finish:%a %H:%M} '
          f'(+evals/saves)')
EOF
    echo "--- last evals"
    tr '\r' '\n' < "logs/$RUN.log" 2>/dev/null | grep -E "^step [0-9]+:" | tail -4
    echo "--- wrapper / errors"
    grep -E "^\[wrapper\]" "logs/$RUN.log" 2>/dev/null | tail -3
    tr '\r' '\n' < "logs/$RUN.log" 2>/dev/null | grep -E "Traceback|Error|CUDA OOM" | tail -3
    echo "--- GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
                       --format=csv,noheader)"
    echo "--- checkpoint sync"
    tail -2 "logs/$RUN.sync.log" 2>/dev/null
    [ -f "$OUT/ckpt.pt" ] && echo "local ckpt.pt: $(date -r "$OUT/ckpt.pt" '+%F %T')"
    echo "--- disk: $(df -h . | awk 'NR==2 {print $4 " free"}')"
}

if [ $WATCH = 1 ]; then
    while true; do
        out=$(show 2>&1)
        clear
        echo "$out"
        echo; echo "(refresh every ${EVERY}s, Ctrl-C to quit; the run keeps going)"
        sleep "$EVERY"
    done
else
    show
fi
