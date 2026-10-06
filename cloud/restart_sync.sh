#!/bin/bash
# Restart ONLY the checkpoint sync loop (e.g. to pick up new code: the post-run sweep), training keeps running.
#   cloud/restart_sync.sh [run name]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-long_450m}
OUT=checkpoints/$RUN
if [ -f cloud/secrets.env ]; then set -a; source cloud/secrets.env; set +a; fi
OLD=$(cat "logs/$RUN.sync.pid" 2>/dev/null || echo)
if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then kill "$OLD"; echo "stopped sync loop $OLD"; fi
STOP_FLAG=()
[ "${AUTO_STOP:-1}" = 1 ] && STOP_FLAG=(--auto-stop)
POST=()
if [ "${POST_SWEEP:-1}" = 1 ]; then
    POST=(--post-run ".venv/bin/python eval/style_search.py --checkpoint $OUT/model_bf16.pt --out results/${RUN}_sweep"
          --post-run-dir "results/${RUN}_sweep")
fi
setsid nohup .venv/bin/python cloud/ckpt.py loop --run "$RUN" --every-min "${SYNC_EVERY_MIN:-60}" "${STOP_FLAG[@]}" \
    "${POST[@]}" >> "logs/$RUN.sync.log" 2>&1 < /dev/null &
echo $! > "logs/$RUN.sync.pid"
sleep 2
kill -0 "$(cat logs/$RUN.sync.pid)" && echo "sync loop restarted: pid $(cat logs/$RUN.sync.pid)" && tail -2 "logs/$RUN.sync.log"
