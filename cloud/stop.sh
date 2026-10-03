#!/bin/bash
# Graceful stop: no restart, train.py saves a resumable checkpoint at the end of its current step (SIGTERM),
# then one final push to the Hub. Resume later (any machine) with cloud/launch.sh after `cloud/ckpt.py pull`.
#   cloud/stop.sh [run name]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-long_450m}
OUT=checkpoints/$RUN
if [ -f cloud/secrets.env ]; then set -a; source cloud/secrets.env; set +a; fi
touch "$OUT/STOP"
for pid in $(pgrep -f "python train.py .*--out_dir=$OUT( |$)"); do
    # only the main process: dataloader workers have the same command line but a train.py parent
    parent=$(ps -o ppid= -p "$pid" | tr -d ' ')
    if ! ps -o args= -p "$parent" | grep -q "python train.py"; then
        echo "SIGTERM -> train.py pid $pid"
        kill -TERM "$pid"
    fi
done
WRAPPER=$(cat "logs/$RUN.wrapper.pid" 2>/dev/null || echo)
for _ in $(seq 1 120); do
    [ -n "$WRAPPER" ] && kill -0 "$WRAPPER" 2>/dev/null || break
    sleep 5
done
kill -0 "$WRAPPER" 2>/dev/null && echo "WARNING: wrapper still running after 10 min"
SYNC=$(cat "logs/$RUN.sync.pid" 2>/dev/null || echo)
[ -n "$SYNC" ] && kill "$SYNC" 2>/dev/null
echo "final push:"
.venv/bin/python cloud/ckpt.py push --run "$RUN"
.venv/bin/python cloud/ckpt.py status --run "$RUN"
