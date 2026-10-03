#!/bin/bash
# Start (or resume) the long run in the background, detached from the SSH session:
#   - cloud/run_resumable.sh around train.py (restarts from checkpoints/<run>/ckpt.pt after a crash)
#   - cloud/ckpt.py loop (pushes a newer checkpoint to the Hub every 60 min, final push at the end, then
#     stops the vast.ai instance if AUTO_STOP=1)
# Needs checkpoints/<run>/run.env from cloud/preflight.py (or pulled from the Hub with `cloud/ckpt.py pull`).
#   cloud/launch.sh [run name]            # default long_450m
# Extra train.py arguments: EXTRA_ARGS="--learning_rate=2e-3" cloud/launch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-long_450m}
OUT=checkpoints/$RUN
[ -f "$OUT/run.env" ] || { echo "$OUT/run.env missing: run cloud/preflight.py first (or cloud/ckpt.py pull)"; exit 1; }
if [ -f cloud/secrets.env ]; then set -a; source cloud/secrets.env; set +a; fi
set -a; source "$OUT/run.env"; set +a
[ -f data/cache/.verified ] || { echo "data not verified: run cloud/setup_instance.sh"; exit 1; }
PID_FILE=logs/$RUN.wrapper.pid
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (wrapper pid $(cat "$PID_FILE"))"; exit 1
fi
rm -f "$OUT/STOP"
mkdir -p logs
ARGS=("$CONFIG" "--out_dir=$OUT" "--wandb_run_name=$RUN" "--batch_size=$MICRO"
      "--gradient_accumulation_steps=$ACCUM" "--act_ckpt=$ACT_CKPT" "--max_iters=$MAX_ITERS"
      "--lr_decay_iters=$MAX_ITERS" "--learning_rate=$LR")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)
echo "train.py ${ARGS[*]}"
setsid nohup cloud/run_resumable.sh "logs/$RUN.log" 20 "${ARGS[@]}" > "logs/$RUN.wrapper.out" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
sleep 2
STOP_FLAG=()
[ "${AUTO_STOP:-1}" = 1 ] && STOP_FLAG=(--auto-stop)
setsid nohup .venv/bin/python cloud/ckpt.py loop --run "$RUN" --every-min "${SYNC_EVERY_MIN:-60}" "${STOP_FLAG[@]}" \
    > "logs/$RUN.sync.log" 2>&1 < /dev/null &
echo $! > "logs/$RUN.sync.pid"
echo "started: wrapper pid $(cat "$PID_FILE"), sync pid $(cat "logs/$RUN.sync.pid")"
echo "watch:   cloud/status.sh $RUN      (or tail -f logs/$RUN.log)"
echo "stop:    cloud/stop.sh $RUN        (graceful: save, push, no restart)"
