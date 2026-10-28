#!/bin/bash
# F1 on the instance in one go after the clone and `cloud/run/f1_ship.sh code` (cloud/README.md "F1"): setup with the
# F1 stores only -> wait for the weights (`cloud/run/f1_ship.sh weights` on the laptop) -> preflight -> f1_checks.sh
# -> launch. Stops at the first failure. Progress: logs/f1_all.log; each step keeps its own log as well.
#   HOURS=18 setsid nohup bash cloud/run/f1_all.sh > logs/f1_all.log 2>&1 < /dev/null &
# checkpoints/f1.hours (one number), if present when preflight starts, overrides HOURS.
set -uo pipefail
cd "$(dirname "$0")/../.."
HOURS=${HOURS:?set HOURS, the GPU hours of the long run}
step() { echo; echo "=== [f1_all $(date '+%F %T')] $*"; }
stop() { echo; echo "F1_ALL FAILED at $1 $(date '+%F %T')"; exit 1; }

step "1/5 setup: packages + the F1 stores (~24 GB)"
DATA_STORES=gigamidi_train,aria_train,gigamidi_clean_validation,aria_validation,gigamidi_heldout_validation \
    bash cloud/setup/setup_instance.sh || stop setup

step "2/5 the weights from the laptop (cloud/run/f1_ship.sh weights)"
until [ -f checkpoints/.f1_weights_ok ]; do sleep 30; done
ls -la checkpoints/f1_plain512_long checkpoints/f1_capphase

[ -s checkpoints/f1.hours ] && HOURS=$(cat checkpoints/f1.hours)
step "3/5 preflight: $HOURS h"
.venv/bin/python cloud/setup/preflight.py --run f1 --config config/f1/f1.py --hours "$HOURS" --skip-lr-probe \
    || stop preflight

step "4/5 checks"
bash cloud/run/f1_checks.sh || stop checks

step "5/5 launch"
POST_SWEEP=0 POST_RUN="bash cloud/run/f1_post.sh" POST_RUN_DIR=results/f1_post POST_RUN_LIMIT_MIN=120 \
    cloud/run/launch.sh f1 || stop launch
echo; echo "F1_ALL LAUNCHED $(date '+%F %T'): cloud/status/status.sh f1 -w 60"
