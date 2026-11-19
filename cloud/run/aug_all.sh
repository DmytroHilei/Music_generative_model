#!/bin/bash
# The augmentation search (tools/sweeps/aug_sweep.py) on the instance in one go after the clone (cloud/README.md
# "Augmentation search"): setup with the two stores it reads -> smoke (40 iterations of every arm) -> the sweep
# (control + fresh, the gate, the arms) -> results/aug_sweep (runs.json, the table, every log) to <user>/aug-results
# (private) -> with AUTO_STOP=1 in cloud/secrets.env the instance stops. A failed smoke skips the sweep, then uploads
# and stops all the same. Progress: logs/aug_all.log and logs/aug_sweep.md.
#   mkdir -p logs && MICRO=16 setsid nohup bash cloud/run/aug_all.sh > logs/aug_all.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
MICRO=${MICRO:-16}
if [ -f cloud/secrets.env ]; then set -a; source cloud/secrets.env; set +a; fi
step() { echo; echo "=== [aug_all $(date '+%F %T')] $*"; }
finish() {
    step "results -> <user>/aug-results$([ "${AUTO_STOP:-0}" = 1 ] && echo ', then auto-stop')"
    mkdir -p results/aug_sweep/logs
    cp logs/aug_sweep.md results/aug_sweep/ 2>/dev/null
    cp logs/aug_sweep/*.log logs/aug_smoke.out logs/aug_all.log results/aug_sweep/logs/ 2>/dev/null
    HF_HUB_DISABLE_XET=1 .venv/bin/python cloud/sync/ckpt.py results --run aug --post-run-dir results/aug_sweep \
        $([ "${AUTO_STOP:-0}" = 1 ] && echo --auto-stop)
}

step "1/3 setup: packages + gigamidi_replay_train, gigamidi_heldout_validation (~2 GB with the label tables)"
DATA_STORES=gigamidi_replay_train,gigamidi_heldout_validation bash cloud/setup/setup_instance.sh \
    || { echo "AUG_ALL FAILED at setup $(date '+%F %T')"; exit 1; }

step "2/3 smoke: 40 iterations of every arm at micro-batch $MICRO"
.venv/bin/python tools/sweeps/aug_sweep.py --smoke --micro "$MICRO" | tee logs/aug_smoke.out
if grep -q FAILED logs/aug_smoke.out || ! grep -q ' ok ' logs/aug_smoke.out; then
    echo "AUG_ALL FAILED at smoke $(date '+%F %T')"
    finish
    exit 1
fi

step "3/3 the sweep: control + fresh on 2 seeds, the gate, then the arms"
.venv/bin/python tools/sweeps/aug_sweep.py --micro "$MICRO"
echo "sweep exited with $?"
cat logs/aug_sweep.md
finish
echo "AUG_ALL DONE $(date '+%F %T')"
