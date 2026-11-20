#!/bin/bash
# The augmentation search (tools/sweeps/aug_sweep.py) on the instance in one go after the clone (cloud/README.md
# "Augmentation search"):
#   1. setup: packages, the replay + held-out stores, musicar-L's weights from the Hub (WEIGHTS)
#   2. smoke: 40 iterations of every training arm
#   3. SYNTH_NOTES notes of musicar-L continuations of real openings from different styles
#      (data_prep/datasets/synth_store.py -> data/cache/synth_musicar_l_train)
#   4. that store -> the Hub data repo (cloud/sync/data.py upload --merge), then the synth25 smoke
#   5. the sweep at 42M (control + fresh, the gate, the arms; synth25 runs even if the gate stops the others)
#   6. fresh, control and synth25 at TREND (default 108M; empty = skip)
# then results/aug_sweep (runs.json, the tables, every log) -> <user>/aug-results (private) and, with AUTO_STOP=1 in
# cloud/secrets.env, the instance stops. A failed smoke or generation skips what follows, then uploads and stops all
# the same. Progress: logs/aug_all.log, logs/aug_sweep.md, logs/aug_sweep_108M.md. Started again, every step skips
# what is finished (shards, runs).
#   mkdir -p logs && MICRO=16 setsid nohup bash cloud/run/aug_all.sh > logs/aug_all.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
MICRO=${MICRO:-16}
SYNTH_NOTES=${SYNTH_NOTES:-40e6}
SYNTH_ROWS=${SYNTH_ROWS:-64}
TREND=${TREND-108M}
PY=.venv/bin/python
if [ -f cloud/secrets.env ]; then set -a; source cloud/secrets.env; set +a; fi
step() { echo; echo "=== [aug_all $(date '+%F %T')] $*"; }
finish() {
    step "results -> <user>/aug-results$([ "${AUTO_STOP:-0}" = 1 ] && echo ', then auto-stop')"
    mkdir -p results/aug_sweep/logs
    cp logs/aug_sweep*.md results/aug_sweep/ 2>/dev/null
    cp logs/aug_sweep/*.log logs/aug_smoke.out logs/synth_store.log logs/aug_all.log results/aug_sweep/logs/ 2>/dev/null
    cp data/cache/synth_musicar_l_train/meta.json results/aug_sweep/synth_meta.json 2>/dev/null
    HF_HUB_DISABLE_XET=1 $PY cloud/sync/ckpt.py results --run aug --post-run-dir results/aug_sweep \
        $([ "${AUTO_STOP:-0}" = 1 ] && echo --auto-stop)
}
smoke_ok() { ! grep -q FAILED logs/aug_smoke.out && grep -q ' ok ' logs/aug_smoke.out; }

step "1/6 setup: packages, gigamidi_replay_train + gigamidi_heldout_validation (~2 GB), musicar-L's weights"
DATA_STORES=gigamidi_replay_train,gigamidi_heldout_validation WEIGHTS=musicar-L bash cloud/setup/setup_instance.sh \
    || { echo "AUG_ALL FAILED at setup $(date '+%F %T')"; exit 1; }

step "2/6 smoke: 40 iterations of every training arm at micro-batch $MICRO"
$PY tools/sweeps/aug_sweep.py --smoke --micro "$MICRO" | tee logs/aug_smoke.out
smoke_ok || { echo "AUG_ALL FAILED at smoke $(date '+%F %T')"; finish; exit 1; }

step "3/6 musicar-L continuations: $SYNTH_NOTES notes, $SYNTH_ROWS rows per batch"
$PY data_prep/datasets/synth_store.py --notes "$SYNTH_NOTES" --rows "$SYNTH_ROWS" --out data/cache/synth_musicar_l \
    2>&1 | tee -a logs/synth_store.log
[ -f data/cache/synth_musicar_l_train/meta.json ] || { echo "AUG_ALL FAILED at generation $(date '+%F %T')"; finish; exit 1; }

step "4/6 the synthetic store -> the Hub data repo, then the synth25 smoke"
HF_HUB_DISABLE_XET=1 $PY cloud/sync/data.py upload --stores synth_musicar_l_train --merge \
    || echo "upload failed: the store stays on this instance only (the sweep goes on)"
$PY tools/sweeps/aug_sweep.py --smoke --arms synth25 --micro "$MICRO" | tee -a logs/aug_smoke.out
smoke_ok || { echo "AUG_ALL FAILED at the synth25 smoke $(date '+%F %T')"; finish; exit 1; }

step "5/6 the sweep at 42M: control + fresh on 2 seeds, the gate, the arms"
$PY tools/sweeps/aug_sweep.py --micro "$MICRO"
$PY tools/sweeps/aug_sweep.py --arms synth25 --force --micro "$MICRO"  # a no-op if the gate let it run already
cat logs/aug_sweep.md

if [ -n "$TREND" ]; then
    step "6/6 fresh, control and synth25 at $TREND"
    $PY tools/sweeps/aug_sweep.py --size "$TREND" --arms fresh,control,synth25 --force --micro "$MICRO"
    cat "logs/aug_sweep_$TREND.md"
fi
finish
echo "AUG_ALL DONE $(date '+%F %T')"
