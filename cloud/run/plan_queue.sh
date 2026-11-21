#!/bin/bash
# Queue runner of the scaling plan (context/claude/scaling_plan_inputs_2026-11-20/01_MASTER_PLAN_assembled.md §4.8,
# K3a). logs/p0.sh execs it once P0 is done; on its own:
#   cd /root/music && setsid nohup bash cloud/run/plan_queue.sh >> logs/plan_queue.out 2>&1 < /dev/null &
#
# Queue: logs/plan_queue.txt, one line per run, popped from the top (blank and # lines are skipped):
#   min=<m> [need=<path>[,<path>]] [disk=<GB>] [tier=ESS|REC|OPT] [id=<plan id>] :: <shell command>
#   gate=<name> [min=<m>] [:: <extra args>]       -> $PY tools/sweeps/plan_gates.py <name> <extra args>
# Edit the queue by writing a new file and mv-ing it over (under `flock logs/plan_queue.txt.lock` to be safe from a
# pop in the same second). Seed: cloud/run/plan_queue_seed.txt.
# - Before each line: git pull --ff-only. The runner itself runs from a private copy (logs/plan_queue.run.sh) taken
#   at a fresh start, so a pull never changes the running script; new runner code applies at the next fresh start.
# - Commands run in `bash -c` with PY, A (aug_sweep at micro 16), T, C42, C108, SYN, PANEL (p0.sh names), TB, FTL,
#   AW, PE, SS (02_brief_items-3-4.md names) and the functions up / upkeep / m exported, under `timeout -k 2m` at
#   2.2 x min (no --foreground: the whole process group dies).
# - need= paths (globs allowed) missing, or disk= GB above free space - 5 GB: the line goes to logs/plan_deferred.txt.
#   Deferred lines move back to the queue front once ready (checked before every pop); when the queue runs empty
#   they get a counted retry, at most twice, then go to logs/plan_dropped.txt.
# - tier=OPT lines start only if `plan_gates.py guard --min <m>` exits 0 (budget / deadline), else dropped.
# - At the end: one `plan_gates.py catchall` pass, then 30 min waiting for new lines, the upload, the auto-stop.
# - Log: logs/plan_queue.log (start, end, exit code of every line); failed lines also in logs/plan_failed.txt.
#   Background upload every 30 min (logs/plan_upload.log).
# Env: PLAN_DRY=1 (no pull / upload / stop / sleeps; lines still run), AUTO_STOP=0 (no stop at the end),
#   PLAN_IDLE_MIN (30), PLAN_KILL (2m), PLAN_PY (.venv/bin/python), PLAN_GATES (command for plan_gates.py).
set -uo pipefail
ROOT=${PLAN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT" || exit 1
mkdir -p logs
STOP=${AUTO_STOP:-1}  # read before secrets.env, which cannot switch the stop off
if [ -z "${PLAN_COPY:-}" ]; then  # fresh start: one runner at a time, then exec a private copy
    exec 9>>logs/plan_queue.lock
    flock -n 9 || { echo "plan_queue: another runner holds logs/plan_queue.lock"; exit 1; }
    cp cloud/run/plan_queue.sh logs/plan_queue.run.sh
    PLAN_ROOT=$ROOT PLAN_COPY=1 AUTO_STOP=$STOP exec bash logs/plan_queue.run.sh "$@"
fi
[ -f cloud/secrets.env ] && { set -a; source cloud/secrets.env; set +a; }
export PLAN_DRY=${PLAN_DRY:-0}
DRY=$PLAN_DRY
Q=logs/plan_queue.txt; QLOCK=logs/plan_queue.txt.lock
DEF=logs/plan_deferred.txt; DROP=logs/plan_dropped.txt; FAIL=logs/plan_failed.txt; LOG=logs/plan_queue.log
UPSTOP=logs/.plan_upload_stop
IDLE_SECS=$(( ${PLAN_IDLE_MIN:-30} * 60 ))
KILL=${PLAN_KILL:-2m}
export PY=${PLAN_PY:-.venv/bin/python}
GATES=${PLAN_GATES:-$PY tools/sweeps/plan_gates.py}
GATES_FILE=${PLAN_GATES_FILE:-tools/sweeps/plan_gates.py}
export A="$PY tools/sweeps/aug_sweep.py --micro 16"
export T="$PY train.py config/pretrain/scale_ladder.py --max_iters=4883 --lr_decay_iters=4883 --batch_size=16 --gradient_accumulation_steps=1 --eval_iters=75 --eval_interval=1220 --always_save_checkpoint=True"
export C42="--n_layer=12 --n_embd=512 --n_head=8" C108="--n_layer=14 --n_embd=768 --n_head=12"
export SYN="--train_file_frac=0.25,1 --csv_path=store:data/cache/gigamidi_replay,store:data/cache/synth_musicar_l --source_weights=0.75,0.25"
export TB="$PY train.py config/pretrain/scale_ladder.py" FTL="--init_from=finetune --init_ckpt=checkpoints/musicar-L/model_bf16.pt"
export AW="--optimizer_name=adamw8bit --learning_rate=1e-4 --min_lr=1e-5"
export PE="--eval_only=True --pad_short=True --pack_short=False --compile=False --fp8=False --batch_size=8 --eval_iters=50"
export SS="$PY data_prep/datasets/synth_store.py --notes 40e6"
export PANEL="--val_extra=disc_aria=store:data/cache/cdisc_va;disc_dense=store:data/cache/cdisc_vd;giga=store:data/cache/gigamidi_heldout"
mkdir -p logs/members checkpoints/members checkpoints/keep results/aug_sweep/logs
touch "$Q" "$DEF"

say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
nap() { [ "$DRY" = 1 ] || sleep "$1"; }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [ -n "$s" ] && [[ $s != Z* ]]; }
pull() { if [ "$DRY" = 1 ]; then echo "DRY git pull --ff-only"; else timeout 120 git pull --ff-only -q 2>&1 | tail -2; fi; }

up() {  # results/aug_sweep (+ logs) -> <user>/aug-results; extra args go to ckpt.py (e.g. --auto-stop)
    mkdir -p results/aug_sweep/logs
    cp logs/aug_sweep*.md results/aug_sweep/ 2>/dev/null
    cp -r logs/aug_sweep/*.log logs/members logs/p0.log logs/plan_queue.log logs/plan_decisions.md logs/reg_arms.json \
        logs/plan_queue.txt logs/plan_deferred.txt logs/plan_dropped.txt logs/plan_failed.txt results/aug_sweep/logs/ 2>/dev/null
    if [ "$PLAN_DRY" = 1 ]; then echo "DRY upload $*"; return 0; fi
    HF_HUB_DISABLE_XET=1 timeout 60m $PY cloud/sync/ckpt.py results --run aug --post-run-dir results/aug_sweep "$@"
}
upkeep() {  # hard-link finished checkpoint dirs (checkpoints/<dir>) into results/aug_sweep/<dir>, then upload
    local d; for d in "$@"; do
        [ -d "checkpoints/$d" ] && mkdir -p "results/aug_sweep/$(dirname "$d")" \
            && cp -al --remove-destination "checkpoints/$d" "results/aug_sweep/$(dirname "$d")/"; done
    up
}
m() {  # m <name> <train.py args>: one member run with $T, skipped if logs/members/<name>.log has the last eval
    local n=$1; shift
    grep -q 'step 4882:' "logs/members/$n.log" 2>/dev/null && return 0
    $T "$@" > "logs/members/$n.log" 2>&1
}
export -f up upkeep m

uploader() { local i; while :; do
    for i in $(seq 30); do [ -f "$UPSTOP" ] && return 0; sleep 60; done
    up >> logs/plan_upload.log 2>&1; done; }

# --- one line: fields
parse() {  # $1 = line -> L_MIN L_NEED L_DISK L_TIER L_ID L_GATE L_TRY L_CMD
    local head tok toks
    L_MIN=; L_NEED=; L_DISK=; L_TIER=; L_ID=; L_GATE=; L_TRY=0; L_CMD=
    if [[ $1 == *' :: '* ]]; then head=${1%%' :: '*}; L_CMD=${1#*' :: '}; else head=$1; fi
    read -ra toks <<< "$head"
    for tok in "${toks[@]}"; do case $tok in
        min=*) L_MIN=${tok#min=} ;; need=*) L_NEED=${tok#need=} ;; disk=*) L_DISK=${tok#disk=} ;;
        tier=*) L_TIER=${tok#tier=} ;; id=*) L_ID=${tok#id=} ;; gate=*) L_GATE=${tok#gate=} ;; try=*) L_TRY=${tok#try=} ;;
    esac; done
}
with_try() {  # $1 = line, $2 = try count -> the line with try=<n> in its head
    local head rest=
    if [[ $1 == *' :: '* ]]; then head=${1%%' :: '*}; rest=" :: ${1#*' :: '}"; else head=$1; fi
    head=$(sed -E 's/(^| )try=[0-9]+//g' <<< "$head")
    echo "$head try=$2$rest"
}
missing() {  # what keeps the parsed line from starting now (empty = ready)
    local p ps free
    if [ -n "$L_NEED" ]; then IFS=, read -ra ps <<< "$L_NEED"
        for p in "${ps[@]}"; do compgen -G "$p" > /dev/null || { echo "need $p"; return; }; done; fi
    if [ -n "$L_DISK" ]; then free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
        awk -v f="$free" -v n="$L_DISK" 'BEGIN{exit !(f - 5 < n)}' && { echo "disk ${L_DISK} GB > ${free} - 5 GB free"; return; }; fi
    if { [ "$L_TIER" = OPT ] || [ -n "$L_GATE" ]; } && [ ! -f "$GATES_FILE" ]; then echo "need $GATES_FILE"; fi
}

# --- queue files (all edits: temp file + mv, under the queue lock)
locked() { local fd; exec {fd}>>"$QLOCK"; flock "$fd"; "$@"; local r=$?; exec {fd}>&-; return $r; }
_pop() {
    LINE=$(awk '!/^[[:space:]]*(#|$)/{print; exit}' "$Q")
    [ -n "$LINE" ] || return 1
    awk '!f && !/^[[:space:]]*(#|$)/{f=1; next} {print}' "$Q" > "$Q.tmp.$$" && mv "$Q.tmp.$$" "$Q"
}
_prepend() { { printf '%s\n' "$@"; cat "$Q"; } > "$Q.tmp.$$" && mv "$Q.tmp.$$" "$Q"; }
_append() { { cat "$Q"; printf '%s\n' "$@"; } > "$Q.tmp.$$" && mv "$Q.tmp.$$" "$Q"; }
pop() { locked _pop; }
has_lines() { grep -qv -E '^[[:space:]]*(#|$)' "$Q" 2>/dev/null; }
defer() { echo "$1" >> "$DEF"; }
promote_ready() {  # deferred lines that are ready now -> queue front, in their order (not a counted retry)
    local line ready=() keep=()
    [ -s "$DEF" ] || return 0
    while IFS= read -r line; do [ -n "$line" ] || continue
        parse "$line"; if [ -z "$(missing)" ]; then ready+=("$line"); else keep+=("$line"); fi
    done < "$DEF"
    [ ${#ready[@]} -gt 0 ] || return 0
    printf '%s\n' "${keep[@]}" | grep -v '^$' > "$DEF.tmp.$$"; mv "$DEF.tmp.$$" "$DEF"
    locked _prepend "${ready[@]}"
    say "READY ${#ready[@]} deferred line(s) back to the queue front"
}
retry_pass() {  # counted retry of every deferred line: try+1, dropped after 2 retries
    local line back=()
    while IFS= read -r line; do [ -n "$line" ] || continue
        parse "$line"
        if [ "$L_TRY" -ge 2 ]; then say "DROP ${L_ID:-$L_GATE} after 2 retries ($(missing))"; echo "$line" >> "$DROP"
        else back+=("$(with_try "$line" $((L_TRY + 1)))"); fi
    done < "$DEF"
    : > "$DEF"
    [ ${#back[@]} -gt 0 ] && { locked _append "${back[@]}"; say "RETRY ${#back[@]} deferred line(s)"; }
    return 0
}

run_cmd() {  # $1 minutes, $2 label, $3 command -> exit code
    local secs t0 code
    secs=$(awk -v m="$1" 'BEGIN{s=int(m * 60 * 2.2 + 0.999); print (s < 1 ? 1 : s)}')
    say "START $2 (min $1, timeout ${secs}s) :: $3"
    t0=$(date +%s)
    timeout -k "$KILL" "$secs" bash -c "$3" 9>&-
    code=$?
    say "END $2 exit $code$([ $code = 124 ] && echo ' (timeout)') after $(( ($(date +%s) - t0 + 30) / 60 )) min"
    return $code
}

handle() {  # one popped line
    local line=$1 why label code
    parse "$line"
    label=${L_ID:-${L_GATE:+gate=$L_GATE}}; label=${label:-line}
    if [ -n "$L_GATE" ]; then
        why=$(missing)
        if [ -n "$why" ]; then say "DEFER gate=$L_GATE: $why"; defer "$line"; return; fi
        run_cmd "${L_MIN:-5}" "gate=$L_GATE" "$GATES $L_GATE $L_CMD"
        RAN=1; return
    fi
    if [ -z "$L_MIN" ] || [ -z "$L_CMD" ]; then say "BAD line (needs min= and ' :: '): $line"; echo "$line" >> "$DROP"; return; fi
    why=$(missing)
    if [ -n "$why" ]; then say "DEFER $label: $why"; defer "$line"; return; fi
    if [ "$L_TIER" = OPT ]; then
        if ! $GATES guard --min "$L_MIN" 9>&-; then say "SKIP $label: budget guard"; echo "$line" >> "$DROP"; return; fi
    fi
    run_cmd "$L_MIN" "$label" "$L_CMD"; code=$?
    [ $code = 0 ] || echo "$(with_try "$line" "$L_TRY") # exit $code $(date -u '+%F %T')" >> "$FAIL"
    RAN=1
}

finish() {
    say "QUEUE DONE: final upload$([ "$STOP" != 0 ] && echo ', then auto-stop')"
    if [ -n "${UP:-}" ]; then touch "$UPSTOP"; local w=0
        while alive "$UP" && [ $w -lt 3600 ]; do sleep 5; w=$((w + 5)); done; rm -f "$UPSTOP"; fi
    up
    if [ "$STOP" != 0 ]; then
        if [ "$DRY" = 1 ]; then say "DRY auto-stop"; else
            HF_HUB_DISABLE_XET=1 $PY cloud/sync/ckpt.py results --run aug --auto-stop; fi
    else say "AUTO_STOP=0: leaving the machine running"; fi
}

say "plan_queue start (pid $$, dry $DRY, stop $STOP)"
UP=
if [ "$DRY" = 1 ]; then echo "DRY background upload every 30 min"; else rm -f "$UPSTOP"; uploader 9>&- & UP=$!; fi
RAN=1; CATCHALL=0
while :; do
    pull
    promote_ready
    if pop; then handle "$LINE"; continue; fi
    if [ -s "$DEF" ] && [ $RAN = 1 ]; then RAN=0; retry_pass; continue; fi
    if [ ! -s "$DEF" ] && [ $CATCHALL = 0 ] && [ -f "$GATES_FILE" ]; then  # one catch-all pass at the end
        CATCHALL=1; run_cmd 10 gate=catchall "$GATES catchall"; RAN=1; continue; fi
    say "queue empty ($(grep -c . "$DEF") deferred): waiting up to $((IDLE_SECS / 60)) min for new lines"
    got=0; waited=0
    while [ $waited -lt $IDLE_SECS ]; do
        nap 60; waited=$((waited + 60)); pull; promote_ready
        has_lines && { got=1; break; }
    done
    [ $got = 1 ] && continue
    if [ $CATCHALL = 0 ] && [ -f "$GATES_FILE" ]; then CATCHALL=1; run_cmd 10 gate=catchall "$GATES catchall"; RAN=1; continue; fi
    break
done
finish
