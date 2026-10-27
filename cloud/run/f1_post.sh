#!/bin/bash
# After the F1 long run: cloud/sync/ckpt.py loop runs this once the run is DONE and its final checkpoint is on the Hub
# (launch.sh POST_RUN, time limit POST_RUN_LIMIT_MIN), then uploads results/f1_post to <user>/f1-results (private) and,
# with AUTO_STOP=1, stops the instance. A step that fails is logged and the next one runs (~40 min on the 5090):
#   1. the caption phase (config/f1/f1_cap.py): 600 it at block 512 from checkpoints/f1/model_bf16.pt -> f1_cap
#   2. held-out CE at block 2048 after the phase (eval_only; the long run's last eval is the before)
#   3. length control before / after: eval/compare/length_follow.py at 30 / 60 / 120 s + none, 32 rows each
#   4. the text gain per position after BOS over 200 held-out openings, before / after (eval/probes/opening_gain.py)
#   5. text_eval at cfg 1 / 2, before / after
# results/f1_post gets the caption phase's weights (final + it 300, hard links) and the logs; the long run's own weights
# are on the Hub already (<user>/f1-ckpt-a / -b: ckpt.pt, model_bf16.pt).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=results/f1_post
CAP=checkpoints/f1_cap
mkdir -p $OUT
set -a; source checkpoints/f1/run.env; [ -f checkpoints/f1/capphase.env ] && source checkpoints/f1/capphase.env; set +a
CAP_MICRO=${CAP_MICRO:-$MICRO}
step() { echo "=== [f1_post $(date +%T)] $*" | tee -a $OUT/steps.log; }
ended() { echo "    exit $1 $(date +%T)" | tee -a $OUT/steps.log; }

step "1/5 caption phase: micro $CAP_MICRO x $((128 / CAP_MICRO)), 600 it at block 512"
$PY train.py config/f1/f1.py config/f1/f1_cap.py --batch_size=$CAP_MICRO \
    --gradient_accumulation_steps=$((128 / CAP_MICRO)) --act_ckpt=$ACT_CKPT \
    --eval_iters=$(( (400 + CAP_MICRO - 1) / CAP_MICRO )) > $OUT/cap_train.log 2>&1
ended $?
for f in model_bf16.pt model_bf16_it300.pt; do [ -f $CAP/$f ] && ln -f $CAP/$f $OUT/f1_cap_$f; done
tr '\r' '\n' < $OUT/cap_train.log | grep -E "^step [0-9]+:" | cut -c1-400 | tee -a $OUT/steps.log
[ -s $CAP/model_bf16.pt ] || { step "no caption-phase weights: the evals run on the long run only"; CAP=""; }

step "2/5 held-out CE at block 2048 after the phase"
if [ -n "$CAP" ]; then
    $PY train.py config/f1/f1.py --init_ckpt=$CAP/model_bf16.pt --eval_only=True --batch_size=$MICRO \
        --act_ckpt=$ACT_CKPT --wandb_log=False --save_every=0 --save_pre_cooldown=False --ckpt_interval_min=0 \
        --out_dir=checkpoints/f1_cap_eval2048 > $OUT/cap_eval2048.log 2>&1
    ended $?
    tr '\r' '\n' < $OUT/cap_eval2048.log | grep -E "^step 0:" | cut -c1-400 | sed 's/^/    after: /' | tee -a $OUT/steps.log
fi
tr '\r' '\n' < logs/f1.log | grep -E "^step [0-9]+:" | tail -1 | cut -c1-400 | sed 's/^/    before (the long run, last eval): /' \
    | tee -a $OUT/steps.log

MODELS=checkpoints/f1${CAP:+,$CAP}
step "3/5 length control: $MODELS"
$PY eval/compare/length_follow.py --checkpoints $MODELS --lengths 30,60,120 --rows 32 --out $OUT/length_follow \
    > $OUT/length_follow.log 2>&1
ended $?

step "4/5 text gain per position after BOS: $MODELS"
$PY eval/probes/opening_gain.py ${MODELS//,/ } --windows 200 --device cuda > $OUT/opening_gain.txt 2>&1
ended $?

step "5/5 text_eval cfg 1 / 2: $MODELS"
$PY eval/compare/text_eval.py --checkpoints $MODELS --cfg 1,2 --out $OUT/text_eval > $OUT/text_eval.log 2>&1
ended $?

cp checkpoints/f1/run.env $OUT/ 2>/dev/null
cp -r results/f1_checks $OUT/ 2>/dev/null
tr '\r' '\n' < logs/f1.log | grep -E "^step [0-9]+:" > $OUT/f1_evals.log 2>/dev/null
step "done"
