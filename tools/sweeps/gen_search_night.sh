#!/bin/bash
# The night of 2026-10-27/28 on the laptop (context/claude/gen_search_2026-10-27.md): pull the F1 weights from the Hub
# once the instance has finished, then run the decoding-settings search in its own systemd scope (the OOM killer
# leaves it alone) with a wall-clock budget, so the morning finds results/gen_search/f1/summary.md.
#   tools/sweeps/gen_search_night.sh              # pull + search, budget 390 min (-> done ~07:30 from a 01:00 start)
#   BUDGET_MIN=300 tools/sweeps/gen_search_night.sh
#   tools/sweeps/gen_search_night.sh pull         # only the weights
#   tools/sweeps/gen_search_night.sh search       # only the search (weights already there); re-runs resume
# Weights: checkpoints/f1_cap/model_bf16.pt (the caption phase, from <user>/f1-results: f1_cap_model_bf16.pt) and
# checkpoints/f1/model_bf16.pt (the long run's final weights, the control, from the newer of <user>/f1-ckpt-a / -b).
# Without the caption-phase weights (the phase failed) the search runs on checkpoints/f1 alone.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
MODE=${1:-all}
BUDGET_MIN=${BUDGET_MIN:-390}
OUT=results/gen_search/f1
mkdir -p logs/gen_search checkpoints/f1_cap checkpoints/f1

pull() {
    $PY - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'cloud/sync')
from huggingface_hub import hf_hub_download, list_repo_files
import hub
hf = hub.api()
user = hub.hf_user(hf)
# the caption phase's weights from the post-run results
res = f'{user}/f1-results'
try:
    files = list_repo_files(res)
except Exception as e:
    files = []
    print(f'{res}: {type(e).__name__}: {e}')
if 'f1_cap_model_bf16.pt' in files:
    p = hf_hub_download(res, 'f1_cap_model_bf16.pt', local_dir='checkpoints/f1_cap')
    Path(p).replace('checkpoints/f1_cap/model_bf16.pt')
    print('checkpoints/f1_cap/model_bf16.pt pulled')
    for f in ('steps.log', 'opening_gain.txt'):
        if f in files:
            hf_hub_download(res, f, local_dir='results/f1_post')
else:
    print(f'no f1_cap_model_bf16.pt in {res} yet ({len(files)} files)')
# the long run's final weights (the control) from the newer checkpoint repo
import json
best, best_it = None, -1
for repo in hub.ckpt_repos(hf, 'f1'):
    try:
        st = json.loads(Path(hf_hub_download(repo, 'state.json')).read_text())
    except Exception as e:
        print(f'{repo}: {type(e).__name__}: {e}')
        continue
    print(f'{repo}: iter {st.get("iter")}, done {st.get("done")}')
    if st.get('iter', -1) > best_it:
        best, best_it = repo, st['iter']
if best:
    p = hf_hub_download(best, 'model_bf16.pt', local_dir='checkpoints/f1')
    print(f'checkpoints/f1/model_bf16.pt pulled from {best} (iter {best_it})')
PY
}

search() {
    CK=checkpoints/f1_cap/model_bf16.pt; CTRL=(--control checkpoints/f1/model_bf16.pt)
    if [ ! -s $CK ]; then
        echo "no caption-phase weights: the search runs on checkpoints/f1"; CK=checkpoints/f1/model_bf16.pt; CTRL=()
    fi
    [ -s $CK ] || { echo "no weights at $CK"; exit 1; }
    echo "search on $CK, budget $BUDGET_MIN min, log logs/gen_search/f1.log"
    systemd-run --user --scope -p ManagedOOMPreference=avoid --unit "musicar-gen-search-$(date +%H%M%S)" \
        $PY tools/sweeps/gen_search.py --checkpoint $CK "${CTRL[@]}" --out $OUT --retrieve gigamidi_replay_train \
        --budget-min "$BUDGET_MIN" >> logs/gen_search/f1.log 2>&1 &
    echo "pid $!; follow with: tail -f logs/gen_search/f1.log   (progress also on the dashboard)"
}

case "$MODE" in
    pull) pull ;;
    search) search ;;
    all) pull && search ;;
    *) sed -n 4,8p "$0"; exit 1 ;;
esac
