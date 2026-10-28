#!/bin/bash
# Laptop -> the rented instance, for F1: what its GitHub clone lacks.
#   cloud/run/f1_ship.sh code    -p PORT root@HOST   # seconds: the uncommitted code + cloud/secrets.env
#   cloud/run/f1_ship.sh weights -p PORT root@HOST   # 1.9 GB: the long run's init + the laptop's caption-phase weights
# code: the files `git ls-files -m -o --exclude-standard` lists (modified + untracked, not ignored; context/, docs/,
# samples/ left out), unpacked over the clone, and cloud/secrets.env from the laptop's logins (the token of
# `hf auth login`, the wandb key in ~/.netrc, AUTO_STOP=1; mode 600, never printed). weights:
# checkpoints/f1_plain512_long and checkpoints/f1_capphase model_bf16.pt, sha256 compared on both sides.
# The clone must exist first (REMOTE_DIR, default music):
#   ssh -p PORT root@HOST 'git clone -b transformer-v2 https://github.com/DmytroHilei/Music_generative_model.git music'
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-}
case "$MODE" in code|weights) shift ;; *) sed -n 3,4p "$0"; exit 1 ;; esac
[ $# -ge 1 ] || { sed -n 3,4p "$0"; exit 1; }
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$@")
R=${REMOTE_DIR:-music}
"${SSH[@]}" "test -d $R/.git" || { echo "no clone at ~/$R on the instance"; exit 1; }

if [ "$MODE" = code ]; then
    mapfile -t FILES < <(git ls-files -m -o --exclude-standard | grep -vE '^(context|docs|samples)/')
    printf '  %s\n' "${FILES[@]}"
    tar cf - "${FILES[@]}" | "${SSH[@]}" "cd $R && tar xf - && echo \"unpacked over \$(git log -1 --format=%h)\""
    HF=$(cat ~/.cache/huggingface/token)
    WB=$(python3 -c "import netrc; a = netrc.netrc().authenticators('api.wandb.ai'); print(a[2] if a else '')")
    printf 'HF_TOKEN=%s\nWANDB_API_KEY=%s\nAUTO_STOP=1\n' "$HF" "$WB" | "${SSH[@]}" "umask 077; cat > $R/cloud/secrets.env"
    echo "cloud/secrets.env written (HF token, wandb key $([ -n "$WB" ] && echo set || echo MISSING), AUTO_STOP=1)"
else
    W=(checkpoints/f1_plain512_long/model_bf16.pt checkpoints/f1_capphase/model_bf16.pt)
    du -ch "${W[@]}" | tail -1
    if command -v pv >/dev/null; then
        tar cf - "${W[@]}" | pv -s "$(du -cb "${W[@]}" | tail -1 | cut -f1)" | "${SSH[@]}" "cd $R && tar xf -"
    else
        tar cf - "${W[@]}" | "${SSH[@]}" "cd $R && tar xf -"
    fi
    here=$(sha256sum "${W[@]}")
    there=$("${SSH[@]}" "cd $R && sha256sum ${W[*]}")
    [ "$here" = "$there" ] || { echo "sha256 MISMATCH"; echo "$here"; echo "$there"; exit 1; }
    "${SSH[@]}" "touch $R/checkpoints/.f1_weights_ok"   # cloud/run/f1_all.sh waits for it
    echo "weights on the instance, sha256 ok"
fi
