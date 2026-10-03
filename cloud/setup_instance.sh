#!/bin/bash
# One-time setup of a fresh GPU machine (vast.ai instance, or the laptop rehearsal). Idempotent: re-run after any
# interruption and it continues where it stopped.
#   git clone https://github.com/DmytroHilei/Music_generative_model.git music && cd music && git checkout <commit>
#   cp cloud/secrets.env.example cloud/secrets.env && nano cloud/secrets.env      # HF_TOKEN, WANDB_API_KEY
#   bash cloud/setup_instance.sh
# Then: .venv/bin/python cloud/preflight.py --hours <GPU hours>   and   cloud/launch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
step() { echo; echo "=== [setup $(date +%T)] $*"; }

step "1/6 secrets"
if [ -f cloud/secrets.env ]; then set -a; source cloud/secrets.env; set +a; fi
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN missing: put it in cloud/secrets.env (see cloud/secrets.env.example)"; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || echo "note: WANDB_API_KEY not set: the run needs --no-wandb or a key"
echo "ok"

step "2/6 system packages"
need=()
for cmd in git curl gcc tmux; do command -v $cmd >/dev/null || need+=($cmd); done
if [ ${#need[@]} -gt 0 ]; then
    if [ "$(id -u)" = 0 ]; then
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl build-essential tmux
    else
        echo "missing ${need[*]} and not root: install them (gcc is needed by triton / torch.compile)"; exit 1
    fi
fi
gcc --version | head -1

step "3/6 Python 3.12 venv with the pinned packages (uv)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -f .venv/.installed ] || [ cloud/requirements-lock.txt -nt .venv/.installed ]; then
    [ -d .venv ] || uv venv --python 3.12 --python-preference only-managed .venv
    uv pip install --python .venv/bin/python -r cloud/requirements-lock.txt vastai \
        --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
    uv cache clean >/dev/null 2>&1 || true
    touch .venv/.installed
fi
.venv/bin/python -c "import torch, torchao; print('torch', torch.__version__, '| torchao', torchao.__version__)"

step "4/6 GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'torch sees no CUDA GPU (driver too old for CUDA 13.0? needs >= 580)'
p = torch.cuda.get_device_properties(0)
print(f'{p.name}: sm_{p.major}{p.minor}, {p.total_memory / 2**30:.1f} GB, {p.multi_processor_count} SMs')
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
print('bf16 matmul ok', float((x @ x).float().abs().mean()) > 0)
EOF

step "5/6 data: download + verify (sha256 of every file)"
if [ ! -f data/cache/.verified ]; then
    .venv/bin/python cloud/data.py download
    .venv/bin/python cloud/data.py verify
    touch data/cache/.verified
else
    echo "already verified ($(date -r data/cache/.verified '+%F %T'))"
fi

step "6/6 accounts"
.venv/bin/python - <<'EOF'
import os, sys
sys.path.insert(0, 'cloud')
import hub
print('Hugging Face user:', hub.hf_user(hub.api()))
if os.environ.get('WANDB_API_KEY'):
    import wandb
    print('wandb login:', wandb.login(anonymous='never', verify=True))
EOF
mkdir -p logs checkpoints
echo
echo "SETUP OK. Next: .venv/bin/python cloud/preflight.py --hours <GPU hours for the run>"
