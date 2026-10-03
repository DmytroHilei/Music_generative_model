# Long run on a rented RTX 5090 (vast.ai): runbook

The 450M multi-instrument pretraining run (`config/long_450m.py`) on one rented RTX 5090, paid from a ~EUR 20 budget.
Everything below was rehearsed end to end on the laptop on 2026-10-03 (fresh clone → setup → preflight → launch →
Hub sync → graceful stop → simulated host loss → pull → resume → finish → laptop pull → generate.py), so the
paid hours should go to training only.

```
laptop                               Hugging Face (private)                 vast.ai RTX 5090
──────                               ──────────────────────                 ────────────────
cloud/data.py upload  ──41 GB──►     <you>/music-stores (dataset)  ──►      cloud/setup_instance.sh
                                                                            cloud/preflight.py --hours N
cloud/ckpt.py pull    ◄──────────    <you>/long_450m-ckpt-a / -b   ◄──      cloud/launch.sh (train + sync every 60 min)
```

## 0. Once, on the laptop (before renting anything)

1. **Hugging Face login** with a *write* token (done 2026-10-03: user `Dimitri-206265`, fine-grained token with
   repo write; it can create/delete repos, checked):
   `.venv/bin/hf auth login`
2. **Upload the training data** (~41 GB, private dataset repo `<you>/music-stores`). Resumable: if it stops, run the
   same command again; finished stores are skipped and Xet re-sends only missing chunks of a half-uploaded file.
   ```
   .venv/bin/python cloud/data.py upload        # first run hashes ~41 GB (a few minutes), then uploads
   ```
   It ends with `upload verified: all N files on <you>/music-stores`. Without that line, run it again.
   Stores: `gigamidi_train, aria_train, discover_train` (27% subset), `gigamidi_clean_validation, aria_validation,
   discover_validation` (see `cloud/hub.py`). Free HF accounts get 100 GB private storage: data 41 GB + at most two
   checkpoints (~3 GB each) fits.
3. **wandb API key**: wandb.ai/authorize (copy it for step 2.4). Optional: without it, add `--no-wandb` to preflight
   and `EXTRA_ARGS="--wandb_log=False"` to launch.
4. **SSH key** for vast.ai: `ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519`, then paste the `.pub` content into
   vast.ai → Account → SSH Keys.

## 1. Rent the GPU (vast.ai console)

1. Account, add credit (EUR 20 ≈ $22).
2. Search → filters:
   - GPU: **RTX 5090**, 1x · rental type **On-Demand** (not interruptible)
   - **CUDA ≥ 13.0** (driver ≥ 580; the scripts install torch 2.14 + CUDA 13.0 wheels)
   - disk **≥ 150 GB** (data 41 + venv 6 + checkpoints + compile caches)
   - reliability ≥ 99% (≥ 99.5% for community hosts) · internet down ≥ 1000 Mbps
   - CPU ≥ 16 cores, RAM ≥ 32 GB
   - **max duration** of the offer longer than the planned run (hosts set an end date)
   - prefer **Secure Cloud / datacenter** if it costs ≤ ~10% more (a lost host costs ~1.5 GPU-hours to recover)
3. Template/image: any Ubuntu 22.04/24.04 image with SSH launch mode, e.g. the "NVIDIA CUDA" template. The scripts
   bring their own Python 3.12 and packages, so the image's Python/torch don't matter.
4. Note the price per hour P (USD) and your balance B. **Training hours to plan**:
   `H = (B − 2) / P − 1.5` (2 USD reserve for storage/egress, 1.5 h for setup + preflight). Example: B = 22, P = 0.40
   → H ≈ 48.

## 2. On the instance (SSH command from the console: `ssh -p <port> root@<ip>`)

```
tmux new -s run                    # optional, survives SSH drops (the run itself does not need it)
git clone https://github.com/DmytroHilei/Music_generative_model.git music && cd music
git checkout transformer-v2        # or the exact commit you tested
cp cloud/secrets.env.example cloud/secrets.env && nano cloud/secrets.env   # HF_TOKEN, WANDB_API_KEY, AUTO_STOP=1
bash cloud/setup_instance.sh       # ~10 min: packages, data download + sha256 verify, accounts
.venv/bin/python cloud/preflight.py --hours H      # ~15-20 min, see below
cloud/launch.sh                    # starts training + sync in the background; you can log out
```

`setup_instance.sh` is idempotent: if anything fails, fix it and run it again. It ends with `SETUP OK`.

`preflight.py` ends with `PREFLIGHT OK`. It:
- checks the GPU (fp8 matmul), free disk (≥ 20 GB), the verified data, wandb and Hub logins;
- benchmarks the model for (micro-batch, checkpointed blocks) settings and keeps the fastest that fits with 2 GB
  spare;
- computes `MAX_ITERS = H × 3600 × notes/s × 0.92 / 65,536` (the WSD cooldown is the last 20% of those steps, so it
  finishes inside the budget);
- smoke-tests the real config on the real data: 30 steps with evals and saves, resume, SIGTERM → graceful save
  (exit 143), a checkpoint push + pull through the Hub;
- LR probe: 300 steps at 1e-3 and 2e-3, keeps the better stable one (≈ 12 min);
- writes `checkpoints/long_450m/run.env` (all of the above), which `launch.sh` reads and every Hub push carries.

## 3. While it runs

- **wandb**: project `music-transformer`, run `long_450m` (val = `gigamidi_clean`, `val2` = aria, `discover/*`).
- **Status**: `ssh ... 'cd music && cloud/status.sh'`: progress, last evals, wrapper restarts, OOM retries, GPU,
  last Hub push.
- **What runs**: `cloud/run_resumable.sh` restarts train.py from the last local checkpoint (every 30 min) after a
  crash, up to 20 times; train.py retries a CUDA OOM step itself (free cache, then eager with activations in RAM);
  `cloud/ckpt.py loop` pushes a newer checkpoint to the Hub every 60 min (alternating `-ckpt-a` / `-ckpt-b`).
- **The end**: train.py exits 0 → wrapper writes `DONE` → the sync loop pushes the final checkpoint and, with
  `AUTO_STOP=1`, stops the instance (GPU billing ends; disk storage is still billed until you destroy it).

## 4. After the run (laptop)

```
.venv/bin/python cloud/ckpt.py status --run long_450m     # the newest checkpoint on the Hub, done: True
.venv/bin/python cloud/ckpt.py pull --run long_450m       # -> checkpoints/long_450m/ (ckpt.pt, model_bf16.pt, ...)
```
Then **destroy the instance** in the vast.ai console (stopped instances still bill for disk). Keep the HF repos until
the checkpoint is safely on the laptop; the data repo can stay for a later run (or `hf repo delete`).

## 5. When something goes wrong

| situation | what to do |
|---|---|
| upload from home interrupted | run `cloud/data.py upload` again |
| `setup_instance.sh` fails (network, missing gcc) | read the last lines, fix, run it again (it resumes) |
| preflight FAILED | the message says which check; nothing has been spent beyond setup time. `logs/preflight_train.log` has the smoke-run output |
| need to stop and resume later / elsewhere | `cloud/stop.sh` (SIGTERM → save → final push, no restart), later `cloud/launch.sh` on any machine that has the checkpoint |
| host disappeared / instance broken | new instance → steps 2.1–2.4 → `.venv/bin/python cloud/ckpt.py pull` → `cloud/launch.sh` (no preflight: run.env comes from the Hub). Lost: ≤ 60 min of training + setup |
| budget will run out before the end | lower `MAX_ITERS` in `checkpoints/long_450m/run.env` (keep it above the current step / 0.8 so the cooldown still runs whole), `cloud/stop.sh`, `cloud/launch.sh` |
| crash loop (wrapper "giving up") | `tail -200 logs/long_450m.log`; the last Hub checkpoint is safe |
| AUTO_STOP did nothing | vast.ai didn't provide `CONTAINER_ID` / `CONTAINER_API_KEY` in this image: stop the instance in the console yourself |
| finish on the laptop instead | `ckpt.py pull`, then edit `run.env`: `MICRO=1 ACCUM=32 ACT_CKPT=-1` (450M fits 8 GB only like this, ~8.8k notes/s), `cloud/launch.sh` |

## Files

| file | role |
|---|---|
| `hub.py` | stores list, repo names, sha256 manifest, loads `cloud/secrets.env` |
| `data.py` | `upload` (laptop) / `download` / `verify` (instance) |
| `setup_instance.sh` | system packages, uv + Python 3.12 + `requirements-lock.txt`, GPU check, data, accounts |
| `preflight.py` | checks, benchmark, budget, smoke tests, LR probe, `run.env` |
| `launch.sh`, `stop.sh`, `status.sh` | start detached / graceful stop + final push / quick status |
| `run_resumable.sh` | crash-restart wrapper around train.py (respects `out_dir/STOP`) |
| `ckpt.py` | `push`, `pull`, `loop`, `status` for checkpoints on the Hub; vast.ai auto-stop |
| `requirements-lock.txt` | the exact training environment (torch 2.14.0+cu130, torchao 0.18, ...) |
| `secrets.env.example` | template for `cloud/secrets.env` (git-ignored) |
