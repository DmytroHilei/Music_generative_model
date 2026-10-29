# Long run on a rented RTX 5090 (vast.ai): runbook

The 450M multi-instrument pretraining run (`config/pretrain/long_450m.py`) on one rented RTX 5090, paid from a ~EUR 20
budget, and the F1 caption run after it (`config/f1/f1.py`, section "F1" below; released as `musicar-L-caption`).
Everything below was rehearsed end to end on the laptop on 2026-10-03 (fresh clone → setup → preflight → launch →
Hub sync → graceful stop → simulated host loss → pull → resume → finish → laptop pull → generate.py), so the
paid hours should go to training only.

```
laptop                               Hugging Face (private)                 vast.ai RTX 5090
──────                               ──────────────────────                 ────────────────
cloud/sync/data.py upload  ──41 GB──►     <you>/music-stores (dataset)  ──►      cloud/setup/setup_instance.sh
                                                                            cloud/setup/preflight.py --hours N
cloud/sync/ckpt.py pull    ◄──────────    <you>/long_450m-ckpt-a / -b   ◄──      cloud/run/launch.sh (train + sync every 60 min)
```

## 0. Once, on the laptop (before renting anything)

1. **Hugging Face login** with a *write* token (done 2026-10-03: user `Dimitri-206265`, fine-grained token with
   repo write; it can create/delete repos, checked):
   `.venv/bin/hf auth login`
2. **Upload the training data** (~41 GB, private dataset repo `<you>/music-stores`). Resumable: if it stops, run the
   same command again; finished stores are skipped and Xet re-sends only missing chunks of a half-uploaded file.
   ```
   .venv/bin/python cloud/sync/data.py upload        # first run hashes ~41 GB (a few minutes), then uploads
   ```
   It ends with `upload verified: all N files on <you>/music-stores`. Without that line, run it again.
   Stores: `gigamidi_train, aria_train, discover_train` (27% subset), `gigamidi_clean_validation, aria_validation,
   discover_validation`, `gigamidi_heldout_validation` (captioned val set; see `cloud/sync/hub.py`), plus their label
   tables from `data/labels` (`labels/`, `gigamidi_train.jsonl` built with
   `data_prep/labels/build_labels.py --index-split train` from offsets.npy only) and Ukrainian captions from
   `data/captions` (`captions/`); on the instance `data/labels` and `data/captions` link to them. A store no longer on
   the laptop keeps its entries from the Hub manifest, so the tables can be re-uploaded alone. Free HF accounts get
   100 GB private storage: data 43 GB + at most two checkpoints (~3 GB each) fits.
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
bash cloud/setup/setup_instance.sh       # ~10 min: packages, data download + sha256 verify, accounts
.venv/bin/python cloud/setup/preflight.py --hours H      # ~15-20 min, see below
cloud/run/launch.sh                    # starts training + sync in the background; you can log out
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
- **Status**: `ssh ... 'cd music && cloud/status/status.sh'`: progress, last evals, wrapper restarts, OOM retries, GPU,
  last Hub push.
- **What runs**: `cloud/run/run_resumable.sh` restarts train.py from the last local checkpoint (every 30 min) after a
  crash, up to 20 times; train.py retries a CUDA OOM step itself (free cache, then eager with activations in RAM);
  `cloud/sync/ckpt.py loop` pushes a newer checkpoint to the Hub every 60 min (alternating `-ckpt-a` / `-ckpt-b`).
- **The end**: train.py exits 0 → wrapper writes `DONE` → the sync loop pushes the final checkpoint and, with
  `AUTO_STOP=1`, stops the instance (GPU billing ends; disk storage is still billed until you destroy it).

## 4. After the run (laptop)

```
.venv/bin/python cloud/sync/ckpt.py status --run long_450m     # the newest checkpoint on the Hub, done: True
.venv/bin/python cloud/sync/ckpt.py pull --run long_450m       # -> checkpoints/long_450m/ (ckpt.pt, model_bf16.pt, ...)
```
Then **destroy the instance** in the vast.ai console (stopped instances still bill for disk). Keep the HF repos until
the checkpoint is safely on the laptop; the data repo can stay for a later run (or `hf repo delete`).

## 5. When something goes wrong

| situation | what to do |
|---|---|
| upload from home interrupted | run `cloud/sync/data.py upload` again |
| `setup_instance.sh` fails (network, missing gcc) | read the last lines, fix, run it again (it resumes) |
| preflight FAILED | the message says which check; nothing has been spent beyond setup time. `logs/preflight_train.log` has the smoke-run output |
| need to stop and resume later / elsewhere | `cloud/run/stop.sh` (SIGTERM → save → final push, no restart), later `cloud/run/launch.sh` on any machine that has the checkpoint |
| host disappeared / instance broken | new instance → steps 2.1–2.4 → `.venv/bin/python cloud/sync/ckpt.py pull` → `cloud/run/launch.sh` (no preflight: run.env comes from the Hub). Lost: ≤ 60 min of training + setup |
| budget will run out before the end | lower `MAX_ITERS` in `checkpoints/long_450m/run.env` (keep it above the current step / 0.8 so the cooldown still runs whole), `cloud/run/stop.sh`, `cloud/run/launch.sh` |
| crash loop (wrapper "giving up") | `tail -200 logs/long_450m.log`; the last Hub checkpoint is safe |
| AUTO_STOP did nothing | vast.ai didn't provide `CONTAINER_ID` / `CONTAINER_API_KEY` in this image: stop the instance in the console yourself |
| finish on the laptop instead | `ckpt.py pull`, then edit `run.env`: `MICRO=1 ACCUM=32 ACT_CKPT=-1` (450M fits 8 GB only like this, ~8.8k notes/s), `cloud/run/launch.sh` |

## F1: the caption run (config/f1/f1.py, 2026-10-27 / 28)

The same machinery with three F1 scripts around it. The long run started from `checkpoints/f1_plain512_long` (from the
laptop), and after it the caption phase (`config/f1/f1_cap.py`) and the before / after evals ran on the instance
before it stopped. It ran on one vast.ai RTX 5090 (400 W cap) from 2026-10-27 10:25 UTC to ~02:30 local on
2026-10-28: 31,098 steps x 65,536 notes (2.04B notes, ~14 h, 1.58 s/step), then `f1_post.sh`: the caption phase
(600 steps at block 512, ~20 min) and the evals. The caption-phase weights (`checkpoints/f1_cap`) were released on
2026-10-28 as `Dimitri-206265/musicar-L-caption` (card: `docs/model_cards/musicar-L-caption.md`).
Times are for the 84 Mbps 5090 host of 2026-10-27; data is ~24 GB (no Discover stores).

| # | where | command | time |
|---|---|---|---|
| 1 | laptop | `ssh -p PORT root@HOST 'git clone -b transformer-v2 https://github.com/DmytroHilei/Music_generative_model.git music'` | 1 min |
| 2 | laptop | `cloud/run/f1_ship.sh code -p PORT root@HOST` (uncommitted code + `cloud/secrets.env`) | 1 min |
| 3 | instance | `cd music && DATA_STORES=gigamidi_train,aria_train,gigamidi_clean_validation,aria_validation,gigamidi_heldout_validation bash cloud/setup/setup_instance.sh` | ~55 min |
| 4 | laptop, during 3 | `cloud/run/f1_ship.sh weights -p PORT root@HOST` (1.9 GB, sha256 checked) | uplink |
| 5 | instance | `.venv/bin/python cloud/setup/preflight.py --run f1 --config config/f1/f1.py --hours H --skip-lr-probe` | ~25 min |
| 6 | instance | `bash cloud/run/f1_checks.sh` (real step time -> run.env, length check, caption-phase micro-batch) | ~12 min |
| 7 | instance | `POST_SWEEP=0 POST_RUN="bash cloud/run/f1_post.sh" POST_RUN_DIR=results/f1_post POST_RUN_LIMIT_MIN=120 cloud/run/launch.sh f1` | H h + ~45 min |

Steps 3 and 5-7 in one go, detached (needs step 4 to finish; progress in `logs/f1_all.log`):
`HOURS=18 setsid nohup bash cloud/run/f1_all.sh > logs/f1_all.log 2>&1 < /dev/null &`.
Disk: ~70 GB at the end (data 24, packages 7, LaBSE 2, checkpoints ~30 with the bf16 snapshots); rent 100 GB.

After: `.venv/bin/python cloud/sync/ckpt.py pull --run f1` (the long run) and
`hf download <user>/f1-results --local-dir results/f1_post` (the caption-phase weights, its logs and the evals),
then destroy the instance. The post-run reads `config/f1/f1_cap.py` when it starts, so the file on the instance can
be changed until the long run ends, e.g. after the length check in step 6.

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
| `f1_ship.sh`, `f1_checks.sh`, `f1_post.sh` | F1: laptop -> instance copy; checks between preflight and launch; caption phase + evals after the run |
| `requirements-lock.txt` | the exact training environment (torch 2.14.0+cu130, torchao 0.18, ...) |
| `secrets.env.example` | template for `cloud/secrets.env` (git-ignored) |
