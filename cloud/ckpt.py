"""
Checkpoint sync between the rented GPU and the Hugging Face Hub (two private model repos used alternately).

  instance: python cloud/ckpt.py loop --run long_450m      # cloud/launch.sh starts this next to training
            python cloud/ckpt.py push --run long_450m      # one push now (cloud/stop.sh does this)
  laptop /  python cloud/ckpt.py pull --run long_450m      # newest checkpoint -> checkpoints/long_450m/
  new host:                                                #   (then cloud/launch.sh resumes from it)
            python cloud/ckpt.py status --run long_450m    # what is on the Hub

A push uploads ckpt.pt (full resumable state), model_bf16.pt (best val weights, if any), run.env (the machine/budget
settings), the training log tail and state.json (iteration, time) in ONE commit to whichever of the two repos holds the
older checkpoint, after deleting and re-creating that repo: one complete checkpoint is always on the Hub and storage
never grows past two. ckpt.pt is hard-linked into a snapshot folder first, so a new save during the upload can't
mix two versions.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hub  # noqa: E402

PUSH_FILES = ('ckpt.pt', 'model_bf16.pt', 'run.env', 'train_tail.log', 'state.json')


def log(msg):
    print(f'[ckpt {time.strftime("%F %T")}] {msg}', flush=True)


def remote_state(hf, repo):
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, 'state.json', force_download=True)
        return json.loads(Path(path).read_text())
    except Exception:
        return None  # repo missing or incomplete (no state.json = never finished a push)


def ckpt_iter(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)['iter_num']


def push(args):
    hf = hub.api()
    out = Path(args.out_dir)
    ckpt = out / 'ckpt.pt'
    if not ckpt.exists():
        log(f'nothing to push: {ckpt} does not exist yet')
        return None
    snap = out / '.push_snapshot'
    shutil.rmtree(snap, ignore_errors=True)
    snap.mkdir()
    os.link(ckpt, snap / 'ckpt.pt')  # same inode: a later os.replace of ckpt.pt leaves this copy intact
    if (out / 'model_bf16.pt').exists():
        os.link(out / 'model_bf16.pt', snap / 'model_bf16.pt')
    if (out / 'run.env').exists():
        shutil.copy(out / 'run.env', snap / 'run.env')
    if args.log and Path(args.log).exists():
        with open(args.log, 'rb') as f:
            f.seek(max(0, Path(args.log).stat().st_size - 5_000_000))
            (snap / 'train_tail.log').write_bytes(f.read())
    it = ckpt_iter(snap / 'ckpt.pt')
    state = {'iter': it, 'time': time.time(), 'utc': time.strftime('%F %T', time.gmtime()),
             'size_gb': round((snap / 'ckpt.pt').stat().st_size / 1e9, 2), 'done': (out / 'DONE').exists()}
    (snap / 'state.json').write_text(json.dumps(state))

    repos = hub.ckpt_repos(hf, args.run)
    states = {r: remote_state(hf, r) for r in repos}
    newest = max((s['iter'] for s in states.values() if s), default=-1)
    if it <= newest and not args.force:
        log(f'Hub already has iter {newest} (local {it}): skipped')
        shutil.rmtree(snap)
        return newest
    # overwrite the repo with the older (or no) checkpoint; the other one stays intact meanwhile
    target = min(repos, key=lambda r: states[r]['iter'] if states[r] else -1)
    t0 = time.time()
    hf.delete_repo(target, missing_ok=True)
    hf.create_repo(target, private=True, exist_ok=True)
    hf.upload_folder(repo_id=target, folder_path=str(snap), commit_message=f'iter {it}')
    check = remote_state(hf, target)
    if not check or check['iter'] != it:
        raise RuntimeError(f'push to {target} not confirmed (state.json: {check})')
    shutil.rmtree(snap)
    log(f'pushed iter {it} ({state["size_gb"]} GB) to {target} in {time.time() - t0:.0f} s')
    return it


def pull(args):
    from huggingface_hub import hf_hub_download, list_repo_files
    hf = hub.api()
    repos = hub.ckpt_repos(hf, args.run)
    states = {r: remote_state(hf, r) for r in repos}
    found = {r: s for r, s in states.items() if s}
    if not found:
        raise SystemExit(f'no checkpoint on the Hub for run {args.run} ({", ".join(repos)})')
    repo = max(found, key=lambda r: found[r]['iter'])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = out / 'ckpt.pt'
    if local.exists() and ckpt_iter(local) >= found[repo]['iter'] and not args.force:
        log(f'local {local} is already at iter {ckpt_iter(local)} (Hub {found[repo]["iter"]}): nothing to do')
        return
    files = [f for f in list_repo_files(repo) if f in PUSH_FILES]
    log(f'pulling iter {found[repo]["iter"]} from {repo}: {", ".join(files)}')
    for f in files:
        tmp = hf_hub_download(repo, f, local_dir=out / '.pull')
        os.replace(tmp, out / f)  # whole files only: a broken download never replaces a good local file
    shutil.rmtree(out / '.pull', ignore_errors=True)
    log(f'pulled into {out}')


def status(args):
    hf = hub.api()
    for repo in hub.ckpt_repos(hf, args.run):
        print(repo, remote_state(hf, repo) or '(none)')


def training_alive(pid_file):
    try:
        pid = int(Path(pid_file).read_text())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def loop(args):
    """Push a newer ckpt.pt every --every-min minutes while training runs; final push when it ends; optional stop."""
    out = Path(args.out_dir)
    last_pushed_mtime = 0.0
    log(f'sync loop: every {args.every_min} min, watching {out / "ckpt.pt"} (wrapper pid file {args.pid_file})')
    while True:
        alive = training_alive(args.pid_file)
        ckpt = out / 'ckpt.pt'
        due = ckpt.exists() and ckpt.stat().st_mtime > last_pushed_mtime
        if due:
            mtime = ckpt.stat().st_mtime
            for attempt in range(3):
                try:
                    push(args)
                    last_pushed_mtime = mtime
                    break
                except Exception as e:  # network or Hub hiccup: retry, then try again next round
                    log(f'push failed ({type(e).__name__}: {e}); retry {attempt + 1}/3 in 60 s')
                    time.sleep(60)
        if not alive:
            pending = ckpt.exists() and ckpt.stat().st_mtime > last_pushed_mtime
            log('training wrapper has exited' + (' (DONE)' if (out / 'DONE').exists() else ' (not DONE)')
                + (', LAST CHECKPOINT NOT ON THE HUB: run `python cloud/ckpt.py push`' if pending else ''))
            if args.auto_stop and (out / 'DONE').exists() and not pending:
                stop_instance()
            return
        for _ in range(args.every_min * 6):  # wake every 10 s to notice the end of training quickly
            if not training_alive(args.pid_file):
                break
            time.sleep(10)


def stop_instance():
    """vast.ai: stop (not destroy) this instance after the final push, so idle GPU time isn't billed. Needs the
    CONTAINER_ID / CONTAINER_API_KEY variables vast.ai sets inside instances; elsewhere it only logs."""
    cid, key = os.environ.get('CONTAINER_ID'), os.environ.get('CONTAINER_API_KEY')
    if not (cid and key):
        log('auto-stop: not on vast.ai (no CONTAINER_ID / CONTAINER_API_KEY), leaving the machine running')
        return
    vastai = shutil.which('vastai') or str(Path(sys.executable).parent / 'vastai')
    log(f'auto-stop: stopping vast.ai instance {cid}')
    subprocess.run([vastai, 'stop', 'instance', cid, '--api-key', key], check=False)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command', choices=['push', 'pull', 'loop', 'status'])
    p.add_argument('--run', default='long_450m')
    p.add_argument('--out-dir', default=None, help='default checkpoints/<run>')
    p.add_argument('--log', default=None, help='training log whose tail is pushed (default logs/<run>.log)')
    p.add_argument('--pid-file', default=None, help='loop: wrapper pid file (default logs/<run>.wrapper.pid)')
    p.add_argument('--every-min', type=int, default=60)
    p.add_argument('--auto-stop', action='store_true', help='loop: stop the vast.ai instance after the final push')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.out_dir = args.out_dir or str(hub.ROOT / 'checkpoints' / args.run)
    args.log = args.log or str(hub.ROOT / 'logs' / f'{args.run}.log')
    args.pid_file = args.pid_file or str(hub.ROOT / 'logs' / f'{args.run}.wrapper.pid')
    {'push': push, 'pull': pull, 'loop': loop, 'status': status}[args.command](args)


if __name__ == '__main__':
    main()
