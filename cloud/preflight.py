"""
Checks and sizing on the training machine before the long run; writes checkpoints/<run>/run.env for cloud/launch.sh.

  python cloud/preflight.py --hours 40          # on the instance, after cloud/setup_instance.sh
  python cloud/preflight.py --hours 0.2 --run rehearsal   # what the laptop rehearsal ran

Steps (each fails loudly, so nothing breaks hours into the paid run):
  1. environment: CUDA GPU, compute capability, an fp8 matmul, disk space, verified data, wandb + Hub credentials
  2. memory/speed: bench.py on the run's model shape for a list of (micro-batch, checkpointed blocks) settings,
     from the fastest to the leanest; keeps the faster of the first two that fit with a safety margin
  3. budget: max_iters = hours x measured notes/s x efficiency / notes per step (cooldown included in the budget)
  4b. LR probe (unless --skip-lr-probe): 300 steps at each --lrs value (constant after a 100-step warmup) on the
     real data; picks the best final GigaMIDI val CE among the runs without a loss spike (> 1.3x its running min
     after step 150); ties and unstable higher LRs fall back to the lowest LR
  4. smoke: real train.py on the real data with the chosen settings: 30 steps with evals and checkpoint saves,
     a resume to step 40, a SIGTERM during a resume (graceful save, exit 143), and a push + pull of that
     checkpoint through the Hub (separate '<run>-preflight' repos, deleted afterwards)
  5. writes run.env (settings + measurements) into the run's out_dir
"""

import argparse
import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'cloud'))
import hub  # noqa: E402  (also loads cloud/secrets.env into the environment)
PY = sys.executable
NOTES_PER_STEP = 65536
BLOCK = 2048
# (micro-batch, checkpointed blocks: 0 = none, -1 = all), most memory-hungry/fastest first
CANDIDATES = [(8, 0), (4, 0), (8, 14), (4, 14), (4, -1), (2, -1), (1, -1)]


def say(msg):
    print(f'[preflight {time.strftime("%T")}] {msg}', flush=True)


def fail(msg):
    say(f'FAILED: {msg}')
    sys.exit(1)


def read_config(path):
    """Top-level assignments of a config file (literals only), like train.py's configurator sees them."""
    cfg = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                cfg[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return cfg


def check_environment(args, cfg):
    import torch
    if not torch.cuda.is_available():
        fail('no CUDA GPU visible to torch')
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 2**30
    say(f'GPU {props.name}, {props.multi_processor_count} SMs, {mem_gb:.1f} GB, sm_{props.major}{props.minor}; '
        f'torch {torch.__version__}, CUDA {torch.version.cuda}')
    if (props.major, props.minor) < (8, 9):
        fail('fp8 training needs sm_89 or newer')
    a = torch.randn(256, 256, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn(256, 256, device='cuda').to(torch.float8_e4m3fn).t()
    one = torch.tensor(1.0, device='cuda')
    torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    say('fp8 matmul ok')
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < args.min_free_gb:
        fail(f'only {free:.0f} GB free disk (need {args.min_free_gb} for checkpoints and compile caches)')
    say(f'free disk {free:.0f} GB')
    if not (ROOT / 'data' / 'cache' / '.verified').exists():
        fail('data not verified: run cloud/setup_instance.sh (it downloads and verifies the stores)')
    needed = [(src, 'train') for src in str(cfg.get('csv_path', '')).split(',')]
    needed += [(src, 'validation') for key in ('val_csv_path', 'val2_csv_path')
               for src in str(cfg.get(key, '')).split(',')]
    needed += [(e.split('=', 1)[1], 'validation') for e in str(cfg.get('val_extra', '')).split(';') if '=' in e]
    for src, split in needed:
        if src.startswith('store:'):
            store = ROOT / f"{src[len('store:'):]}_{split}"
            if not (store / 'meta.json').exists():
                fail(f'{store} missing (read by {args.config})')
    say('data stores present and verified')
    if cfg.get('wandb_log', True) and not args.no_wandb:
        import wandb
        if not wandb.login(anonymous='never', verify=True):
            fail('wandb login failed: set WANDB_API_KEY in cloud/secrets.env (or pass --no-wandb)')
        say('wandb login ok')
    try:
        say(f'Hugging Face user {hub.hf_user(hub.api())}')
    except Exception as e:
        fail(f'Hugging Face login failed ({e}): set HF_TOKEN (a write token) in cloud/secrets.env')
    return mem_gb


def bench(cfg, micro, act_ckpt):
    cmd = [PY, 'bench.py', '--n_layer', str(cfg['n_layer']), '--n_embd', str(cfg['n_embd']),
           '--n_head', str(cfg['n_head']), '--block', str(BLOCK), '--pos_emb', 'rope',
           '--n_programs', str(cfg.get('n_programs', 0)), '--tokens_per_step', str(NOTES_PER_STEP),
           '--optim', 'muon_bf16' if cfg.get('muon_bf16') else 'muon', '--micro', str(micro), '--compile', '1',
           '--fp8', '1', '--steps', '8', '--act_ckpt', str(act_ckpt), '--ckpt_save', cfg.get('act_ckpt_save', ''),
           '--compile_ns', '1', '--norm', cfg.get('norm', 'layernorm'), '--mlp', cfg.get('mlp', 'gelu'),
           '--qk_norm', str(int(cfg.get('qk_norm', False))), '--fp8_cache', str(int(cfg.get('fp8_cache_weights', False)))]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r'^\s+\d+\s+1\s+\S+\s+1\s+-?\d+\s+([\d,]+)\s+(\d+)\s+([\d.]+)\s*$', out, re.M)
    if not m:
        return None
    return float(m.group(1).replace(',', '')), float(m.group(3))


def choose_setting(cfg, mem_gb, margin_gb):
    fits = []
    for micro, ck in CANDIDATES:
        say(f'bench micro {micro}, checkpointed blocks {ck} ...')
        r = bench(cfg, micro, ck)
        if r is None:
            say('  out of memory')
            continue
        nps, peak = r
        ok = peak <= mem_gb - margin_gb
        say(f'  {nps:,.0f} notes/s, peak {peak:.2f} GB' + ('' if ok else f' (over the {margin_gb} GB safety margin)'))
        if ok:
            fits.append((nps, micro, ck, peak))
        if len(fits) == 2:
            break
    if not fits:
        fail('no setting fits this GPU')
    return max(fits)


def run_train(args, overrides, timeout=None, sigterm_after=None):
    cmd = [PY, 'train.py', args.config] + [f'--{k}={v}' for k, v in overrides.items()]
    log = open(ROOT / 'logs' / 'preflight_train.log', 'a')
    log.write(f'\n=== {" ".join(cmd)}\n')
    log.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if sigterm_after:
        # wait for the first training step, then ask for a graceful stop
        deadline = time.time() + 900
        while time.time() < deadline and p.poll() is None:
            text = (ROOT / 'logs' / 'preflight_train.log').read_text(errors='ignore')[-20000:]
            if re.search(r'Training:\s+\d+%\|[^|]*\|\s*\d+/\d+', text.split('=== ')[-1]):
                time.sleep(sigterm_after)
                p.send_signal(signal.SIGTERM)
                break
            time.sleep(5)
    return p.wait(timeout=timeout)


def smoke(args, setting, cfg):
    nps, micro, ck, _ = setting
    out = ROOT / 'checkpoints' / f'{args.run}_preflight'
    shutil.rmtree(out, ignore_errors=True)
    (ROOT / 'logs' / 'preflight_train.log').unlink(missing_ok=True)
    base = dict(out_dir=out, batch_size=micro, gradient_accumulation_steps=NOTES_PER_STEP // (micro * BLOCK),
                act_ckpt=ck, warmup_iters=10, eval_interval=1000, eval_iters=4, ckpt_interval_min=0.5,
                wandb_log=False)
    say('smoke 1/4: 30 steps from scratch (evals at 0 and 29, checkpoint saves)')
    code = run_train(args, {**base, 'max_iters': 30, 'lr_decay_iters': 30})
    if code != 0 or not (out / 'ckpt.pt').exists():
        fail(f'smoke train exit {code}; see logs/preflight_train.log')
    say('smoke 2/4: resume to step 40')
    code = run_train(args, {**base, 'max_iters': 40, 'lr_decay_iters': 40, 'init_from': 'resume'})
    text = (ROOT / 'logs' / 'preflight_train.log').read_text(errors='ignore')
    if code != 0 or 'resume: loading' not in text or not re.search(r'step 39:', text):
        fail(f'resume exit {code}; see logs/preflight_train.log')
    say('smoke 3/4: SIGTERM during a resume -> graceful save, exit 143')
    code = run_train(args, {**base, 'max_iters': 400, 'lr_decay_iters': 400, 'init_from': 'resume'}, sigterm_after=20)
    import torch
    it = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=False, mmap=True)['iter_num']
    if code != 143 or it <= 40:
        fail(f'graceful stop: exit {code} (want 143), checkpoint at iter {it} (want > 40)')
    say(f'  saved at iter {it}, exit 143')
    if args.skip_hub_test:
        say('smoke 4/4: Hub round trip skipped (--skip-hub-test)')
    else:
        say('smoke 4/4: push this checkpoint to the Hub and pull it back')
        run = f'{args.run}-preflight'
        env = os.environ.copy()
        for cmd in (['push', '--out-dir', str(out), '--force'], ['pull', '--out-dir', str(out / 'pulled'), '--force']):
            r = subprocess.run([PY, 'cloud/ckpt.py', cmd[0], '--run', run] + cmd[1:], cwd=ROOT, env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                fail(f'ckpt.py {cmd[0]}: {r.stdout[-500:]} {r.stderr[-1500:]}')
        pulled = torch.load(out / 'pulled' / 'ckpt.pt', map_location='cpu', weights_only=False, mmap=True)['iter_num']
        if pulled != it:
            fail(f'pulled iter {pulled} != pushed {it}')
        hf = hub.api()
        for repo in hub.ckpt_repos(hf, run):
            hf.delete_repo(repo, missing_ok=True)
        say(f'  Hub round trip ok (iter {it}), test repos deleted')
    shutil.rmtree(out, ignore_errors=True)


def lr_probe(args, setting):
    nps, micro, ck, _ = setting
    results = {}
    for lr in [float(x) for x in args.lrs.split(',')]:
        out = ROOT / 'checkpoints' / f'{args.run}_lrprobe'
        shutil.rmtree(out, ignore_errors=True)
        (ROOT / 'logs' / 'preflight_train.log').unlink(missing_ok=True)
        say(f'LR probe {lr:g}: {args.probe_iters} steps')
        code = run_train(args, dict(out_dir=out, batch_size=micro, gradient_accumulation_steps=NOTES_PER_STEP // (micro * BLOCK),
                                    act_ckpt=ck, learning_rate=lr, warmup_iters=100, cooldown_frac=0.0,
                                    max_iters=args.probe_iters, lr_decay_iters=args.probe_iters,
                                    eval_interval=args.probe_iters, eval_iters=10, ckpt_interval_min=0, wandb_log=False))
        text = (ROOT / 'logs' / 'preflight_train.log').read_text(errors='ignore').replace('\r', '\n')
        shutil.rmtree(out, ignore_errors=True)
        evals = re.findall(r'step (\d+): .*? \| \S+ CE ([\d.]+) \(', text)
        losses = [float(x) for x in re.findall(r'loss=([\d.]+), lr=', text)]
        if code != 0 or not evals or int(evals[-1][0]) != args.probe_iters - 1:
            say(f'  failed (exit {code}): treated as unstable')
            continue
        late = losses[len(losses) // 2:]
        spike = any(x > 1.3 * min(late[:i + 1]) for i, x in enumerate(late)) if late else True
        results[lr] = (float(evals[-1][1]), spike)
        say(f'  val CE {evals[-1][1]}' + (' (loss spike: unstable)' if spike else ''))
    stable = {lr: ce for lr, (ce, spike) in results.items() if not spike}
    if not stable:
        fail('every probed LR failed or spiked; check logs/preflight_train.log')
    best = min(stable, key=lambda lr: (round(stable[lr], 2), lr))  # within 0.01 nats: the lower LR
    say(f'LR probe result: {", ".join(f"{lr:g} -> {ce:.3f}" for lr, ce in stable.items())}; chosen {best:g}')
    return best


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--hours', type=float, required=True, help='GPU hours to plan the run for (cooldown included)')
    p.add_argument('--run', default='long_450m')
    p.add_argument('--config', default='config/long_450m.py')
    p.add_argument('--efficiency', type=float, default=0.92, help='share of bench speed the real run keeps '
                   '(evals, data loading, checkpoint saves)')
    p.add_argument('--margin-gb', type=float, default=None, help='GPU memory kept free (default 2 GB, 0.8 on <=8 GB)')
    p.add_argument('--min-free-gb', type=float, default=20)
    p.add_argument('--setting', default=None, help="skip the benchmark: 'micro,act_ckpt,notes_per_s'")
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--skip-smoke', action='store_true')
    p.add_argument('--skip-hub-test', action='store_true')
    p.add_argument('--skip-lr-probe', action='store_true')
    p.add_argument('--lrs', default='1e-3,2e-3', help='LR probe candidates')
    p.add_argument('--probe-iters', type=int, default=300)
    args = p.parse_args()
    os.chdir(ROOT)
    (ROOT / 'logs').mkdir(exist_ok=True)
    cfg = read_config(args.config)

    say(f'1/5 environment ({args.config})')
    mem_gb = check_environment(args, cfg)
    margin = args.margin_gb if args.margin_gb is not None else (0.8 if mem_gb < 9 else 2.0)

    say('2/5 memory and speed')
    if args.setting:
        micro, ck, nps = args.setting.split(',')
        setting = (float(nps), int(micro), int(ck), 0.0)
    else:
        setting = choose_setting(cfg, mem_gb, margin)
    nps, micro, ck, peak = setting
    say(f'chosen: micro {micro}, checkpointed blocks {ck}: {nps:,.0f} notes/s, peak {peak:.2f} GB')

    say('3/5 budget')
    max_iters = int(args.hours * 3600 * nps * args.efficiency / NOTES_PER_STEP)
    warmup = cfg.get('warmup_iters', 0)
    if max_iters <= warmup:
        fail(f'{args.hours} h gives only {max_iters} steps (warmup is {warmup})')
    notes = max_iters * NOTES_PER_STEP
    say(f'{args.hours} h x {nps:,.0f} notes/s x {args.efficiency} -> max_iters {max_iters:,} '
        f'({notes / 1e9:.2f}B notes, cooldown from step {int(max_iters * (1 - cfg.get("cooldown_frac", 0))):,})')

    if args.skip_smoke:
        say('4/5 smoke skipped (--skip-smoke)')
    else:
        say('4/5 smoke tests on the real data')
        smoke(args, setting, cfg)
    lr = cfg.get('learning_rate') if args.skip_lr_probe else lr_probe(args, setting)

    say('5/5 run.env')
    out = ROOT / 'checkpoints' / args.run
    out.mkdir(parents=True, exist_ok=True)
    import torch
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    env = {'RUN': args.run, 'CONFIG': args.config, 'MICRO': micro,
           'ACCUM': NOTES_PER_STEP // (micro * BLOCK), 'ACT_CKPT': ck, 'MAX_ITERS': max_iters, 'LR': lr,
           'HOURS': args.hours, 'BENCH_NOTES_PER_S': round(nps), 'GPU': torch.cuda.get_device_name().replace(' ', '_'),
           'GIT_COMMIT': commit, 'CREATED': time.strftime('%F_%T')}
    (out / 'run.env').write_text(''.join(f'{k}={v}\n' for k, v in env.items()))
    say(f'wrote {out / "run.env"}:\n' + json.dumps(env, indent=1))
    say('PREFLIGHT OK. Start the run with: cloud/launch.sh')


if __name__ == '__main__':
    main()
