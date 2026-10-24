"""One test session for the whole repo on this machine (CPU + the CUDA GPU), like a CI run before a merge:

  .venv/bin/python tests/ci.py               # everything, ~7 min
  .venv/bin/python tests/ci.py -k train,gen  # tests whose name contains 'train' or 'gen' (+ the tests they need)
  .venv/bin/python tests/ci.py --list        # the plan

Built after two open-source CIs:
- OpenCV (opencv/ci-gha-workflow): one test plan (suites of accuracy and perf test binaries, per-platform skip lists,
  perf tests in sanity mode = 1 sample, speed not judged), a runner with a per-test timeout (10 min default), one log
  file per test kept as the artifact, test data kept outside the code (opencv_extra), docs built as their own job.
- torchtitan (pytorch/torchtitan tests/): unit tests split into CPU and GPU, integration tests = the real training
  script on a tiny model for a few steps, one run per feature (checkpoint save / resume, compile, float8, activation
  checkpointing), numerics checked against stored loss values per execution mode and GPU type (tests/assets/losses/).

Tests run in plan order and the session doesn't stop at a failure; a test whose dependency didn't pass is skipped.
GPU tests are skipped without CUDA (or --cpu-only) and while another process computes on the GPU (a training run;
--gpu-busy-ok runs them anyway). Training data = synthetic token stores and caption tables, so no dataset is needed;
sample-search reads data/cache/gigamidi_heldout_validation and is skipped without it. Output:
logs/ci/<date_time>/<test>.log + summary.json (rewritten after every test) + report.txt / report.html
(tests/ci_report.py: failed tests with their command and traceback / log tail), logs/ci/index.html (every run, a test x
run grid) and logs/ci/latest -> the newest run; checkpoints there are deleted at the end (--keep keeps them).
Exit code 1 if anything failed.
"""

import argparse
import ast
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

import ci_report  # noqa: E402
from helpers import make_store, notes  # noqa: E402

PY = sys.executable
AUDIO_PY = ROOT / '.venv-audio' / 'bin' / 'python'
LOSSES = ROOT / 'tests' / 'assets' / 'losses'  # stored val losses: cpu.json, <GPU name>.json
PERF_LOG = ROOT / 'logs' / 'ci' / 'perf.jsonl'
HELDOUT = ROOT / 'data' / 'cache' / 'gigamidi_heldout_validation'
SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
TINY = dict(n_layer=2, n_head=2, n_embd=96, block_size=128, batch_size=4, gradient_accumulation_steps=2,
            warmup_iters=2, max_iters=6, lr_decay_iters=6, eval_interval=3, eval_iters=2, wandb_log=False,
            num_workers=0)
CPU = dict(device='cpu', compile=False, fp8=False, dtype='float32')
# config/pretrain/long_450m.py at a tiny size, on the synthetic multi-instrument store
RECIPE = dict(source_weights='', val_csv_path='', val2_csv_path='', val_extra='', n_layer=2, n_head=2, n_embd=96,
              block_size=128, batch_size=2, gradient_accumulation_steps=2, warmup_iters=2, max_iters=6,
              lr_decay_iters=6, eval_interval=3, eval_iters=2, act_ckpt=-1, style_eval_songs=0, ckpt_interval_min=0.01,
              num_workers=0, wandb_log=False)
# caption fine-tunes as the pilots run them (the architecture comes from the checkpoint they start from)
TEXT = ['config/pilots/text_pilot.py']
SECTIONS = TEXT + ['config/pilots/section_pilot.py', 'config/pilots/section_long.py', 'config/pilots/section_v2.py']
# audio test floors; measured 2026-10-23 on the laptop: melody 0.81, bass 0.93, piano 0.81, drums 0.99, reduction 0.82
AUDIO_MIN = dict(melody=0.6, bass=0.75, piano=0.6, drums=0.9, reduction=0.6)
RENDER = 'import sys, pathlib; from musicar.data.midi_io import render_mp3; render_mp3(pathlib.Path(sys.argv[1]), sys.argv[2])'
TESTS = []


class Fail(Exception):
    pass


class Skip(Exception):
    pass


def test(name, gpu=False, needs=(), timeout=600):
    def add(fn):
        TESTS.append(dict(name=name, fn=fn, gpu=gpu, needs=needs, timeout=timeout, doc=' '.join((fn.__doc__ or '').split())))
        return fn
    return add


class Session:
    def __init__(self, run_dir, args):
        self.run_dir, self.work, self.args = run_dir, run_dir / 'work', args
        self.log, self.deadline = None, 0.0

    def run(self, cmd, env=None, cwd=ROOT, codes=(0,)):
        """Runs cmd with its output appended to the test's log and returns the output; another exit code = Fail."""
        cmd = [str(c) for c in cmd]
        try:
            p = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True,
                               errors='replace', timeout=max(1.0, self.deadline - time.time()))
            out, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out, code = f'{e.stdout or ""}{e.stderr or ""}\nTIMEOUT', None
        out = re.sub(r'\x1b\[[0-9;]*m', '', out)  # colour codes (FORCE_COLOR is set in this environment)
        with open(self.log, 'a') as f:
            f.write(f'$ {" ".join(cmd)}\n{out}\n[exit {code}]\n\n')
        if code not in codes:
            tail = [ln.strip() for ln in out.replace('\r', '\n').splitlines() if ln.strip()][-3:]
            raise Fail(f'exit {code}: ' + ' | '.join(tail)[-400:])
        return out

    def train(self, name, config=(), data='syn', **kw):
        """train.py on a synthetic store, tiny model; checks that the val loss went down. Returns the output."""
        over = {**TINY, 'out_dir': self.work / name, 'csv_path': f'store:{self.work}/{data}', 'cache_dir': self.work, **kw}
        out = self.run([PY, 'train.py', *config, *(f'--{k}={v}' for k, v in over.items())])
        losses = val_losses(out)
        if len(losses) < 2:
            raise Fail('fewer than 2 val losses in the output')
        if not (math.isfinite(losses[-1][1]) and losses[-1][1] < losses[0][1]):
            raise Fail(f'val loss {losses[0][1]:.4f} -> {losses[-1][1]:.4f} (step {losses[-1][0]})')
        return out

    def generate(self, name, checkpoint, *flags):
        """generate.py, 2 rows x 64 notes, no mp3; both MIDI files must hold notes. Returns their paths."""
        from musicar.data.loader import tokenize_midi
        d = self.work / 'gen' / name
        d.mkdir(parents=True)
        self.run([PY, 'generate.py', '--checkpoint', checkpoint, '--output', d / 'g.mid', '--num-samples', 2,
                  '--max-new-tokens', 64, '--no-mp3', '--seed', 1, *flags])
        files = sorted(d.glob('g_*.mid'))
        counts = [len(tokenize_midi(str(f))) for f in files]
        if len(files) != 2 or min(counts) == 0:
            raise Fail(f'{len(files)} MIDI files, notes {counts}')
        return files


def val_losses(out):
    return [(int(s), float(v)) for s, v in re.findall(r'step (\d+): train loss \S+, val loss ([-\w.]+)', out)]


def ckpt_iter(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)['iter_num']


def same_files(a, b):
    return all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b))


def gpu_name():
    r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    return r.stdout.splitlines()[0].strip()


def golden(s, path, runs):
    """The val losses in the logs of `runs` vs the ones stored in `path`; a missing file is written, --update-golden
    rewrites it after an intended change."""
    from importlib.metadata import version
    got = {r: {str(k): v for k, v in val_losses((s.run_dir / f'{r}.log').read_text())} for r in runs}
    if s.args.update_golden or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(torch=version('torch'), atol=1e-3, val_loss=got), indent=1) + '\n')
        return f'written {path.relative_to(ROOT)}'
    ref = json.loads(path.read_text())
    bad = [f'{r} step {k}: {got[r].get(k)} vs {v}' for r, losses in ref['val_loss'].items() if r in got
           for k, v in losses.items() if abs(got[r].get(k, math.inf) - v) > ref['atol']]
    if bad:
        raise Fail('; '.join(bad) + f' (stored with torch {ref["torch"]}, now {version("torch")})')
    return f'{sum(len(v) for v in got.values())} losses within {ref["atol"]} of {path.relative_to(ROOT)}'


# ---------------------------------------------------------------- synthetic data

def band_notes(n, seed, section=32):
    """(n, 4) notes + (n,) programs in sections of `section` notes that alternate between quiet, slow, low piano and
    loud, fast, high piano + bass + drums: every boundary changes dynamics, pace, register, drums and instruments
    (what section captions describe)."""
    rng = np.random.default_rng(seed)
    quiet = np.stack([rng.integers(36, 56, n), rng.integers(4, 12, n), rng.integers(10, 60, n), rng.integers(15, 40, n)], 1)
    loud = np.stack([rng.integers(64, 90, n), rng.integers(22, 32, n), rng.integers(1, 10, n), rng.integers(0, 6, n)], 1)
    b = (np.arange(n) // section) % 2 == 1
    return np.where(b[:, None], loud, quiet), np.where(b, rng.choice([0, 33, 128], n), 0)


def make_labels(labels, store):
    """Caption tables of a synthetic store in the format musicar.text.captions reads (as tests/text/test_text.py
    builds them): every other file labeled (a MidiCaps caption or XMIDI tags), features for all."""
    offsets = np.load(store / 'offsets.npy')
    n = len(offsets) - 1
    progs = np.fromfile(store / 'programs.u8', np.uint8) if (store / 'programs.u8').exists() \
        else np.zeros(offsets[-1], np.uint8)
    caps = dict(caption='A sad rock song with piano, bass and drums.', genre=['rock'], mood=['sad'], key='A minor',
                tempo_word='Moderate', instrument_summary=['Piano', 'Bass', 'Drums'])
    rows = [dict(file=i, md5=f'f{i}', **(dict(midicaps=caps) if i % 4 == 0 else dict(xmidi=[['happy', 'pop']])))
            for i in range(0, n, 2)]
    (labels / f'{store.name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    bits = np.zeros((n, 136), np.uint8)
    for i in range(n):
        bits[i, np.unique(progs[offsets[i]:offsets[i + 1]])] = 1
    np.savez(labels / f'{store.name}.features.npz', programs=np.packbits(bits, axis=1, bitorder='little'),
             seconds=np.full(n, 60.0, np.float32), ops=np.linspace(1, 10, n).astype(np.float32),
             vel_mean=np.full(n, 21.0, np.float32), pitch_mean=np.full(n, 60.0, np.float32),
             key=np.full(n, 21, np.int16), key_conf=np.full(n, 0.2, np.float32), drums=np.full(n, 0.3, np.float32))


def band_midi():
    """8 bars of Am F C G at 100 bpm (19 s): piano half-note chords, bass eighths on the root, a voice melody in half
    notes, drums (kick / snare on the beats, closed hi-hat eighths)."""
    import pretty_midi
    pm, beat = pretty_midi.PrettyMIDI(initial_tempo=100), 0.6
    piano, bass, voice = pretty_midi.Instrument(0), pretty_midi.Instrument(33), pretty_midi.Instrument(53)
    drums = pretty_midi.Instrument(0, is_drum=True)
    chords = [(33, [57, 60, 64]), (29, [53, 57, 60]), (36, [55, 60, 64]), (31, [55, 59, 62])]  # bass root, chord
    melody = [76, 74, 72, 71, 72, 74, 76, 79]
    for bar in range(8):
        root, chord = chords[bar % 4]
        t0 = bar * 4 * beat
        for h in range(2):
            piano.notes += [pretty_midi.Note(80, p, t0 + 2 * h * beat, t0 + 2 * (h + 1) * beat - 0.05) for p in chord]
            voice.notes.append(pretty_midi.Note(90, melody[(2 * bar + h) % 8], t0 + 2 * h * beat,
                                                t0 + 2 * (h + 1) * beat - 0.1))
        for e in range(8):
            t = t0 + e * beat / 2
            bass.notes.append(pretty_midi.Note(95, root, t, t + beat / 2 - 0.03))
            drums.notes.append(pretty_midi.Note(70, 42, t, t + 0.1))
        drums.notes += [pretty_midi.Note(110, 36 if b % 2 == 0 else 38, t0 + b * beat, t0 + b * beat + 0.1)
                        for b in range(4)]
    pm.instruments += [piano, bass, voice, drums]
    return pm


def piano_roll(notes, end, hop=0.01):
    r = np.zeros((128, int(end / hop) + 1), bool)
    for n in notes:
        r[n.pitch, int(n.start / hop):max(int(n.end / hop), int(n.start / hop) + 1)] = True
    return r


def raw_pitch_accuracy(ref, est, end):
    """MIREX melody metric: share of the frames where the reference line sounds that the estimate has its pitch."""
    a, b = piano_roll(ref, end), piano_roll(est, end)
    return float((a & b).any(0)[a.any(0)].mean())


def frame_f1(ref, est, end):
    """MIREX multi-F0 frame-level F1 over (10 ms frame, pitch) pairs."""
    a, b = piano_roll(ref, end), piano_roll(est, end)
    tp = (a & b).sum()
    p, r = tp / max(b.sum(), 1), tp / max(a.sum(), 1)
    return float(2 * p * r / max(p + r, 1e-9))


def onset_f1(ref, est, tol=0.05):
    est_t, used, hit = sorted(n.start for n in est), set(), 0
    for t in sorted(n.start for n in ref):
        j = next((j for j, e in enumerate(est_t) if j not in used and abs(e - t) <= tol), None)
        if j is not None:
            used.add(j)
            hit += 1
    p, r = hit / max(len(est), 1), hit / max(len(ref), 1)
    return 2 * p * r / max(p + r, 1e-9)


# ---------------------------------------------------------------- the plan, in run order

@test('lint')
def lint(s):
    """ruff over the tracked .py files with the error set of GitHub's Python starter workflow (E9, F63, F7, F82:
    syntax errors, undefined names, misused comparisons)."""
    ruff = [shutil.which('ruff')] if shutil.which('ruff') else [shutil.which('uvx'), 'ruff'] if shutil.which('uvx') else None
    if not ruff:
        raise Skip('no ruff / uvx on PATH')
    files = subprocess.run(['git', 'ls-files', '*.py'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    s.run([*ruff, 'check', '--no-cache', '--select', 'E9,F63,F7,F82', '--output-format', 'concise', *files])
    return f'{len(files)} files'


@test('imports')
def imports(s):
    """Every 'from musicar... import x' / 'import musicar...' in the tracked .py files (legacy/ excepted) resolves, so
    scripts no unit test imports survive a module move or a rename (ruff doesn't check what an import finds)."""
    return s.run([PY, __file__, '--check-imports'], env={'CUDA_VISIBLE_DEVICES': ''}).strip().splitlines()[0]


@test('config-keys')
def config_keys(s):
    """Every name a config/ file assigns is a train.py setting: config files are exec'd (only --key=value is
    checked), so a renamed or removed setting in a config would silently do nothing."""
    known = set()
    for node in ast.parse((ROOT / 'train.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'config_keys':
                break
            known.add(node.targets[0].id)
    files = [f for f in subprocess.run(['git', 'ls-files', 'config'], cwd=ROOT, capture_output=True,
                                       text=True).stdout.split() if f.endswith('.py')]
    bad = [f'{f}:{node.lineno} {t.id}' for f in files for node in ast.parse((ROOT / f).read_text()).body
           if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name) and t.id not in known]
    if bad:
        raise Fail('not train.py settings: ' + ', '.join(bad))
    return f'{len(files)} configs, {len(known)} settings'


def unit_tests(s, env=None):
    """tests/run.py; its 'n/m passed' line, or Fail with the failed tests named."""
    out = s.run([PY, 'tests/run.py'], env=env, codes=(0, 1))
    passed = re.findall(r'^(\d+/\d+ passed)', out, re.M)
    failed = re.findall(r'^  failed: (\S+)', out, re.M)
    if failed or not passed:
        raise Fail(f'{passed[-1] if passed else "no summary line"}; failed: {", ".join(failed)}')
    return passed[-1]


@test('unit-cpu')
def unit_cpu(s):
    """tests/run.py with CUDA hidden (torchtitan: unit_tests/cpu)."""
    return unit_tests(s, env={'CUDA_VISIBLE_DEVICES': ''})


@test('unit-gpu', gpu=True)
def unit_gpu(s):
    """tests/run.py with the GPU visible: the CUDA-only checks (CUDA-graph decode, fla kernels) run instead of
    returning early."""
    return unit_tests(s)


@test('data')
def data(s):
    """Synthetic token stores, 24 + 4 files x 400 notes: 'syn' (piano, random notes) and 'multi' (quiet piano and
    loud band sections of 32 notes in turn), and caption tables for both (labels/)."""
    for split, k, off in (('train', 24, 0), ('validation', 4, 1000)):
        make_store(s.work / f'syn_{split}', [notes(400, seed=off + i) for i in range(k)])
        band = [band_notes(400, seed=off + i) for i in range(k)]
        make_store(s.work / f'multi_{split}', [t for t, _ in band], [p for _, p in band])
    (s.work / 'labels').mkdir()
    for store in ('syn_train', 'syn_validation', 'multi_train', 'multi_validation'):
        make_labels(s.work / 'labels', s.work / store)
    return 'syn, multi, labels'


@test('train-cpu', needs=('data',))
def train_cpu(s):
    """6 steps on the CPU, eager fp32."""
    losses = val_losses(s.train('train-cpu', **CPU))
    return f'val {losses[0][1]:.3f} -> {losses[-1][1]:.3f}'


@test('golden-cpu', needs=('train-cpu',))
def golden_cpu(s):
    """train-cpu's val losses vs tests/assets/losses/cpu.json: any change to init, data sampling, the loss or the
    optimizer moves them (rerun with --update-golden after an intended one)."""
    return golden(s, LOSSES / 'cpu.json', ['train-cpu'])


@test('train-gpu', gpu=True, needs=('data',))
def train_gpu(s):
    """6 steps with the defaults: CUDA, bf16, torch.compile, fp8; resumable ckpt.pt."""
    losses = val_losses(s.train('train-gpu', ckpt_interval_min=0.01))
    if not (s.work / 'train-gpu' / 'ckpt.pt').exists():
        raise Fail('no ckpt.pt')
    return f'val {losses[0][1]:.3f} -> {losses[-1][1]:.3f}'


@test('train-gpu-eager', gpu=True, needs=('data',))
def train_gpu_eager(s):
    """6 steps on CUDA without compile and fp8."""
    s.train('train-gpu-eager', compile=False, fp8=False)


@test('train-gpu-f1', gpu=True, needs=('data',))
def train_gpu_f1(s):
    """6 steps with the GPU defaults (bf16, torch.compile, fp8) and the planned F1 additions: GQA (1 KV head for 2
    query heads, SDPA enable_gqa), the length input (n_remain 33 + progress levels n_progress 17, length jitter /
    slack, the length tables at 10x LR), RoPE, BOS/EOS; then generate.py --length-seconds from it with the EOS ramp
    and the hard stop (CUDA graph with the per-note (remain, progress) levels)."""
    out = s.train('train-gpu-f1', n_kv_head=1, n_remain=33, n_progress=17, remain_jitter=0.1, remain_slack=0.05,
                  remain_lr_mult=10, pos_emb='rope', special_tokens=True)
    if 'length tables:' not in out:
        raise Fail('no own optimizer group for the length tables')
    s.generate('gpu-f1', s.work / 'train-gpu-f1', '--length-seconds', 5, '--eos-ramp', 1, '--length-stop', 1.5)
    losses = val_losses(out)
    return f'val {losses[0][1]:.3f} -> {losses[-1][1]:.3f}'


@test('golden-gpu', gpu=True, needs=('train-gpu', 'train-gpu-eager'))
def golden_gpu(s):
    """train-gpu's and train-gpu-eager's val losses vs tests/assets/losses/<GPU name>.json (one file per GPU type,
    as torchtitan keeps them; both runs repeat to 4 decimals on the laptop)."""
    return golden(s, LOSSES / (re.sub(r'[^a-z0-9]+', '-', gpu_name().lower()).strip('-') + '.json'),
                  ['train-gpu', 'train-gpu-eager'])


@test('train-recipe', gpu=True, needs=('data',))
def train_recipe(s):
    """config/pretrain/long_450m.py at 2 x 96 on the multi-instrument store: Muon (bf16 momentum), WSD, RoPE, RMSNorm,
    SwiGLU, QK-norm, instrument head + instrument-set and density conditions, BOS/EOS, packed short files, fp8 with
    cached weights, all blocks checkpointed, bf16 checkpoint format."""
    s.train('train-recipe', config=['config/pretrain/long_450m.py'], data='multi', **RECIPE)
    if not (s.work / 'train-recipe' / 'model_bf16.pt').exists():
        raise Fail('no model_bf16.pt')


@test('train-text', gpu=True, needs=('train-recipe',))
def train_text(s):
    """config/pilots/text_pilot.py with T2 captions (text_prefix 16) from train-recipe's weights: frozen LaBSE,
    captions from the label tables (labeled + unlabeled files), validation with true / shuffled / no captions."""
    w = s.work
    out = s.train('train-text', config=TEXT, init_ckpt=w / 'train-recipe' / 'model_bf16.pt', text_prefix=16,
                  csv_path=f'store:{w}/multi#labeled,store:{w}/multi#unlabeled', val_csv_path=f'store:{w}/multi#labeled',
                  val2_csv_path=f'store:{w}/multi#unlabeled', captions=w / 'labels', save_iters='')
    gain = re.findall(r'text gain ([-+][\d.]+)', out)
    if not gain:
        raise Fail('no text gain in the validation output')
    return f'text gain {gain[0]} -> {gain[-1]}'


@test('train-sections', gpu=True, needs=('train-text',))
def train_sections(s):
    """The section-caption stack of the sweeps (text_pilot + section_pilot + section_long + section_v2: change-aware
    section captions arriving a section early, oversampled change windows) from train-text, 32-note sections."""
    w = s.work
    out = s.train('train-sections', config=SECTIONS, init_ckpt=w / 'train-text' / 'model_bf16.pt',
                  csv_path=f'store:{w}/multi#labeled,store:{w}/multi#unlabeled,store:{w}/syn#labeled',
                  val_csv_path=f'store:{w}/multi#labeled', val2_csv_path=f'store:{w}/multi#unlabeled',
                  captions=w / 'labels', section_len=32, section_lead=32, save_iters='', ckpt_interval_min=0)
    changes = re.findall(r'multi_train: ([\d,]+) section changes', out)
    return f'{changes[0]} section changes in multi_train' if changes else ''


def f1_train(s, name, init_ckpt, *overlays, check=True, **kw):
    """config/f1/f1.py (+ overlays) from init_ckpt on the synthetic stores, 32-note sections (caption repeated every 8
    notes, a quarter as in F1), GQA into 1 KV head; check=False: no val-loss check (a transfer stage trains on another
    loss)."""
    w = s.work
    config = ['config/f1/f1.py', *overlays]
    kw = dict(init_ckpt=init_ckpt, csv_path=f'store:{w}/multi#labeled,store:{w}/syn#labeled,store:{w}/multi#unlabeled',
              val_csv_path=f'store:{w}/multi#labeled', val2_csv_path=f'store:{w}/multi#unlabeled', val_extra='',
              captions=w / 'labels', section_len=32, section_repeat=8, n_kv_head=1, gqa_calib_batches=2, **kw)
    if check:
        return s.train(name, config=config, **kw)
    over = {**TINY, 'out_dir': w / name, 'cache_dir': w, **kw}
    return s.run([PY, 'train.py', *config, *(f'--{k}={v}' for k, v in over.items())])


def f1_checks(out, s, name):
    missing = [m for m in ('AdamW8bit', 'length tables:', 'text gain', 'section changes') if m not in out]
    if missing:
        raise Fail(f'not in the output: {missing}')
    s.generate(name, s.work / name, '--text', 'a short loud piece with drums', '--cfg-scale', 2, '--length-seconds', 5,
               '--eos-ramp', 1, '--length-stop', 1.5)
    losses = val_losses(out)
    return f'val {losses[0][1]:.3f} -> {losses[-1][1]:.3f}'


@test('train-f1', gpu=True, needs=('train-recipe',))
def train_f1(s):
    """config/f1/f1.py from train-recipe's weights (the tiny long_450m): T2 text prefix, change-aware section captions
    with splice + repeat and attributes v2, the length input v2 with length words, GQA by the pca merge, 8-bit AdamW
    with the length tables' group, bf16 snapshots every 3 steps; a resume of its ckpt.pt (8-bit AdamW states) to step
    9; then generate.py with a caption, guidance and a length."""
    out = f1_train(s, 'train-f1', s.work / 'train-recipe' / 'model_bf16.pt', save_every=3, ckpt_interval_min=0.01)
    if 'gqa_init pca: 2 layers merged' not in out:
        raise Fail('no pca merge in the output')
    if not (s.work / 'train-f1' / 'model_bf16_it3.pt').exists():
        raise Fail('no snapshot model_bf16_it3.pt (save_every)')
    res = f1_train(s, 'train-f1', None, check=False, init_from='resume', max_iters=9, lr_decay_iters=9,
                   save_every=3, ckpt_interval_min=0.01)
    if 'resume: loading' not in res or 'step 8:' not in res or ckpt_iter(s.work / 'train-f1' / 'ckpt.pt') != 9:
        raise Fail('no resume of the 8-bit AdamW ckpt.pt to step 9')
    return f1_checks(out, s, 'train-f1')


@test('train-f1-lw', gpu=True, needs=('train-recipe',))
def train_f1_lw(s):
    """The GQA 'lw' path of F1: config/f1/f1_gqa_s1.py (pca merge + per-layer attention transfer, eager) from
    train-recipe's weights, then config/f1/f1.py from its ckpt.pt (already 1 KV head: no merge again)."""
    s1 = f1_train(s, 'train-f1-s1', s.work / 'train-recipe' / 'model_bf16.pt', 'config/f1/f1_gqa_s1.py', check=False,
                  ckpt_interval_min=0.01)
    if 'gqa_init pca: 2 layers merged' not in s1:
        raise Fail('no pca merge in stage 1')
    transfer = [float(x) for x in re.findall(r'step [1-9]\d*: train loss ([\d.]+)', s1)]
    if not transfer or not all(math.isfinite(x) for x in transfer):
        raise Fail(f'stage-1 train (transfer) losses {transfer}')
    out = f1_train(s, 'train-f1-lw', s.work / 'train-f1-s1' / 'ckpt.pt', save_every=0)
    if 'mean-pooled' in out or 'layers merged' in out:
        raise Fail('F1 merged the converted checkpoint again')
    return f1_checks(out, s, 'train-f1-lw')


@test('profile-f1', gpu=True, needs=('train-recipe',))
def profile_f1(s):
    """train.py --profile_steps on config/f1/f1.py: steps timed as they run, per phase (forward, data + captions,
    backward, clip, optimizer), one step under torch.profiler; out_dir/profile.txt with the kernel and op tables."""
    out_dir = s.work / 'profile-f1'
    f1_train(s, 'profile-f1', s.work / 'train-recipe' / 'model_bf16.pt', check=False, profile_steps=2, profile_wait=1,
             max_iters=100, lr_decay_iters=100)
    text = (out_dir / 'profile.txt').read_text() if (out_dir / 'profile.txt').exists() else ''
    missing = [m for m in ('notes/s', 'per phase', 'forward', 'data', 'optimizer', 'CUDA kernel time', 'self CPU')
               if m not in text]
    if missing:
        raise Fail(f'not in profile.txt: {missing}')
    return re.search(r'notes as they run: (.*?),', text).group(1)


@test('resume', gpu=True, needs=('train-gpu',))
def resume(s):
    """train-gpu resumed from its ckpt.pt to step 9 (checkpoint save / load)."""
    out = s.train('train-gpu', ckpt_interval_min=0.01, init_from='resume', max_iters=9, lr_decay_iters=9)
    if 'resume: loading' not in out or 'step 8:' not in out:
        raise Fail('no resume message or no eval at step 8')
    if ckpt_iter(s.work / 'train-gpu' / 'ckpt.pt') != 9:
        raise Fail('ckpt.pt not at iter 9')


@test('sigterm', gpu=True, needs=('resume',))
def sigterm(s):
    """SIGTERM while training (what the cloud wrapper sends): finish the step, save ckpt.pt, exit 143
    (cloud/setup/preflight.py smoke 3/4)."""
    before = ckpt_iter(s.work / 'train-gpu' / 'ckpt.pt')
    over = {**TINY, 'out_dir': s.work / 'train-gpu', 'csv_path': f'store:{s.work}/syn', 'cache_dir': s.work,
            'init_from': 'resume', 'max_iters': 10**6, 'lr_decay_iters': 10**6, 'eval_interval': 10**6,
            'ckpt_interval_min': 0.01}
    log = s.work / 'sigterm.out'
    with open(log, 'w') as f:
        p = subprocess.Popen([PY, 'train.py', *(f'--{k}={v}' for k, v in over.items())], cwd=ROOT, stdout=f,
                             stderr=subprocess.STDOUT)
    sent = False
    while p.poll() is None and time.time() < s.deadline - 60:
        steps = [int(n) for n in re.findall(r'(\d+)/1000000', log.read_text(errors='replace'))]
        if steps and steps[-1] >= before + 3:
            p.send_signal(signal.SIGTERM)
            sent = True
            break
        time.sleep(0.5)
    try:
        code = p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
        code = 'timeout'
    with open(s.log, 'a') as f:
        f.write(log.read_text(errors='replace') + f'\n[exit {code}]\n')
    after = ckpt_iter(s.work / 'train-gpu' / 'ckpt.pt')
    if not sent or code != 143 or after < before + 3:
        raise Fail(f'signal sent {sent}, exit {code} (want 143), ckpt iter {before} -> {after}')
    return f'saved at iter {after}, exit 143'


@test('finetune', gpu=True, needs=('train-gpu',))
def finetune(s):
    """init_from=finetune from train-gpu's ckpt.pt with BOS/EOS added (pitch table grown 128 -> 130)."""
    s.train('finetune', init_from='finetune', init_ckpt=s.work / 'train-gpu' / 'ckpt.pt', special_tokens=True)


@test('generate-gpu', gpu=True, needs=('train-gpu',))
def generate_gpu(s):
    """generate.py on CUDA (bf16, CUDA graph for the one-note step)."""
    s.generate('gpu', s.work / 'train-gpu')


@test('generate-no-graph', gpu=True, needs=('generate-gpu',))
def generate_no_graph(s):
    """--no-cuda-graph gives the same notes as the CUDA graph (generate.py's documented invariant)."""
    files = s.generate('no-graph', s.work / 'train-gpu', '--no-cuda-graph')
    if not same_files(files, sorted((s.work / 'gen' / 'gpu').glob('g_*.mid'))):
        raise Fail('MIDI differs from generate-gpu')
    return 'same MIDI'


@test('generate-cpu', needs=('train-cpu',))
def generate_cpu(s):
    """generate.py on the CPU from train-cpu."""
    s.generate('cpu', s.work / 'train-cpu', '--device', 'cpu')


@test('finetune-cpu-f1', needs=('train-cpu',))
def finetune_cpu_f1(s):
    """init_from=finetune from train-cpu on the CPU into the planned F1 additions: GQA (2 heads mean-pooled into 1 KV
    head) and the length input (n_remain 33 + n_progress 17, zero-init, seconds to the piece's end and from its start
    from the loader), with BOS/EOS."""
    from musicar.model.checkpoint import resolve_checkpoint
    out = s.train('finetune-cpu-f1', init_from='finetune', init_ckpt=resolve_checkpoint(s.work / 'train-cpu'),
                  special_tokens=True, n_kv_head=1, n_remain=33, n_progress=17, **CPU)
    if 'K / V mean-pooled to 1 heads' not in out or 'transformer.cond_remain.weight' not in out \
            or 'transformer.cond_progress.weight' not in out:
        raise Fail('no K / V pooling or no new cond_remain / cond_progress in the finetune output')
    losses = val_losses(out)
    return f'val {losses[0][1]:.3f} -> {losses[-1][1]:.3f}'


@test('gqa-transfer-cpu', needs=('train-cpu',))
def gqa_transfer_cpu(s):
    """MHA -> GQA conversion layer by layer (musicar/model/gqa.py) from train-cpu on the CPU: gqa_init='pca' (merge
    from the input moments of 2 calibration batches through the teacher stream) + attn_transfer (each layer learns its
    own full attention; the loss is their summed relative MSE), then the normal finetune from its ckpt.pt (already 1
    KV head: no pooling again)."""
    from musicar.model.checkpoint import resolve_checkpoint
    over = {**TINY, **CPU, 'out_dir': s.work / 'gqa-transfer-cpu', 'csv_path': f'store:{s.work}/syn',
            'cache_dir': s.work, 'init_from': 'finetune', 'init_ckpt': resolve_checkpoint(s.work / 'train-cpu'),
            'n_kv_head': 1, 'gqa_init': 'pca', 'gqa_calib_batches': 2, 'attn_transfer': True, 'learning_rate': 1e-3,
            'ckpt_interval_min': 0.01}
    out = s.run([PY, 'train.py', *(f'--{k}={v}' for k, v in over.items())])
    if 'gqa_init pca: 2 layers merged' not in out or 'K / V mean-pooled to 1 heads' not in out:
        raise Fail('no mean-pool + pca merge in the stage-1 output')
    transfer = [float(x) for x in re.findall(r'step [1-9]\d*: train loss ([\d.]+)', out)]
    if not transfer or not all(math.isfinite(x) for x in transfer):
        raise Fail(f'stage-1 train (transfer) losses {transfer}')
    out2 = s.train('gqa-stage2-cpu', init_from='finetune', init_ckpt=s.work / 'gqa-transfer-cpu' / 'ckpt.pt',
                   n_kv_head=1, **CPU)
    if 'mean-pooled' in out2:
        raise Fail('stage 2 pooled the converted checkpoint again')
    losses = val_losses(out2)
    return f'transfer {transfer[0]:.3f} -> {transfer[-1]:.3f}; stage 2 val {losses[0][1]:.3f} -> {losses[-1][1]:.3f}'


@test('generate-cpu-length', needs=('finetune-cpu-f1',))
def generate_cpu_length(s):
    """generate.py --length-seconds on the CPU from finetune-cpu-f1 (GQA cache, per-note remain + progress levels),
    with the EOS ramp and the hard stop (fade-out)."""
    s.generate('cpu-length', s.work / 'finetune-cpu-f1', '--device', 'cpu', '--length-seconds', 5, '--eos-ramp', 1,
               '--length-stop', 1.2)


@test('length-follow-cpu', needs=('finetune-cpu-f1',))
def length_follow_cpu(s):
    """eval/compare/length_follow.py end to end on finetune-cpu-f1 (2 lengths + 'none', 2 rows, MIDI of one row)."""
    out = s.work / 'length_follow'
    s.run([PY, 'eval/compare/length_follow.py', '--checkpoints', s.work / 'finetune-cpu-f1', '--lengths', '1,2',
           '--rows', 2, '--max-nps', 20, '--midi', 1, '--device', 'cpu', '--out', out])
    rows = (out / 'rows.csv').read_text().splitlines()
    if len(rows) != 1 + 3 * 2 or not (out / 'summary.md').exists():
        raise Fail(f'{len(rows) - 1} rows in rows.csv, expected 6')
    return f'{len(list(out.rglob("*.mid")))} MIDI'


@test('generate-recipe', gpu=True, needs=('train-recipe',))
def generate_recipe(s):
    """generate.py from train-recipe's model_bf16.pt (multi-instrument, RoPE, BOS/EOS)."""
    s.generate('recipe', s.work / 'train-recipe')


@test('generate-text', gpu=True, needs=('train-text',))
def generate_text(s):
    """generate.py --text (LaBSE) with classifier-free guidance 2.0 from train-text."""
    s.generate('text', s.work / 'train-text', '--text', 'a sad rock song with piano, bass and drums', '--cfg-scale', 2)


@test('export-bf16', gpu=True, needs=('generate-gpu',))
def export_bf16(s):
    """tools/checkpoints/export_bf16.py on train-gpu's best.pt, then generate.py from the export."""
    dst = s.work / 'export' / 'model_bf16.pt'
    dst.parent.mkdir()
    s.run([PY, 'tools/checkpoints/export_bf16.py', s.work / 'train-gpu' / 'best.pt', dst])
    files = s.generate('bf16', dst)
    same = same_files(files, sorted((s.work / 'gen' / 'gpu').glob('g_*.mid')))
    return f'{dst.stat().st_size / 1e6:.1f} MB, ' + ('same MIDI as from best.pt' if same else 'MIDI differs from best.pt')


@test('sample-search', gpu=True, needs=('train-recipe',))
def sample_search(s):
    """eval/settings/sample_search.py end to end (3 settings, 2 halving stages, 16 new notes) on train-recipe, prompts
    from the held-out GigaMIDI store (test data outside the repo, as OpenCV's opencv_extra; skipped without it). A
    refactor had left it crashing after the model load (NameError, fixed 2026-10-23)."""
    if not (HELDOUT / 'meta.json').exists():
        raise Skip(f'no {HELDOUT.relative_to(ROOT)}')
    out = s.work / 'sample_search'
    s.run([PY, 'eval/settings/sample_search.py', '--checkpoint', s.work / 'train-recipe', '--out', out, '--configs', 3,
           '--stages', '1,1', '--keep', 2, '--new-notes', 16, '--store', HELDOUT])
    if not (out / 'summary.md').exists():
        raise Fail('no summary.md')
    return 'summary.md written'


@test('audio', gpu=True)
def audio(s):
    """data_prep/songs/audio_to_piano.py --multi in .venv-audio (Demucs htdemucs_6s, basic-pitch, ADTOF) on a band
    clip rendered with fluidsynth from a known MIDI (band_midi), scored against that MIDI with MIREX metrics: raw pitch
    accuracy of the melody and bass lines, frame-level multi-pitch F1 of the piano stem and of the piano reduction,
    drum onset F1 (50 ms); floors in AUDIO_MIN (OpenCV's accuracy tests)."""
    if not AUDIO_PY.exists():
        raise Skip('no .venv-audio')
    if not shutil.which('fluidsynth') or not Path(SOUNDFONT).exists():
        raise Skip('no fluidsynth or soundfont')
    import pretty_midi
    d = s.work / 'audio'
    (d / 'in').mkdir(parents=True)
    src = band_midi()
    src.write(str(d / 'in' / 'band.mid'))
    s.run([PY, '-c', RENDER, d / 'in' / 'band.mid', SOUNDFONT])
    (d / 'in' / 'band.mid').rename(d / 'band.mid')
    s.run([AUDIO_PY, 'data_prep/songs/audio_to_piano.py', '--input', d / 'in', '--output', d / 'out', '--multi',
           '--cpu-workers', 2])
    multi = {('drums' if i.is_drum else i.name): i.notes for i in pretty_midi.PrettyMIDI(str(d / 'out' / 'multi' / 'band.mid')).instruments}
    reduction = [n for i in pretty_midi.PrettyMIDI(str(d / 'out' / 'midi' / 'band.mid')).instruments for n in i.notes]
    ref = {('drums' if i.is_drum else i.program): i.notes for i in src.instruments}
    end = src.get_end_time() + 1
    got = dict(melody=raw_pitch_accuracy(ref[53], multi.get('vocals', []), end),
               bass=raw_pitch_accuracy(ref[33], multi.get('bass', []), end),
               piano=frame_f1(ref[0], multi.get('piano', []), end),
               drums=onset_f1(ref['drums'], multi.get('drums', [])),
               reduction=frame_f1(ref[0] + ref[33] + ref[53], reduction, end))
    note = ', '.join(f'{k} {v:.2f}' for k, v in got.items())
    low = [k for k, v in got.items() if v < AUDIO_MIN[k]]
    if low:
        raise Fail(f'{note}; below the floor: {", ".join(f"{k} < {AUDIO_MIN[k]}" for k in low)}')
    return note


@test('perf', gpu=True)
def perf(s):
    """tools/bench/bench.py on its default 12 x 512 model (~42M), micro 12, compile + fp8, 8 steps: OpenCV's perf
    tests in sanity mode (must run; the speed is recorded in logs/ci/perf.jsonl, not judged)."""
    out = s.run([PY, 'tools/bench/bench.py', '--micro', 12, '--compile', 1, '--fp8', 1, '--steps', 8])
    m = re.search(r'^\s*12\s+1\s+\S+\s+1\s+\S+\s+\S+\s+\S+\s+([\d,]+)\s+(\d+)\s+([\d.]+)\s*$', out, re.M)
    if not m:
        raise Fail('no result row (out of memory?)')
    tps, peak = int(m.group(1).replace(',', '')), float(m.group(3))
    prev = [json.loads(ln) for ln in PERF_LOG.read_text().splitlines()] if PERF_LOG.exists() else []
    PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_LOG, 'a') as f:
        f.write(json.dumps(dict(time=time.strftime('%F %T'), commit=commit(), tok_s=tps, peak_gb=peak)) + '\n')
    vs = f' ({tps / prev[-1]["tok_s"] - 1:+.0%} vs {prev[-1]["commit"]})' if prev else ''
    return f'{tps:,} notes/s{vs}, peak {peak} GB'


@test('docs', timeout=900)
def docs(s):
    """pdflatex docs/figures/architecture.tex in a scratch copy (OpenCV builds its docs as a separate job)."""
    if not shutil.which('pdflatex'):
        raise Skip('no pdflatex on PATH')
    d = s.work / 'docs'
    shutil.copytree(ROOT / 'docs' / 'figures', d)
    s.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'architecture.tex'], cwd=d)
    return f'{(d / "architecture.pdf").stat().st_size / 1e6:.1f} MB pdf'


# ---------------------------------------------------------------- runner

def check_imports():
    """--check-imports (run by the 'imports' test in its own process): the musicar imports that don't resolve."""
    import importlib
    import importlib.util
    files = subprocess.run(['git', 'ls-files', '*.py'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    files = [f for f in files if not f.startswith('legacy/')]
    bad, n = [], 0
    for f in files:
        for node in ast.walk(ast.parse((ROOT / f).read_text(), f)):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or '').split('.')[0] == 'musicar':
                wanted = [(node.module, a.name) for a in node.names if a.name != '*']
            elif isinstance(node, ast.Import):
                wanted = [(a.name, None) for a in node.names if a.name.split('.')[0] == 'musicar']
            else:
                continue
            for mod, name in wanted:
                n += 1
                try:
                    m = importlib.import_module(mod)
                except Exception as e:
                    bad.append(f'{f}:{node.lineno}: {mod}: {type(e).__name__}: {e}')
                    continue
                if name and not hasattr(m, name) and importlib.util.find_spec(f'{mod}.{name}') is None:
                    bad.append(f'{f}:{node.lineno}: {mod} has no {name}')
    print(f'{n} musicar imports in {len(files)} files resolve' if not bad else f'{len(bad)} of {n} musicar imports fail')
    print('\n'.join(bad))
    sys.exit(1 if bad else 0)


def commit():
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '-uno'], cwd=ROOT, capture_output=True, text=True).stdout
    return sha + ('+dirty' if dirty.strip() else '')


def gpu_problem():
    """Why the GPU tests can't run here, or None."""
    if not shutil.which('nvidia-smi'):
        return 'no nvidia-smi'
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                       capture_output=True, text=True)
    if r.returncode:
        return 'nvidia-smi failed'
    apps = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return 'GPU busy: ' + '; '.join(apps) if apps else None


def select(patterns):
    """Tests matching any pattern, plus everything they need, in plan order."""
    by_name = {t['name']: t for t in TESTS}
    chosen = {t['name'] for t in TESTS if not patterns or any(p in t['name'] for p in patterns)}
    todo = list(chosen)
    while todo:
        for n in by_name[todo.pop()]['needs']:
            if n not in chosen:
                chosen.add(n)
                todo.append(n)
    return [t for t in TESTS if t['name'] in chosen]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-k', default='', help="comma-separated substrings of test names ('' = all)")
    p.add_argument('--list', action='store_true', help='print the plan and exit')
    p.add_argument('--cpu-only', action='store_true', help='skip the GPU tests')
    p.add_argument('--gpu-busy-ok', action='store_true', help='run the GPU tests next to another GPU process')
    p.add_argument('--update-golden', action='store_true', help='rewrite the stored losses in tests/assets/losses/')
    p.add_argument('--keep', action='store_true', help='keep the checkpoints in the run directory')
    p.add_argument('--timeout', type=float, default=1.0, help='multiplier on the per-test timeouts')
    p.add_argument('--check-imports', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.check_imports:
        check_imports()
    tests = select([x for x in args.k.split(',') if x])
    if args.list:
        for t in tests:
            needs = f' [needs {", ".join(t["needs"])}]' if t['needs'] else ''
            print(f'{t["name"]:18s} {"gpu" if t["gpu"] else "cpu"}{needs}\n    {t["doc"]}')
        return

    run_dir = ROOT / 'logs' / 'ci' / time.strftime('%Y%m%d_%H%M%S')
    s = Session(run_dir, args)
    s.work.mkdir(parents=True)
    no_gpu = 'skipped (--cpu-only)' if args.cpu_only else gpu_problem()
    if no_gpu and no_gpu.startswith('GPU busy') and args.gpu_busy_ok:
        no_gpu = None
    sha = commit()
    print(f'CI session {run_dir.relative_to(ROOT)}  commit {sha}  {len(tests)} tests'
          + (f'  (GPU tests: {no_gpu})' if no_gpu else ''))
    results, t_start = {}, time.time()
    state = dict(commit=sha, started=run_dir.name, gpu=no_gpu or 'ok', minutes=0.0, finished=False, running=None,
                 pid=os.getpid(), boot=ci_report.boot_id(),
                 plan=[dict(name=t['name'], gpu=t['gpu'], needs=list(t['needs']), doc=t['doc']) for t in tests],
                 results=results)

    def save(running=None, finished=False):
        """summary.json + the reports, after every test (a broken report never stops the session)."""
        state.update(minutes=round((time.time() - t_start) / 60, 1), running=running, finished=finished)
        (run_dir / 'summary.json').write_text(json.dumps(state, indent=1) + '\n')
        try:
            ci_report.write_run(run_dir, state)
        except Exception as e:
            print(f'  (report not written: {type(e).__name__}: {e})', flush=True)

    ci_report.link_latest(run_dir)
    save()
    print(f'  report: {(run_dir / "report.html").relative_to(ROOT)}')
    try:
        for t in tests:
            name = t['name']
            save(running=name)
            print(f'  {name:20s} ', end='', flush=True)
            s.log, s.deadline = run_dir / f'{name}.log', time.time() + t['timeout'] * args.timeout
            missing = [n for n in t['needs'] if results.get(n, {}).get('status') != 'PASS']
            t0, status, note, tb = time.time(), 'PASS', '', None
            try:
                if t['gpu'] and no_gpu:
                    raise Skip(no_gpu)
                if missing:
                    raise Skip('needs ' + ', '.join(missing))
                note = t['fn'](s) or ''
            except Skip as e:
                status, note = 'SKIP', str(e)
            except Fail as e:
                status, note = 'FAIL', str(e)
            except Exception as e:
                status, note, tb = 'FAIL', f'{type(e).__name__}: {e}', traceback.format_exc()
            secs = time.time() - t0
            results[name] = dict(status=status, seconds=round(secs, 1), note=note, **({'traceback': tb} if tb else {}))
            save()
            print(f'{status}  {secs:6.1f} s  {note}', flush=True)
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        if not args.keep:
            for f in s.work.rglob('*.pt'):
                f.unlink()
    counts = {k: sum(r['status'] == k for r in results.values()) for k in ('PASS', 'FAIL', 'SKIP')}
    save(finished=True)
    print(f'\n{counts["PASS"]} passed, {counts["FAIL"]} failed, {counts["SKIP"]} skipped in '
          f'{(time.time() - t_start) / 60:.1f} min; logs in {run_dir.relative_to(ROOT)}/')
    print(f'report: {(run_dir / "report.html").relative_to(ROOT)} (all runs: logs/ci/index.html)')
    for n, r in results.items():
        if r['status'] == 'FAIL':
            print(f'  FAIL {n}: {run_dir.relative_to(ROOT)}/{n}.log')
    sys.exit(1 if counts['FAIL'] else 0)


if __name__ == '__main__':
    main()
