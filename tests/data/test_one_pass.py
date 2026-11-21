"""One-pass window plan (musicar/data/loader.py one_pass_plan, NoteWindowDataset one_pass, PlanSampler; train.py
--one_pass): every target row covered once per epoch, deterministic, a resumed stream equal to the uninterrupted one,
and the default path (one_pass off) identical to the code before it (stored digests / losses of 94097c7).
Run: python tests/data/test_one_pass.py (or pytest tests/)."""

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import ROOT, make_store, run  # noqa: E402
from musicar.data.loader import (BOS_PITCH, EOS_PITCH, NoteWindowDataset, PlanSampler, collate_windows,  # noqa: E402
                                 one_pass_plan)

T = 32


def expand(plan, L, T):
    """Per window: list of (file, virtual row) of its kept targets, rebuilt from the plan arrays alone."""
    out = []
    for w in plan['plan']:
        if w >= 0:
            f, v = plan['full_file'][w], plan['full_v'][w]
            out.append([(f, r) for r in range(v + 1, v + T + 1)])
            continue
        s, e = plan['packed_start'][-1 - w], plan['packed_end'][-1 - w]
        tg = []
        for k in np.flatnonzero((plan['piece_start'] < e) & (plan['piece_start'] + plan['piece_n'] > s)):
            s0, f, a, n = plan['piece_start'][k], plan['piece_file'][k], plan['piece_a'][k], plan['piece_n'][k]
            for p in range(max(s, s0), min(e, s0 + n)):
                if p == s or (p == s0 and not plan['piece_join'][k]):
                    continue
                tg.append((f, a + p - s0))
        out.append(tg)
    return out


def test_plan_coverage():
    rng = np.random.default_rng(1)
    L = np.concatenate([rng.integers(66, 200, 300), [T + 1, T + 2, 2 * T, 2 * T + 1, 66, 67]])
    skip = np.zeros(len(L), bool)
    skip[[3, 10]] = True
    p2 = one_pass_plan(L, T, seed=5, epochs=2, phase0=0.2, skip=skip)
    p1 = one_pass_plan(L, T, seed=5, epochs=1, phase0=0.2, skip=skip)
    n1 = len(p1['plan'])
    assert all((p2[k][:len(p1[k])] == p1[k]).all() for k in p1), 'more epochs must extend the plan'
    tg = expand(p2, L, T)
    for e, part in enumerate((tg[:n1], tg[n1:])):
        rows = [t for w in part for t in w if t[1] > 0]
        want = {(f, r) for f in range(len(L)) if not skip[f] for r in range(1, L[f])}
        assert len(rows) == len(set(rows)) == len(want) == int((L[~skip] - 1).sum()), f'epoch {e}'
        assert set(rows) == want
        assert all(len(w) <= T for w in part)
    # phase 0 share and opening windows: tiled windows from v = 0 exist, and most long files start elsewhere
    full_v = p1['full_v']
    assert 0 < (full_v == 0).sum() < (L >= T + 1).sum()
    assert one_pass_plan(L, T, seed=5, epochs=1, phase0=0.2, skip=skip)['plan'].tolist() == p1['plan'].tolist()
    assert one_pass_plan(L, T, seed=6, epochs=1, phase0=0.2, skip=skip)['plan'].tolist() != p1['plan'].tolist()


def id_store(d, n_files=40, seed=0):
    """Notes that name themselves: delta_time = file index, duration = note index (pitch 60 + note % 12)."""
    rng = np.random.default_rng(seed)
    files = []
    for f in range(n_files):
        n = int(rng.choice([10, 30, 70, 100, 200]) + rng.integers(0, 20))
        files.append(np.stack([60 + np.arange(n) % 12, np.full(n, 5), np.arange(n), np.full(n, f)], 1))
    make_store(Path(d) / 'ids_train', files, [np.full(len(f), 0, np.uint8) for f in files])
    return files


def dataset(d, **kw):
    return NoteWindowDataset(f'store:{d}/ids', '.', 'train', T, cache_dir=f'{d}/cache', special_tokens=True,
                             pack_short=True, min_notes=8, programs=True, augment=kw.pop('augment', False),
                             boundary_frac=0.0, one_pass=True, **kw)


def targets(ds):
    """Kept targets of every item: ('note', file, index) / ('eos', file) / ('bos',) + checks of the masks."""
    seen = []
    for i in range(len(ds)):
        x, y = ds[i]
        assert x[0].shape == (T,) and all(t.shape == (T,) for t in y)
        for j in range(T):
            if y[0][j] < 0:
                assert all(t[j] < 0 for t in y)
                continue
            if y[0][j] == BOS_PITCH:
                assert x[0][j] == EOS_PITCH, 'BOS kept only after an EOS'
                seen.append(('bos',))
            elif y[0][j] == EOS_PITCH:
                seen.append(('eos', int(x[3][j])))
            else:
                f, k = int(y[3][j]), int(y[2][j])
                if k > 0:  # the previous row of a note target is its own file's note (or BOS for note 0)
                    assert int(x[3][j]) == f and int(x[2][j]) == k - 1
                else:
                    assert x[0][j] == BOS_PITCH
                seen.append(('note', f, k))
    return seen


def test_dataset_covers_once_and_skip():
    with tempfile.TemporaryDirectory() as d:
        files = id_store(d)
        ds = dataset(d, one_pass_epochs=2)
        seen = targets(ds)
        kept = [f for f in range(len(files)) if len(files[f]) >= 8]
        notes = [t for t in seen if t[0] == 'note']
        eos = [t for t in seen if t[0] == 'eos']
        assert sorted(notes) == sorted([('note', f, k) for f in kept for k in range(len(files[f]))] * 2)
        assert sorted(eos) == sorted([('eos', f) for f in kept] * 2)
        skip = Path(d) / 'skip.txt'
        skip.write_text('0\n5\n')
        seen = targets(dataset(d, one_pass_skip=str(skip)))
        assert not any(t[0] != 'bos' and t[1] in (0, 5) for t in seen)
        assert len([t for t in seen if t[0] == 'eos']) == len(kept) - 2


def batches(ds, start, n, workers=0):
    sampler = PlanSampler(len(ds))
    sampler.start = start
    dl = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=workers, drop_last=True,
                    collate_fn=lambda b: collate_windows(b, ('program',)))
    out = []
    for (x, extra), y in dl:
        out.append(torch.cat([*x, extra['program'], *y]))
        if len(out) == n:
            break
    return out


def test_deterministic_and_resume():
    with tempfile.TemporaryDirectory() as d:
        id_store(d)
        a, b = dataset(d, augment=True), dataset(d, augment=True)  # b reads the cached plan
        assert len(list((Path(d) / 'cache' / 'one_pass').glob('*.npz'))) == 1
        full = batches(a, 0, 10, workers=2)
        assert all((u == v).all() for u, v in zip(full, batches(b, 0, 10)))
        resumed = batches(b, 3 * 4, 7, workers=2)  # a restart at iteration 3 (batch 4, no accumulation)
        assert all((u == v).all() for u, v in zip(full[3:], resumed))
        assert not (full[0] == batches(dataset(d, augment=True, one_pass_seed=2), 0, 1)[0]).all()


# ---------------------------------------------------------------- the default path vs 94097c7

def multi_store(d):
    rng = np.random.default_rng(3)
    files, progs = [], []
    for i in range(24):
        n = int(rng.choice([20, 50, 90, 150]))
        files.append(np.stack([rng.integers(36, 90, n), rng.integers(0, 32, n), rng.integers(1, 60, n),
                               rng.integers(0, 30, n)], 1))
        progs.append(rng.choice([0, 25, 33, 128], n).astype(np.uint8))
    make_store(Path(d) / 'm_train', files, progs)
    make_store(Path(d) / 'm_validation', files[:6], progs[:6])
    make_store(Path(d) / 'p_train', [f[:, :4] for f in files])


def item_digest(d, cls=NoteWindowDataset):
    import random
    h = hashlib.sha256()
    cases = [dict(src='m', split='train', augment=True, special_tokens=True, pack_short=True, programs=True,
                  boundary_frac=0.1, aug_tempo=0.1, min_notes=16),
             dict(src='m', split='validation', eval_stride=T, special_tokens=True, programs=True),
             dict(src='p', split='train', augment=True)]
    for kw in cases:
        random.seed(0)
        torch.manual_seed(0)
        ds = cls(f"store:{d}/{kw.pop('src')}", '.', kw.pop('split'), T, cache_dir=f'{d}/cache', **kw)
        h.update(str(len(ds)).encode())
        for i in range(min(len(ds), 40)):
            x, y = ds[i]
            for t in (*x, *y):
                h.update(t.numpy().astype(np.int64).tobytes())
    return h.hexdigest()


TRAIN_ARGS = ['--device=cpu', '--compile=False', '--fp8=False', '--dtype=float32', '--n_layer=2', '--n_embd=32',
              '--n_head=2', f'--block_size={T}', '--batch_size=4', '--gradient_accumulation_steps=2', '--max_iters=7',
              '--eval_interval=2', '--eval_iters=2', '--special_tokens=True', '--pack_short=True', '--min_notes=16',
              '--n_programs=129', '--cond_inst=True', '--n_density=16', '--num_workers=0', '--wandb_log=False',
              '--optimizer_name=muon', '--lr_schedule=wsd', '--warmup_iters=2', '--lr_decay_iters=7',
              '--learning_rate=1e-3', '--dropout=0.0', '--save_pre_cooldown=False', '--checkpoint_format=bf16']


def train_lines(d, root=ROOT, extra=()):
    out = subprocess.run([sys.executable, 'train.py', f'--csv_path=store:{d}/m', f'--cache_dir={d}/cache',
                          f'--out_dir={d}/out', *TRAIN_ARGS, *extra], cwd=root, capture_output=True, text=True,
                         env={'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(root)})
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    return re.findall(r'step \d+: train loss \S+ val loss \S+ \| \S+ CE \S+ \([^)]*\)', out.stdout), out.stdout


# from 94097c7: python tests/data/test_one_pass.py --golden <worktree of 94097c7>
GOLDEN = {
    'items': 'cb69736ff024a9b07190f7998bbe60f947919378ed1eb4a881d5df3f48c082a4',
    'train': ['step 0: train loss nan, val loss 25.7125 | m CE 20.834 (pit 4.873 vel 3.469 dur 6.246 del 6.246)',
              'step 2: train loss 25.6973, val loss 25.5758 | m CE 20.797 (pit 4.868 vel 3.467 dur 6.242 del 6.220)',
              'step 4: train loss 25.5191, val loss 25.3373 | m CE 20.684 (pit 4.856 vel 3.465 dur 6.219 del 6.144)',
              'step 6: train loss 25.2964, val loss 25.0788 | m CE 20.504 (pit 4.826 vel 3.462 dur 6.169 del 6.047)'],
}


def test_default_path_unchanged():
    with tempfile.TemporaryDirectory() as d:
        multi_store(d)
        assert item_digest(d) == GOLDEN['items'], 'default-path dataset items changed'
        lines, _ = train_lines(d)
        assert lines == GOLDEN['train'], '\n'.join(lines)


def test_train_one_pass_resume():
    """train.py --one_pass: the plan line, the capacity assert, dense evals, and a resume from pre_cooldown.pt that
    starts at window iter * batch * accumulation."""
    with tempfile.TemporaryDirectory() as d:
        multi_store(d)
        one = ['--one_pass=True', '--boundary_frac=0', '--eval_interval=100', '--eval_dense=2', '--eval_dense_until=4']
        lines, out = train_lines(d, extra=one + ['--save_pre_cooldown=True'])
        assert 'one_pass plan:' in out and 'one_pass: windows 0 ..' in out, out[-1500:]
        assert [int(re.match(r'step (\d+)', s).group(1)) for s in lines] == [0, 2, 4, 6]
        n = int(re.search(r'one_pass plan: ([\d,]+) windows', out).group(1).replace(',', ''))
        bad = subprocess.run([sys.executable, 'train.py', f'--csv_path=store:{d}/m', f'--cache_dir={d}/cache',
                              f'--out_dir={d}/out2', *TRAIN_ARGS, *one, f'--max_iters={n // 8 + 1}'], cwd=ROOT,
                             capture_output=True, text=True, env={'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin',
                                                                  'PYTHONPATH': str(ROOT)})
        assert bad.returncode != 0 and 'raise one_pass_epochs' in bad.stderr
        Path(f'{d}/out3').mkdir()
        Path(f'{d}/out/pre_cooldown.pt').rename(f'{d}/out3/ckpt.pt')  # cooldown of 7 iters starts at 6
        _, out = train_lines(d, extra=one + ['--init_from=resume', '--max_iters=12', '--lr_decay_iters=12', '--one_pass_epochs=2',
                                             f'--out_dir={d}/out3'])
        assert 'one_pass: windows 48 .. 96' in out, out[-1500:]


if __name__ == '__main__':
    if sys.argv[1:2] == ['--golden']:  # print GOLDEN from another checkout (e.g. a worktree of 94097c7)
        base = Path(sys.argv[2]).resolve()
        sys.path.insert(0, str(base))
        import importlib.util
        spec = importlib.util.spec_from_file_location('base_loader', base / 'musicar/data/loader.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            multi_store(d)
            print(repr(dict(items=item_digest(d, mod.NoteWindowDataset), train=train_lines(d, base)[0])))
    else:
        run(globals())
