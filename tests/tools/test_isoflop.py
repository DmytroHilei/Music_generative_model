"""tools/sweeps/isoflop.py and isoflop_fit.py without training: the grid's iterations and resume points (master plan
3(C)), commands (chaining flags, overrides), queue lines, gates on a fake runs.json, the synthetic fit.
Run: python tests/tools/test_isoflop.py (or pytest tests/)."""

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helpers import run  # noqa: E402
from tools.sweeps import isoflop, isoflop_fit  # noqa: E402

PLAN = {  # id: (iters, resumes at), master plan 3(C)
    '01': (25944, 0), '02': (20377, 0), '03': (14609, 0), '04': (10506, 0), '05': (7693, 0), '06': (5613, 0),
    '07': (48697, 11688), '08': (35019, 8405), '09': (25643, 6155), '10': (18711, 4491), '11': (14252, 0),
    '12': (10548, 0), '13': (56132, 14969), '14': (42757, 11402), '15': (31643, 8439), '16': (23531, 0),
    '17': (17974, 0), '23': (11226, 0), '27': (9766, 3907), '34': (76928, 20515), '28': (51888, 0), 'O1': (14312, 0)}


def test_grid_iterations_and_chaining():
    for id, (it, st) in PLAN.items():
        a = isoflop.ARMS[id]
        assert (isoflop.iters(a), isoflop.start(a)) == (it, st), id
    assert max(isoflop.iters(a) * a['batch'] for a in isoflop.ARMS.values()
               if a['store'] == isoflop.POOL and '--one_pass_epochs=2' not in a['extra']) == 898_112


def last(cmd, key):
    return [c for c in cmd if c.startswith(f'--{key}=')][-1].split('=', 1)[1]


def test_commands():
    c = isoflop.command(isoflop.ARMS['07'])
    assert '--init_from=resume' in c and last(c, 'save_pre_cooldown') == 'True' and last(c, 'max_iters') == '48697'
    assert last(c, 'one_pass') == 'True' and last(c, 'boundary_frac') == '0' and last(c, 'eval_dense') == '250'
    assert last(c, 'eval_interval') == str(48697 // 8)
    c = isoflop.command(isoflop.ARMS['22p'])
    assert (last(c, 'max_iters'), last(c, 'lr_decay_iters'), last(c, 'eval_interval')) == ('4678', '18711', '4677')
    c = isoflop.command(isoflop.ARMS['10'])
    assert (last(c, 'eval_dense'), last(c, 'eval_dense_until')) == ('4677', '4677')
    c = isoflop.command(isoflop.ARMS['23'])
    assert (last(c, 'batch_size'), last(c, 'warmup_iters'), last(c, 'eval_iters')) == ('8', '200', '150')
    assert last(isoflop.command(isoflop.ARMS['26']), 'train_file_frac') == '0.25'
    assert last(isoflop.command(isoflop.ARMS['18']), 'learning_rate') == '0.002'
    q = isoflop.queue_line(isoflop.ARMS['13'])
    assert q.startswith('min=') and ' need=checkpoints/iso_10_10x448/pre_cooldown.pt tier=ESS id=iso_13 :: ' in q
    assert 'need=data/cache/gigamidi_pool_train/meta.json' in isoflop.queue_line(isoflop.ARMS['01'])


def fake_runs(d, ce):
    runs = {id: dict(id=id, ce=v, curve=[(4677, v + 0.1, None), (isoflop.iters(isoflop.ARMS[id]) - 1, v, None)])
            for id, v in ce.items()}
    isoflop.RUNS = Path(d) / 'runs.json'
    isoflop.SKIP_IDS = Path(d) / 'skip.txt'
    isoflop.RUNS.write_text(json.dumps(runs))


def test_gates():
    isoflop.cleanup_stems = lambda: None  # never touch checkpoints/ from a test
    with tempfile.TemporaryDirectory() as d:
        base = {'01': 2.10, '02': 2.05, '03': 2.02, '04': 2.01, '05': 2.03, '06': 2.07}
        fake_runs(d, {**base, '18': 2.05, '20': 2.08, '23': 2.04, '21': 2.031, '05': 2.03})
        q = Path(d) / 'queue.txt'
        q.write_text('min=1 need=x tier=ESS id=other :: true\n')
        args = Namespace(append=str(q), delete_kept=False)
        isoflop.gate('1', args)
        text = q.read_text()
        assert text.startswith('min=1 need=x') and 'id=iso_29 ' in text and 'id=iso_33 ' in text  # |t| = 0.06
        assert 'id=iso_28 ' in text and 'id=iso_G1a' not in text and 'id=iso_O2a' not in text
        fake_runs(d, {'22p': 2.0, '10': 2.0})
        assert isoflop.gate('chain', args)
        fake_runs(d, {'22p': 2.0, '10': 2.02})
        assert not isoflop.gate('chain', args)
        # 1e17 vertex at ~45M: r* = 1e17 / (6 N*) / N* ~ 8 < 15 -> #34 skipped
        fake_runs(d, {'07': 2.2, '08': 2.1, '09': 2.03, '10': 1.99, '11': 1.97, '12': 1.965})
        isoflop.gate('2', Namespace(append=None, delete_kept=False))
        assert '34' in isoflop.SKIP_IDS.read_text()


def test_skiplist():
    """tools/data/iso_skiplist.py: an exact and a transposed copy of an eval file are found, other files not."""
    import subprocess
    import numpy as np
    from helpers import make_store, notes
    with tempfile.TemporaryDirectory() as d:
        held = [notes(300, seed=i) for i in range(3)]
        hp = [np.zeros(300, np.uint8)] * 3
        make_store(Path(d) / 'held_validation', held, hp)
        up = held[1].copy()
        up[:, 0] += 3
        pool = [notes(300, seed=10 + i) for i in range(6)]
        pool[2], pool[4] = held[0], up
        make_store(Path(d) / 'pool_train', pool, [np.zeros(300, np.uint8)] * 6)
        out = Path(d) / 'skip.txt'
        root = Path(__file__).resolve().parents[2]
        r = subprocess.run([sys.executable, 'tools/data/iso_skiplist.py', f'--pool={d}/pool_train',
                            f'--heldout={d}/held_validation', '--cont96=', '--eval-min=100', '--workers=1',
                            f'--out={out}'], cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert out.read_text().split() == ['2', '4'], out.read_text()


def test_fit_synthetic():
    isoflop_fit.synthetic()


if __name__ == '__main__':
    run(globals())
