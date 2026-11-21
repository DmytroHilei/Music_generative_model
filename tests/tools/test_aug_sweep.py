"""aug_sweep's commands, run names, records, tables and gate (CPU, no training): python tests/tools/test_aug_sweep.py
(or pytest tests/)."""

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/sweeps'))

import aug_sweep as A  # noqa: E402

# tools/sweeps/aug_sweep.py at 9a8227d (the runs in results/aug_sweep/runs.json), micro 16
HEAD_CONTROL_42M = ['.venv/bin/python', 'train.py', 'config/pretrain/scale_ladder.py', '--n_layer=12', '--n_embd=512',
                    '--n_head=8', '--train_file_frac=0.25', '--seed=1337', '--max_iters=4883', '--lr_decay_iters=4883',
                    '--out_dir=checkpoints/aug_control_s1337', '--batch_size=16', '--gradient_accumulation_steps=1',
                    '--eval_iters=75', '--eval_interval=1220']
HEAD_FRESH_108M = ['.venv/bin/python', 'train.py', 'config/pretrain/scale_ladder.py', '--n_layer=14', '--n_embd=768',
                   '--n_head=12', '--train_file_frac=1.0', '--seed=2', '--max_iters=4883', '--lr_decay_iters=4883',
                   '--out_dir=checkpoints/aug_fresh_s2_108M', '--batch_size=16', '--gradient_accumulation_steps=1',
                   '--eval_iters=75', '--eval_interval=1220']
HEAD_SYNTH25_108M = ['.venv/bin/python', 'train.py', 'config/pretrain/scale_ladder.py', '--n_layer=14', '--n_embd=768',
                     '--n_head=12', '--train_file_frac=0.25,1', '--seed=1337', '--max_iters=4883',
                     '--lr_decay_iters=4883', '--out_dir=checkpoints/aug_synth25_s1337_108M',
                     '--csv_path=store:data/cache/gigamidi_replay,store:data/cache/synth_musicar_l',
                     '--source_weights=0.75,0.25', '--batch_size=16', '--gradient_accumulation_steps=1',
                     '--eval_iters=75', '--eval_interval=1220']
HEAD_DYN_6M_F0125_MICRO4 = ['.venv/bin/python', 'train.py', 'config/pretrain/scale_ladder.py', '--n_layer=6',
                            '--n_embd=256', '--n_head=4', '--train_file_frac=0.125', '--seed=2', '--max_iters=4883',
                            '--lr_decay_iters=4883', '--out_dir=checkpoints/aug_dyn_s2_f0.125_6M',
                            '--hypothesis=aug_dyn:shift=4', '--batch_size=4', '--gradient_accumulation_steps=4',
                            '--eval_iters=300', '--eval_interval=1220']


def _cmd(arm, seed=1337, frac=0.25, size='42M', micro=16, notes=A.NOTES, keep=False, save_iters=''):
    name = A.run_name(arm, seed, frac, size, notes)
    iters, _ = A.arm_iters(arm, notes)
    return A.command(arm, seed, frac, size, iters, A.out_dir(name, keep), A.run_args(arm, micro, iters, keep,
                                                                                      save_iters))


def test_existing_commands_identical_to_head():
    assert _cmd('control') == HEAD_CONTROL_42M
    assert _cmd('fresh', seed=2, size='108M') == HEAD_FRESH_108M
    assert _cmd('synth25', size='108M') == HEAD_SYNTH25_108M
    assert _cmd('dyn', seed=2, frac=0.125, size='6M', micro=4) == HEAD_DYN_6M_F0125_MICRO4


def test_existing_run_names_unchanged():
    assert A.run_name('control', 1337, 0.25) == 'control_s1337'
    assert A.run_name('fresh', 2, 0.125, '108M') == 'fresh_s2_108M'
    assert A.run_name('control', 2, 0.125, '6M') == 'control_s2_f0.125_6M'
    assert A.run_name('fresh', 1337, 0.0625, '20M') == 'fresh_s1337_20M'
    assert A.run_name('synth25', 2, 0.25, '108M') == 'synth25_s2_108M'
    assert A.run_name('family', 1337, 0.5) == 'family_s1337_f0.5'


def test_run_names_unique():
    names = {}
    for size in A.SIZES:
        for frac in (0.25, 0.125, 0.5, 0.0625):
            for notes in (A.NOTES, 80e6, 120e6, 320e6):
                for arm in A.ARMS:
                    for seed in (1337, 2, 3):
                        key = (arm, seed, A.arm_frac(arm, frac), size, A.arm_iters(arm, notes))
                        names.setdefault(A.run_name(arm, seed, frac, size, notes), set()).add(key)
    assert all(len(v) == 1 for v in names.values()), [k for k, v in names.items() if len(v) > 1]
    assert A.run_name('fresh', 1337, 0.25, notes=120e6) == 'fresh_s1337_n120M'
    assert A.run_name('control', 1337, 0.25, '6M', 320e6) == 'control_s1337_6M_n320M'
    assert A.run_name('disc_fresh120', 2, 0.25, notes=120e6) == 'disc_fresh120_s2'  # its own notes: no suffix


def test_new_arm_commands():
    c = _cmd('b16k')
    assert '--max_iters=9766' in c and '--batch_size=8' in c and '--gradient_accumulation_steps=1' in c
    assert '--eval_iters=150' in c and '--eval_interval=2441' in c and c[-1] == '--warmup_iters=200'
    c = _cmd('b8k', micro=4)
    assert '--max_iters=19531' in c and '--batch_size=4' in c and '--gradient_accumulation_steps=1' in c
    assert '--warmup_iters=400' in c
    c = _cmd('fresh_dyn', frac=0.125)
    assert '--train_file_frac=1.0' in c and '--hypothesis=aug_dyn:shift=4' in c
    assert '--out_dir=checkpoints/aug_fresh_dyn_s1337' in c
    c = _cmd('fresh_census', size='6M', notes=320e6)
    assert '--csv_path=store:data/cache/gigamidi_pool' in c and '--boundary_frac=0' in c and '--max_iters=9766' in c
    assert '--train_file_frac=0.088' in _cmd('sub168_census')
    c = _cmd('disc_fresh120')
    assert '--max_iters=3662' in c and '--train_file_frac=1.0' in c and A.PANEL in c
    assert A.PANEL == ('--val_extra=disc_aria=store:data/cache/cdisc_va;disc_dense=store:data/cache/cdisc_vd;'
                       'giga=store:data/cache/gigamidi_heldout')
    assert '--train_file_frac=0.25,1' in _cmd('synth_clean25')
    c = _cmd('control', size='217M', keep=True, save_iters='2440,3660')
    assert '--n_layer=16' in c and '--n_embd=1024' in c and '--n_head=16' in c
    assert '--out_dir=checkpoints/keep/aug_control_s1337_217M' in c
    assert c[-2:] == ['--always_save_checkpoint=True', '--save_iters=2440,3660']
    assert '--n_embd=192' in _cmd('control', size='3M')


def test_synth_args_generalised():
    assert A.synth_args(['a', 'b'], 0.5) == ['--csv_path=store:data/cache/gigamidi_replay,store:a,store:b',
                                             '--source_weights=0.5,0.25,0.25']


def test_table_paths():
    assert A.table_path('42M') == ROOT / 'logs/aug_sweep.md'
    assert A.table_path('108M') == ROOT / 'logs/aug_sweep_108M.md'
    assert A.table_path('42M', 0.125) == ROOT / 'logs/aug_sweep_f0.125.md'
    assert A.table_path('6M', 0.5, 320e6) == ROOT / 'logs/aug_sweep_6M_f0.5_n320M.md'


def test_expected_minutes():
    assert abs(A.expected_minutes('42M', 4883) - (95 * 4883 / 60000 + 0.7)) < 1e-9
    assert abs(A.expected_minutes('217M', 4883) - (350 * 4883 / 60000 + 1.5)) < 1e-9


class _Sandbox:
    """Points aug_sweep's runs.json, logs and tables at a temporary directory; train() writes a fake log."""

    def __init__(self, ce=None):
        self.dir = Path(tempfile.mkdtemp())
        self.ce = ce or {}
        self.cmds = []

    def __enter__(self):
        self.saved = A.RUNS, A.LOGS, A.table_path, A.train
        A.RUNS, A.LOGS = self.dir / 'runs.json', self.dir / 'logs'
        A.table_path = lambda size, frac=0.25, notes=A.NOTES: self.dir / f'table_{size}_{frac:g}_{notes:g}.md'

        def train(cmd, out, log, keep=False, timeout=None):
            self.cmds.append(cmd)
            A.LOGS.mkdir(exist_ok=True)
            iters = int(next(x for x in cmd if x.startswith('--max_iters=')).split('=')[1])
            ce = self.ce.get(log.stem, 2.5)
            log.write_text(f'gigamidi_replay_train: file_frac 0.25 keeps 25,383 files, 41,500,000 notes\n'
                           f'Model parameters : 42,000,000\n'
                           f'step {iters - 1}: train loss 2.0 val loss 2.1 | gigamidi_heldout CE {ce} '
                           f'(pit 1.0 vel 1.0) | prog CE 0.1\n')
            return 0
        A.train = train
        return self

    def __exit__(self, *exc):
        A.RUNS, A.LOGS, A.table_path, A.train = self.saved


def test_fresh_recorded_at_frac_1():
    with _Sandbox() as sb:
        A.run('fresh_dyn', 1337, 0.125, '42M', 16)
        A.run('fresh', 2, 0.125, '42M', 16)
        A.run('b16k', 2, 0.25, '42M', 16)
        runs = A.load()
        assert runs['fresh_dyn_s1337']['frac'] == 1.0 and runs['fresh_s2']['frac'] == 1.0
        assert runs['b16k_s2']['iters'] == 9766 and runs['b16k_s2']['notes'] == 9766 * 8 * 2048
        assert runs['b16k_s2']['rows'] == 8 and runs['b16k_s2']['micro'] == 8
        assert '--train_file_frac=1.0' in sb.cmds[0]


def _fake_runs():
    ce = {'control': 2.60, 'fresh': 2.47, 'dyn': 2.55, 'fresh_dyn': 2.46, 'wd3x': 2.58,
          'disc_control': 2.90, 'disc_fresh': 2.70, 'disc_dyn': 2.80}
    runs = {}
    for arm, v in ce.items():
        for seed, d in ((1337, 0.0), (2, 0.01)):
            runs[A.run_name(arm, seed, 0.25)] = dict(arm=arm, seed=seed, frac=A.arm_frac(arm, 0.25), size='42M',
                                                    micro=16, ce=v + d, prog=0.1, unique_notes=41_500_000)
    runs[A.run_name('wd10x', 1337, 0.25)] = dict(arm='wd10x', seed=1337, frac=0.25, size='42M', failed=True)
    return runs


def test_table_from_fake_runs():
    with _Sandbox() as sb:
        A.RUNS.write_text(json.dumps(_fake_runs()))
        A.table(0.25, '42M')
        rows = {ln.split(' | ')[0][2:]: ln.split(' | ') for ln in (sb.dir / 'table_42M_0.25_1.6e+08.md').read_text()
                .splitlines() if ln.startswith('| ') and not ln.startswith('| arm')}
        head = '| arm | s1337 | s2 | mean | ref | vs ref | won back | P_X | prog CE |'
        assert head in (sb.dir / 'table_42M_0.25_1.6e+08.md').read_text()
        assert rows['dyn'][4:8] == ['control', '-0.050', '38%', '+0.090']  # P_X = dyn - fresh_dyn
        assert rows['fresh_dyn'][4:6] == ['fresh', '-0.010']
        assert rows['control'][7] == '+0.130'  # the repeat penalty
        assert rows['wd3x'][1:3] == ['2.580', '2.590']
        assert rows['wd10x'][1] == 'failed'
        assert rows['disc_dyn'][4:7] == ['disc_control', '-0.100', '50%']  # Discover's own penalty 0.2
        assert 'fresh_census' not in rows


def test_gate_unchanged():
    runs = _fake_runs()
    assert abs(A.penalty(runs, 0.25, '42M') - 0.13) < 1e-9
    assert abs(A.penalty(runs, 0.25, '42M', control='disc_control') - 0.2) < 1e-9
    del runs['fresh_s2']
    assert A.penalty(runs, 0.25, '42M') is None  # needs both gate seeds
    # a small penalty stops the queue before the arms, as in HEAD
    small = {k: dict(v, ce=2.5) for k, v in _fake_runs().items() if v['arm'] in ('control', 'fresh')}
    with _Sandbox() as sb:
        A.RUNS.write_text(json.dumps(small))
        argv = sys.argv
        sys.argv = ['aug_sweep.py', '--arms', 'control,fresh,dyn', '--micro', '16']
        try:
            A.main()
        finally:
            sys.argv = argv
        assert sb.cmds == []
        assert 'Stopped at the gate' in (sb.dir / 'table_42M_0.25_1.6e+08.md').read_text()
        # --seeds 3 runs control and fresh at seed 3, then the gate still reads seeds 1337 and 2
        sys.argv = ['aug_sweep.py', '--arms', 'control,fresh,dyn', '--micro', '16', '--seeds', '3']
        try:
            A.main()
        finally:
            sys.argv = argv
        assert [c[c.index('--seed=3')] for c in sb.cmds] == ['--seed=3', '--seed=3'] and len(sb.cmds) == 2


def test_arm_json():
    path = Path(tempfile.mkdtemp()) / 'reg_arms.json'
    path.write_text(json.dumps({'BEST': ['--weight_decay=1.0', '--hypothesis=stack:parts=aug_dyn/shift=4'],
                                'fresh_BEST': {'args': ['--weight_decay=1.0']},
                                'BEST_b16k': {'args': [], 'rows': 8, 'warmup': 200}}))
    try:
        A.load_arms(path)
        assert A.OPTS['fresh_BEST'] == dict(frac=1.0, ref='fresh', twin='BEST')
        assert '--train_file_frac=1.0' in _cmd('fresh_BEST') and '--max_iters=9766' in _cmd('BEST_b16k')
        A.load_arms(path)  # the same arms again: no change, no error
    finally:
        for name in ('BEST', 'fresh_BEST', 'BEST_b16k'):
            A.ARMS.pop(name, None)
            A.OPTS.pop(name, None)


def test_timeout_kills_the_run():
    log = Path(tempfile.mkdtemp()) / 'x.log'
    t0 = time.time()
    code = A.train(['bash', '-c', 'sleep 60 & sleep 60'], log.parent / 'out', log, timeout=0.02)
    assert code == 'timeout' and time.time() - t0 < 15


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
