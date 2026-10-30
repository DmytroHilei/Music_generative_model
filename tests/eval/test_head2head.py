"""Head-to-head task, reuse and bootstrap logic (CPU, no model, no store): python tests/eval/test_head2head.py  (or
pytest tests/)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval/compare'))

import head2head as h  # noqa: E402
from musicar.eval import sample_eval  # noqa: E402


def refused(fn, *args):
    try:
        fn(*args)
    except SystemExit:
        return True
    return False


def fake_store(n_files, notes):
    """open_store's (offsets, read) of n_files files of notes[i % len(notes)] notes, 4 instruments each."""
    offsets = np.cumsum([0] + [notes[i % len(notes)] for i in range(n_files)])

    def read(a, b):
        k = np.arange(a, b)
        tok = np.stack([40 + k % 40, np.full(len(k), 10), np.full(len(k), 5), np.full(len(k), 3)], 1)
        return tok.astype(np.int64), (k % 4).astype(np.int64)
    return offsets, read


def export(out, task, **spec):
    saved = h.OUT, h.open_store
    h.OUT, h.open_store = out, lambda name: fake_store(40, [1200, 200])
    try:
        h.export_cont(task, 'fake', **spec)
    finally:
        h.OUT, h.open_store = saved
    return json.loads((out / task / 'prompts.json').read_text())


def test_task_rows():
    assert h.ROWS == 32
    assert [(t, h.CONT_TASKS[t]['n_rows']) for t in ('cont', 'piano', 'cont96', 'piano96')] == [
        ('cont', 32), ('piano', 32), ('cont96', 96), ('piano96', 96)]
    assert h.BASE_TASKS == ['cont', 'piano'], 'a bare prompts / score must not reach the 96-prompt sets'
    for task, base in (('cont96', 'cont'), ('piano96', 'piano')):
        spec = h.CONT_TASKS[task]
        assert spec['base'] == base
        assert all(spec[k] == h.CONT_TASKS[base][k] for k in ('store', 'min_instruments', 'min_notes', 'candidates'))
    assert 'moonbeam' not in h.CONT_TASKS['cont96']['systems']


def test_spaced_files():
    lengths = np.array([5, 100, 100, 5, 100, 100, 100, 100, 5, 100, 100, 100])
    a = h.spaced_files(lengths, 50, {1, 7}, 4)
    assert a.tolist() == h.spaced_files(lengths, 50, {7, 1}, 4).tolist(), 'deterministic'
    assert len(set(a.tolist())) == 4 and not set(a.tolist()) & {0, 1, 3, 7, 8}
    assert a[0] == 2 and a[-1] == 11, 'from the first to the last eligible file'
    assert a.tolist() == sorted(a.tolist())


def test_export_superset():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        spec = dict(min_instruments=3, min_notes=1128, candidates=8)
        base = export(out, 'base', n_rows=2, **spec)
        grid = export(out, 'grid', n_rows=3, **spec)
        spaced = export(out, 'spaced', n_rows=6, grid_rows=2, **spec)
        assert base['rows'] == 2 and len(base['prompts']) == 2 and 'grid_rows' not in spaced
        for meta, n in ((grid, 3), (spaced, 6)):
            assert len(meta['prompts']) == n and meta['prompts'][:2] == base['prompts']
            files = [r['file'] for r in meta['prompts']]
            assert len(set(files)) == n and [r['id'] for r in meta['prompts']] == [f'p{i:02d}' for i in range(n)]
        assert h.shared_prompts(out / 'grid', out / 'base') == base['prompts']
        assert h.shared_prompts(out / 'spaced', out / 'base') == base['prompts']
        grid_files = np.linspace(0, 39, 8).astype(int)
        assert [r['file'] for r in grid['prompts']] == [i for i in grid_files.tolist() if i % 2 == 0][:3]


def test_shared_prompts_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        spec = dict(min_instruments=3, min_notes=1128, candidates=8)
        export(out, 'base', n_rows=2, **spec)
        export(out, 'task', n_rows=3, **spec)
        extra = out / 'task' / 'prompts' / 'p02.mid'
        extra.write_bytes(b'not the base set, not checked')
        assert h.shared_prompts(out / 'task', out / 'base')
        real = out / 'task' / 'real' / 'p01_0.mid'
        keep = real.read_bytes()
        real.write_bytes(keep + b'\0')
        assert refused(h.shared_prompts, out / 'task', out / 'base'), 'real MIDI differs'
        real.write_bytes(keep)
        meta = json.loads((out / 'task' / 'prompts.json').read_text())
        meta['prompts'][0]['file'] += 1
        (out / 'task' / 'prompts.json').write_text(json.dumps(meta))
        assert refused(h.shared_prompts, out / 'task', out / 'base'), 'row differs'


def test_reuse_links_and_keeps():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        spec = dict(min_instruments=3, min_notes=1128, candidates=8)
        export(out, 'base', n_rows=2, **spec)
        export(out, 'task', n_rows=3, **spec)
        (out / 'base' / 'sys').mkdir()
        (out / 'base' / 'other').mkdir()
        for pid in ('p00', 'p01'):
            (out / 'base' / 'sys' / f'{pid}_0.mid').write_bytes(pid.encode())
            (out / 'base' / 'other' / f'{pid}_0.mid').write_bytes(pid.encode())
        (out / 'task' / 'sys').mkdir()
        (out / 'task' / 'sys' / 'p01_0.mid').write_bytes(b'already there')
        saved = h.OUT, dict(h.CONT_TASKS)
        h.OUT, h.CONT_TASKS['task'] = out, dict(base='base', systems=('sys',))
        try:
            h.reuse(type('A', (), dict(task='task')))
        finally:
            h.OUT = saved[0]
            del h.CONT_TASKS['task']
        assert (out / 'task' / 'sys' / 'p00_0.mid').samefile(out / 'base' / 'sys' / 'p00_0.mid')
        assert (out / 'task' / 'sys' / 'p01_0.mid').read_bytes() == b'already there'
        assert not (out / 'task' / 'sys' / 'p02_0.mid').exists(), 'new prompts are left to the run'
        assert not (out / 'task' / 'other').exists(), 'only the task\'s systems'


def test_score_name():
    assert h.score_name(None, None) == 'scores'
    assert h.score_name(['cont96', 'piano96'], 'scores_96') == 'scores_96'
    assert h.score_name(None, 'scores_96_first200') == 'scores_96_first200'
    for tasks, out in ((['cont96'], None), (['cont96'], 'scores'), (None, 'scores'), (None, 'scores_first200')):
        assert refused(h.score_name, tasks, out), (tasks, out)


def test_score_writes_the_name():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        export(out, 'cont', n_rows=2, min_instruments=3, min_notes=1128, candidates=8)
        saved = h.OUT
        h.OUT = out
        try:
            h.score(type('A', (), dict(task=None, out=None, notes=50)))
            h.score(type('A', (), dict(task=['cont'], out='scores_96', notes=50)))
        finally:
            h.OUT = saved
        assert sorted(p.name for p in out.iterdir() if p.is_file()) == ['scores.json', 'scores.md', 'scores_96.json',
                                                                        'scores_96.md']
        assert json.loads((out / 'scores_96.json').read_text())['cont']['real']['rows'] == 2


def tables(seed=0, n=12):
    """A prompt table per system, rows drawn at random."""
    rng = np.random.default_rng(seed)
    keys = len(sample_eval.DIST_WEIGHTS)
    ids = [f'p{i:02d}' for i in range(n)]
    return {s: {i: rng.random(keys) * 5 for i in ids} for s in ('real', 'a', 'b')}


def test_prompt_table():
    keys = list(sample_eval.DIST_WEIGHTS)
    rows = [dict(zip(keys, [float(v)] * len(keys)), id=i) for i, v in (('p01', 2), ('p00', 1), ('p01', 4))]
    t = h.prompt_table(rows)
    assert t['p00'].tolist() == [1.0] * len(keys) and t['p01'].tolist() == [3.0] * len(keys)


def test_bootstrap_point_is_score():
    t = tables()
    ids, point, per_draw = h.bootstrap(t, 50, 0)
    keys = list(sample_eval.DIST_WEIGHTS)
    real = np.array(list(t['real'].values()))
    ref = dict(zip(keys, real.mean(0).tolist()), sd=dict(zip(keys, real.std(0, ddof=1).tolist())))
    m = dict(zip(keys, np.array(list(t['a'].values())).mean(0).tolist()))
    assert abs(point['a'] - sample_eval.distance(m, ref)) < 1e-12
    assert len(ids) == 12 and set(per_draw) == {'a', 'b'} and len(per_draw['a']) == 50


def test_bootstrap_aligns_by_id():
    t = tables()
    shuffled = dict(t)
    shuffled['a'] = dict(reversed(list(t['a'].items())))  # the same rows in another order
    a = h.bootstrap(t, 200, 3)
    b = h.bootstrap(shuffled, 200, 3)
    assert a[1] == b[1] and (a[2]['a'] == b[2]['a']).all()
    t['a'] = {i: v for i, v in t['a'].items() if i != 'p05'}
    t['b']['p99'] = t['b']['p00']
    ids, _, _ = h.bootstrap(t, 10, 3)
    assert ids == [f'p{i:02d}' for i in range(12) if i != 5], 'the ids every system has'


def test_bootstrap_same_draws_for_real():
    t = tables()
    t['a'] = dict(t['real'])  # real itself: D 0 in every draw only if real and the system get the same indices
    _, point, per_draw = h.bootstrap(t, 200, 1)
    assert point['a'] == 0 and (per_draw['a'] == 0).all()
    assert per_draw['b'].min() > 0
    _, _, again = h.bootstrap(t, 200, 1)
    assert (again['b'] == per_draw['b']).all(), 'seeded'


def test_boot_table():
    point = dict(a=1.0, b=2.0, c=3.0)
    per_draw = dict(a=np.array([1.0, 1.0, 3.0, 1.0]), b=np.array([2.0, 2.0, 2.0, 2.0]), c=np.full(4, 3.0))
    systems, pairs = h.boot_table(point, per_draw, ['a'])
    assert list(pairs) == ['a vs b', 'a vs c']
    assert pairs['a vs b']['closer'] == 0.75 and pairs['a vs b']['diff'] == -1.0
    assert abs(systems['a']['bias'] - 0.5) < 1e-12 and abs(pairs['a vs b']['bias'] - 0.5) < 1e-12


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
