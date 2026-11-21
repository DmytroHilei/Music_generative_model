"""data_prep/datasets/synth_store.py (CPU, tiny model): genre groups and draws, shards, resume, merge, the store's
contents. Run: python tests/data/test_synth_store.py (or pytest tests/)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_prep' / 'datasets'))

from helpers import make_store, notes, run, tiny  # noqa: E402
from musicar.data.loader import BOS_PITCH, NoteWindowDataset  # noqa: E402
import synth_store  # noqa: E402


def setup(d):
    """A 12-piece store (pieces 0-3 rock, 4-5 classical, 6 baroque only, the rest without labels) and a tiny model."""
    files = [notes(100, i) for i in range(12)]
    make_store(Path(d) / 'toy_train', files, [np.full(100, i % 3 * 30, np.uint8) for i in range(12)])
    labels = Path(d) / 'labels'
    labels.mkdir()
    rows = [dict(file=i, midicaps=dict(genre=['rock'])) for i in range(4)] + \
        [dict(file=i, midicaps=dict(genre=['classical'])) for i in (4, 5)] + [dict(file=6, midicaps=dict(genre=['baroque']))]
    (labels / 'toy_train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2)
    args = {k: v for k, v in vars(m.config).items() if k != 'dropout'}
    torch.save(dict(model=m.state_dict(), model_args=args, iter_num=0), Path(d) / 'ckpt.pt')
    return files, labels


def test_groups_and_draws():
    with tempfile.TemporaryDirectory() as d:
        _, labels = setup(d)
        groups = synth_store.prompt_groups(Path(d) / 'toy_train', 1.0, 16, labels, min_group=2)
        assert groups == {'classical': [4, 5], 'other': [6], 'rock': [0, 1, 2, 3], 'untagged': list(range(7, 12))}
        w = synth_store.group_weights(groups, 0.0)
        assert all(abs(v - 0.25) < 1e-9 for v in w.values()), 'alpha 0: every genre alike'
        names = list(groups)
        cum = np.cumsum([w[g] for g in names]).tolist()
        drawn = [synth_store.draw(7, r, names, cum, groups) for r in range(400)]
        assert drawn == [synth_store.draw(7, r, names, cum, groups) for r in range(400)], 'a function of (seed, row)'
        assert all(i in groups[g] for g, i in drawn) and {g for g, _ in drawn} == set(names)
        short = synth_store.prompt_groups(Path(d) / 'toy_train', 1.0, 40, labels, min_group=2)
        assert not short, 'pieces need prompt + 64 notes'


def test_shards_resume_merge():
    with tempfile.TemporaryDirectory() as d:
        files, labels = setup(d)
        argv = ['synth_store.py', '--checkpoint', f'{d}/ckpt.pt', '--prompt-store', f'{d}/toy_train', '--prompt-frac',
                '1', '--labels-dir', str(labels), '--min-group', '2', '--prompt-notes', '16', '--new-notes', '24',
                '--min-new', '24', '--rows', '3', '--notes', '150', '--shard-notes', '100', '--device', 'cpu']
        saved = sys.argv
        try:
            for out in ('a', 'b'):
                sys.argv = argv + ['--out', f'{d}/{out}']
                synth_store.main()
        finally:
            sys.argv = saved
        shards = synth_store.finished_shards(Path(d) / 'a_shards')
        assert [s.name for s in shards] == ['shard_00000', 'shard_00001'], 'two shards of 3 x 40 notes reach 150'
        a, b = (np.fromfile(Path(d) / f'{o}_train' / 'tokens.u16', np.uint16) for o in ('a', 'b'))
        assert np.array_equal(a, b), 'the same plan gives the same notes'
        sys.argv = argv + ['--out', f'{d}/a']
        try:
            synth_store.main()  # resume: nothing left to generate, merged again
        finally:
            sys.argv = saved
        assert len(synth_store.finished_shards(Path(d) / 'a_shards')) == 2
        meta = json.loads((Path(d) / 'a_train' / 'meta.json').read_text())
        recs = [json.loads(x) for x in (Path(d) / 'a_train' / 'files.jsonl').read_text().splitlines()]
        assert meta['n_files'] == len(recs) == 6 and meta['n_notes'] == 6 * 40
        tokens = a.reshape(-1, 4).astype(np.int64)
        offsets = np.load(Path(d) / 'a_train' / 'offsets.npy')
        for k, r in enumerate(recs):
            assert (r['prompt_notes'], r['new_notes']) == (16, 24) and r['group'] in ('classical', 'other', 'rock',
                                                                                         'untagged')
            got, real = tokens[offsets[k]:offsets[k] + 16], files[r['prompt_file']][:16]
            assert np.array_equal(got[:, [0, 1, 3]], real[:, [0, 1, 3]]), 'each file starts with its real prompt'
        ds = NoteWindowDataset(f'store:{d}/a', root_dir=d, split='train', block_size=16, cache_dir=d,
                               special_tokens=True, programs=True, pack_short=True, min_notes=16)
        assert ds.n_files == 6 and len(ds[0][0]) == 5
        sys.argv = argv + ['--out', f'{d}/a', '--temperature', '1.0']
        try:
            synth_store.main()
            raise AssertionError('mixed plans accepted')
        except SystemExit as e:
            assert 'other settings' in str(e)
        finally:
            sys.argv = saved


def test_ladder_member_16_rows():
    """A scale-ladder member's run dir (model_bf16.pt; cond_inst + n_density, RMSNorm, SwiGLU, QK-norm, as
    config/pretrain/scale_ladder.py; the E-S6 generator path): one shard of 16 rows."""
    with tempfile.TemporaryDirectory() as d:
        files, labels = setup(d)
        m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2, cond_inst=True, n_density=16, norm='rmsnorm', mlp='swiglu',
                 qk_norm=True)
        args = {k: v for k, v in vars(m.config).items() if k != 'dropout'}
        (Path(d) / 'member').mkdir()
        torch.save(dict(model=m.state_dict(), model_args=args, iter_num=199), Path(d) / 'member' / 'model_bf16.pt')
        argv = ['synth_store.py', '--checkpoint', f'{d}/member', '--prompt-store', f'{d}/toy_train', '--prompt-frac',
                '1', '--labels-dir', str(labels), '--min-group', '2', '--prompt-notes', '16', '--new-notes', '24',
                '--min-new', '24', '--rows', '16', '--notes', '600', '--shard-notes', '600', '--device', 'cpu',
                '--out', f'{d}/m']
        saved, sys.argv = sys.argv, argv
        try:
            synth_store.main()
        finally:
            sys.argv = saved
        meta = json.loads((Path(d) / 'm_train' / 'meta.json').read_text())
        assert meta['n_files'] == 16 and meta['n_notes'] == 16 * 40 and meta['plan']['iter'] == 199
        assert meta['plan']['checkpoint'].endswith('member/model_bf16.pt')


if __name__ == '__main__':
    run(globals())
