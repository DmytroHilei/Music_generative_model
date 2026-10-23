"""NoteWindowDataset checks on synthetic token stores: python tests/data/test_dataset.py  (or pytest tests/)."""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import make_store, notes, run  # noqa: E402
from musicar.data.loader import (BOS_PITCH, DRUM_PROGRAM, EOS_PITCH, NoteWindowDataset, collate_windows,  # noqa: E402
                                 density_level, remain_level, remaining_seconds, source_name, window_conditions)


def dataset(d, files, progs=None, name='toy', **kw):
    make_store(Path(d) / f'{name}_train', files, progs)
    return NoteWindowDataset(f'store:{d}/{name}', root_dir=d, split='train', block_size=16, cache_dir=d, **kw)


def test_windows_are_shifted_targets():
    with tempfile.TemporaryDirectory() as d:
        ds = dataset(d, [notes(40, 0), notes(10, 1), notes(30, 2)])
        assert ds.n_files == 2, 'files with <= block_size notes are dropped'
        x, y = ds[0]
        assert len(x) == len(y) == 4 and all(t.shape == (16,) for t in x + y)
        assert all(torch.equal(a[1:], b[:-1]) for a, b in zip(x, y)), 'y = the next note'
        assert ds.num_notes() == 70


def test_special_tokens_and_eval_stride():
    with tempfile.TemporaryDirectory() as d:
        ds = dataset(d, [notes(40, 0), notes(30, 1)], special_tokens=True, eval_stride=8)
        starts = ds.windows.tolist()
        assert starts == [[0, 0], [0, 8], [0, 16], [0, 24], [1, 0], [1, 8]], starts
        x, y = ds[0]
        assert x[0][0] == BOS_PITCH and all(t[0] == 0 for t in x[1:]), 'BOS: pitch 128, rest 0'
        tail = dataset(d, [notes(15, 3)], name='one', special_tokens=True, eval_stride=8)
        assert tail[0][1][0][-1] == EOS_PITCH, '[BOS] + 15 notes + [EOS] = one window, its last target is EOS'
        assert [tuple(t.tolist()) for t in ds[2][0]] == [tuple(t.tolist()) for t in ds[2][0]], 'deterministic'


def test_transposition_and_tempo():
    with tempfile.TemporaryDirectory() as d:
        base = notes(17, 0, pitch=(122, 126))  # block_size + 1 notes: one window, high so the shift is limited
        ds = dataset(d, [base], augment=True)
        raw = dataset(d, [base], name='raw')
        random.seed(0)
        shifts = set()
        for _ in range(50):
            x, _ = ds[0]
            diff = x[0] - raw[0][0][0]
            assert len(set(diff.tolist())) == 1, 'every note moves by the same interval'
            assert torch.equal(x[1], raw[0][0][1]) and torch.equal(x[2], raw[0][0][2])
            shifts.add(int(diff[0]))
        assert shifts <= set(range(-5, 3)) and len(shifts) > 3, shifts  # pitches 122..125: at most +2
        assert int(base[:, 0].max()) + max(shifts) <= 127
        tempo = dataset(d, [base], name='tempo', augment=True, aug_tempo=0.5)
        x, _ = tempo[0]
        assert torch.all(x[2] <= 511) and torch.all((x[3] == 0) == (raw[0][0][3] == 0)), 'chords stay chords'


def test_stretch_unbiased():
    torch.manual_seed(0)
    b = torch.full((20000,), 7)
    s = NoteWindowDataset._stretch(b, 1.3, 511)
    assert set(s.unique().tolist()) == {9, 10} and abs(s.double().mean() - 9.1) < 0.05
    assert NoteWindowDataset._stretch(torch.tensor([0, 500]), 2.0, 511).tolist() == [0, 511]


def test_programs_drums():
    with tempfile.TemporaryDirectory() as d:
        n = notes(17, 0)
        prog = np.array([0, DRUM_PROGRAM] * 8 + [33])
        ds = dataset(d, [n], [prog], programs=True, augment=True)
        x, y = ds[0]
        assert len(x) == len(y) == 5
        drums = x[4] == DRUM_PROGRAM
        assert torch.all(x[2][drums] == 0) and torch.all(y[2][y[4] == DRUM_PROGRAM] == -1)
        assert torch.equal(x[0][drums], torch.from_numpy(n[:16, 0][prog[:16] == DRUM_PROGRAM]).long()), \
            'drum sounds are never transposed'
        piano = dataset(d, [n], [prog], name='piano')
        try:
            piano[0]
            raise AssertionError('a piano dataset must refuse a store with programs')
        except AssertionError as e:
            assert 'programs=True' in str(e), e


def test_pad_and_pack_short():
    with tempfile.TemporaryDirectory() as d:
        files = [notes(5, i) for i in range(6)] + [notes(70, 9)]
        pad = dataset(d, files, special_tokens=True, pad_short=True, min_notes=4, eval_stride=16)
        x, y = pad[0]  # [BOS] + 5 + [EOS] = 7 rows
        assert x[0][0] == BOS_PITCH and y[0][5] == EOS_PITCH and torch.all(y[0][6:] == -1)
        pack = dataset(d, files, name='pack', special_tokens=True, pack_short=True, min_notes=4)
        random.seed(1)
        packed = [pack[0] for _ in range(30)]
        multi = [x for x, _ in packed if (x[0] == BOS_PITCH).sum() > 1]
        assert multi and all(torch.all(y[0] >= 0) for _, y in packed), 'packed windows are full, no padding'


def test_source_weights():
    with tempfile.TemporaryDirectory() as d:
        make_store(Path(d) / 'low_train', [notes(40, 0, pitch=(20, 30))])
        make_store(Path(d) / 'high_train', [notes(400, 1, pitch=(90, 100))])
        kw = dict(root_dir=d, split='train', block_size=16, cache_dir=d)
        ds = NoteWindowDataset(f'store:{d}/low,store:{d}/high', source_weights=[1, 0], **kw)
        random.seed(0)
        assert all(ds[0][0][0].max() < 30 for _ in range(50))
        ds = NoteWindowDataset(f'store:{d}/low,store:{d}/high', **kw)
        low = sum(int(ds[0][0][0].max() < 30) for _ in range(400))
        assert 0 < low < 60, 'default: by size (24 vs 384 window starts)'


def test_conditions_and_collate():
    assert density_level([0.1, 0.5, 1.0, 64.0, 1000.0]).tolist() == [1, 1, 3, 15, 15]
    pitch = torch.tensor([[BOS_PITCH, 60, 62, 64], [60, 61, 62, EOS_PITCH]])
    dt = torch.tensor([[0, 25, 25, 25], [0, 0, 0, 0]])
    program = torch.tensor([[0, 33, 33, DRUM_PROGRAM], [0, 0, 0, 0]])
    inst, dens = window_conditions(pitch, dt, program, 129, 16)
    assert inst[0].nonzero().flatten().tolist() == [33, DRUM_PROGRAM] and inst[1].nonzero().flatten().tolist() == [0]
    assert dens.tolist() == [density_level(3 / 1.5).item(), density_level(3 / 0.5).item()]
    inst, dens = window_conditions(pitch, dt, program, 129, 16, drop_p=1.0)
    assert inst.sum() == 0 and dens.sum() == 0
    items = [((torch.zeros(4),) * 4 + (torch.tensor(i), f'cap {i}'), (torch.ones(4),) * 4) for i in range(3)]
    (streams, extra), y = collate_windows(items, extra_names=('style', 'text'))
    assert streams[0].shape == (3, 4) and extra['style'].tolist() == [0, 1, 2] and extra['text'][2] == 'cap 2'
    assert source_name('store:data/cache/gigamidi_clean,data/finetune/x.csv') == 'gigamidi_clean+x'



def test_remaining_seconds_and_levels():
    pitch = torch.tensor([60, 60, 60, EOS_PITCH, BOS_PITCH, 60, 60])
    dt = torch.tensor([0, 5, 10, 0, 0, 0, 25])
    rem = remaining_seconds(pitch, dt, tail=1.0)  # a packed window: a piece ends at the EOS, the next one is cut
    assert torch.allclose(rem, torch.tensor([0.3, 0.2, 0.0, 0.0, 1.5, 1.5, 1.0])), rem
    assert remain_level([0.0, 0.49, 0.5, 1200.0, 5000.0]).tolist() == [1, 1, 2, 32, 32]
    levels = remain_level(np.geomspace(0.5, 1199, 200))
    assert torch.all(levels[1:] >= levels[:-1]) and set(levels.tolist()) == set(range(2, 33))


def test_remain_input():
    """The seconds from each input row's onset to its piece's end: exact for the strided windows of a file, 0 at
    EOS, restarting in packed windows, and stretched with the notes by the tempo augmentation."""
    with tempfile.TemporaryDirectory() as d:
        f = notes(40, 0)
        ds = dataset(d, [f], special_tokens=True, eval_stride=8, remain=True)
        onset = np.concatenate([[0.0], np.cumsum(f[:, 3]) * 0.02])  # virtual rows: BOS, notes (EOS = last onset)
        onset = np.concatenate([onset, [onset[-1]]])
        for i, (_, v) in enumerate(ds.windows.tolist()):
            x, _ = ds[i]
            assert len(x) == 5 and x[4].dtype == torch.float32
            rows = np.arange(v, min(v + 16, len(onset)))
            assert np.allclose(x[4][:len(rows)].numpy(), onset[-1] - onset[rows], atol=1e-4), (v, x[4])
        files = [notes(5, i) for i in range(6)] + [notes(70, 9)]
        pack = dataset(d, files, name='pack', special_tokens=True, pack_short=True, min_notes=4, remain=True)
        random.seed(1)
        for _ in range(30):
            x, _ = pack[0]
            pitch, dt, rem = x[0], x[3], x[4]
            assert torch.all(rem[pitch == EOS_PITCH] == 0) and torch.all(rem >= 0)
            step = rem[:-1] - rem[1:]
            inside = (pitch[:-1] != EOS_PITCH)  # within a piece the remainder falls by each next note's step
            assert torch.allclose(step[inside], dt[1:][inside].float() * 0.02, atol=1e-4)
        tempo = dataset(d, [notes(60, 4)], name='tempo', augment=True, aug_tempo=0.5, remain=True)
        random.seed(2)
        for _ in range(10):
            x, _ = tempo[0]
            assert torch.allclose(x[4][:-1] - x[4][1:], x[3][1:].float() * 0.02, atol=1e-4)
        (streams, extra), _ = collate_windows([ds[0], ds[1]], extra_names=('remain',))
        assert extra['remain'].shape == (2, 16)


if __name__ == '__main__':
    run(globals())
