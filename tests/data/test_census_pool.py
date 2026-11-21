"""data_prep/datasets/census_pool.py on a synthetic census tar: held-out twins (transposed copies) dropped, bytes of the
kept files intact, count == build, verify passes. Run: python tests/data/test_census_pool.py (or pytest tests/)."""

import io
import json
import subprocess
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_prep' / 'datasets'))

from helpers import make_store, notes, run  # noqa: E402
import census_pool  # noqa: E402


def census_tar(d, parts):
    """A stores_gigamidi.tar.zst in the export's member order (tokens, programs, files.csv, offsets, meta per part)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as tf:
        for part, (files, progs) in parts.items():
            tok = np.concatenate(files).astype(np.uint16)
            off = np.concatenate([[0], np.cumsum([len(f) for f in files])]).astype(np.int64)
            ob = io.BytesIO()
            np.save(ob, off)
            csv = 'file_id,md5,store,idx\n' + ''.join(f'{part}{i:04d},{part}{i:04d},gigamidi_{part},{i}\n'
                                                      for i in range(len(files)))
            meta = json.dumps(dict(n_files=len(files), n_notes=int(off[-1]), programs=True))
            for name, data in (('tokens.u16', tok.tobytes()), ('programs.u8', np.concatenate(progs).tobytes()),
                               ('files.csv', csv.encode()), ('offsets.npy', ob.getvalue()),
                               ('meta.json', meta.encode())):
                info = tarfile.TarInfo(f'gigamidi/gigamidi_{part}/{name}')
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    path = Path(d) / 'stores.tar.zst'
    path.write_bytes(subprocess.run(['zstd', '-c'], input=raw.getvalue(), capture_output=True, check=True).stdout)
    return path


def setup(d):
    rng = np.random.default_rng(0)
    parts = {}
    for k, part in enumerate(('train', 'validation', 'test')):
        files = [notes(int(rng.integers(20, 300)), seed=100 * k + i) for i in range(30)]
        progs = [rng.choice([0, 33, 128], len(f)).astype(np.uint8) for f in files]
        parts[part] = (files, progs)
    held, held_p = [], []
    for part, i, shift in (('train', 3, 2), ('train', 17, 0), ('validation', 5, -3)):  # twins (transposed)
        f, p = parts[part][0][i].copy(), parts[part][1][i]
        f[:, 0] = np.where(p == 128, f[:, 0], f[:, 0] + shift)
        f[:, 1] = (f[:, 1] + 1) % 32  # velocity is not part of the loose hashes
        held.append(f)
        held_p.append(p)
    same_onsets = parts['train'][0][8].copy()
    same_onsets[:, 0] = 60  # an onset candidate that is no twin
    held.append(same_onsets)
    held_p.append(np.zeros(len(same_onsets), np.uint8))
    make_store(Path(d) / 'held_validation', held, held_p)
    return census_tar(d, parts), parts


def args(d, tar, **kw):
    return Namespace(tar=tar, heldout=str(Path(d) / 'held_validation'), out=str(Path(d) / 'pool_train'),
                     parts='train,validation', **kw)


def test_build_verify_count():
    with tempfile.TemporaryDirectory() as d:
        tar, parts = setup(d)
        a = args(d, tar)
        census_pool.build(a)
        out = Path(a.out)
        meta = json.loads((out / 'meta.json').read_text())
        assert meta['twins_dropped'] == 3, meta
        kept = [(p, i) for p in ('train', 'validation') for i in range(30)
                if (p, i) not in {('train', 3), ('train', 17), ('validation', 5)}]
        off = np.load(out / 'offsets.npy')
        tok = np.fromfile(out / 'tokens.u16', np.uint16).reshape(-1, 4)
        prog = np.fromfile(out / 'programs.u8', np.uint8)
        assert len(off) - 1 == len(kept) == meta['n_files'] and off[-1] == len(tok) == len(prog)
        for j, (p, i) in enumerate(kept):
            assert (tok[off[j]:off[j + 1]] == parts[p][0][i]).all() and (prog[off[j]:off[j + 1]] == parts[p][1][i]).all()
        rows = (out / 'files.csv').read_text().splitlines()
        assert rows[1].startswith('train0000') and len(rows) == len(kept) + 1 and 'train0003' not in ''.join(rows)
        assert census_pool.verify(a)


def test_count_matches_build():
    with tempfile.TemporaryDirectory() as d:
        tar, _ = setup(d)
        chunk = census_pool.CHUNK
        census_pool.CHUNK = 1000  # files straddle the stream chunks
        try:
            counted = census_pool.count(args(d, tar))
        finally:
            census_pool.CHUNK = chunk
        assert counted == census_pool.build(args(d, tar)) == (57, counted[1]), counted


if __name__ == '__main__':
    run(globals())
