"""musicar/generation/retrieval.py checks (CPU, a word-hashing stand-in for LaBSE):
python tests/generation/test_retrieval.py  (or tests/run.py)."""

import json
import re
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import make_store, run  # noqa: E402
from musicar.data.features import FIELDS, file_features  # noqa: E402
from musicar.generation.retrieval import CaptionIndex  # noqa: E402


class WordEncoder:
    """Bag of hashed words -> unit vectors (similar wording = similar vectors); counts the texts it encoded."""

    def __init__(self, dim=64):
        self.dim, self.n = dim, 0

    def encode(self, texts):
        self.n += len(texts)
        out = torch.zeros(len(texts), self.dim)
        for i, t in enumerate(texts):
            for w in re.findall(r'\w+', t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        return torch.nn.functional.normalize(out, dim=-1)


def toy_store(root, n=12, notes=96):
    """Files 0-5 solo piano, 6-11 drums + clean guitar + bass; their feature table (the real measures) and MidiCaps
    rows for half of them."""
    rng = np.random.default_rng(0)
    files, progs = [], []
    for i in range(n):
        tok = np.stack([rng.integers(48, 76, notes), rng.integers(10, 30, notes), rng.integers(5, 40, notes),
                        rng.integers(3, 12, notes)], 1)
        files.append(tok)
        progs.append(np.zeros(notes, np.int64) if i < n // 2 else np.array([128, 27, 33])[np.arange(notes) % 3])
    store = root / 'cache' / 'toy_train'
    make_store(store, files, progs)
    labels = root / 'labels'
    labels.mkdir()
    feats = [file_features(t, p) for t, p in zip(files, progs)]
    table = {name: np.array([f[0][j] for f in feats]) for j, name in enumerate(FIELDS)}
    table['programs'] = np.stack([f[1] for f in feats])
    np.savez(labels / 'toy_train.features.npz', **table)
    rows = [dict(file=i, md5=str(i), midicaps=dict(caption='', genre=['classical' if i < n // 2 else 'rock'],
                                                   mood=['relaxing' if i < n // 2 else 'energetic']))
            for i in range(0, n, 2)]
    (labels / 'toy_train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    return store, labels


def test_index_search_and_cache():
    with tempfile.TemporaryDirectory() as d:
        store, labels = toy_store(Path(d))
        enc = WordEncoder()
        kw = dict(labels_dir=labels, captions_dir=Path(d) / 'none', min_notes=64, encoder=enc)
        index = CaptionIndex(store, **kw)
        assert sorted(index.files.tolist()) == list(range(12)) and index.emb.shape == (12, 64)
        assert (labels / 'toy_train.retrieval.npz').exists() and enc.n == 12 * 4
        piano = [h['file'] for h in index.search('a solo piano piece', k=4)]
        band = [h['file'] for h in index.search('a rock song with drums, electric guitar and bass', k=4)]
        assert all(f < 6 for f in piano) and all(f >= 6 for f in band), (piano, band)
        top = index.search('drums, electric guitar and bass', k=12)[0]
        assert top['match'] == 1.0 and top['score'] == top['sim'] + top['match']
        calm = index.search('a relaxing classical piece', k=3)  # labels: the even piano files are classical, relaxing
        assert [h['file'] % 2 for h in calm] == [0, 0, 0] and all(h['file'] < 6 and h['match'] == 1.0 for h in calm)
        solo = index.search('a piece', k=12, only=[0, 1])  # the caption's hard limits: piano files only
        assert sorted(h['file'] for h in solo) == list(range(6))
        assert index.search('a piece', k=12, only=[40]) == []
        tok, prog = index.opening(7, 10)
        assert tok.shape == (10, 4) and set(prog.tolist()) == {128, 27, 33}

        enc2 = WordEncoder()
        again = CaptionIndex(store, **{**kw, 'encoder': enc2})
        assert enc2.n == 0 and np.array_equal(again.emb, index.emb), 'loaded from the cache'
        (labels / 'toy_train.jsonl').write_text('')  # new label tables: the index is rebuilt
        CaptionIndex(store, **{**kw, 'encoder': enc2})
        assert enc2.n == 12 * 4
        enc3 = WordEncoder()
        CaptionIndex(store, **{**kw, 'encoder': enc3, 'max_files': 5})  # another subset size: rebuilt
        assert enc3.n == 5 * 4


def test_min_notes():
    with tempfile.TemporaryDirectory() as d:
        store, labels = toy_store(Path(d))
        try:
            CaptionIndex(store, labels_dir=labels, captions_dir=Path(d) / 'none', min_notes=1000, encoder=WordEncoder())
        except ValueError as e:
            assert 'no captions' in str(e)
        else:
            raise AssertionError('an index without files must fail')


if __name__ == '__main__':
    run(globals())
