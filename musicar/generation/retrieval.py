"""
Caption retrieval over a captioned token store (generate.py --retrieve): which real openings fit a caption.

Every file with at least min_notes notes gets the mean LaBSE vector (musicar/text/encoder.py) of n_caps captions
drawn by the training caption builder (StoreCaptions, English and Ukrainian in turn, so its attribute dropout and
phrasing average out). The vectors are cached in <labels_dir>/<store>.retrieval.npz and rebuilt when the label
tables, the features, the Ukrainian captions or the settings change. search() ranks by cosine similarity to the
query plus attr_weight x how well each file matches what the query asks for (musicar/generation/caption_match.py:
its measured attributes and its genre / mood labels), over the 50 x k (at least 1,000) files most similar
to the query.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from musicar.generation.caption_match import FileAttributes, match, parse_caption
from musicar.text.captions import StoreCaptions

VERSION = 1


class CaptionIndex:
    def __init__(self, store, labels_dir='data/labels', captions_dir='data/captions', min_notes=512, n_caps=4,
                 max_files=60000, seed=0, encoder=None, device='cpu', batch_size=512):
        """store: a token store directory (data/cache/<name>); its label tables are <labels_dir>/<name>.*.
        encoder: anything with encode(list of str) -> (B, dim) unit vectors (default: LaBSE on `device`, made when
        the index has to be built or a query encoded). max_files: a random subset (fixed by seed) of the files with
        enough notes, for big stores."""
        self.store = Path(store)
        name = self.store.name
        self.caps = StoreCaptions(name, labels_dir=labels_dir, captions_dir=captions_dir)
        if self.caps.features is None:
            raise FileNotFoundError(f'no {labels_dir}/{name}.features.npz (data_prep/labels/song_features.py)')
        self.offsets = np.load(self.store / 'offsets.npy')
        self.attrs = FileAttributes(self.caps.features, self.caps)
        self.encoder, self.device, self.batch_size = encoder, device, batch_size
        sources = [self.store / 'offsets.npy', Path(labels_dir) / f'{name}.jsonl',
                   Path(labels_dir) / f'{name}.features.npz', Path(captions_dir) / f'{name}.uk.fixed.jsonl']
        # sources by file name, so the cache holds whether the paths were given relative or absolute
        self.meta = dict(version=VERSION, n_caps=n_caps, min_notes=min_notes, max_files=max_files, seed=seed,
                         sources={p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sources if p.exists()})
        self.path = Path(labels_dir) / f'{name}.retrieval.npz'
        if not self._load():
            self._build()

    def _load(self):
        if not self.path.exists():
            return False
        z = np.load(self.path)
        if json.loads(str(z['meta'])) != self.meta:
            return False
        self.files, self.emb = z['files'], z['emb']
        return True

    def _encode(self, texts):
        if self.encoder is None:
            from musicar.text.encoder import TextEncoder
            dtype = torch.float16 if str(self.device).startswith('cuda') else torch.float32
            self.encoder = TextEncoder(device=self.device, dtype=dtype)
        return self.encoder.encode(texts).float().cpu().numpy()

    def _build(self):
        m = self.meta
        n_notes = np.diff(self.offsets)
        files = np.flatnonzero(n_notes >= m['min_notes'])
        if len(files) > m['max_files']:
            files = np.sort(np.random.default_rng(m['seed']).choice(files, m['max_files'], replace=False))
        texts, owner = [], []
        for k, i in enumerate(files):
            for j in range(m['n_caps']):
                rng = random.Random(m['seed'] * 1_000_003 + int(i) * 8 + j)
                cap = self.caps.caption(int(i), rng, 'en' if j % 2 == 0 else 'uk')
                if cap:
                    texts.append(cap)
                    owner.append(k)
        if not texts:
            raise ValueError(f'{self.store.name}: no captions for files with >= {m["min_notes"]} notes')
        print(f'retrieval index {self.path}: {len(texts):,} captions of {len(files):,} files', flush=True)
        t0 = time.time()
        sums, owner = None, torch.as_tensor(owner)
        for a in range(0, len(texts), self.batch_size):
            v = torch.from_numpy(self._encode(texts[a:a + self.batch_size]))
            if sums is None:
                sums = torch.zeros(len(files), v.shape[1])
            sums.index_add_(0, owner[a:a + len(v)], v)
            if (a // self.batch_size) % 50 == 0:
                print(f'  {a + len(v):,} / {len(texts):,} captions ({time.time() - t0:.0f} s)', flush=True)
        sums = sums.numpy()
        norm = np.linalg.norm(sums, axis=1)
        keep = norm > 0
        self.files = files[keep].astype(np.int64)
        self.emb = (sums[keep] / norm[keep, None]).astype(np.float16)
        np.savez(self.path, files=self.files, emb=self.emb, meta=json.dumps(self.meta))
        print(f'  {len(self.files):,} files indexed in {time.time() - t0:.0f} s', flush=True)

    def search(self, query, k=32, attr_weight=1.0, only=None, strict=()):
        """The k best files for a caption: [dict(file, sim, match, score, strict)], best first. score = cosine
        similarity + attr_weight x caption_match score of the file. only: GM programs (0-128) a file may use (its
        instruments with >= 2% of the notes), e.g. the caption's hard limits; files with others are left out.
        strict: caption_match categories (e.g. ('genre', 'mood')) whose checks a file must fully meet (its labels name
        the asked genre / mood), each asked check on its own and searched over a wider pool; a check no file meets
        stays soft. strict on the hits = the checks enforced (empty: soft ranking only)."""
        wants = parse_caption(query)
        q = self._encode([query])[0]
        sims = self.emb.astype(np.float32) @ q
        if only is not None:
            outside = self.attrs.bits[self.files] & ~np.isin(np.arange(129), list(only))[None]
            sims = np.where(outside.any(1), -np.inf, sims)
        pool = [j for j in np.argsort(-sims)[:max((200 if strict else 50) * k, 1000)] if np.isfinite(sims[j])]
        hits, named = [], []
        for j in pool:
            m, _, cats = match(wants, self.attrs(int(self.files[j])), categories=True)
            sim = float(sims[j])
            hits.append(dict(file=int(self.files[j]), sim=sim, match=m, score=sim + attr_weight * m))
            named.append(cats)
        order = sorted(range(len(hits)), key=lambda i: -hits[i]['score'])
        enforced = []
        for cat in strict:
            for name in (named[0].get(cat, {}) if named else {}):
                keep = [i for i in order if named[i][cat][name] >= 1]
                if keep:  # some file carries it: only those
                    order, enforced = keep, enforced + [name]
        for h in hits:
            h['strict'] = tuple(enforced)
        return [hits[i] for i in order[:k]]

    def opening(self, i, notes):
        """(tokens (n, 4), programs (n,)) of the first `notes` notes of file i (programs 0 for piano stores)."""
        a = int(self.offsets[i])
        b = min(int(self.offsets[i + 1]), a + notes)
        n = int(self.offsets[-1])
        tokens = np.memmap(self.store / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
        prog_path = self.store / 'programs.u8'
        prog = (np.asarray(np.memmap(prog_path, dtype=np.uint8, mode='r', shape=(n,))[a:b], dtype=np.int64)
                if prog_path.exists() else np.zeros(b - a, np.int64))
        return np.asarray(tokens[a:b], dtype=np.int64), prog
