"""
G4c of the scaling plan (§8.6): the share of a synthetic store's files with a held-out neighbour at cosine >= 0.95,
against the same share for files of the real quarter they were prompted from. A synthetic share more than 1 point
above the real one means the generator reproduces held-out music, and its CE gain is only an upper bound.

Embedding (cheap, no model): hashed counts of 3-grams of (pitch interval, delta_time bin capped at 63) over
consecutive notes, D = 4,096 buckets, sqrt counts, L2-normalised. Intervals make it transposition-invariant; a near
copy scores ~1, unrelated music of the same style well below 0.95. Each file is embedded over its notes after the
prompt (files.jsonl prompt_notes; synthetic files start with a real opening of the quarter) up to --notes; the real
files over the same span (notes prompt..prompt + --notes); held-out files in windows of --notes notes at a stride of half that.

    .venv/bin/python tools/sweeps/synth_leak.py --synth data/cache/synth_clean_train
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from musicar.data.loader import file_subset  # noqa: E402

D = 4096


def store(path):
    path = Path(path)
    off = np.load(path / 'offsets.npy')
    tok = np.memmap(path / 'tokens.u16', dtype=np.uint16, mode='r', shape=(int(off[-1]), 4))
    return off, tok


def embed(tok, spans):
    """(n, D) float32: one row per (start, end) span of the token rows."""
    out = np.zeros((len(spans), D), np.float32)
    for i, (a, b) in enumerate(spans):
        t = np.asarray(tok[a:b], dtype=np.int64)
        if len(t) < 4:
            continue
        iv = np.diff(t[:, 0]) + 128
        dt = np.minimum(t[1:, 3], 63)
        x = iv * 64 + dt
        h = (x[:-2] * 1_000_003 + x[1:-1]) * 1_000_003 + x[2:]
        np.add.at(out[i], (h * 2654435761 % (1 << 32)) % D, 1.0)
    out = np.sqrt(out)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def heldout_spans(off, notes):
    """Windows of `notes` notes at a stride of notes / 2 over every file (a whole short file as one)."""
    spans = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        spans += [(s, min(b, s + notes)) for s in range(a, max(a + 1, b - notes // 2), notes // 2)]
    return spans


def near_share(q, ref, threshold, chunk=4096):
    """(share of the q rows with a ref row at cosine >= threshold, max cosine per q row)."""
    r = torch.from_numpy(ref)
    best = np.concatenate([(torch.from_numpy(q[i:i + chunk]) @ r.T).amax(1).numpy() for i in range(0, len(q), chunk)])
    nonempty = q.any(1)
    return float((best[nonempty] >= threshold).mean()) if nonempty.any() else 0.0, best


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--synth', required=True, help='synthetic store dir (with files.jsonl)')
    p.add_argument('--heldout', default='data/cache/gigamidi_heldout_validation')
    p.add_argument('--real', default='data/cache/gigamidi_replay_train', help='the store the prompts came from')
    p.add_argument('--real-frac', type=float, default=0.25)
    p.add_argument('--notes', type=int, default=1024, help='notes per file after the prompt')
    p.add_argument('--threshold', type=float, default=0.95)
    p.add_argument('--max-files', type=int, default=40000, help='files per side (evenly spaced)')
    p.add_argument('--out', default='')
    a = p.parse_args()
    s_off, s_tok = store(ROOT / a.synth)
    recs = [json.loads(x) for x in (ROOT / a.synth / 'files.jsonl').read_text().splitlines()]
    idx = np.linspace(0, len(recs) - 1, min(len(recs), a.max_files)).astype(int)
    s_spans = [(int(s_off[i]) + recs[i]['prompt_notes'], min(int(s_off[i + 1]), int(s_off[i]) + recs[i]['prompt_notes']
                                                             + a.notes)) for i in idx]
    prompt = int(np.median([r['prompt_notes'] for r in recs]))
    r_off, r_tok = store(ROOT / a.real)
    n = np.diff(r_off)
    keep = np.flatnonzero(file_subset(Path(a.real).name, len(n), a.real_frac) & (n >= prompt + 64))
    keep = keep[np.linspace(0, len(keep) - 1, min(len(keep), a.max_files)).astype(int)]
    r_spans = [(int(r_off[i]) + prompt, min(int(r_off[i + 1]), int(r_off[i]) + prompt + a.notes)) for i in keep]
    h_off, h_tok = store(ROOT / a.heldout)
    held = embed(h_tok, heldout_spans(h_off, a.notes))
    syn, _ = near_share(embed(s_tok, s_spans), held, a.threshold)
    real, _ = near_share(embed(r_tok, r_spans), held, a.threshold)
    res = dict(synth=a.synth, heldout=a.heldout, real=a.real, threshold=a.threshold, embedding=f'hashed (interval, dt) '
               f'3-grams, D {D}', synth_files=len(s_spans), real_files=len(r_spans), heldout_files=len(held),
               synth_share=syn, real_share=real, excess=syn - real, g4c_pass=syn - real <= 0.01)
    out = ROOT / (a.out or f'results/aug_sweep/synth_leak_{Path(a.synth).name}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
