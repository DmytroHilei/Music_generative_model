"""
Linear probe: do a model's hidden states know genre / emotion / key mode? Text-control plan 15h, phase 1.4 (the gate
for cluster labels).

    python eval/embed_probe.py --checkpoint checkpoints/long_450m_final --out results/embed_probe

Files with labels come from data_prep/build_labels.py (data/labels/<store>.jsonl); --train-store files fit the
probes, --test-store files score them. Each file is embedded from its first --notes notes (BOS in front, the piece
start), or with --windows 3 also from the middle and the end (no BOS), averaged: the hidden states of --layers (block outputs, 'final' = after ln_f) mean-pooled over positions >= --skip
(early positions have little causal context). No style / instrument / density conditioning (cond 'none').
Baseline: the same probes on plain note statistics (pitch-class and program histograms, notes/s, velocity,
duration, register, polyphony), so the result says what the model adds beyond them.

Probes: standardized features -> multinomial logistic regression (full-batch Adam, 500 steps, L2), accuracy and macro-F1 on
the test files, next to the majority-class rate. Tasks: XMIDI emotion (11) and genre (6), MidiCaps top genre and key
mode (major / minor; a control the model should know). Writes <out>/embeddings.npz (+ embed_meta.json) before probing, then <out>/probe.json; --reuse probes saved embeddings.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from musicar.checkpoint import load_model
from musicar.jobstatus import JobStatus
from musicar.sampling import prompt_inputs


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', default='checkpoints/long_450m_final')
    p.add_argument('--train-store', default='data/cache/gigamidi_replay_train')
    p.add_argument('--test-store', default='data/cache/gigamidi_heldout_validation')
    p.add_argument('--labels-dir', default='data/labels')
    p.add_argument('--notes', type=int, default=1024)
    p.add_argument('--skip', type=int, default=32)
    p.add_argument('--windows', type=int, default=1, choices=(1, 3), help='3 = start, middle and end of each file')
    p.add_argument('--layers', default='7,14,21,final', help='block indices (0-based) and/or final')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-files', type=int, default=0, help='per store, 0 = all labeled files')
    p.add_argument('--l2', type=float, default=1e-3)
    p.add_argument('--out', default='results/embed_probe')
    p.add_argument('--device', default='cuda')
    p.add_argument('--reuse', action='store_true', help='probe the embeddings saved in --out by an earlier run')
    return p.parse_args()


def labeled_files(labels_dir, store, max_files):
    rows = [json.loads(line) for line in open(Path(labels_dir) / f'{Path(store).name}.jsonl', encoding='utf-8')]
    return rows[:max_files] if max_files else rows


def read_store(store):
    offsets = np.load(Path(store) / 'offsets.npy')
    n = int(offsets[-1])
    tokens = np.memmap(Path(store) / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    path = Path(store) / 'programs.u8'  # piano-only stores have none: program 0 (as in musicar/data_loader.py)
    programs = (np.memmap(path, dtype=np.uint8, mode='r', shape=(n,)) if path.exists()
                else np.broadcast_to(np.zeros(1, np.uint8), (n,)))
    return offsets, tokens, programs


def note_stats(tok, prog):
    """Hand-made baseline features of one window: 12 pitch classes, 129 programs, 8 scalars."""
    pitch, vel, dur, dt = (tok[:, j].astype(np.float64) for j in range(4))
    pitched = prog != 128
    pc = np.bincount(pitch[pitched].astype(int) % 12, minlength=12) / max(pitched.sum(), 1)
    pr = np.bincount(prog.astype(int), minlength=129) / len(prog)
    seconds = max(dt.sum() * 0.02, 1e-3)
    onsets = max((dt > 0).sum(), 1)
    p = pitch[pitched] if pitched.any() else np.zeros(1)
    scalars = [len(tok) / seconds, vel.mean(), vel.std(), np.log1p(dur).mean(), p.mean(), p.std(),
               len(tok) / onsets, (prog == 128).mean()]
    return np.concatenate([pc, pr, scalars])


def window_starts(n, length, windows):
    """Start of each window in a file of n notes: the beginning, then (windows = 3) the middle and the end."""
    if windows == 1 or n <= length:
        return [0] * windows
    return [0, (n - length) // 2, n - length]


@torch.no_grad()
def pool(model, tok, prog, lens, bos, args, layers, captured):
    """Mean hidden state per row over positions skip .. len (shifted by 1 after BOS). tok (B, T, 4), lens per row."""
    streams, progs, cond = prompt_inputs(model, torch.from_numpy(tok).to(args.device),
                                         torch.from_numpy(prog).to(args.device))
    if not bos:  # a window from inside the piece: no BOS in front (prompt_inputs adds one for BOS models)
        if streams[0].size(1) > tok.shape[1]:
            streams, progs = tuple(x[:, 1:] for x in streams), progs[:, 1:]
    h = model.encode(*streams, program=progs, cond=cond)
    shift = h.size(1) - tok.shape[1]  # 1 with BOS, else 0
    pos = torch.arange(h.size(1), device=h.device)
    lo = torch.tensor([shift + args.skip if n >= args.skip + 16 else shift for n in lens], device=h.device)
    hi = torch.tensor(lens, device=h.device) + shift
    mask = ((pos >= lo[:, None]) & (pos < hi[:, None])).unsqueeze(-1).float()
    return {name: (((h if name == 'final' else captured[name]).float() * mask).sum(1) / mask.sum(1)).cpu().numpy()
            for name in layers}


@torch.no_grad()
def embed(model, store, rows, args, layers, job=None, done=0):
    """Per file: the pooled vector of each layer averaged over --windows windows of --notes notes (the first one
    starts at the piece with BOS, the others in the middle / at the end without BOS), and the note statistics
    averaged over the same windows."""
    offsets, tokens, programs = read_store(store)
    hooks, captured = [], {}
    for name in layers:
        if name != 'final':
            hooks.append(model.transformer.h[int(name)].register_forward_hook(
                lambda m, i, o, name=name: captured.__setitem__(name, o)))
    out = {name: [] for name in layers}
    stats = []
    for s in tqdm(range(0, len(rows), args.batch_size), desc=Path(store).name):
        batch = rows[s:s + args.batch_size]
        acc = {name: np.zeros((len(batch), model.config.n_embd)) for name in layers}
        st = np.zeros((len(batch), 149))
        for w in range(args.windows):
            wins = []
            for r in batch:
                a0, b0 = int(offsets[r['file']]), int(offsets[r['file'] + 1])
                start = a0 + window_starts(b0 - a0, args.notes, args.windows)[w]
                wins.append((start, min(b0, start + args.notes), start == a0))
            for bos in (True, False):
                idx = [i for i, (_, _, first) in enumerate(wins) if first == bos]
                if not idx:
                    continue
                lens = [wins[i][1] - wins[i][0] for i in idx]
                tok = np.zeros((len(idx), max(lens), 4), np.int64)  # padded at the end (causal: no effect)
                prog = np.zeros((len(idx), max(lens)), np.int64)
                for j, i in enumerate(idx):
                    a, b, _ = wins[i]
                    tok[j, :b - a], prog[j, :b - a] = tokens[a:b], programs[a:b]
                    st[i] += note_stats(tok[j, :b - a], prog[j, :b - a]) / args.windows
                pooled = pool(model, tok, prog, lens, bos, args, layers, captured)
                for name in layers:
                    acc[name][idx] += pooled[name] / args.windows
        for name in layers:
            out[name].append(acc[name])
        stats.append(st)
        if job:
            job.update(done + s + len(batch), detail=f'{Path(store).name}, {args.windows} window(s)')
    for hk in hooks:
        hk.remove()
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(stats)


def tasks(rows):
    """task -> list of labels (None = file has no label for it)."""
    def mode(r):
        key = r.get('midicaps', {}).get('key')
        return key.split()[-1] if key else None
    return {
        'xmidi_emotion': [r['xmidi'][0][0] if 'xmidi' in r else None for r in rows],
        'xmidi_genre': [r['xmidi'][0][1] if 'xmidi' in r else None for r in rows],
        'midicaps_genre': [r['midicaps']['genre'][0] if r.get('midicaps', {}).get('genre') else None for r in rows],
        'midicaps_mode': [mode(r) for r in rows],
    }


def probe(xtr, ytr, xte, yte, l2, device):
    classes = sorted(set(ytr))
    idx = {c: i for i, c in enumerate(classes)}
    keep = [i for i, y in enumerate(yte) if y in idx]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a = torch.tensor((xtr - mu) / sd, dtype=torch.float32, device=device)
    b = torch.tensor((xte[keep] - mu) / sd, dtype=torch.float32, device=device)
    ya = torch.tensor([idx[y] for y in ytr], device=device)
    yb = torch.tensor([idx[yte[i]] for i in keep], device=device)
    w = torch.zeros(a.size(1), len(classes), device=device, requires_grad=True)
    bias = torch.zeros(len(classes), device=device, requires_grad=True)
    opt = torch.optim.Adam([w, bias], lr=0.01)
    for _ in range(500):  # full batch; L-BFGS overflowed on the 1152-d embeddings
        opt.zero_grad()
        loss = F.cross_entropy(a @ w + bias, ya) + l2 * w.pow(2).sum()
        loss.backward()
        opt.step()
    pred = (b @ w + bias).argmax(1)
    acc = (pred == yb).float().mean().item()
    f1 = []
    for c in range(len(classes)):
        tp = ((pred == c) & (yb == c)).sum().item()
        fp = ((pred == c) & (yb != c)).sum().item()
        fn = ((pred != c) & (yb == c)).sum().item()
        if tp + fn:
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    majority = max(np.bincount(yb.cpu().numpy(), minlength=len(classes))) / len(keep)
    return dict(acc=round(acc, 3), macro_f1=round(float(np.mean(f1)), 3), majority=round(float(majority), 3),
                n_train=len(ytr), n_test=len(keep), classes=len(classes))


def main():
    args = parse_args()
    layers = args.layers.split(',')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {s: labeled_files(args.labels_dir, s, args.max_files) for s in (args.train_store, args.test_store)}
    emb, stats = {}, {}
    if args.reuse:
        z = np.load(out / 'embeddings.npz')
        it = json.loads((out / 'embed_meta.json').read_text())['iter']
        for s in rows:
            emb[s] = {k: z[f'{Path(s).name}__{k}'].astype(np.float32) for k in layers}
            stats[s] = z[f'{Path(s).name}__stats'].astype(np.float32)
    else:
        model, it = load_model(args.checkpoint, args.device, torch.bfloat16)
        print(f'{args.checkpoint} @{it}: {model.config.n_layer} layers x {model.config.n_embd}')
        job = JobStatus('embed_probe', sum(len(r) for r in rows.values()), f'{Path(args.checkpoint).name} -> {out}')
        done = 0
        for s in rows:
            emb[s], stats[s] = embed(model, s, rows[s], args, layers, job, done)
            done += len(rows[s])
        job.finish(f'embeddings -> {out}')
        del model
        np.savez(out / 'embeddings.npz', **{f'{Path(s).name}__{k}': v.astype(np.float16) for s in emb for k, v in emb[s].items()},
                 **{f'{Path(s).name}__stats': stats[s] for s in stats})
        (out / 'embed_meta.json').write_text(json.dumps(dict(args=vars(args), iter=it)))
    for s in rows:
        for k, v in list(emb[s].items()) + [('stats', stats[s])]:
            bad = ~np.isfinite(v)
            if bad.any():
                print(f'{Path(s).name} {k}: {bad.sum()} non-finite values set to 0')
                v[bad] = 0
    tr, te = args.train_store, args.test_store
    t_tr, t_te = tasks(rows[tr]), tasks(rows[te])
    results = {}
    feats = {f'layer_{k}': (emb[tr][k], emb[te][k]) for k in layers}
    feats['note_stats'] = (stats[tr], stats[te])
    for task in t_tr:
        i_tr = [i for i, y in enumerate(t_tr[task]) if y is not None]
        i_te = [i for i, y in enumerate(t_te[task]) if y is not None]
        for name, (xa, xb) in feats.items():
            r = probe(xa[i_tr], [t_tr[task][i] for i in i_tr], xb[i_te], [t_te[task][i] for i in i_te],
                      args.l2, args.device)
            results.setdefault(task, {})[name] = r
            print(f'{task:15s} {name:12s} acc {r["acc"]:.3f}  macro-F1 {r["macro_f1"]:.3f}  '
                  f'(majority {r["majority"]:.3f}, {r["classes"]} classes, {r["n_train"]}/{r["n_test"]} files)')
    (out / 'probe.json').write_text(json.dumps(dict(args=vars(args), iter=it, results=results), indent=1))
    print(f'wrote {out}/probe.json')


if __name__ == '__main__':
    main()
