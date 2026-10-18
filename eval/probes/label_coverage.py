"""
Precision vs coverage of pseudo-labels: how many unlabeled songs could get a genre / emotion word, and how often it
would be right. Text-control plan 15h (labels for the songs without MidiCaps / XMIDI labels).

    python eval/probes/label_coverage.py --emb results/embed_probe --layer 14

Reads the embeddings eval/probes/embed_probe.py saved (one pooled vector + note statistics per labeled file). Classifiers are
fit on the --train-store files and judged on the --test-store files, which are split in two halves: the confidence
threshold for a target precision is picked on half A and reported on half B (so the numbers aren't tuned on the files
they describe). Confidence = the classifier's top softmax probability.

Classifiers: 'linear' (logistic regression), 'mlp' (one hidden layer, dropout, early stop on 10% of train) and 'knn'
(the k most similar labeled songs by cosine after standardizing, votes weighted by similarity; confidence = vote
share), on the --layers vectors (concatenated) + note statistics. Extra task xmidi_mood = XMIDI emotion in 4 groups
(MOOD). Writes <emb>/coverage[_<tag>].json and, with --save, the linear + MLP classifiers (weights, standardization,
classes) to <emb>/classifiers[_<tag>].pt.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embed_probe import labeled_files, tasks

TARGETS = (0.6, 0.7, 0.8, 0.9)
# XMIDI's 11 emotions in 4 valence x arousal groups (annotators mostly disagree inside a group: of the 1,491 files
# with two emotion labels, most pairs are warm/romantic-like neighbours)
MOOD = {'exciting': 'energetic', 'happy': 'energetic', 'funny': 'energetic', 'magnificent': 'energetic',
        'warm': 'calm', 'romantic': 'calm', 'lazy': 'calm', 'quiet': 'calm',
        'sad': 'sad', 'angry': 'tense', 'fear': 'tense'}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--emb', default='results/embed_probe')
    p.add_argument('--layers', default='14', help="comma-separated, concatenated; 'none' = note statistics only")
    p.add_argument('--k', type=int, default=25, help='neighbours for knn')
    p.add_argument('--tag', default='', help='suffix of the output files')
    p.add_argument('--save', action='store_true', help='save the linear + MLP classifiers')
    p.add_argument('--train-store', default='data/cache/gigamidi_replay_train')
    p.add_argument('--test-store', default='data/cache/gigamidi_heldout_validation')
    p.add_argument('--labels-dir', default='data/labels')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def fit(kind, xa, ya, n_classes, device, seed):
    torch.manual_seed(seed)
    d = xa.size(1)
    model = (nn.Linear(d, n_classes) if kind == 'linear' else
             nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n_classes)))
    model.to(device)
    perm = torch.randperm(len(ya), generator=torch.Generator().manual_seed(seed)).to(device)
    n_val = len(ya) // 10
    vi, ti = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3 if kind == 'linear' else 1e-3, weight_decay=1e-2)
    best, best_state, bad = 1e9, None, 0
    for step in range(3000):
        model.train()
        idx = ti[torch.randint(len(ti), (512,), device=device)]
        loss = F.cross_entropy(model(xa[idx]), ya[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                v = F.cross_entropy(model(xa[vi]), ya[vi]).item()
            if v < best - 1e-4:
                best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10:
                    break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def knn(xa, ya, xb, n_classes, k):
    """Class probabilities from the k nearest training files (cosine on the standardized features), votes weighted
    by similarity."""
    a, b = F.normalize(xa, dim=1), F.normalize(xb, dim=1)
    sim, idx = (b @ a.T).topk(k, dim=1)
    w = sim.clamp(min=0)
    p = torch.zeros(len(xb), n_classes, device=xb.device).scatter_add_(1, ya[idx], w)
    return (p / p.sum(1, keepdim=True).clamp(min=1e-9)).cpu().numpy()


def curve(conf, correct, half):
    """Accuracy at fixed coverage, and coverage at target precision with the threshold from half A applied to B."""
    a, b = half, ~half
    out = {'acc_all': round(float(correct[b].mean()), 3)}
    order = np.argsort(-conf[b])
    for cov in (0.2, 0.4, 0.6, 0.8):
        k = max(int(cov * b.sum()), 1)
        out[f'acc_at_cov{int(cov * 100)}'] = round(float(correct[b][order[:k]].mean()), 3)
    oa = np.argsort(-conf[a])
    ca = np.cumsum(correct[a][oa]) / np.arange(1, a.sum() + 1)
    for t in TARGETS:
        ok = np.nonzero(ca >= t)[0]
        ok = ok[ok >= 9]  # need >= 10 files above the threshold on half A
        if not len(ok):
            out[f'prec{int(t * 100)}'] = None
            continue
        thr = conf[a][oa][ok[-1]]
        sel = conf[b] >= thr
        out[f'prec{int(t * 100)}'] = dict(threshold=round(float(thr), 3), coverage=round(float(sel.mean()), 3),
                                          precision=round(float(correct[b][sel].mean()), 3) if sel.any() else None)
    return out


def main():
    args = parse_args()
    z = np.load(Path(args.emb) / 'embeddings.npz')
    tr, te = Path(args.train_store).name, Path(args.test_store).name

    def feats(store):
        layers = [] if args.layers == 'none' else args.layers.split(',')  # 'none' = note statistics only
        x = np.concatenate([z[f'{store}__{k}'].astype(np.float32) for k in layers]
                           + [z[f'{store}__stats'].astype(np.float32)], 1)
        return np.nan_to_num(x)
    xtr, xte = feats(tr), feats(te)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    t_tr = tasks(labeled_files(args.labels_dir, args.train_store, 0))
    t_te = tasks(labeled_files(args.labels_dir, args.test_store, 0))
    for t in (t_tr, t_te):
        t['xmidi_mood'] = [MOOD[y] if y else None for y in t['xmidi_emotion']]
    saved = {}
    rng = np.random.default_rng(args.seed)
    results = {}
    for task in t_tr:
        i_tr = [i for i, y in enumerate(t_tr[task]) if y is not None]
        classes = sorted({t_tr[task][i] for i in i_tr})
        idx = {c: k for k, c in enumerate(classes)}
        i_te = [i for i, y in enumerate(t_te[task]) if y in idx]
        xa = torch.tensor(xtr[i_tr], device=args.device)
        ya = torch.tensor([idx[t_tr[task][i]] for i in i_tr], device=args.device)
        xb = torch.tensor(xte[i_te], device=args.device)
        yb = np.array([idx[t_te[task][i]] for i in i_te])
        half = rng.random(len(i_te)) < 0.5
        for kind in ('linear', 'mlp', 'knn'):
            if kind == 'knn':
                p = knn(xa, ya, xb, len(classes), args.k)
            else:
                model = fit(kind, xa, ya, len(classes), args.device, args.seed)
                with torch.no_grad():
                    p = F.softmax(model(xb), 1).cpu().numpy()
                saved[f'{task}/{kind}'] = dict(state=model.cpu().state_dict(), classes=classes)
            r = curve(p.max(1), p.argmax(1) == yb, half)
            r.update(classes=len(classes), n_test_half=int((~half).sum()))
            results.setdefault(task, {})[kind] = r
            line = ' '.join(f'cov{c}% {r[f"acc_at_cov{c}"]:.2f}' for c in (20, 40, 60, 80))
            prec = ' '.join(f'P>={int(t * 100)}%: ' + (f'{r[f"prec{int(t * 100)}"]["coverage"]:.0%} '
                            f'(P {r[f"prec{int(t * 100)}"]["precision"]:.2f})' if r[f'prec{int(t * 100)}'] else '-')
                            for t in TARGETS)
            print(f'{task:15s} {kind:6s} acc all {r["acc_all"]:.2f} | {line} | {prec}')
    tag = f'_{args.tag}' if args.tag else ''
    (Path(args.emb) / f'coverage{tag}.json').write_text(json.dumps(dict(args=vars(args), results=results), indent=1))
    print(f'wrote {args.emb}/coverage{tag}.json')
    if args.save:
        torch.save(dict(args=vars(args), mean=mu, std=sd, classifiers=saved), Path(args.emb) / f'classifiers{tag}.pt')
        print(f'wrote {args.emb}/classifiers{tag}.pt')


if __name__ == '__main__':
    main()
