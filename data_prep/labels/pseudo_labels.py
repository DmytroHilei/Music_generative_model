"""
Pseudo-labels for every file of a token store (text-control plan 15h, step 1.5): genre / emotion / mood / mode
predicted from the model's own hidden states by the classifiers eval/probes/label_coverage.py --save wrote.

    python data_prep/labels/pseudo_labels.py --stores data/cache/gigamidi_replay_train,data/cache/gigamidi_heldout_validation

Each file is embedded like in eval/probes/embed_probe.py (first --notes notes with BOS, layers of the classifier file
mean-pooled over positions >= --skip, plus the note statistics), standardized with the classifier file's mean / std,
and scored by the --kind classifier of every task. Files are batched in order of length (less padding), results are
written in store order.

Writes data/labels/<store>.pseudo.npz, one row per store file: <task>_pred (class index), <task>_conf (top softmax
probability), <task>_keep (conf >= the threshold that gave --precision on half A in the coverage file); and
<store>.pseudo.json: classes per task, thresholds for every target precision of the coverage file, kept counts.
--save-emb also writes the pooled vectors (float16) to <store>.emb.npz.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'eval' / 'probes'))

from embed_probe import embed, read_store  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stores', required=True, help='comma-separated store directories')
    p.add_argument('--checkpoint', default='checkpoints/long_450m_final')
    p.add_argument('--classifiers', default='results/embed_probe/classifiers_l7_14.pt')
    p.add_argument('--coverage', default='results/embed_probe/coverage_l7_14.json')
    p.add_argument('--kind', default='linear', choices=('linear', 'mlp'))
    p.add_argument('--precision', type=float, default=0.8, help='target precision of the keep mask')
    p.add_argument('--notes', type=int, default=1024)
    p.add_argument('--skip', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-files', type=int, default=0, help='per store, 0 = all')
    p.add_argument('--save-emb', action='store_true')
    p.add_argument('--out-dir', default='data/labels')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def build(kind, state, n_classes):
    d = state['weight' if kind == 'linear' else '1.weight'].shape[1]
    model = (nn.Linear(d, n_classes) if kind == 'linear' else
             nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n_classes)))
    model.load_state_dict(state)
    return model.eval()


def main():
    args = parse_args()
    clf = torch.load(args.classifiers, weights_only=False)
    cov = json.loads(Path(args.coverage).read_text())['results']
    layers = clf['args']['layers'].split(',')
    mean, std = clf['mean'].astype(np.float32), clf['std'].astype(np.float32)
    heads = {}
    for name, c in clf['classifiers'].items():
        task, kind = name.split('/')
        if kind == args.kind:
            heads[task] = (build(kind, c['state'], len(c['classes'])).to(args.device), c['classes'])
    key = f'prec{int(round(args.precision * 100))}'
    model, it = load_model(args.checkpoint, args.device, torch.bfloat16)
    print(f'{args.checkpoint} @{it}: layers {layers}, {args.kind} classifiers for {", ".join(heads)}')
    stores = args.stores.split(',')
    sizes = {s: len(np.load(Path(s) / 'offsets.npy')) - 1 for s in stores}
    sizes = {s: min(n, args.max_files) if args.max_files else n for s, n in sizes.items()}
    job = JobStatus('pseudo_labels', sum(sizes.values()), f'{Path(args.checkpoint).name}, {len(stores)} store(s)')
    done = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for store in stores:
        name = Path(store).name
        offsets = read_store(store)[0]
        n = sizes[store]
        order = np.argsort(-np.minimum(np.diff(offsets[:n + 1]), args.notes), kind='stable')
        rows = [{'file': int(i)} for i in order]
        emb_args = argparse.Namespace(notes=args.notes, skip=args.skip, windows=1, batch_size=args.batch_size,
                                      device=args.device)
        emb, stats = embed(model, store, rows, emb_args, layers, job, done)
        done += n
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        emb = {k: v[inv] for k, v in emb.items()}
        stats = stats[inv]
        x = np.nan_to_num(np.concatenate([emb[k].astype(np.float32) for k in layers] + [stats.astype(np.float32)], 1))
        x = torch.tensor((x - mean) / std, device=args.device)
        out, meta = {}, dict(checkpoint=args.checkpoint, iter=it, classifiers=args.classifiers, kind=args.kind,
                             precision=args.precision, n_files=n, tasks={})
        for task, (head, classes) in heads.items():
            with torch.no_grad():
                p = torch.cat([F.softmax(head(x[i:i + 8192]), 1) for i in range(0, n, 8192)]).cpu().numpy()
            conf, pred = p.max(1), p.argmax(1)
            thresholds = {t: v['threshold'] for t, v in cov[task][args.kind].items()
                          if t.startswith('prec') and v}
            thr = thresholds.get(key)
            keep = conf >= thr if thr is not None else np.zeros(n, bool)
            out[f'{task}_pred'], out[f'{task}_conf'], out[f'{task}_keep'] = pred.astype(np.int8), conf.astype(np.float16), keep
            counts = np.bincount(pred[keep], minlength=len(classes))
            meta['tasks'][task] = dict(classes=classes, thresholds=thresholds, kept=int(keep.sum()),
                                       kept_share=round(float(keep.mean()), 3),
                                       kept_per_class={c: int(k) for c, k in zip(classes, counts)})
            print(f'{name} {task:15s} kept {keep.mean():6.1%} at {key} (threshold {thr}): '
                  + ', '.join(f'{c} {k}' for c, k in zip(classes, counts)))
        np.savez(out_dir / f'{name}.pseudo.npz', **out)
        (out_dir / f'{name}.pseudo.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        if args.save_emb:
            np.savez(out_dir / f'{name}.emb.npz', **{k: v.astype(np.float16) for k, v in emb.items()},
                     stats=stats.astype(np.float32))
        print(f'wrote {out_dir}/{name}.pseudo.npz')
    job.finish(f'pseudo-labels -> {out_dir}')


if __name__ == '__main__':
    main()
