"""
Does a text-conditioned model follow composer / artist prompts? (composer captions of musicar/text/artists.py)

Generates --notes notes from BOS for 'a piece by <composer>' (EN) and "п'єса <genitive>" (UK) per composer, plus
'none', and judges the rows against real music: the openings (first --notes notes) of up to --ref-files Aria train
files per composer give a centroid of style features (notes/s, onsets/s, notes per onset, pitch mean / spread /
range, velocity mean / spread, tempo, note length, onset regularity, diatonic share; z-scored over all reference
openings). A generated row is assigned to the nearest
centroid; accuracy = share of a composer's rows assigned to that composer (chance 1 / composers). Ceiling: the same
classifier on held-out real openings (centroids from one half of the reference files, judged on the other half).
Next to it per prompt: median distance to the own centroid minus the 'none' rows' median distance to it (< 0: the
prompt moves toward the composer). Band prompts ('a song by Queen', "пісня гурту Queen") are only judged on
instruments: drums / guitar / piano share against the composer prompts.

    python eval/compare/composer_eval.py --checkpoints checkpoints/v2_A0 --out results/composer_eval/v2_A0

Writes <out>/rows.csv, <out>/summary.md and MIDI of the first --midi rows per prompt (<out>/<checkpoint>/cfg<s>/).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.data.loader import BOS_PITCH  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
from musicar.data.midi_io import tokens_to_midi  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.text import artists as A  # noqa: E402
from musicar.text.captions import StoreCaptions  # noqa: E402
from musicar.text.encoder import TextEncoder  # noqa: E402
from text_eval import row_metrics, slug  # noqa: E402

sys.path.insert(0, str(ROOT / 'data_prep' / 'labels'))
from song_features import FIELDS, file_features  # noqa: E402

COMPOSERS = ('joplin', 'bach', 'chopin', 'satie', 'liszt', 'mozart', 'hisaishi', 'gershwin')
BANDS = ('beatles', 'queen')
STYLE = ('nps', 'ops', 'notes_per_onset', 'pitch_mean', 'pitch_std', 'range', 'vel_mean', 'vel_std', 'tempo',
         'dur_mean', 'ioi_cv', 'diatonic')
METRICS = ('nps', 'pitch', 'velocity', 'piano', 'drums', 'guitar', 'n_notes')


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs with text_dim')
    p.add_argument('--out', default='results/composer_eval/run')
    p.add_argument('--composers', default=','.join(COMPOSERS), help='artists.py ids judged against real openings')
    p.add_argument('--bands', default=','.join(BANDS), help='artists.py ids judged on instruments only')
    p.add_argument('--rows', type=int, default=16, help='rows per prompt')
    p.add_argument('--notes', type=int, default=512, help='new notes per row (no EOS before)')
    p.add_argument('--ref-store', default='data/cache/aria_train')
    p.add_argument('--ref-files', type=int, default=300, help='reference files per composer')
    p.add_argument('--labels', default='data/labels')
    p.add_argument('--cfg', default='1,2', help='guidance scales')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--midi', type=int, default=2, help='MIDI files written per prompt and scale')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--summarize', action='store_true', help='only rewrite <out>/summary.md from <out>/rows.csv')
    return p.parse_args()


def style_vector(tok, prog):
    """STYLE features of one note sequence (tok: (n, 4) pitch, velocity, duration, dt): song_features' fields, the
    mean note length (s), the spread of the time between onsets (std / mean) and the share of duration-weighted pitch
    in the 7 most used pitch classes."""
    f = dict(zip(FIELDS, file_features(tok, prog)[0]))
    dt = tok[:, 3][tok[:, 3] > 0] * 0.02
    pc = np.sort(np.bincount(tok[:, 0] % 12, weights=tok[:, 2] + 1.0, minlength=12))[::-1]
    return np.array([f['nps'], f['ops'], f['notes_per_onset'], f['pitch_mean'], f['pitch_std'],
                     f['pitch_hi'] - f['pitch_lo'], f['vel_mean'], f['vel_std'], f['tempo'], tok[:, 2].mean() * 0.02,
                     dt.std() / dt.mean() if len(dt) > 1 else 0.0, pc[:7].sum() / pc.sum()])


def prompts(composers, bands):
    """(name, text, artist id) for 'none', each composer and band in EN and UK."""
    out = [('none', '', '')]
    for a in composers + bands:
        e = A.ARTISTS[a]
        if e['band']:
            name = e['en'][0]
            out += [(a, f"a song by {name}", a), (f'uk_{a}', f"пісня гурту {name[0].upper() + name[1:]}", a)]
        else:
            out += [(a, f"a piece by {e['en'][0]}", a)]
            if e['uk']:
                out.append((f'uk_{a}', f"п'єса {e['uk'][0][1]}", a))
    return out


def reference(store, labels, composers, n_files, notes, seed):
    """{composer: (k, len(STYLE)) style vectors of the openings of up to n_files of its files in the store}."""
    store = Path(store)
    caps = StoreCaptions(store.name, labels_dir=labels)
    offsets = np.load(store / 'offsets.npy')
    tokens = np.memmap(store / 'tokens.u16', dtype=np.uint16, mode='r', shape=(int(offsets[-1]), 4))
    progs = np.memmap(store / 'programs.u8', dtype=np.uint8, mode='r', shape=(int(offsets[-1]),)) \
        if (store / 'programs.u8').exists() else None
    rng = np.random.default_rng(seed)
    ref = {}
    for c in composers:
        files = np.array(sorted(i for i, a in caps.artist.items() if a == c))
        rows = []
        for i in rng.permutation(files):
            a, b = int(offsets[i]), int(offsets[i + 1])
            if b - a < notes // 4:
                continue
            b = min(b, a + notes)
            prog = progs[a:b].astype(np.int64) if progs is not None else np.zeros(b - a, np.int64)
            rows.append(style_vector(np.asarray(tokens[a:b]).astype(np.int64), prog))
            if len(rows) == n_files:
                break
        assert len(rows) >= 20, f'{c}: only {len(rows)} reference files in {store}'
        ref[c] = np.stack(rows)
    return ref


class Centroids:
    """Nearest-centroid composer classifier on z-scored style vectors."""

    def __init__(self, ref):
        allv = np.concatenate(list(ref.values()))
        self.mu, self.sd = allv.mean(0), allv.std(0) + 1e-9
        self.names = list(ref)
        self.c = np.stack([((v - self.mu) / self.sd).mean(0) for v in ref.values()])

    def dist(self, v):
        """(len(v), composers) distances."""
        z = (np.atleast_2d(v) - self.mu) / self.sd
        return np.linalg.norm(z[:, None] - self.c[None], axis=-1)

    def predict(self, v):
        return [self.names[j] for j in self.dist(v).argmin(1)]


def ceiling(ref):
    """Nearest-centroid accuracy on real openings: centroids from the first half of each composer's files, judged on
    the second half (per composer and mean)."""
    half = {c: v[:len(v) // 2] for c, v in ref.items()}
    clf = Centroids(half)
    acc = {c: float(np.mean([p == c for p in clf.predict(v[len(v) // 2:])])) for c, v in ref.items()}
    return acc, float(np.mean(list(acc.values())))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    composers = [c for c in args.composers.split(',') if c]
    bands = [b for b in args.bands.split(',') if b]
    t0 = time.time()
    ref = reference(args.ref_store, args.labels, composers, args.ref_files, args.notes, args.seed)
    print('reference openings: ' + ', '.join(f'{c} {len(v)}' for c, v in ref.items()) + f' ({time.time() - t0:.0f} s)')
    np.savez(out / 'reference.npz', **ref)
    if args.summarize:
        with open(out / 'rows.csv') as f:
            rows = [{k: v if k in ('checkpoint', 'prompt', 'text', 'artist') else float(v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        (out / 'summary.md').write_text(summarize(rows, ref, composers, bands))
        print((out / 'summary.md').read_text())
        return
    plist = prompts(composers, bands)
    scales = [float(s) for s in args.cfg.split(',')]
    ckpts = [c for c in args.checkpoints.split(',') if c]
    encoder = TextEncoder(device=args.device)
    vectors = encoder.encode([text for _, text, _ in plist])
    token_states = encoder.encode([text for _, text, _ in plist], tokens=True)  # T2 prefix models
    del encoder
    torch.cuda.empty_cache()
    job = JobStatus('composer_eval', len(ckpts) * len(scales) * len(plist), f'{len(ckpts)} checkpoint(s) -> {out}')
    rows, done = [], 0
    for ck in ckpts:
        model, it = load_model(ck, args.device, torch.bfloat16 if args.device == 'cuda' else torch.float32)
        assert model.config.text_dim, f'{ck} has no text input'
        name = Path(ck).name if Path(ck).is_dir() else Path(ck).parent.name
        for s in scales:
            t1 = time.time()
            inputs = token_states if model.config.text_prefix else vectors
            for (pname, text, artist), vec in zip(plist, inputs):
                torch.manual_seed(args.seed)  # the same noise for every prompt: differences come from the text
                start = [torch.full((args.rows, 1), BOS_PITCH, device=args.device)] + \
                    [torch.zeros(args.rows, 1, dtype=torch.long, device=args.device)] * 3
                res = generate(model, *start, max_new_tokens=args.notes, temperature=args.temperature,
                               top_p=args.top_p, program=None, programs=torch.zeros_like(start[0]),
                               return_programs=True, min_new=args.notes, text=vec if text else None,
                               cfg_scale=s if text else 1.0)
                arrays = [t[:, 1:].cpu().numpy() for t in res]
                for r in range(args.rows):
                    p, v, du, dt, pr = (a[r] for a in arrays)
                    keep = np.cumsum(p >= BOS_PITCH) == 0  # up to the first EOS
                    m = row_metrics(p, v, du, dt, pr)
                    style = style_vector(np.stack([x[keep] for x in (p, v, du, dt)], 1).astype(np.int64),
                                         pr[keep].astype(np.int64)) if keep.sum() >= 16 else np.full(len(STYLE), np.nan)
                    rows.append(dict(checkpoint=name, iter=it, cfg=s, prompt=pname, text=text, artist=artist, row=r,
                                     **{k: m[k] for k in METRICS}, **{f's_{k}': x for k, x in zip(STYLE, style)}))
                    if r < args.midi:
                        d = out / name / f'cfg{s:g}'
                        d.mkdir(parents=True, exist_ok=True)
                        tokens_to_midi(p[keep], v[keep], du[keep], dt[keep], programs=pr[keep].tolist()).write(
                            str(d / f'{slug(pname)}_{r}.mid'))
                done += 1
                job.update(done, f'{name} cfg {s:g}: {pname}')
            print(f'{name} cfg {s:g}: {len(plist)} prompts x {args.rows} rows in {time.time() - t1:.0f} s')
        del model
        torch.cuda.empty_cache()
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, ref, composers, bands)
    (out / 'summary.md').write_text(summary)
    print(summary)
    job.finish(f'{out}/summary.md')


def summarize(rows, ref, composers, bands):
    clf = Centroids(ref)
    acc_real, ceil = ceiling(ref)
    lines = [f'Composers: {", ".join(composers)} (chance {1 / len(composers):.2f}); real openings ceiling '
             f'{ceil:.2f} (' + ', '.join(f'{c} {a:.2f}' for c, a in acc_real.items()) + ')', '']
    groups = {}
    for r in rows:
        groups.setdefault((r['checkpoint'], r['cfg']), {}).setdefault(r['prompt'], []).append(r)
    for (ck, s), by_prompt in groups.items():
        def style(rs):
            v = np.array([[r[f's_{k}'] for k in STYLE] for r in rs], float)
            return v[np.isfinite(v).all(1)]
        none_v = style(by_prompt['none'])
        none_d = clf.dist(none_v) if len(none_v) else None
        lines += [f'## {ck}, cfg {s:g}', '',
                  '| prompt | rows | accuracy | none rows as it | median distance to own - none | top prediction |',
                  '|---|---|---|---|---|---|']
        accs = {'en': [], 'uk': []}
        for p, rs in by_prompt.items():
            a = rs[0]['artist']
            if a not in composers:
                continue
            v = style(rs)
            if not len(v):
                continue
            pred = clf.predict(v)
            j = clf.names.index(a)
            acc = float(np.mean([x == a for x in pred]))
            base = float(np.mean(none_d.argmin(1) == j)) if none_d is not None else np.nan
            delta = float(np.median(clf.dist(v)[:, j]) - np.median(none_d[:, j])) if none_d is not None else np.nan
            top = max(set(pred), key=pred.count)
            accs['uk' if p.startswith('uk_') else 'en'].append(acc)
            lines.append(f'| {p} | {len(v)} | {acc:.2f} | {base:.2f} | {delta:+.2f} | {top} |')
        lines += ['', 'mean accuracy: ' + ', '.join(f'{k.upper()} {np.mean(v):.2f}' for k, v in accs.items() if v)
                  + f' (chance {1 / len(composers):.2f}, real ceiling {ceil:.2f})', '']
        if bands:
            comp = [r for p, rs in by_prompt.items() for r in rs if rs[0]['artist'] in composers]
            lines += ['| prompt | piano | drums | guitar | nps |', '|---|---|---|---|---|']
            for label, rs in [('composer prompts', comp), ('none', by_prompt['none'])] + \
                    [(p, rs) for p, rs in by_prompt.items() if rs[0]['artist'] in bands]:
                m = {k: float(np.nanmean([r[k] for r in rs])) for k in ('piano', 'drums', 'guitar')}
                nps = float(np.nanmedian([r['nps'] for r in rs]))
                lines.append(f'| {label} | {m["piano"]:.2f} | {m["drums"]:.2f} | {m["guitar"]:.2f} | {nps:.1f} |')
            lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
