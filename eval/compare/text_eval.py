"""
Does a text-conditioned model follow its prompt? (context/claude/text_control.md, build step 10)

Generates from BOS (no MIDI prompt, instruments sampled) for a fixed set of EN + UK prompts and measures what is
measurable on the notes: notes/s, onsets/s, mean velocity, share of piano / drum / string / guitar notes, minor key
(Krumhansl-Schmuckler). Each prompt pair targets one attribute (fast vs slow, minor vs major, ...); the summary lists
the contrast per pair next to the unconditioned rows ('' prompt), for every checkpoint and guidance scale.

    python eval/compare/text_eval.py --checkpoints checkpoints/text_p1_full,checkpoints/text_p1_frozen --out results/text_eval/p1

Writes <out>/rows.csv (one line per generated row), <out>/summary.md, and MIDI of the first --midi rows per prompt
(<out>/<checkpoint>/cfg<s>/<prompt>_<i>.mid) for listening.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.data.loader import BOS_PITCH, EOS_PITCH  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
from musicar.data.midi_io import tokens_to_midi  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.text.encoder import TextEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_prep' / 'labels'))
from song_features import FIELDS, file_features  # noqa: E402

# (name, prompt); pairs share a stem and differ in the attribute: <stem>+ should raise the metric over <stem>-
PROMPTS = [
    ('none', ''),
    ('fast+', 'a fast, busy piece'), ('fast-', 'a slow, sparse piece'),
    ('minor+', 'a piece in a minor key'), ('minor-', 'a piece in a major key'),
    ('loud+', 'a loud, powerful piece'), ('loud-', 'a soft, quiet piece'),
    ('high+', 'a high-pitched piece'), ('high-', 'a deep, low-pitched piece'),
    ('piano+', 'solo piano'), ('drums+', 'a drum beat'), ('nodrums+', 'a piece without drums'),
    ('strings+', 'a piece for string ensemble'), ('guitar+', 'a rock song with electric guitar, bass and drums'),
    ('sad', 'a sad slow piano ballad'), ('happy', 'a happy upbeat pop song'),
    ('uk_fast+', 'швидка енергійна музика'), ('uk_fast-', 'повільна спокійна музика'),
    ('uk_minor+', 'твір у мінорі'), ('uk_minor-', 'твір у мажорі'),
    ('uk_loud+', 'гучна потужна музика'), ('uk_loud-', "тиха м'яка музика"),
    ('uk_high+', 'твір у високому регістрі'), ('uk_high-', 'твір у низькому регістрі'),
    ('uk_piano+', 'фортепіано соло'), ('uk_drums+', 'барабанний ритм'), ('uk_nodrums+', 'музика без ударних'),
    ('uk_strings+', 'п\'єса для струнного ансамблю'),
]
# pair stem -> the metric it should move (+ minus -, or + minus none; 'nodrums' should lower its metric)
PAIRS = {'fast': 'nps', 'minor': 'minor', 'loud': 'velocity', 'high': 'pitch', 'piano': 'piano', 'drums': 'drums',
         'nodrums': 'has_drums', 'strings': 'strings', 'guitar': 'guitar', 'uk_fast': 'nps', 'uk_minor': 'minor',
         'uk_loud': 'velocity', 'uk_high': 'pitch', 'uk_piano': 'piano', 'uk_drums': 'drums',
         'uk_nodrums': 'has_drums', 'uk_strings': 'strings'}
METRICS = ('nps', 'ops', 'velocity', 'pitch', 'piano', 'drums', 'has_drums', 'strings', 'guitar', 'minor', 'n_notes')
MEDIAN = ('nps', 'ops', 'velocity', 'pitch')  # per-prompt median over rows (a few rows stack hundreds of notes / s)
KEY_CONF = 0.072  # the caption builder's key threshold (musicar/text/captions.py)
DT = 0.02


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs with text_dim')
    p.add_argument('--out', default='results/text_eval/run')
    p.add_argument('--rows', type=int, default=16, help='rows per prompt')
    p.add_argument('--notes', type=int, default=512, help='new notes per row (no EOS before)')
    p.add_argument('--cfg', default='1,2', help='guidance scales')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--midi', type=int, default=2, help='MIDI files written per prompt and scale')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--summarize', action='store_true', help='only rewrite <out>/summary.md from <out>/rows.csv')
    return p.parse_args()


def row_metrics(pitch, vel, dur, dt, prog):
    """Metrics of one generated row (numpy arrays of the new notes, cut at the first EOS), computed like the caption
    labels (data_prep/song_features.file_features): minor = the Krumhansl-Schmuckler key is minor, only when its
    margin passes the caption threshold (else NaN); has_drums = drums are >= 2% of the notes."""
    end = np.flatnonzero(pitch >= BOS_PITCH)
    n = int(end[0]) if len(end) else len(pitch)
    if n < 16:
        return dict({k: np.nan for k in METRICS}, n_notes=n)
    tok = np.stack([a[:n] for a in (pitch, vel, dur, dt)], 1)
    prog = prog[:n]
    f = dict(zip(FIELDS, file_features(tok, prog)[0]))
    return dict(nps=f['nps'], ops=f['ops'], velocity=f['vel_mean'], pitch=f['pitch_mean'] if f['drums'] < 1 else np.nan,
                piano=float(np.isin(prog, range(8)).mean()), drums=f['drums'], has_drums=float(f['drums'] >= 0.02),
                strings=float(np.isin(prog, range(40, 52)).mean()), guitar=float(np.isin(prog, range(24, 32)).mean()),
                minor=float(f['key'] >= 12) if f['key'] >= 0 and f['key_conf'] >= KEY_CONF else np.nan, n_notes=n)


def slug(name):
    return re.sub(r'[^a-z0-9_+-]', '_', name.lower())


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        with open(out / 'rows.csv') as f:
            rows = [{k: v if k in ('checkpoint', 'prompt', 'text') else float(v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        (out / 'summary.md').write_text(summarize(rows))
        print(summarize(rows))
        return
    scales =[float(s) for s in args.cfg.split(',')]
    ckpts = [c for c in args.checkpoints.split(',') if c]
    encoder = TextEncoder(device=args.device)
    vectors = encoder.encode([text for _, text in PROMPTS])
    token_states = encoder.encode([text for _, text in PROMPTS], tokens=True)  # T2 prefix models
    del encoder
    torch.cuda.empty_cache()
    job = JobStatus('text_eval', len(ckpts) * len(scales) * len(PROMPTS), f'{len(ckpts)} checkpoint(s) -> {out}')
    rows, done = [], 0
    for ck in ckpts:
        model, it = load_model(ck, args.device, torch.bfloat16 if args.device == 'cuda' else torch.float32)
        assert model.config.text_dim, f'{ck} has no text input'
        name = Path(ck).name if Path(ck).is_dir() else Path(ck).parent.name
        for s in scales:
            t0 = time.time()
            prompt_inputs = token_states if model.config.text_prefix else vectors
            for (pname, text), vec in zip(PROMPTS, prompt_inputs):
                torch.manual_seed(args.seed)  # the same noise for every prompt: differences come from the text
                start = [torch.full((args.rows, 1), BOS_PITCH, device=args.device)] + \
                    [torch.zeros(args.rows, 1, dtype=torch.long, device=args.device)] * 3
                res = generate(model, *start, max_new_tokens=args.notes, temperature=args.temperature,
                               top_p=args.top_p, program=None, programs=torch.zeros_like(start[0]),
                               return_programs=True, min_new=args.notes, text=vec if text else None,
                               cfg_scale=s if text else 1.0)
                arrays = [t[:, 1:].cpu().numpy() for t in res]
                for r in range(args.rows):
                    m = row_metrics(*(a[r] for a in arrays))
                    rows.append(dict(checkpoint=name, iter=it, cfg=s, prompt=pname, text=text, row=r, **m))
                    if r < args.midi:
                        d = out / name / f'cfg{s:g}'
                        d.mkdir(parents=True, exist_ok=True)
                        p, v, du, dt, pr = (a[r] for a in arrays)
                        keep = p < BOS_PITCH
                        tokens_to_midi(p[keep], v[keep], du[keep], dt[keep], programs=pr[keep].tolist()).write(
                            str(d / f'{slug(pname)}_{r}.mid'))
                done += 1
                job.update(done, f'{name} cfg {s:g}: {pname}')
            print(f'{name} cfg {s:g}: {len(PROMPTS)} prompts x {args.rows} rows in {time.time() - t0:.0f} s')
        del model
        torch.cuda.empty_cache()
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (out / 'summary.md').write_text(summary)
    print(summary)
    job.finish(f'{out}/summary.md')


def summarize(rows):
    """Per checkpoint x scale: metrics per prompt (median over rows for MEDIAN, else mean; NaN rows left out), then
    each pair's contrast (+ minus -, or + minus none)."""
    lines = []
    groups = {}
    for r in rows:
        groups.setdefault((r['checkpoint'], r['cfg']), {}).setdefault(r['prompt'], []).append(r)
    for (ck, s), by_prompt in groups.items():
        agg = lambda k, v: float(np.nanmedian(v) if k in MEDIAN else np.nanmean(v)) if np.isfinite(v).any() else np.nan
        mean = {p: {k: agg(k, np.array([r[k] for r in rs], float)) for k in METRICS} for p, rs in by_prompt.items()}
        lines += [f'## {ck}, cfg {s:g}', '', '| prompt | ' + ' | '.join(METRICS) + ' |',
                  '|---|' + '---|' * len(METRICS)]
        for p, m in mean.items():
            lines.append(f'| {p} | ' + ' | '.join(f'{m[k]:.2f}' for k in METRICS) + ' |')
        lines += ['', '| pair | metric | + | - (or none) | difference |', '|---|---|---|---|---|']
        for stem, k in PAIRS.items():
            plus = mean.get(f'{stem}+')
            minus = mean.get(f'{stem}-', mean['none'])
            if plus:
                lines.append(f'| {stem} | {k} | {plus[k]:.3f} | {minus[k]:.3f} | {plus[k] - minus[k]:+.3f} |')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
