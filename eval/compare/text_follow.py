"""
Does a text-conditioned model keep following its prompt past the opening? (context/claude/text_control.md, "Where
the caption acts"; text_eval.py only measures 512 notes from BOS, where the caption's effect sits.)

Two parts, every measure per segment of --seg new notes:
  conflict: real openings (first --open-notes notes of held-out labeled files) continued with a prompt that
    contradicts them: calm openings (no drums, <= 10 notes/s) get drums / guitar / loud / fast / strings, busy
    openings (drums, >= 15 notes/s) get piano / no drums / quiet / slow / strings; plus 'none' and 'own' (the
    file's own caption). Shift = segment metric under the prompt minus under 'none' for the same openings: does the
    text move the music against what the opening says, and does that hold in later segments?
  drift: --drift-notes new notes from BOS per prompt (past block_size, so the cache is refilled with the prefix
    rebuilt), shift vs 'none' per segment: does the prompt's effect fade along the piece?
Next to the attribute metrics: runaway share (segments > 40 notes/s; real songs ~10%) and notes per onset (chord
stacking; a 'fast' prompt that only stacks notes raises nps without faster rhythm).

    python eval/compare/text_follow.py --checkpoints checkpoints/text_t2_v3_long --out results/text_follow/t2_v3_long

Writes <out>/rows.csv (one line per row x segment), <out>/summary.md, and MIDI (+ MP3 with --mp3) of the first
--midi rows per prompt (<out>/<checkpoint>/cfg<s>/<part>_<pool>_<prompt>_<i>.mid).
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.data.loader import BOS_PITCH, NoteWindowDataset  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
from musicar.data.midi_io import render_mp3, tokens_to_midi  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.text.encoder import TextEncoder  # noqa: E402
from text_eval import MEDIAN, PAIRS, PROMPTS, row_metrics, slug  # noqa: E402

TEXT = dict(PROMPTS)
# opening pool -> prompts that contradict it
CONFLICT = {
    'calm': ['drums+', 'guitar+', 'loud+', 'fast+', 'strings+', 'uk_drums+', 'uk_loud+'],
    'busy': ['piano+', 'nodrums+', 'loud-', 'fast-', 'strings+', 'uk_piano+', 'uk_nodrums+', 'uk_fast-'],
}
DRIFT = ['fast+', 'fast-', 'loud+', 'loud-', 'high+', 'high-', 'piano+', 'drums+', 'nodrums+', 'strings+', 'guitar+',
         'uk_piano+', 'uk_drums+', 'uk_loud-']
RUNAWAY_NPS = 40.0  # musicar/eval/sample_eval.py
VAL_SOURCE = 'store:data/cache/gigamidi_heldout#labeled'
SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs with text_dim')
    p.add_argument('--out', default='results/text_follow/run')
    p.add_argument('--parts', default='conflict,drift')
    p.add_argument('--openings', type=int, default=16, help='openings per pool (conflict)')
    p.add_argument('--open-notes', type=int, default=256, help='notes of each real opening')
    p.add_argument('--new', type=int, default=768, help='new notes after an opening (conflict)')
    p.add_argument('--drift-rows', type=int, default=16, help='rows per prompt (drift)')
    p.add_argument('--drift-notes', type=int, default=2048, help='new notes from BOS (drift)')
    p.add_argument('--seg', type=int, default=256, help='notes per measured segment')
    p.add_argument('--cfg', default='1,2', help='guidance scales')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--midi', type=int, default=1, help='MIDI files written per prompt, pool and scale')
    p.add_argument('--mp3', action='store_true', help='also render the MIDI files to MP3 (fluidsynth)')
    p.add_argument('--soundfont', default=SOUNDFONT)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--summarize', action='store_true', help='only rewrite <out>/summary.md from <out>/rows.csv')
    return p.parse_args()


def target(pname):
    """Prompt name -> (metric it should move, +1 / -1 = raise / lower vs 'none'); (None, 0) for none / own."""
    stem = pname[:-1] if pname[-1:] in '+-' else pname
    if stem not in PAIRS:
        return None, 0
    return PAIRS[stem], -1 if pname.endswith('-') or stem.endswith('nodrums') else 1


def openings(n, open_notes, seed=0):
    """n calm and n busy openings of held-out labeled files: {pool: (5 streams (n, 1 + open_notes) incl. BOS and
    programs, captions, opening metrics)}."""
    ds = NoteWindowDataset(VAL_SOURCE, root_dir=str(ROOT), split='validation', block_size=1024, eval_stride=1024,
                           special_tokens=True, programs=True, pad_short=True, captions=str(ROOT / 'data/labels'),
                           caption_uk=0.3)
    first = [i for i, (f, v) in enumerate(ds.windows.tolist()) if v == 0 and ds.lengths[f] > open_notes + 64]
    random.Random(seed).shuffle(first)
    pools = {'calm': [], 'busy': []}
    for i in first:
        x, _ = ds[i]
        streams = [t[:open_notes + 1] for t in x[:5]]  # BOS + opening, programs
        m = row_metrics(*(t[1:].numpy() for t in streams))
        pool = 'calm' if m['has_drums'] == 0 and m['nps'] <= 10 else 'busy' if m['has_drums'] == 1 and m['nps'] >= 15 \
            else None
        if pool and len(pools[pool]) < n:
            pools[pool].append((streams, x[5], m))
        if all(len(v) == n for v in pools.values()):
            break
    assert all(len(v) == n for v in pools.values()), {k: len(v) for k, v in pools.items()}
    return {pool: ([torch.stack([it[0][j] for it in items]) for j in range(5)], [it[1] for it in items],
                   [it[2] for it in items]) for pool, items in pools.items()}


def segments(arrays, r, start, seg):
    """Metrics of each seg-note segment of row r's new notes (from index start); quality measures added."""
    out = []
    n = arrays[0].shape[1] - start
    for s in range(n // seg):
        sl = slice(start + s * seg, start + (s + 1) * seg)
        m = row_metrics(*(a[r, sl] for a in arrays))
        m['runaway'] = float(m['nps'] > RUNAWAY_NPS) if np.isfinite(m['nps']) else np.nan
        m['notes_per_onset'] = m['nps'] / m['ops'] if np.isfinite(m['nps']) and m['ops'] > 0 else np.nan
        out.append(m)
    return out


def write_midi(arrays, r, path, mp3, soundfont):
    p, v, du, dt, pr = (a[r] for a in arrays)
    keep = p < BOS_PITCH
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(p[keep], v[keep], du[keep], dt[keep], programs=pr[keep].tolist()).write(str(path))
    if mp3:
        render_mp3(path, soundfont)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        with open(out / 'rows.csv') as f:
            rows = [{k: v if k in ('checkpoint', 'part', 'pool', 'prompt') else float(v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        (out / 'summary.md').write_text(summarize(rows))
        print(summarize(rows))
        return
    parts = [p for p in args.parts.split(',') if p]
    scales = [float(s) for s in args.cfg.split(',')]
    ckpts = [c for c in args.checkpoints.split(',') if c]
    opens = openings(args.openings, args.open_notes) if 'conflict' in parts else {}
    for pool, (_, caps, ms) in opens.items():
        print(f'{pool}: {len(caps)} openings, median {np.median([m["nps"] for m in ms]):.1f} notes/s, '
              f'e.g. "{caps[0][:80]}"')

    encoder = TextEncoder(device=args.device)
    names = sorted({p for ps in CONFLICT.values() for p in ps} | set(DRIFT))
    states = dict(zip(names, encoder.encode([TEXT[p] for p in names], tokens=True)))
    vectors = dict(zip(names, encoder.encode([TEXT[p] for p in names])))
    own = {pool: (encoder.encode(caps, tokens=True), encoder.encode(caps)) for pool, (_, caps, _) in opens.items()}
    del encoder
    torch.cuda.empty_cache()

    # jobs: (part, pool, prompt, start streams + programs, text per kind or None, rows)
    jobs = []
    for pool, (streams, _, _) in opens.items():
        for pname in ['none', 'own'] + CONFLICT[pool]:
            text = None if pname == 'none' else own[pool] if pname == 'own' else (states[pname], vectors[pname])
            jobs.append(('conflict', pool, pname, streams, text, args.new))
    if 'drift' in parts:
        bos = [torch.full((args.drift_rows, 1), BOS_PITCH)] + [torch.zeros(args.drift_rows, 1, dtype=torch.long)] * 4
        for pname in ['none'] + DRIFT:
            text = None if pname == 'none' else (states[pname], vectors[pname])
            jobs.append(('drift', '-', pname, bos, text, args.drift_notes))
    job = JobStatus('text_follow', len(ckpts) * len(scales) * len(jobs), f'{len(ckpts)} checkpoint(s) -> {out}')
    rows, done = [], 0
    for ck in ckpts:
        model, it = load_model(ck, args.device, torch.bfloat16 if args.device == 'cuda' else torch.float32)
        assert model.config.text_dim, f'{ck} has no text input'
        name = Path(ck).name if Path(ck).is_dir() else Path(ck).parent.name
        kind = 0 if model.config.text_prefix else 1  # token states for T2, pooled vectors for T1
        for s in scales:
            t0 = time.time()
            for part, pool, pname, streams, text, new in jobs:
                torch.manual_seed(args.seed)  # the same noise for every prompt: differences come from the text
                streams = [t.to(args.device) for t in streams]
                res = generate(model, *streams[:4], max_new_tokens=new, temperature=args.temperature,
                               top_p=args.top_p, program=None, programs=streams[4], return_programs=True,
                               min_new=new, text=None if text is None else text[kind].to(args.device),
                               cfg_scale=1.0 if text is None else s)
                arrays = [t.cpu().numpy() for t in res]
                start = streams[0].size(1)
                metric, sign = target(pname)
                for r in range(arrays[0].shape[0]):
                    for si, m in enumerate(segments(arrays, r, start, args.seg)):
                        rows.append(dict(checkpoint=name, iter=it, cfg=s, part=part, pool=pool, prompt=pname, row=r,
                                         segment=si, **m))
                    if r < args.midi:
                        write_midi(arrays, r, out / name / f'cfg{s:g}' / f'{part}_{pool}_{slug(pname)}_{r}.mid',
                                   args.mp3, args.soundfont)
                done += 1
                job.update(done, f'{name} cfg {s:g}: {part} {pool} {pname}')
            print(f'{name} cfg {s:g}: {len(jobs)} prompts in {time.time() - t0:.0f} s')
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
    """Per checkpoint x scale x part x pool: for every prompt, its target metric per segment (median for MEDIAN,
    else mean over rows) with the shift vs 'none' in brackets, 'kept' = sign-corrected shift of the last segment /
    first segment, then the runaway share and median notes per onset over all its segments."""
    lines = []
    groups = {}
    for r in rows:
        groups.setdefault((r['checkpoint'], r['cfg'], r['part'], r['pool']), {}).setdefault(r['prompt'], []).append(r)
    for (ck, s, part, pool), by_prompt in groups.items():
        n_seg = int(max(r['segment'] for rs in by_prompt.values() for r in rs)) + 1

        def agg(rs, k, seg):
            v = np.array([r[k] for r in rs if r['segment'] == seg], float)
            if not np.isfinite(v).any():
                return np.nan
            return float(np.nanmedian(v) if k in MEDIAN else np.nanmean(v))

        title = f'## {ck}, cfg {s:g}, {part}' + (f', {pool} openings' if part == 'conflict' else '')
        seg_cols = [f'seg {i + 1}' for i in range(n_seg)]
        lines += [title, '', '| prompt | metric | ' + ' | '.join(seg_cols) + ' | kept | runaway | notes/onset |',
                  '|---|---|' + '---|' * (n_seg + 3)]
        for pname, rs in by_prompt.items():
            metric, sign = target(pname)
            quality = (f'{np.nanmean([r["runaway"] for r in rs]):.0%} | '
                       f'{np.nanmedian([r["notes_per_onset"] for r in rs]):.2f}')
            if metric is None:
                lines.append(f'| {pname} | - | ' + ' | '.join('' for _ in seg_cols) + f' | | {quality} |')
                continue
            cells, shifts = [], []
            for i in range(n_seg):
                v, base = agg(rs, metric, i), agg(by_prompt['none'], metric, i)
                shifts.append(sign * (v - base))
                cells.append(f'{v:.2f} ({v - base:+.2f})')
            kept = shifts[-1] / shifts[0] if shifts[0] > 0 and np.isfinite(shifts[-1]) else np.nan
            lines.append(f'| {pname} | {metric} | ' + ' | '.join(cells) + f' | {kept:.2f} | {quality} |')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
