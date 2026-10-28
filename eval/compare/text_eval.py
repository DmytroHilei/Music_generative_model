"""
Does a text-conditioned model follow its prompt? (context/claude/text_control.md, build step 10)

Generates from BOS (no MIDI prompt, instruments sampled) for a fixed set of EN + UK prompts and measures what is
measurable on the notes: notes/s, onsets/s, mean velocity, share of piano / drum / string / guitar notes, minor key
(Krumhansl-Schmuckler). Each prompt pair targets one attribute (fast vs slow, minor vs major, ...); the summary lists
the contrast per pair next to the unconditioned rows ('' prompt), for every checkpoint and decoding setting.

    python eval/compare/text_eval.py --checkpoints checkpoints/text_p1_full,checkpoints/text_p1_frozen --out results/text_eval/p1
    python eval/compare/text_eval.py --checkpoints checkpoints/f1_cap --out results/gen_search/stageA \\
        --setting base --setting "cfg2:cfg=2" --setting "open64:cfg=3,cfg_notes=64" --setting "dry:dry=3,dry_key=pitch_class"

Decoding settings (context/claude/gen_search_2026-10-27.md): `--setting NAME[:key=value,...]`, repeatable; without
it `--cfg 1,2` gives the settings cfg1, cfg2 (the old form). Keys (defaults in brackets): temperature [--temperature],
top_p [--top-p, 'none' = off], top_k, min_p, cfg [1], cfg_notes / cfg_end / cfg_linear (the guidance schedule of
musicar.generation.sampling.guidance_scale), neg [none] = what the unconditional half gets: 'pair' = the opposite
prompt of the pair (plain guidance for prompts without one), 'neutral' = NEUTRAL, dry [0] = DRY multiplier with
dry_key [notes] / dry_allowed [16] / dry_base [1.15], best_of [1] = candidates per kept row, kept by caption match
minus runaway share minus verbatim copies over 25% (generate.py's rule), retrieve [''] = a captioned store whose
openings (--prompt-notes notes, caption-matched; NEUTRAL for the '' prompt) start the rows instead of BOS, inst_t =
the instrument head's temperature, seed [--seed]. The '' prompt (none) is never guided.

Scores per checkpoint x setting (the scoreboard at the top of summary.md, and scores.json):
  F  follow score: the mean over the prompt pairs of the contrast in the expected direction over a fixed "clearly
     audible" reference (REF: notes/s 15, pitch 12 semitones, velocity 8 bins, instrument share 0.4, has_drums 0.5,
     minor 0.3), clipped to [-1, 1.5] so overshoot does not score. 1 = every attribute moves audibly.
  D  musicar.eval.sample_eval.distance of the '' rows to real band songs (takeover, top instrument share, notes/s,
     instrument count, runaway; the out-of-band term needs a prompt band and is left out, weights renormalised).
  Q  = D + 4 x (runaway share over 0.10) + 4 x (copies share over 0.25), both over all rows; runaway = 10-s windows
     over 40 notes/s (real songs ~10%), copies = pitched notes inside exact copies of >= 16 chords (real median ~13%).
  J  = F - 0.25 Q; a setting with a runaway share over 0.20 or a copies share over 0.25 is gated out (J still shown).
  F_late (--notes > 512): F over the last 512 generated notes of each row: does the effect hold past the opening?

Writes <out>/rows.csv (one line per kept row; re-runs skip the checkpoint x setting pairs already in it), summary.md,
scores.json, and MIDI of the first --midi rows per prompt and setting (<out>/<checkpoint>/<setting>/<prompt>_<i>.mid).
"""

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.data.loader import BOS_PITCH, EOS_PITCH  # noqa: E402
from musicar.eval import sample_eval  # noqa: E402
from musicar.generation.caption_match import match, note_attributes, parse_caption  # noqa: E402
from musicar.generation.repetition import copy_share  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
from musicar.data.midi_io import tokens_to_midi  # noqa: E402
from musicar.generation.sampling import generate, prompt_inputs  # noqa: E402
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
NEGATIVE = {'nodrums', 'uk_nodrums'}  # pairs whose metric should fall
METRICS = ('nps', 'ops', 'velocity', 'pitch', 'piano', 'drums', 'has_drums', 'strings', 'guitar', 'minor', 'n_notes')
MEDIAN = ('nps', 'ops', 'velocity', 'pitch')  # per-prompt median over rows (a few rows stack hundreds of notes / s)
# the quality side: caption_match's runaway share (10-s windows over 40 notes/s), notes per onset, verbatim copies,
# and sample_eval's row metrics (instrument count, top instrument share, takeover, its per-row runaway flag)
EXTRA = ('runaway', 'npo', 'copies', 'instruments', 'top_share', 'takeover', 'se_runaway', 'cand_score')
LATE = 512  # --notes > LATE: the F metrics again over each row's last LATE notes (late_<metric>)
# a "clearly audible" move per metric: F = mean over pairs of clip(signed contrast / REF, -1, 1.5)
REF = {'nps': 15.0, 'pitch': 12.0, 'velocity': 8.0, 'piano': 0.4, 'drums': 0.4, 'strings': 0.4, 'guitar': 0.4,
       'has_drums': 0.5, 'minor': 0.3}
F_CLIP = (-1.0, 1.5)
Q_WEIGHT, RUNAWAY_FREE, COPY_FREE, PENALTY = 0.25, 0.10, 0.25, 4.0
GATE_RUNAWAY, GATE_COPIES = 0.20, 0.25
NEUTRAL = 'a piece of music'
KEY_CONF = 0.072  # the caption builder's key threshold (musicar/text/captions.py)


@dataclass
class Setting:
    name: str
    temperature: float = 0.9
    top_p: float = 0.9          # nan = off
    top_k: int = 0
    min_p: float = 0.0
    cfg: float = 1.0
    cfg_notes: int = 0
    cfg_end: float = 1.0
    cfg_linear: bool = False
    neg: str = 'none'           # none | pair | neutral
    dry: float = 0.0
    dry_key: str = 'notes'
    dry_allowed: int = 16
    dry_base: float = 1.15
    best_of: int = 1
    retrieve: str = ''
    inst_t: float = 0.0         # 0 = temperature
    seed: int = 1234

    @property
    def guided(self):
        return self.cfg != 1.0 or (self.cfg_notes and self.cfg_end != 1.0)

    def short(self):
        """The keys that differ from the defaults, as 'key=value,...'."""
        d = Setting(self.name)
        return ','.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self)
                        if f.name != 'name' and getattr(self, f.name) != getattr(d, f.name)) or 'defaults'


def parse_setting(spec, **defaults):
    """'name:key=value,key=value' -> Setting; defaults (temperature, top_p, seed) fill the unnamed keys."""
    name, _, rest = spec.partition(':')
    kw = dict(defaults)
    for part in filter(None, rest.split(',')):
        k, _, v = part.partition('=')
        k = k.strip().replace('-', '_')
        if k not in {f.name for f in fields(Setting)} or k == 'name':
            raise ValueError(f'setting {name!r}: unknown key {k!r}')
        t = Setting.__dataclass_fields__[k].type
        v = v.strip()
        if k == 'top_p' and v.lower() in ('none', 'off', ''):
            kw[k] = math.nan
        elif t is bool:
            kw[k] = v.lower() in ('1', 'true', 'yes', 'on')
        elif t is float:
            kw[k] = float(v)
        elif t is int:
            kw[k] = int(v)
        else:
            kw[k] = v
    if kw.get('neg', 'none') not in ('none', 'pair', 'neutral'):
        raise ValueError(f'setting {name!r}: neg must be none, pair or neutral')
    if kw.get('best_of', 1) < 1:
        raise ValueError(f'setting {name!r}: best_of must be >= 1')
    return Setting(name=name.strip() or 'setting', **kw)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs with text_dim')
    p.add_argument('--out', default='results/text_eval/run')
    p.add_argument('--rows', type=int, default=16, help='rows per prompt (kept rows with best_of)')
    p.add_argument('--notes', type=int, default=512, help='new notes per row (no EOS before)')
    p.add_argument('--cfg', default='1,2', help='guidance scales = the settings cfg<s> when no --setting is given')
    p.add_argument('--setting', action='append', default=[], help='NAME[:key=value,...], repeatable (see above)')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--prompts', type=int, default=0, help='> 0: only the first N prompts (smoke tests)')
    p.add_argument('--batch', type=int, default=16, help='rows per generate call (guidance doubles it inside)')
    p.add_argument('--prompt-notes', type=int, default=128, help='retrieve: notes of each real opening')
    p.add_argument('--labels-dir', default='data/labels', help='retrieve: the store\'s label tables')
    p.add_argument('--midi', type=int, default=2, help='MIDI files written per prompt and setting')
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


def new_notes(arrays, r, n0):
    """(tokens (n, 4), programs (n,)) of row r's generated notes after the n0 prompt tokens, cut at the first EOS,
    without BOS / EOS."""
    pitch = arrays[0][r, n0:]
    end = np.flatnonzero(pitch == EOS_PITCH)
    n = int(end[0]) if len(end) else len(pitch)
    tok = np.stack([a[r, n0:n0 + n] for a in arrays[:4]], 1).astype(np.int64)
    keep = tok[:, 0] < BOS_PITCH
    return tok[keep], arrays[4][r, n0:n0 + n].astype(np.int64)[keep]


def quality_metrics(arrays, r, n0, wants):
    """The EXTRA metrics of row r (prompt tokens before n0 count as earlier passages for the copies, not as notes) and
    the best-of score: caption match - runaway share - copies over COPY_FREE (generate.py's rule; match 1 without a
    caption)."""
    tok, prog = new_notes(arrays, r, n0)
    a = note_attributes(tok, prog)
    m = match(wants, a)[0] if wants is not None else 1.0
    pitch = arrays[0][r]
    end = np.flatnonzero(pitch[n0:] == EOS_PITCH)
    stop = n0 + int(end[0]) if len(end) else len(pitch)
    copies = copy_share(pitch[:stop].tolist(), arrays[3][r, :stop], arrays[4][r, :stop], 16, n0)
    se = sample_eval.row_metrics(torch.as_tensor(arrays[0][r, n0:]), torch.as_tensor(arrays[3][r, n0:]),
                                 torch.as_tensor(arrays[4][r, n0:]), set(arrays[4][r, :n0].tolist()))
    se = se or dict(instruments=np.nan, top_share=np.nan, takeover=np.nan, runaway=np.nan)
    onsets = 1 + int((tok[1:, 3] > 0).sum()) if len(tok) else 0
    return dict(runaway=a['runaway'], npo=len(tok) / onsets if onsets else np.nan, copies=copies,
                instruments=se['instruments'], top_share=se['top_share'], takeover=se['takeover'],
                se_runaway=se['runaway'], cand_score=m - a['runaway'] - max(0.0, copies - COPY_FREE))


def slug(name):
    return re.sub(r'[^a-z0-9_+-]', '_', name.lower())


def pair_text(pname, prompts):
    """The opposite prompt of a pair ('fast+' -> the text of 'fast-'), or None."""
    if pname.endswith('+') and pname[:-1] + '-' in prompts:
        return prompts[pname[:-1] + '-']
    if pname.endswith('-') and pname[:-1] + '+' in prompts:
        return prompts[pname[:-1] + '+']
    return None


class Openings:
    """--retrieve: caption-matched real openings per prompt (the index and the searches run before the model is
    loaded; LaBSE leaves the GPU after)."""

    def __init__(self, store, prompts, n, notes, labels_dir, seed, device):
        from musicar.generation.retrieval import CaptionIndex
        path = Path(store) if Path(store).is_dir() else Path('data/cache') / store
        index = CaptionIndex(path, labels_dir=labels_dir, device=device)
        self.pieces = {}
        for pname, text in prompts:
            hits = index.search(text or NEUTRAL, k=max(n, 32))
            random.Random(seed).shuffle(hits)
            if not hits:
                raise SystemExit(f'retrieve: no openings in {path.name} for {pname!r}')
            toks, progs = [], []
            for h in [hits[i % len(hits)] for i in range(n)]:
                t, g = index.opening(h['file'], notes)
                toks.append(np.asarray(t, dtype=np.int64)[:notes])
                progs.append(np.asarray(g, dtype=np.int64)[:notes])
            self.pieces[pname] = (np.stack(toks), np.stack(progs))
        index.encoder = None
        print(f'retrieve: {len(index.files):,} files of {path.name} indexed; {n} openings x {len(prompts)} prompts')

    def inputs(self, model, pname, lo, hi, device):
        toks, progs = self.pieces[pname]
        tokens = torch.as_tensor(toks[lo:hi], device=device)
        return prompt_inputs(model, tokens, torch.as_tensor(progs[lo:hi], device=device))


def run_setting(model, st, prompts, vectors, neutral_vec, rows, notes, batch, openings, device, midi, out_dir,
                progress=None):
    """Every prompt under one setting: generated rows (kept best_of candidates per prompt) as metric dicts."""
    results = []
    for (pname, text), vec in zip(prompts, vectors):
        wants = parse_caption(text) if text else None
        n_cand = rows * st.best_of
        neg = None
        if text and st.guided and st.neg != 'none':
            other = pair_text(pname, dict(prompts)) if st.neg == 'pair' else NEUTRAL
            neg = neutral_vec if other == NEUTRAL else (vectors[[t for _, t in prompts].index(other)]
                                                        if other is not None else None)
        torch.manual_seed(st.seed)  # the same noise for every prompt: differences come from the text
        chunks = []
        for lo in range(0, n_cand, batch):
            hi = min(n_cand, lo + batch)
            if openings is not None:
                streams, programs, cond = openings.inputs(model, pname, lo, hi, device)
            else:
                streams = [torch.full((hi - lo, 1), BOS_PITCH, device=device)] + \
                    [torch.zeros(hi - lo, 1, dtype=torch.long, device=device)] * 3
                programs, cond = torch.zeros_like(streams[0]), None
            res = generate(model, *streams, max_new_tokens=notes, temperature=st.temperature,
                           top_p=None if math.isnan(st.top_p) else st.top_p, top_k=st.top_k or None,
                           min_p=st.min_p or None, program=None, programs=programs, cond=cond,
                           return_programs=True, min_new=notes, text=vec if text else None,
                           text_neg=neg if text else None, cfg_scale=st.cfg if text else 1.0,
                           cfg_notes=st.cfg_notes if text else 0, cfg_end=st.cfg_end, cfg_linear=st.cfg_linear,
                           program_temperature=st.inst_t or None, dry_multiplier=st.dry, dry_key=st.dry_key,
                           dry_allowed=st.dry_allowed, dry_base=st.dry_base)
            chunks.append([t.cpu().numpy() for t in res])
        arrays = [np.concatenate([c[j] for c in chunks]) for j in range(5)]
        n0 = arrays[0].shape[1] - notes
        cands = []
        for c in range(n_cand):
            q = quality_metrics(arrays, c, n0, wants)
            m = row_metrics(*(a[c, n0:] for a in arrays))
            if notes > LATE:
                m.update({f'late_{k}': v for k, v in row_metrics(*(a[c, -LATE:] for a in arrays)).items()})
            cands.append((c, dict(m, **q)))
        if st.best_of > 1:
            cands.sort(key=lambda cm: -cm[1]['cand_score'])
        for r, (c, m) in enumerate(cands[:rows]):
            results.append(dict(prompt=pname, text=text, row=r, cand=c, **m))
            if r < midi:
                d = out_dir / slug(st.name)
                d.mkdir(parents=True, exist_ok=True)
                p, v, du, dt, pr = (a[c] for a in arrays)
                keep = p < BOS_PITCH
                tokens_to_midi(p[keep], v[keep], du[keep], dt[keep], programs=pr[keep].tolist()).write(
                    str(d / f'{slug(pname)}_{r}.mid'))
        if progress is not None:
            progress(pname)
    return results


def read_rows(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in ('checkpoint', 'prompt', 'text', 'setting', 'spec'):
                    row[k] = v
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = np.nan
            if 'setting' not in row:  # rows of the old form: the scale names the setting
                row['setting'] = f"cfg{row['cfg']:g}"
            rows.append(row)
    return rows


def write_rows(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    tmp.replace(path)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        rows = read_rows(out / 'rows.csv')
        (out / 'summary.md').write_text(summarize(rows))
        (out / 'scores.json').write_text(json.dumps(scoreboard(rows), indent=1))
        print(summarize(rows))
        return
    defaults = dict(temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    specs = args.setting or [f'cfg{float(s):g}:cfg={float(s)}' for s in args.cfg.split(',')]
    settings = [parse_setting(sp, **defaults) for sp in specs]
    if len({st.name for st in settings}) < len(settings):
        raise SystemExit('setting names must be distinct')
    prompts = PROMPTS[:args.prompts] if args.prompts else PROMPTS
    ckpts = [c for c in args.checkpoints.split(',') if c]
    rows = read_rows(out / 'rows.csv') if (out / 'rows.csv').exists() else []
    done_keys = {(r['checkpoint'], r['setting']) for r in rows}
    encoder = TextEncoder(device=args.device)
    texts = [text for _, text in prompts] + [NEUTRAL]
    vectors = encoder.encode(texts)
    token_states = encoder.encode(texts, tokens=True)  # T2 prefix models
    del encoder
    torch.cuda.empty_cache()
    openings = {}
    for st in settings:
        if st.retrieve and st.retrieve not in openings:
            openings[st.retrieve] = Openings(st.retrieve, prompts, args.rows * st.best_of, args.prompt_notes,
                                             args.labels_dir, args.seed, args.device)
    torch.cuda.empty_cache()
    job = JobStatus('text_eval', len(ckpts) * len(settings) * len(prompts), f'{len(ckpts)} checkpoint(s) -> {out}')
    done = [0]
    for ck in ckpts:
        name = Path(ck).name if Path(ck).is_dir() else Path(ck).parent.name
        todo = [st for st in settings if (name, st.name) not in done_keys]
        if not todo:
            done[0] += len(settings) * len(prompts)
            continue
        model, it = load_model(ck, args.device, torch.bfloat16 if args.device == 'cuda' else torch.float32)
        assert model.config.text_dim, f'{ck} has no text input'
        inputs = token_states if model.config.text_prefix else vectors
        for st in todo:
            t0 = time.time()

            def progress(pname, st=st):
                done[0] += 1
                job.update(done[0], f'{name} {st.name}: {pname}')
            res = run_setting(model, st, prompts, inputs[:-1], inputs[-1], args.rows, args.notes, args.batch,
                              openings.get(st.retrieve), args.device, args.midi, out / name, progress)
            rows += [dict(checkpoint=name, iter=it, setting=st.name, spec=st.short(), cfg=st.cfg, **r) for r in res]
            write_rows(out / 'rows.csv', rows)
            print(f'{name} {st.name} ({st.short()}): {len(prompts)} prompts x {args.rows} rows'
                  f'{f" (best of {st.best_of})" if st.best_of > 1 else ""} in {time.time() - t0:.0f} s')
        del model
        torch.cuda.empty_cache()
    summary = summarize(rows)
    (out / 'summary.md').write_text(summary)
    (out / 'scores.json').write_text(json.dumps(scoreboard(rows), indent=1))
    print(summary)
    job.finish(f'{out}/summary.md')


def _agg(k, v):
    v = np.asarray(v, float)
    return float(np.nanmedian(v) if k in MEDIAN else np.nanmean(v)) if np.isfinite(v).any() else np.nan


def prompt_means(rows, prefix=''):
    """prompt -> metric -> aggregate over its rows (median for MEDIAN, else mean; NaN rows left out)."""
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r['prompt'], []).append(r)
    return {p: {k: _agg(k, [r.get(prefix + k, np.nan) for r in rs]) for k in METRICS} for p, rs in by_prompt.items()}


def pair_contrasts(mean):
    """pair stem -> (metric, +, - or none, signed contrast / REF clipped) for the pairs present."""
    out = {}
    for stem, k in PAIRS.items():
        plus = mean.get(f'{stem}+')
        minus = mean.get(f'{stem}-', mean.get('none'))
        if plus is None or minus is None:
            continue
        diff = plus[k] - minus[k]
        sign = -1.0 if stem in NEGATIVE else 1.0
        norm = float(np.clip(sign * diff / REF[k], *F_CLIP)) if np.isfinite(diff) else np.nan
        out[stem] = (k, plus[k], minus[k], diff, norm)
    return out


def follow_score(contrasts):
    norms = [c[4] for c in contrasts.values() if np.isfinite(c[4])]
    return float(np.mean(norms)) if norms else np.nan


def quality(rows):
    """D of the '' rows to real band songs (sample_eval reference without the out-of-band term), the runaway and
    copies shares over all rows, Q and the gate."""
    ref = sample_eval._load_reference() or {}
    none_rows = [r for r in rows if r['prompt'] == 'none'] or rows
    keys = {'takeover': 'takeover', 'top_share': 'top_share', 'notes_per_s': 'nps', 'instruments': 'instruments',
            'runaway': 'se_runaway'}
    d, wsum = 0.0, 0.0
    for k, col in keys.items():
        if k not in ref or k not in sample_eval.DIST_WEIGHTS:
            continue
        v = np.nanmean([r.get(col, np.nan) for r in none_rows])
        if not np.isfinite(v):
            continue
        sd = max(ref.get('sd', {}).get(k, 0.0), sample_eval.SD_FLOOR[k])
        d += sample_eval.DIST_WEIGHTS[k] * abs(v - ref[k]) / sd
        wsum += sample_eval.DIST_WEIGHTS[k]
    D = d / wsum if wsum else np.nan
    runaway = float(np.nanmean([r.get('runaway', np.nan) for r in rows]))
    copies = float(np.nanmean([r.get('copies', np.nan) for r in rows]))
    Q = (D if np.isfinite(D) else 0.0) + PENALTY * max(0.0, runaway - RUNAWAY_FREE) + PENALTY * max(0.0, copies - COPY_FREE)
    gated = bool(runaway > GATE_RUNAWAY or copies > GATE_COPIES)
    return dict(D=D, runaway=runaway, copies=copies, Q=Q, gated=gated)


def groups(rows):
    g = {}
    for r in rows:
        g.setdefault((r['checkpoint'], r['setting']), []).append(r)
    return g


def scoreboard(rows):
    """One dict per checkpoint x setting: F, F_late, D, runaway, copies, Q, J, gated, spec, rows; best J first."""
    board = []
    for (ck, st), rs in groups(rows).items():
        F = follow_score(pair_contrasts(prompt_means(rs)))
        late = pair_contrasts(prompt_means(rs, 'late_')) if any(np.isfinite(r.get('late_nps', np.nan)) for r in rs) else {}
        q = quality(rs)
        J = F - Q_WEIGHT * q['Q'] if np.isfinite(F) else np.nan
        board.append(dict(checkpoint=ck, setting=st, spec=rs[0].get('spec', ''), rows=len(rs), F=F,
                          F_late=follow_score(late) if late else np.nan, J=J, **q))
    board.sort(key=lambda b: (b['gated'], -(b['J'] if np.isfinite(b['J']) else -1e9)))
    return board


def summarize(rows):
    """The scoreboard, then per checkpoint x setting: metrics per prompt and each pair's contrast (+ minus -, or +
    minus none) with its share of the REF move."""
    f = lambda v, p=3: f'{v:.{p}f}' if isinstance(v, float) and np.isfinite(v) else '-'  # noqa: E731
    lines = ['## Scoreboard (J = F - 0.25 Q; gated: runaway > 20% or copies > 25%)', '',
             '| checkpoint | setting | spec | F | F_late | D | runaway | copies | Q | J | gate |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for b in scoreboard(rows):
        lines.append(f"| {b['checkpoint']} | {b['setting']} | {b['spec']} | {f(b['F'])} | {f(b['F_late'])} | "
                     f"{f(b['D'], 2)} | {f(b['runaway'], 2)} | {f(b['copies'], 2)} | {f(b['Q'], 2)} | {f(b['J'])} | "
                     f"{'OUT' if b['gated'] else 'ok'} |")
    lines.append('')
    for (ck, st), rs in groups(rows).items():
        mean = prompt_means(rs)
        extra = {k: float(np.nanmean([r.get(k, np.nan) for r in rs])) for k in ('runaway', 'npo', 'copies')}
        lines += [f'## {ck}, {st} ({rs[0].get("spec", "")})', '',
                  f"runaway {f(extra['runaway'], 2)}, notes per onset {f(extra['npo'], 2)}, copies {f(extra['copies'], 2)}",
                  '', '| prompt | ' + ' | '.join(METRICS) + ' |', '|---|' + '---|' * len(METRICS)]
        for p, m in mean.items():
            lines.append(f'| {p} | ' + ' | '.join(f(m[k], 2) for k in METRICS) + ' |')
        lines += ['', '| pair | metric | + | - (or none) | difference | / REF |', '|---|---|---|---|---|---|']
        for stem, (k, plus, minus, diff, norm) in pair_contrasts(mean).items():
            lines.append(f'| {stem} | {k} | {f(plus)} | {f(minus)} | {diff:+.3f} | {f(norm, 2)} |')
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
