"""
Do section captions steer a whole piece? (context/claude/text_control.md, "Section captions"). Generation from BOS,
--notes new notes, every measure per segment of --seg notes (eval/compare/text_follow.py's metrics):
  hold: the same section prompt for every section (no file caption) vs 'none': does the effect last to the end,
    unlike T2's file caption (text_follow drift)? Checkpoints without section captions get the prompt as their file
    caption instead, so the T2 baseline sits in the same table.
  switch: prompt A for the sections before note --switch, B after (and B then A): d = metric(A->B) - metric(B->A)
    per segment should flip sign at the switch; effect = (mean d after - mean d before) / 2, sign-corrected (> 0 =
    the music follows the change). With section_lead > 0 the change may start before the switch.
Prompt wording (--prompts): 'section' (default) = SECTION_TEXT, the words of the section captions the model trained
on (musicar/text/captions.py section_caption: pace busy / dense / sparse / minimal, dynamics loud / powerful / soft /
gentle, register 'in a high / low register', the drum nouns, 'without drums', 'solo piano', named string
instruments); 'file' = the file-caption prompts of text_eval.PROMPTS (the overnight sweep of 2026-10-20), whose 'slow'
and 'high-pitched' land on other section captions (held-out check: the comment at SECTION_TEXT). The prompt
names (and so the target metrics) are the same in both sets; the T2 baseline (no sections) gets the same texts as its
file caption.
Variants (switch tests of text_control.md "Next: hypotheses and tests", section-caption models only):
  --contrast prev: guidance of the sections after the switch against the previous prompt instead of against none
    (uncond + s * (cond - uncond) with the old caption as uncond; before the switch against none as usual), for
    --contrast-sections sections (0 = to the end).
  --file-caption TEXT: every row also gets this file caption, in both halves of the guidance (only the section
    captions are guided); 'none' gets it too.
  --slide N: notes made room for at every cache refill (generate's slide). With the default (block_size // 4 = 256 =
    section_len) every section start from note block_size on falls on a refill.

    python eval/compare/section_follow.py --checkpoints checkpoints/sec_A0,checkpoints/text_t2_v3_long \\
        --out results/section_follow/s1 --mp3
    python eval/compare/section_follow.py --checkpoints checkpoints/sec_A0 --prompts file   # the old wording

Writes <out>/rows.csv (column 'prompts' = the set; missing in older files = 'file'), <out>/prompts.json (the texts),
<out>/summary.md and MIDI (+ MP3) of the first --midi rows per prompt (<out>/<checkpoint>/cfg<s>/<part>_<name>_<i>.mid).
Default <out>: results/section_follow/run_<prompts>.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicar.data.loader import BOS_PITCH  # noqa: E402
from musicar.generation.sampling import generate  # noqa: E402
from musicar.jobstatus import JobStatus  # noqa: E402
from musicar.model.checkpoint import load_model  # noqa: E402
from musicar.text.encoder import TextEncoder  # noqa: E402
from text_eval import MEDIAN, PAIRS, PROMPTS, slug  # noqa: E402
from text_follow import SOUNDFONT, segments, summarize as summarize_hold, write_midi  # noqa: E402

# section-worded prompts: same names (and target metrics, text_eval.PAIRS) as the file-caption prompts. Chosen by the
# held-out check (LaBSE, ~4,300 section captions of the held-out store per sample): share of the 20 nearest captions
# whose section has the asked attribute, old -> new, two samples: slow 45 / 35 -> 90 / 90%, high 10 / 25 -> 75 / 70%,
# low 50 / 40 -> 85 / 80%, strings 50 / 35 -> 95 / 90%; the others were already 90-100% and stay there (base rates
# 6-53%). Switch pairs differ only in the attribute words
SECTION_TEXT = {
    'fast+': 'a busy, dense piece', 'fast-': 'a sparse, minimal piece',
    'loud+': 'a loud, powerful piece', 'loud-': 'a soft, gentle piece',
    'high+': 'a piece in a high register', 'high-': 'a piece in a low register',
    'drums+': 'a drum beat', 'nodrums+': 'a piece without drums',
    'piano+': 'a solo piano piece', 'strings+': 'a piece with violin, viola and cello',
    'uk_loud+': 'гучна потужна мелодія', 'uk_loud-': "тиха м'яка мелодія", 'uk_drums+': 'барабанний біт',
}
PROMPT_SETS = {'section': SECTION_TEXT, 'file': dict(PROMPTS)}
HOLD = ['fast+', 'fast-', 'loud+', 'loud-', 'drums+', 'nodrums+', 'piano+', 'strings+', 'uk_loud-', 'uk_drums+']
# name -> (prompt before, prompt after, metric); the 'after' prompt should raise the metric
SWITCH = {
    'loud': ('loud-', 'loud+', 'velocity'),
    'fast': ('fast-', 'fast+', 'nps'),
    'drums': ('nodrums+', 'drums+', 'drums'),
    'high': ('high-', 'high+', 'pitch'),
    'strings': ('piano+', 'strings+', 'strings'),
    'uk_loud': ('uk_loud-', 'uk_loud+', 'velocity'),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', required=True, help='comma-separated checkpoints / run dirs with text_dim')
    p.add_argument('--out', default=None, help='default: results/section_follow/run_<prompts>')
    p.add_argument('--prompts', default='section', choices=sorted(PROMPT_SETS),
                   help="'section' = section-caption wording, 'file' = text_eval's file-caption prompts")
    p.add_argument('--parts', default='hold,switch')
    p.add_argument('--rows', type=int, default=16, help='rows per prompt')
    p.add_argument('--notes', type=int, default=2048, help='new notes from BOS')
    p.add_argument('--switch', type=int, default=1024, help='note index from which the second prompt holds')
    p.add_argument('--seg', type=int, default=256, help='notes per measured segment')
    p.add_argument('--cfg', default='1,2', help='guidance scales')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--section-max-length', type=int, default=48)
    p.add_argument('--contrast', default='null', choices=['null', 'prev'],
                   help="switch: the sections after the switch are guided against none or the previous prompt")
    p.add_argument('--contrast-sections', type=int, default=0, help='--contrast prev for this many sections (0 = all)')
    p.add_argument('--file-caption', default='', help='file caption of every row (section models), in both halves')
    p.add_argument('--slide', type=int, default=None, help="generate's slide (default block_size // 4)")
    p.add_argument('--midi', type=int, default=1, help='MIDI files written per prompt and scale')
    p.add_argument('--mp3', action='store_true', help='also render the MIDI files to MP3 (fluidsynth)')
    p.add_argument('--soundfont', default=SOUNDFONT)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--summarize', action='store_true', help='only rewrite <out>/summary.md from <out>/rows.csv')
    args = p.parse_args()
    args.out = args.out or f'results/section_follow/run_{args.prompts}'
    return args


def read_rows(path):
    """rows.csv -> rows (numbers as float); files from before --prompts used the file-caption set."""
    with open(path) as f:
        rows = [{k: v if k in ('checkpoint', 'part', 'pool', 'prompt', 'prompts') else float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]
    for r in rows:
        r.setdefault('prompts', 'file')
    return rows


def schedule(cfg, first, second, switch, n_notes):
    """Section caption names, one per section of the model: `first` for sections starting before note `switch`."""
    S = cfg.section_len
    return [first if k * S < switch else second for k in range(n_notes // S + 2)]


def contrast_schedule(cfg, first, switch, n_notes, n_sections):
    """The unconditional half's section captions for --contrast prev: `first` (the previous prompt) for the
    n_sections sections (0 = all) starting at or after note `switch`, None (= null) for the others."""
    S = cfg.section_len
    return [first if k * S >= switch and (not n_sections or k * S < switch + n_sections * S) else None
            for k in range(n_notes // S + 2)]


def settings(args):
    contrast = getattr(args, 'contrast', 'null')  # namespaces from before these options (tests)
    n = args.contrast_sections if contrast == 'prev' else 0
    return (f'switch at note {args.switch}, contrast {contrast}' + (f' for {n} section(s)' if n else '')
            + f", file caption {getattr(args, 'file_caption', '')!r}, slide "
            + str(getattr(args, 'slide', None) or 'block_size // 4'))


def agg(rows, metric, seg):
    v = np.array([r[metric] for r in rows if r['segment'] == seg], float)
    if not np.isfinite(v).any():
        return np.nan
    return float(np.nanmedian(v) if metric in MEDIAN else np.nanmean(v))


def switch_d(rows, seg_len, switch):
    """{(checkpoint, cfg): {pair: (d per segment, effect, last / first after)}} of the switch rows."""
    groups, out = {}, {}
    for r in rows:
        if r['part'] == 'switch':
            groups.setdefault((r['checkpoint'], r['cfg']), {}).setdefault(r['pool'], {}).setdefault(
                r['prompt'], []).append(r)
    for key, by_pair in groups.items():
        n_seg = int(max(r['segment'] for p in by_pair.values() for rs in p.values() for r in rs)) + 1
        first_after = -(-(switch - 1) // seg_len)  # segments of new notes (from index 1) starting at or after switch
        out[key] = {}
        for name, by_dir in by_pair.items():
            metric = SWITCH[name][2]
            d = [agg(by_dir['AB'], metric, i) - agg(by_dir['BA'], metric, i) for i in range(n_seg)]
            before, after = d[:first_after], d[first_after:]
            effect = (np.nanmean(after) - np.nanmean(before)) / 2 if before and after else np.nan
            kept = after[-1] / after[0] if after and after[0] > 0 else np.nan
            out[key][name] = (d, float(effect), float(kept))
    return out


def hold_late(rows):
    """{(checkpoint, cfg): {prompt: sign-corrected shift vs 'none' in the last segment}} of the hold rows."""
    from text_follow import target
    groups, out = {}, {}
    for r in rows:
        if r['part'] == 'hold':
            groups.setdefault((r['checkpoint'], r['cfg']), {}).setdefault(r['prompt'], []).append(r)
    for key, by_prompt in groups.items():
        last = int(max(r['segment'] for rs in by_prompt.values() for r in rs))
        out[key] = {}
        for pname, rs in by_prompt.items():
            metric, sign = target(pname)
            if metric is not None and 'none' in by_prompt:
                out[key][pname] = sign * (agg(rs, metric, last) - agg(by_prompt['none'], metric, last))
    return out


def summarize_switch(rows, seg_len, switch):
    lines = []
    for (ck, s), pairs in switch_d(rows, seg_len, switch).items():
        n_seg = len(next(iter(pairs.values()))[0])
        cols = [f'seg {i + 1}' for i in range(n_seg)]
        lines += [f'## {ck}, cfg {s:g}, switch at note {switch} (d = A->B minus B->A)', '',
                  '| pair | metric | ' + ' | '.join(cols) + ' | effect | last / first after |',
                  '|---|---|' + '---|' * (n_seg + 2)]
        for name, (d, effect, kept) in pairs.items():
            lines.append(f'| {name} | {SWITCH[name][2]} | ' + ' | '.join(f'{x:+.2f}' for x in d)
                         + f' | {effect:+.2f} | {kept:.2f} |')
        lines.append('')
    return '\n'.join(lines)


def summarize(rows, args):
    hold = [r for r in rows if r['part'] == 'hold']
    sets = sorted({r.get('prompts', 'file') for r in rows})
    return (f'Prompt wording: {", ".join(sets)}; {settings(args)}\n\n' + (summarize_hold(hold) if hold else '') + '\n'
            + summarize_switch(rows, args.seg, args.switch))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        rows = read_rows(out / 'rows.csv')
        (out / 'summary.md').write_text(summarize(rows, args))
        print(summarize(rows, args))
        return
    parts = [p for p in args.parts.split(',') if p]
    scales = [float(s) for s in args.cfg.split(',')]
    ckpts = [c for c in args.checkpoints.split(',') if c]

    names = sorted(set(HOLD) | {p for a, b, _ in SWITCH.values() for p in (a, b)})
    text = PROMPT_SETS[args.prompts]
    (out / 'prompts.json').write_text(json.dumps({'prompts': args.prompts, 'texts': {p: text[p] for p in names},
                                                  'settings': settings(args)}, indent=1, ensure_ascii=False))
    encoder = TextEncoder(device=args.device)
    file_states = dict(zip(names, encoder.encode([text[p] for p in names], tokens=True)))
    sec_states = dict(zip(names, encoder.encode([text[p] for p in names], tokens=True,
                                                max_length=args.section_max_length)))
    file_cap = encoder.encode([args.file_caption], tokens=True)[0] if args.file_caption else None
    del encoder
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    # jobs: (part, name, prompt label, (first, second) caption names or None)
    jobs = []
    if 'hold' in parts:
        jobs += [('hold', '-', 'none', None)] + [('hold', '-', p, (p, p)) for p in HOLD]
    if 'switch' in parts:
        jobs += [('switch', name, d, (a, b) if d == 'AB' else (b, a))
                 for name, (a, b, _) in SWITCH.items() for d in ('AB', 'BA')]
    bos = [torch.full((args.rows, 1), BOS_PITCH)] + [torch.zeros(args.rows, 1, dtype=torch.long)] * 4
    job = JobStatus('section_follow', len(ckpts) * len(scales) * len(jobs), f'{len(ckpts)} checkpoint(s) -> {out}')
    rows, done = [], 0
    for ck in ckpts:
        model, it = load_model(ck, args.device, torch.bfloat16 if args.device.startswith('cuda') else torch.float32)
        cfg = model.config
        assert cfg.text_prefix, f'{ck}: needs the T2 prefix'
        name = Path(ck).name if Path(ck).is_dir() else Path(ck).parent.name
        for s in scales:
            t0 = time.time()
            for part, pool, label, caps in jobs:
                if part == 'switch' and not cfg.section_len:
                    done += 1
                    continue
                kw = {}
                if file_cap is not None and cfg.section_len:
                    kw['text'] = kw['text_neg'] = file_cap.to(args.device)
                if caps is not None and cfg.section_len:
                    sched = schedule(cfg, caps[0], caps[1], args.switch, args.notes)
                    kw['sections'] = torch.stack([sec_states[c] for c in sched]).to(args.device)
                    if part == 'switch' and args.contrast == 'prev':
                        neg = contrast_schedule(cfg, caps[0], args.switch, args.notes, args.contrast_sections)
                        zero = torch.zeros_like(sec_states[caps[0]])
                        kw['sections_neg'] = torch.stack([zero if c is None else sec_states[c]
                                                          for c in neg]).to(args.device)
                elif caps is not None:
                    kw['text'] = file_states[caps[0]].to(args.device)  # no sections: the file caption holds it
                torch.manual_seed(args.seed)
                streams = [t.to(args.device) for t in bos]
                res = generate(model, *streams[:4], max_new_tokens=args.notes, temperature=args.temperature,
                               top_p=args.top_p, slide=args.slide, program=None, programs=streams[4],
                               return_programs=True, min_new=args.notes, cfg_scale=1.0 if caps is None else s, **kw)
                arrays = [t.cpu().numpy() for t in res]
                for r in range(args.rows):
                    for si, m in enumerate(segments(arrays, r, 1, args.seg)):
                        rows.append(dict(checkpoint=name, iter=it, cfg=s, part=part, pool=pool, prompt=label, row=r,
                                         segment=si, prompts=args.prompts, **m))
                    if r < args.midi:
                        stem = f'{part}_{slug(label if pool == "-" else pool + "_" + label)}_{r}.mid'
                        write_midi(arrays, r, out / name / f'cfg{s:g}' / stem, args.mp3, args.soundfont)
                done += 1
                job.update(done, f'{name} cfg {s:g}: {part} {pool} {label}')
            print(f'{name} cfg {s:g}: {len(jobs)} jobs in {time.time() - t0:.0f} s')
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, args)
    (out / 'summary.md').write_text(summary)
    print(summary)
    job.finish(f'{out}/summary.md')


if __name__ == '__main__':
    main()
