"""
Overnight search of the decoding settings of a caption model (context/claude/gen_search_2026-10-27.md): which
settings make the model follow its caption best (text_eval's F) while the music stays good (Q), judged by
J = F - 0.25 Q with the runaway / copies gates, all scored by eval/compare/text_eval.py.

    .venv/bin/python tools/sweeps/gen_search.py --checkpoint checkpoints/f1_cap --out results/gen_search/f1 \\
        [--control checkpoints/f1] [--retrieve gigamidi_replay_train] [--budget-min 360]
    .venv/bin/python tools/sweeps/gen_search.py --checkpoint checkpoints/f1_cap --out /tmp/gs --smoke   # minutes

Stages (every stage is one text_eval.py run; a re-run skips the settings whose rows exist, so the search resumes):
  A  the hypotheses (STAGE_A, ~14 settings at --rows-a rows): the baseline twice (two seeds = the noise floor),
     global guidance, guidance schedules, the negative branch, DRY, best-of, retrieval, temperature / truncation.
     Pick: the best J that beats the baseline by more than the noise floor (else the baseline).
  A2 --control: the baseline and the pick on a second checkpoint (the long run before the caption phase).
  B  random search around the pick: N_B settings (each key copies the pick with probability 0.5, else a grid value),
     successive halving: rows 8 -> the best 8 at 16 -> the best 3 at 32 rows with --notes 2048 (F_late: does the
     effect hold past the opening) and MIDI (+ MP3) of the finalists.
  L  the length axis on the pick's checkpoint: eval/compare/length_follow.py with eos_ramp {0, 0.5, 1} (the music
     knobs are independent of it by design; its temperature / top-p follow the pick).
--budget-min: the wall-clock budget; stage B's pool and rungs shrink, then the length axis is skipped, when the
per-row cost measured in stage A says the rest would not fit. State in <out>/search.json, report <out>/summary.md
(the scoreboards, the pick with the noise floor, the finalists, a config/generate TOML snippet of the winner).
"""

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.jobstatus import JobStatus  # noqa: E402

PY = sys.executable
TEXT_EVAL = ROOT / 'eval/compare/text_eval.py'
LENGTH_FOLLOW = ROOT / 'eval/compare/length_follow.py'
SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'

# stage A: name -> spec (text_eval --setting NAME:spec); the baseline is T 0.9, top-p 0.9, cfg 1 (A1's reference).
# Order = cheap and most informative first, so a short budget still answers the main questions.
STAGE_A = [
    ('base', ''),
    ('base_s2', 'seed=4321'),
    ('cfg1.5', 'cfg=1.5'), ('cfg2', 'cfg=2'), ('cfg3', 'cfg=3'),
    ('open64', 'cfg=3,cfg_notes=64'), ('open128', 'cfg=3,cfg_notes=128'),
    ('anneal', 'cfg=3,cfg_notes=512,cfg_linear=1'),
    ('warmup', 'cfg=1,cfg_notes=250,cfg_end=3,cfg_linear=1'),
    ('dry16', 'dry=3,dry_key=pitch_class'), ('dry8', 'dry=3,dry_key=pitch_class,dry_allowed=8'),
    ('t1.0', 'temperature=1.0'), ('t1.1', 'temperature=1.1'),
    ('p0.98', 'top_p=0.98'), ('pnone', 'top_p=none'), ('minp', 'top_p=none,min_p=0.035'),
    ('neg_pair1.5', 'cfg=1.5,neg=pair'), ('neg_pair3', 'cfg=3,neg=pair'),
    ('neg_neutral1.5', 'cfg=1.5,neg=neutral'), ('neg_neutral3', 'cfg=3,neg=neutral'),
    ('retrieve', 'retrieve={store}'),
    ('best4', 'best_of=4'), ('best4_cfg1.5', 'best_of=4,cfg=1.5'), ('best8', 'best_of=8'),
]
# stage B grid (one value per key; 'schedule' and 'trunc' expand to several keys)
GRID = dict(
    cfg=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    schedule=['const', 'open32', 'open64', 'open128', 'anneal'],
    neg=['none', 'pair', 'neutral'],
    temperature=[0.9, 0.95, 1.0, 1.1],
    trunc=['p0.9', 'p0.95', 'p0.98', 'none', 'minp0.035', 'minp0.05'],
    dry=[0.0, 2.0, 3.0, 4.0],
    dry_allowed=[4, 8, 16],
    inst_t=[0.0, 0.8],
)
N_B, RUNGS = 24, ((8, 8), (16, 3), (32, 0))  # (rows, keep) per rung; the last rung runs at --late-notes
SMOKE = dict(rows_a=2, notes=24, prompts=3, n_b=3, rungs=((2, 2), (2, 0)), late_notes=48)


def log(msg):
    print(f'[gen_search {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='the caption model (file or run dir)')
    p.add_argument('--control', default='', help='a second checkpoint for the baseline and the pick (stage A2)')
    p.add_argument('--out', required=True)
    p.add_argument('--retrieve', default='', help='captioned store for the retrieval hypothesis (off without it)')
    p.add_argument('--rows-a', type=int, default=16, help='rows per prompt in stage A')
    p.add_argument('--notes', type=int, default=512)
    p.add_argument('--late-notes', type=int, default=2048, help='notes of the final rung (F_late)')
    p.add_argument('--prompts', type=int, default=0, help='> 0: only the first N prompts')
    p.add_argument('--n-b', type=int, default=N_B, help='stage B pool')
    p.add_argument('--budget-min', type=float, default=0.0, help='wall-clock budget (0 = none)')
    p.add_argument('--skip', default='', help='comma-separated stage A names to leave out')
    p.add_argument('--no-length', action='store_true')
    p.add_argument('--no-mp3', action='store_true')
    p.add_argument('--seed', type=int, default=7, help='stage B sampling seed')
    p.add_argument('--device', default=None)
    p.add_argument('--smoke', action='store_true', help='a minutes-long run through every stage')
    p.add_argument('--dry-run', action='store_true', help='print the stage A settings and the plan, run nothing')
    return p.parse_args()


def text_eval(ck, settings, out, rows, notes, prompts, device, midi=1):
    """One text_eval.py run (resumes its own rows.csv); returns its scoreboard (scores.json)."""
    cmd = [PY, str(TEXT_EVAL), '--checkpoints', ck, '--out', str(out), '--rows', str(rows), '--notes', str(notes),
           '--midi', str(midi)]
    for name, spec in settings:
        cmd += ['--setting', f'{name}:{spec}' if spec else name]
    if prompts:
        cmd += ['--prompts', str(prompts)]
    if device:
        cmd += ['--device', device]
    log(' '.join(str(c) for c in cmd[1:]))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        log(f'text_eval exited with {r.returncode} (the settings done so far are kept)')
    path = Path(out) / 'scores.json'
    if not path.exists():
        return []
    return json.loads(path.read_text().replace('NaN', 'null'))


def cost_units(settings, rows, notes):
    """Relative cost of a settings list: rows x candidates x guided doubling x notes."""
    total = 0.0
    for _, spec in settings:
        kw = dict(part.split('=', 1) for part in spec.split(',') if part)
        best_of = int(kw.get('best_of', 1))
        guided = float(kw.get('cfg', 1)) != 1 or (kw.get('cfg_notes') and float(kw.get('cfg_end', 1)) != 1)
        total += rows * best_of * (2 if guided else 1) * notes / 512
    return total


def pick(board, base='base', noise_from=('base', 'base_s2')):
    """The best non-gated setting if its J beats the baseline's by more than the noise floor, else the baseline.
    Returns (name, noise, reason)."""
    by = {b['setting']: b for b in board}
    if base not in by or by[base]['J'] is None:
        best = next((b for b in board if not b['gated'] and b['J'] is not None), None)
        return (best['setting'] if best else base), math.nan, 'no baseline score'
    js = [by[n]['J'] for n in noise_from if n in by and by[n]['J'] is not None]
    noise = abs(js[0] - js[1]) if len(js) == 2 else 0.0
    ok = [b for b in board if not b['gated'] and b['J'] is not None]
    if not ok:
        return base, noise, 'every setting gated'
    best = max(ok, key=lambda b: b['J'])
    if best['setting'] != base and best['J'] - by[base]['J'] > noise:
        return best['setting'], noise, f"J {best['J']:.3f} vs base {by[base]['J']:.3f}, noise {noise:.3f}"
    return base, noise, f"nothing beats base {by[base]['J']:.3f} by more than the noise floor {noise:.3f}"


def spec_to_dict(spec):
    return dict(part.split('=', 1) for part in spec.split(',') if part)


def dict_to_spec(d):
    return ','.join(f'{k}={v}' for k, v in d.items())


def sample_settings(center_spec, n, rng, best_of_ok=False, retrieve=''):
    """n distinct random settings around a center: each grid key copies the center's value with probability 1/2,
    else a grid value. Keys the center does not set keep the grid's default when copied."""
    center = spec_to_dict(center_spec)
    c_sched = ('anneal' if center.get('cfg_linear') and center.get('cfg_notes')
               else f"open{center['cfg_notes']}" if center.get('cfg_notes') else 'const')
    c_trunc = (f"minp{center['min_p']}" if center.get('min_p') else 'none' if center.get('top_p') == 'none'
               else f"p{center.get('top_p', 0.9)}")
    center_vals = dict(cfg=float(center.get('cfg', 1)), schedule=c_sched, neg=center.get('neg', 'none'),
                       temperature=float(center.get('temperature', 0.9)), trunc=c_trunc,
                       dry=float(center.get('dry', 0)), dry_allowed=int(center.get('dry_allowed', 16)),
                       inst_t=float(center.get('inst_t', 0)))
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        v = {k: (center_vals[k] if rng.random() < 0.5 else rng.choice(GRID[k])) for k in GRID}
        d = {}
        if v['cfg'] != 1.0:
            d['cfg'] = v['cfg']
            if v['schedule'].startswith('open'):
                d['cfg_notes'] = int(v['schedule'][4:])
            elif v['schedule'] == 'anneal':
                d['cfg_notes'], d['cfg_linear'] = 512, 1
            if v['neg'] != 'none':
                d['neg'] = v['neg']
        if v['temperature'] != 0.9:
            d['temperature'] = v['temperature']
        if v['trunc'] == 'none':
            d['top_p'] = 'none'
        elif v['trunc'].startswith('minp'):
            d['top_p'], d['min_p'] = 'none', float(v['trunc'][4:])
        elif v['trunc'] != 'p0.9':
            d['top_p'] = float(v['trunc'][1:])
        if v['dry']:
            d['dry'], d['dry_key'], d['dry_allowed'] = v['dry'], 'pitch_class', v['dry_allowed']
        if v['inst_t']:
            d['inst_t'] = v['inst_t']
        if best_of_ok and center.get('best_of') and rng.random() < 0.5:
            d['best_of'] = rng.choice([2, 4])
        if retrieve and center.get('retrieve') and rng.random() < 0.5:
            d['retrieve'] = retrieve
        spec = dict_to_spec(d)
        if spec in seen:
            continue
        seen.add(spec)
        out.append((f'b{len(out):02d}', spec))
    return out


def board_table(board, title):
    f = lambda v, p=3: f'{v:.{p}f}' if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else '-'  # noqa: E731
    lines = [f'## {title}', '', '| setting | spec | rows | F | F_late | D | runaway | copies | Q | J | gate |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for b in board:
        lines.append(f"| {b['setting']} | {b['spec']} | {b['rows']} | {f(b['F'])} | {f(b['F_late'])} | {f(b['D'], 2)} | "
                     f"{f(b['runaway'], 2)} | {f(b['copies'], 2)} | {f(b['Q'], 2)} | {f(b['J'])} | "
                     f"{'OUT' if b['gated'] else 'ok'} |")
    return lines + ['']


def toml_snippet(ck, spec):
    d = spec_to_dict(spec)
    lines = [f'checkpoint = "{ck}"', 'sample-instruments = true']
    keymap = dict(temperature='temperature', top_p='top-p', top_k='top-k', min_p='min-p', cfg='cfg-scale',
                  cfg_notes='cfg-notes', cfg_end='cfg-end', cfg_linear='cfg-linear', dry='dry-multiplier',
                  dry_key='dry-key', dry_allowed='dry-allowed', dry_base='dry-base', best_of='best-of',
                  retrieve='retrieve', inst_t='instrument-temperature')
    if 'temperature' not in d:
        lines.append('temperature = 0.9')
    if 'top_p' not in d:
        lines.append('top-p = 0.9')
    for k, v in d.items():
        if k in ('seed', 'neg'):
            continue  # neg: text_eval's pair prompts; generate.py has no --text-neg yet
        if k == 'top_p' and str(v) in ('none', 'nan'):
            continue
        key = keymap[k]
        if k == 'cfg_linear':
            lines.append(f'{key} = true')
        elif k in ('dry_key', 'retrieve'):
            lines.append(f'{key} = "{v}"')
        else:
            lines.append(f'{key} = {v}')
    return lines


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    smoke = SMOKE if args.smoke else {}
    rows_a = smoke.get('rows_a', args.rows_a)
    notes = smoke.get('notes', args.notes)
    prompts = smoke.get('prompts', args.prompts)
    n_b = smoke.get('n_b', args.n_b)
    rungs = smoke.get('rungs', RUNGS)
    late_notes = smoke.get('late_notes', args.late_notes)
    skip = set(filter(None, args.skip.split(',')))
    stage_a = [(n, s.format(store=args.retrieve)) for n, s in STAGE_A
               if n not in skip and (args.retrieve or n != 'retrieve')]
    state_path = out / 'search.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    t_start = time.time()
    budget = args.budget_min * 60 if args.budget_min else math.inf
    left = lambda: budget - (time.time() - t_start)  # noqa: E731

    def save():
        state_path.write_text(json.dumps(state, indent=1, default=str))
    if args.dry_run:
        for n, s in stage_a:
            print(f'{n:16s} {s or "(defaults)"}')
        print(f'stage A cost units {cost_units(stage_a, rows_a, notes):.0f}; stage B {n_b} -> rungs {rungs}')
        return
    job = JobStatus('gen_search', 4 + len(rungs), f'{args.checkpoint} -> {out}')
    report = [f'# Generation settings search: {args.checkpoint}', '',
              f'started {time.strftime("%F %T")}, rows_a {rows_a}, notes {notes}, prompts {prompts or "all"}, '
              f'budget {args.budget_min or "none"} min', '']

    # stage A
    job.update(0, 'stage A')
    t0 = time.time()
    board_a = text_eval(args.checkpoint, stage_a, out / 'stageA', rows_a, notes, prompts, args.device)
    secs_a = time.time() - t0
    unit = secs_a / max(1.0, cost_units(stage_a, rows_a, notes)) if 'unit_secs' not in state else state['unit_secs']
    if 'unit_secs' not in state and secs_a > 60:
        state['unit_secs'] = unit  # a resumed run has a partial stage A: keep the first measurement
    chosen, noise, reason = pick(board_a)
    spec_a = dict(stage_a)
    state.update(stage_a=board_a, pick=chosen, pick_spec=spec_a.get(chosen, ''), noise=noise, reason=reason)
    save()
    log(f'stage A done in {secs_a / 60:.0f} min: pick {chosen} ({spec_a.get(chosen, "defaults")}); {reason}')
    report += board_table(board_a, 'Stage A: hypotheses') + [
        f'**Pick: {chosen}** ({spec_a.get(chosen, "defaults")}); {reason}.', '']
    best_of_won = chosen.startswith('best')

    # stage A2: control checkpoint
    if args.control:
        job.update(1, 'stage A2 (control)')
        ctrl = [('base', '')] + ([(chosen, spec_a[chosen])] if chosen != 'base' else [])
        board_c = text_eval(args.control, ctrl, out / 'control', rows_a, notes, prompts, args.device)
        state['control'] = board_c
        save()
        report += board_table(board_c, f'Stage A2: {args.control} (the control checkpoint)')

    # stage B
    rng = random.Random(args.seed)
    pool = [('base', ''), (chosen, spec_a.get(chosen, ''))] if chosen != 'base' else [('base', '')]
    pool += sample_settings(spec_a.get(chosen, ''), n_b - len(pool), rng, best_of_won, args.retrieve)
    pool = list({name: (name, spec) for name, spec in pool}.values())
    boards_b = []
    for i, (rows, keep) in enumerate(rungs):
        last = i == len(rungs) - 1
        rung_notes = late_notes if last else notes
        need = cost_units(pool, rows, rung_notes) * unit
        if math.isfinite(budget) and need > left():
            shrink = max(1, int(len(pool) * left() / max(need, 1e-9)))
            log(f'rung {i}: {need / 60:.0f} min needed, {left() / 60:.0f} left: pool {len(pool)} -> {shrink}')
            pool = pool[:shrink]
            if shrink < 2:
                report += [f'Stage B rung {i} skipped: out of budget.', '']
                break
        job.update(2 + i, f'stage B rung {i}: {len(pool)} settings x {rows} rows')
        board = text_eval(args.checkpoint, pool, out / f'stageB_r{i}', rows, rung_notes, prompts, args.device,
                          midi=2 if last else 0)
        boards_b.append(board)
        state[f'stage_b_r{i}'] = board
        save()
        report += board_table(board, f'Stage B rung {i}: {len(pool)} settings x {rows} rows x {rung_notes} notes')
        if keep:
            ranked = [b for b in board if not b['gated'] and b['J'] is not None] + \
                     [b for b in board if b['gated'] or b['J'] is None]
            names = [b['setting'] for b in ranked[:keep]]
            pool = [(n, s) for n, s in pool if n in names]
            log(f'rung {i}: keep {names}')
    final = boards_b[-1] if boards_b else board_a
    winner = next((b for b in final if not b['gated'] and b['J'] is not None), None) or (final[0] if final else None)
    if winner:
        w_spec = dict(pool).get(winner['setting'], spec_a.get(winner['setting'], ''))
        state.update(winner=winner['setting'], winner_spec=w_spec)
        save()
        report += [f"**Winner: {winner['setting']}** ({w_spec or 'defaults'}): F {winner['F']:.3f}, "
                   f"F_late {winner['F_late'] if winner['F_late'] is not None else float('nan'):.3f}, "
                   f"J {winner['J']:.3f}.", '', 'config/generate snippet:', '```toml',
                   *toml_snippet(args.checkpoint, w_spec), '```', '']
        if not args.no_mp3 and shutil.which('fluidsynth') and boards_b:
            from musicar.data.midi_io import render_mp3
            ck_name = Path(args.checkpoint).name if Path(args.checkpoint).is_dir() else Path(args.checkpoint).parent.name
            d = out / f'stageB_r{len(boards_b) - 1}' / ck_name
            mids = sorted(d.glob('*/*.mid')) if d.exists() else []
            log(f'rendering {len(mids)} MIDI files of the finalists to MP3')
            for m in mids:
                try:
                    render_mp3(m, SOUNDFONT)
                except Exception as e:  # a missing soundfont or ffmpeg: the MIDI stays
                    log(f'mp3 failed for {m.name}: {e}')
                    break

    # stage L: the length axis
    if not args.no_length:
        kw = spec_to_dict(state.get('winner_spec', ''))
        lengths, l_rows, ramps = ('5,10', 2, (0.0, 1.0)) if args.smoke else ('30,60,120', 32, (0.0, 0.5, 1.0))
        lf = [PY, str(LENGTH_FOLLOW), '--checkpoints', args.checkpoint, '--lengths', lengths, '--rows', str(l_rows),
              '--temperature', kw.get('temperature', '0.9')]
        if kw.get('top_p', '0.9') != 'none':
            lf += ['--top-p', kw.get('top_p', '0.9')]
        if args.device:
            lf += ['--device', args.device]
        need = (len(ramps) * 3 * l_rows * 1.5) * unit  # rough: ramps x lengths x rows, longer rows
        if math.isfinite(budget) and need > left():
            report += [f'Stage L skipped: {need / 60:.0f} min needed, {left() / 60:.0f} left.', '']
        else:
            job.update(3 + len(rungs), 'stage L: length axis')
            report += ['## Stage L: length control (share within +-20% of L)', '']
            for ramp in ramps:
                o = out / f'length_ramp{ramp:g}'
                if not (o / 'summary.md').exists():
                    r = subprocess.run([*lf, '--eos-ramp', str(ramp), '--out', str(o)], cwd=ROOT)
                    if r.returncode:
                        log(f'length_follow eos_ramp {ramp} exited with {r.returncode}')
                if (o / 'summary.md').exists():
                    report += [f'### eos_ramp {ramp:g}', '', (o / 'summary.md').read_text(), '']
    report += [f'finished {time.strftime("%F %T")} ({(time.time() - t_start) / 60:.0f} min)', '']
    (out / 'summary.md').write_text('\n'.join(report))
    state['finished'] = time.strftime('%F %T')
    save()
    job.finish(f'{out}/summary.md')
    log(f'done: {out}/summary.md')


if __name__ == '__main__':
    main()
