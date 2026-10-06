"""
Per-style generative evaluation: does the model continue songs of different kinds the way real songs continue?

    python -m musicar.eval.style_eval --checkpoint checkpoints/long_450m_snap          # prints the table for a checkpoint
    python -m musicar.eval.style_eval --list                                           # the chosen songs and their profiles

Styles (from the instruments of a file's first 1,128 notes, clean GigaMIDI val; GM program groups below):
  electronic   drums + >= 30% of notes on synths (leads, pads, synth bass/strings/brass)
  rock_pop     drums + bass + guitar, < 15% synth notes
  piano_keys   only pianos / e-pianos / harpsichord / clavinet, <= 2 of them, no drums
  orchestral   no drums, >= 60% of notes on orchestral strings/brass/reeds/pipes, no electric guitar or synth
  acoustic     no drums, bass + piano or acoustic guitar, no electric guitar or synth, <= 5 instruments
per_style songs each (evenly spaced over the store, deterministic); prompt = the first 128 notes, reference = the
real next 1,000 notes.

Distance (robust, per style): each sample is compared with ITS OWN song's real continuation,
    d_row = sum_k w_k * min(|t(sample_k) - t(real_k)| / s_style,k , 3)
t = log for notes_per_s and instruments (ratios, not differences: 300 vs 20 notes/s), identity for the shares;
s_style,k = the spread of t(metric k) over the real continuations of THAT style only (rock density is never compared
with folk density), floored; every term capped at 3 spreads so one runaway sample can't swamp a style.
style score = mean d_row over the style's songs; overall = mean over styles (each style counts the same).
"""

import argparse
import math

import numpy as np
import torch

from musicar.model import BOS_PITCH, DRUM_PROGRAM, window_conditions
from musicar.eval.sample_eval import row_metrics

STORE = 'data/cache/gigamidi_clean_validation'
STYLES = ('rock_pop', 'orchestral', 'piano_keys', 'electronic', 'acoustic')
WEIGHTS = {'takeover': 0.20, 'top_share': 0.20, 'notes_per_s': 0.20, 'out_of_band': 0.15, 'instruments': 0.15,
           'runaway': 0.10}
LOG = {'notes_per_s', 'instruments'}
FLOOR = {'takeover': 0.22, 'runaway': 0.22, 'top_share': 0.05, 'out_of_band': 0.05, 'notes_per_s': 0.15,
         'instruments': 0.15}  # log-scale floors: ~15% ratio
CLIP = 3.0
# the fixed tracking setting (the best of the 60-song stage of the 2026-10-04 search: top-p 0.9 was in every good one)
TRACK = dict(temperature=0.9, top_p=0.9, dt_bias=0.1)


def family(program):
    return 'drums' if program == DRUM_PROGRAM else ('keys', 'keys', 'keys', 'guitar', 'bass', 'strings', 'strings',
                                                     'brass', 'winds', 'winds', 'synth', 'synth', 'fx', 'ethnic',
                                                     'perc', 'fx')[program // 8]


# GM program groups used by the style rules
PIANO = set(range(0, 8))                                   # pianos, e-pianos, harpsichord, clavinet
SYNTH = set(range(80, 104)) | {38, 39, 50, 51, 62, 63}    # leads, pads, fx, synth bass/strings/brass
ORCH = set(range(40, 48)) | {48, 49} | set(range(56, 62)) | set(range(64, 80))  # strings, brass, reeds, pipes
ELECTRIC_GTR = set(range(26, 32))                          # jazz/clean/muted/overdriven/distortion/harmonics
ACOUSTIC_GTR = {24, 25}
BASS = set(range(32, 38))                                  # acoustic/electric/fretless/slap (not synth bass)
DRUMS = DRUM_PROGRAM


def style_of(programs):
    """Style of a window from its notes' instruments (per-note programs). None = none of the five."""
    counts = np.bincount(programs, minlength=DRUM_PROGRAM + 1)
    n = counts.sum()
    has = lambda group: any(counts[g] for g in group)
    share = lambda group: sum(counts[g] for g in group) / n
    used = {g for g in range(len(counts)) if counts[g]}
    drums = counts[DRUMS] > 0
    if drums and share(SYNTH) >= 0.30:
        return 'electronic'
    if drums and has(BASS | {38, 39}) and has(ELECTRIC_GTR | ACOUSTIC_GTR) and share(SYNTH) < 0.15:
        return 'rock_pop'
    if not drums and used <= PIANO and len(used) <= 2:
        return 'piano_keys'
    if not drums and share(ORCH) >= 0.6 and not has(ELECTRIC_GTR | SYNTH):
        return 'orchestral'
    if not drums and has(BASS) and has(PIANO | ACOUSTIC_GTR) and not has(ELECTRIC_GTR | SYNTH) and len(used) <= 5:
        return 'acoustic'
    return None


def style_set(per_style=16, prompt_notes=128, new_notes=1000, store=STORE, candidates=40000, skip=0):
    """{style: [dict(prompt (T,4), prompt_prog (T,), real metrics, file index)]}, deterministic. skip: leave out the
    first `skip` songs of each style (skip=16 gives a held-out set next to the in-training one)."""
    offsets = np.load(f'{store}/offsets.npy')
    n = int(offsets[-1])
    tokens = np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    programs = np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(n,))
    out = {s: [] for s in STYLES}
    for i in np.linspace(0, len(offsets) - 2, candidates).astype(int):
        a, b = offsets[i], offsets[i + 1]
        if b - a < prompt_notes + new_notes:
            continue
        p = np.asarray(programs[a:a + prompt_notes + new_notes], dtype=np.int64)
        s = style_of(p)
        if s is None or len(out[s]) >= per_style + skip:
            continue
        t = np.asarray(tokens[a:a + prompt_notes + new_notes], dtype=np.int64)
        real = row_metrics(torch.from_numpy(t[prompt_notes:, 0]), torch.from_numpy(t[prompt_notes:, 3]),
                           torch.from_numpy(p[prompt_notes:]), set(p[:prompt_notes].tolist()))
        out[s].append(dict(prompt=t[:prompt_notes], prompt_prog=p[:prompt_notes], real=real, file=int(i),
                           instruments=sorted(set(p.tolist()))))
        if all(len(v) >= per_style + skip for v in out.values()):
            break
    return {s: v[skip:] for s, v in out.items()}


def tf(k, v):
    return math.log(max(v, 1e-3)) if k in LOG else v


def style_scales(songs):
    """s_style,k: spread of t(metric k) over the style's real continuations, floored."""
    return {k: max(float(np.std([tf(k, s['real'][k]) for s in songs], ddof=1)) if len(songs) > 1 else 0.0, FLOOR[k])
            for k in WEIGHTS}


def row_distance(m, real, scale):
    return sum(w * min(abs(tf(k, m[k]) - tf(k, real[k])) / scale[k], CLIP) for k, w in WEIGHTS.items())


@torch.no_grad()
def evaluate(model, sset, new_notes=1000, seed=1234, chunk=20, cuda_graph=True, device='cuda', settings=TRACK,
             modes=None):
    """{mode: {'all': mean over styles, <style>: mean row distance, 'nps': .., 'top': .., ...}} for modes 'none' and
    (cond_inst models) 'band'. The model is put back into its previous train/eval mode even after an error."""
    was_training = model.training
    model.eval()
    try:
        return _evaluate(model, sset, new_notes, seed, chunk, cuda_graph, device, settings, modes)
    finally:
        model.train(was_training)


def _evaluate(model, sset, new_notes, seed, chunk, cuda_graph, device, settings, modes=None):
    cfg = model.config
    rows = [(s, song) for s in STYLES for song in sset[s]]
    scales = {s: style_scales(sset[s]) for s in STYLES if sset[s]}
    modes = modes or (['none'] + (['band'] if cfg.cond_inst else []))
    out = {}
    for mode in modes:
        per_style, metrics = {s: [] for s in STYLES}, []
        for c in range(0, len(rows), chunk):
            part = rows[c:c + chunk]
            tok = torch.stack([torch.from_numpy(song['prompt']) for _, song in part]).to(device)
            prog = torch.stack([torch.from_numpy(song['prompt_prog']) for _, song in part]).to(device)
            streams = [tok[:, :, j] for j in range(4)]
            if cfg.pitch_size > BOS_PITCH:
                bos = [torch.full_like(streams[0][:, :1], BOS_PITCH if j == 0 else 0) for j in range(4)]
                streams = [torch.cat([b, x], 1) for b, x in zip(bos, streams)]
                prog = torch.cat([torch.zeros_like(prog[:, :1]), prog], 1)
            cond = None
            if cfg.cond_inst or cfg.n_density:
                inst, _ = window_conditions(streams[0], streams[3], prog, cfg.n_programs, cfg.n_density or 16)
                zero = torch.zeros(len(part), dtype=torch.long, device=device)
                cond = (inst if mode == 'band' else torch.zeros_like(inst), zero)
            torch.manual_seed(seed + c)
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=device == 'cuda' and next(model.parameters()).dtype == torch.float32):
                res = model.generate(*streams, max_new_tokens=new_notes, temperature=settings['temperature'],
                                     top_p=settings['top_p'], dt_bias=settings['dt_bias'], program=None,
                                     programs=prog, return_programs=True, cond=cond, cuda_graph=cuda_graph,
                                     program_temperature=settings.get('instrument_temperature'))
            n0 = streams[0].size(1)
            for r, (style, song) in enumerate(part):
                m = row_metrics(res[0][r, n0:].cpu(), res[3][r, n0:].cpu(), res[4][r, n0:].cpu(),
                                set(song['prompt_prog'].tolist()))
                if m is None:  # nothing but EOS: count as maximally far
                    per_style[style].append(CLIP * sum(WEIGHTS.values()))
                    continue
                per_style[style].append(row_distance(m, song['real'], scales[style]))
                metrics.append(m)
        res_mode = {s: float(np.mean(v)) for s, v in per_style.items() if v}
        res_mode['all'] = float(np.mean([res_mode[s] for s in STYLES if s in res_mode]))
        for k in ('notes_per_s', 'top_share', 'takeover', 'runaway', 'out_of_band', 'instruments'):
            res_mode[k] = float(np.mean([m[k] for m in metrics])) if metrics else float('nan')
        out[mode] = res_mode
    return out


def format_line(step, rows, secs, res):
    """One log line; cloud/status_tables.py parses it."""
    parts = []
    for mode, r in res.items():
        parts.append(f"{mode}: P {r['all']:.2f} [" + ' '.join(f"{s} {r[s]:.2f}" for s in STYLES if s in r) +
                     f"] nps {r['notes_per_s']:.0f} top {r['top_share']:.0%} takeover {r['takeover']:.0%}")
    return f"styles @{step} ({rows} rows, {secs:.0f} s): " + ' | '.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--per-style', type=int, default=16)
    p.add_argument('--list', action='store_true', help='print the chosen songs and their profiles, then exit')
    p.add_argument('--no-cuda-graph', action='store_true')
    args = p.parse_args()
    sset = style_set(args.per_style)
    if args.list:
        import pretty_midi
        name = lambda g: 'Drums' if g == DRUM_PROGRAM else pretty_midi.program_to_instrument_name(g)
        for s in STYLES:
            sc = style_scales(sset[s])
            r = [x['real'] for x in sset[s]]
            print(f"\n== {s}: {len(sset[s])} songs | real: notes/s median {np.median([x['notes_per_s'] for x in r]):.0f}"
                  f" (log spread {sc['notes_per_s']:.2f}), top {np.mean([x['top_share'] for x in r]):.0%}, "
                  f"out-of-band {np.mean([x['out_of_band'] for x in r]):.0%}")
            for x in sset[s]:
                print(f"  file {x['file']:>6}  {x['real']['notes_per_s']:5.1f} notes/s  top {x['real']['top_share']:.0%}  "
                      + ', '.join(name(g) for g in x['instruments'])[:110])
        return
    from musicar.checkpoint import resolve_checkpoint
    from musicar.model import GPT, MusicConfig
    ck = torch.load(resolve_checkpoint(args.checkpoint), map_location='cuda')
    cfg = MusicConfig(**{k: v for k, v in ck['model_args'].items() if k in MusicConfig.__dataclass_fields__},
                      dropout=0.0)
    model = GPT(cfg)
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()})
    model = model.cuda().to(torch.bfloat16)
    import time
    t = time.time()
    res = evaluate(model, sset, cuda_graph=not args.no_cuda_graph)
    print(format_line(ck.get('iter_num'), sum(len(v) for v in sset.values()), time.time() - t, res))


if __name__ == '__main__':
    main()
