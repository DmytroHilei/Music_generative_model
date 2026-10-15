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
Consistency C (logged next to P): the same distance over sample_eval.consistency_metrics (4 s windows: density
variation and drift, velocity variation, instrument-mix change between windows, prompt instruments kept, dominance
drift), C_WEIGHTS / C_FLOOR; songs whose sample or real continuation spans < 3 windows are left out of C.

csv_set: the same for validation songs of a multi-track CSV (one group, spreads over its songs), e.g. the Ukrainian
set during a fine-tune (train.py style_eval_csv). Settings: TRACK (base model) or UA_MULTI (SETTINGS by name).
"""

import argparse
import math

import numpy as np
import torch

from musicar.data_loader import DRUM_PROGRAM
from musicar.eval.sample_eval import consistency_metrics, row_metrics
from musicar.sampling import generate, prompt_inputs

STORE = 'data/cache/gigamidi_clean_validation'
STYLES = ('rock_pop', 'orchestral', 'piano_keys', 'electronic', 'acoustic')
WEIGHTS = {'takeover': 0.20, 'top_share': 0.20, 'notes_per_s': 0.20, 'out_of_band': 0.15, 'instruments': 0.15,
           'runaway': 0.10}
LOG = {'notes_per_s', 'instruments'}
FLOOR = {'takeover': 0.22, 'runaway': 0.22, 'top_share': 0.05, 'out_of_band': 0.05, 'notes_per_s': 0.15,
         'instruments': 0.15}  # log-scale floors: ~15% ratio
CLIP = 3.0
# consistency over time (sample_eval.consistency_metrics, 4 s windows), scored like the distance above against each
# song's own real continuation; logged as C next to P, not added to it
C_WEIGHTS = {'density_var': 0.20, 'density_drift': 0.15, 'velocity_var': 0.15, 'mix_change': 0.20, 'prompt_kept': 0.15,
             'dominance_drift': 0.15}
C_FLOOR = {'density_var': 0.1, 'density_drift': 0.15, 'velocity_var': 0.5, 'mix_change': 0.05, 'prompt_kept': 0.1,
           'dominance_drift': 0.05}
# the fixed tracking setting (the best of the 60-song stage of the 2026-10-04 search: top-p 0.9 was in every good one)
TRACK = dict(temperature=0.9, top_p=0.9, dt_bias=0.1)
# the Ukrainian multi-track fine-tune (2026-10-13 grid on ua_450_multi, c8; the user's pick after listening): the
# instrument set of the prompt, denser than TRACK (dt 0.1 made the fine-tune ~0.6x the real notes/s), no early end
UA_MULTI = dict(temperature=1.0, top_p=0.9, dt_bias=-0.1, cond='band', min_new=1000)
SETTINGS = {'track': TRACK, 'ua_multi': UA_MULTI}


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
        real_c = consistency_metrics(*[torch.from_numpy(t[prompt_notes:, j]) for j in (0, 1, 3)],
                                     torch.from_numpy(p[prompt_notes:]), p[:prompt_notes].tolist())
        out[s].append(dict(prompt=t[:prompt_notes], prompt_prog=p[:prompt_notes], real=real, real_c=real_c,
                           file=int(i), instruments=sorted(set(p.tolist()))))
        if all(len(v) >= per_style + skip for v in out.values()):
            break
    return {s: v[skip:] for s, v in out.items()}


def csv_store(csv, split='validation', block_size=1128):
    """The token store of a multi-track CSV's split: (rows of that split whose MIDI exists = the store's file order,
    store dir, offsets, tokens (N, 4) memmap, programs (N,) memmap)."""
    from pathlib import Path
    import pandas as pd
    from musicar.data_loader import NoteWindowDataset
    ds = NoteWindowDataset(csv, root_dir='.', split=split, block_size=block_size, programs=True,
                           eval_stride=block_size)
    store = ds.store_dirs[0]
    df = pd.read_csv(csv)
    df = df[df['split'] == split].reset_index(drop=True)
    df = df[[Path(p).exists() for p in df['midi_filename']]].reset_index(drop=True)
    off = np.load(f'{store}/offsets.npy')
    assert len(off) - 1 == len(df), f'{store}: {len(off) - 1} files vs {len(df)} CSV rows'
    tokens = np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(int(off[-1]), 4))
    programs = np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(int(off[-1]),))
    return df, store, off, tokens, programs


def csv_set(csv, n_songs=24, prompt_notes=128, new_notes=1000, match=''):
    """{csv name: [songs]} in style_set's format (+ store, name, artist): n_songs validation songs of a multi-track CSV
    (0 = all), evenly spaced over the ones long enough; match = comma-separated parts of 'artist - title' (any case)
    instead."""
    import re
    from pathlib import Path
    df, store, off, tokens, programs = csv_store(csv, 'validation', prompt_notes + new_notes)
    long_enough = [i for i in range(len(df)) if off[i + 1] - off[i] >= prompt_notes + new_notes]
    n = min(n_songs or len(long_enough), len(long_enough))
    picked = [long_enough[int(k)] for k in np.linspace(0, len(long_enough) - 1, n)]
    if match:
        full = (df['artist'] + ' - ' + df['title']).str.lower()
        picked = []
        for pat in [m.strip().lower() for m in match.split(',') if m.strip()]:
            hits = [i for i in range(len(df)) if pat in full[i]]
            found = ', '.join(full[i] + ('' if i in long_enough else ' (too short)') for i in hits)
            print(f"--match '{pat}': {found or 'no song'}")
            picked += [i for i in hits if i in long_enough and i not in picked]
    songs = []
    for i in picked:
        a = off[i]
        t = np.asarray(tokens[a:a + prompt_notes + new_notes], dtype=np.int64)
        p = np.asarray(programs[a:a + prompt_notes + new_notes], dtype=np.int64)
        real = row_metrics(torch.from_numpy(t[prompt_notes:, 0]), torch.from_numpy(t[prompt_notes:, 3]),
                           torch.from_numpy(p[prompt_notes:]), set(p[:prompt_notes].tolist()))
        real_c = consistency_metrics(*[torch.from_numpy(t[prompt_notes:, j]) for j in (0, 1, 3)],
                                     torch.from_numpy(p[prompt_notes:]), p[:prompt_notes].tolist())
        name = re.sub(r'[^\w-]+', '_', f"{df['artist'][i]} - {df['title'][i]}").strip('_')[:60]
        songs.append(dict(prompt=t[:prompt_notes], prompt_prog=p[:prompt_notes], real=real, real_c=real_c, file=i,
                          store=str(store), name=name, artist=df['artist'][i]))
    return {Path(csv).stem: songs}


def tf(k, v):
    return math.log(max(v, 1e-3)) if k in LOG else v


def style_scales(songs):
    """s_style,k: spread of t(metric k) over the style's real continuations, floored."""
    return {k: max(float(np.std([tf(k, s['real'][k]) for s in songs], ddof=1)) if len(songs) > 1 else 0.0, FLOOR[k])
            for k in WEIGHTS}


def row_distance(m, real, scale):
    return sum(w * min(abs(tf(k, m[k]) - tf(k, real[k])) / scale[k], CLIP) for k, w in WEIGHTS.items())


def consistency_scales(songs):
    """Spread of each consistency term over the real continuations (songs with >= 3 windows), floored."""
    reals = [s['real_c'] for s in songs if s.get('real_c')]
    return {k: max(float(np.std([r[k] for r in reals], ddof=1)) if len(reals) > 1 else 0.0, C_FLOOR[k])
            for k in C_WEIGHTS}


def consistency_distance(c, real_c, scale):
    """C of one sample against its song's real continuation; None when either is too short (< 3 windows)."""
    if c is None or real_c is None:
        return None
    return sum(w * min(abs(c[k] - real_c[k]) / scale[k], CLIP) for k, w in C_WEIGHTS.items())


def score_row(res, r, n0, song, scale, c_scale):
    """Row r of generate()'s output (with programs; new notes from n0) against its song's real continuation:
    (metrics, distance, consistency metrics, consistency distance or None). A row of nothing but EOS has no
    metrics and the maximal distance."""
    pitch, vel, dt, prog = (res[j][r, n0:].cpu() for j in (0, 1, 3, 4))
    m = row_metrics(pitch, dt, prog, set(song['prompt_prog'].tolist()))
    d = CLIP * sum(WEIGHTS.values()) if m is None else row_distance(m, song['real'], scale)
    c = consistency_metrics(pitch, vel, dt, prog, song['prompt_prog'].tolist())
    return m, d, c, consistency_distance(c, song.get('real_c'), c_scale)


@torch.no_grad()
def evaluate(model, sset, new_notes=1000, seed=1234, chunk=20, cuda_graph=True, device='cuda', settings=TRACK,
             modes=None, rows_out=None):
    """{mode: {'all': mean over groups, <group>: mean row distance, 'notes_per_s': .., ...}} for modes 'none' and
    (cond_inst models) 'band', or only settings['cond'] when the settings name one. sset = style_set's styles or
    csv_set's group. settings['min_new']: no end-of-piece before that many notes. The model is put back into its
    previous train/eval mode even after an error. rows_out: a dict, filled with {mode: [(group, song index, distance,
    notes/s or None, consistency distance or None)]} in row order."""
    was_training = model.training
    model.eval()
    try:
        return _evaluate(model, sset, new_notes, seed, chunk, cuda_graph, device, settings, modes, rows_out)
    finally:
        model.train(was_training)


def _evaluate(model, sset, new_notes, seed, chunk, cuda_graph, device, settings, modes=None, rows_out=None):
    cfg = model.config
    groups = [s for s in sset if sset[s]]
    rows = [(s, song) for s in groups for song in sset[s]]
    index = [k for s in groups for k in range(len(sset[s]))]
    scales = {s: style_scales(sset[s]) for s in groups}
    c_scales = {s: consistency_scales(sset[s]) for s in groups}
    modes = modes or ([settings['cond']] if 'cond' in settings else ['none'] + (['band'] if cfg.cond_inst else []))
    out = {}
    for mode in modes:
        per_style, metrics = {s: [] for s in groups}, []
        per_style_c, c_metrics = {s: [] for s in groups}, []
        if rows_out is not None:
            rows_out[mode] = []
        for c in range(0, len(rows), chunk):
            part, start = rows[c:c + chunk], c
            tok = torch.stack([torch.from_numpy(song['prompt']) for _, song in part]).to(device)
            prog = torch.stack([torch.from_numpy(song['prompt_prog']) for _, song in part]).to(device)
            streams, prog, cond = prompt_inputs(model, tok, prog, band=mode == 'band')
            torch.manual_seed(seed + c)
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=device == 'cuda' and next(model.parameters()).dtype == torch.float32):
                res = generate(model, *streams, max_new_tokens=new_notes, temperature=settings['temperature'],
                               top_p=settings['top_p'], dt_bias=settings['dt_bias'], program=None,
                               programs=prog, return_programs=True, cond=cond, cuda_graph=cuda_graph,
                               program_temperature=settings.get('instrument_temperature'),
                               min_new=settings.get('min_new', 0))
            n0 = streams[0].size(1)
            for r, (style, song) in enumerate(part):
                m, d, c, dc = score_row(res, r, n0, song, scales[style], c_scales[style])
                per_style[style].append(d)
                if rows_out is not None:
                    rows_out[mode].append((style, index[start + r], d, m['notes_per_s'] if m else None, dc))
                if m is not None:
                    metrics.append(m)
                if dc is not None:
                    per_style_c[style].append(dc)
                    c_metrics.append(c)
        res_mode = {s: float(np.mean(v)) for s, v in per_style.items() if v}
        res_mode['all'] = float(np.mean([res_mode[s] for s in groups if s in res_mode]))
        for k in ('notes_per_s', 'top_share', 'takeover', 'runaway', 'out_of_band', 'instruments', 'pc_entropy'):
            res_mode[k] = float(np.mean([m[k] for m in metrics])) if metrics else float('nan')
        c_means = [float(np.mean(v)) for v in per_style_c.values() if v]
        res_mode['consistency'] = float(np.mean(c_means)) if c_means else float('nan')
        for k in C_WEIGHTS:
            res_mode[f'c_{k}'] = float(np.mean([c[k] for c in c_metrics])) if c_metrics else float('nan')
        out[mode] = res_mode
    return out


METRICS = {'all', 'notes_per_s', 'top_share', 'takeover', 'runaway', 'out_of_band', 'instruments', 'pc_entropy',
           'consistency'} | {f'c_{k}' for k in C_WEIGHTS}


def format_line(step, rows, secs, res):
    """One log line; cloud/status_tables.py parses it."""
    parts = []
    for mode, r in res.items():
        groups = [s for s in r if s not in METRICS]
        parts.append(f"{mode}: P {r['all']:.2f} [" + ' '.join(f"{s} {r[s]:.2f}" for s in groups) +
                     f"] nps {r['notes_per_s']:.0f} top {r['top_share']:.0%} takeover {r['takeover']:.0%}"
                     + (f" C {r['consistency']:.2f}" if 'consistency' in r else ''))
    return f"styles @{step} ({rows} rows, {secs:.0f} s): " + ' | '.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--per-style', type=int, default=16)
    p.add_argument('--store', default=STORE,
                   help='laptop: data/cache/gigamidi_heldout_validation (the val store is on the cloud repo only)')
    p.add_argument('--list', action='store_true', help='print the chosen songs and their profiles, then exit')
    p.add_argument('--no-cuda-graph', action='store_true')
    args = p.parse_args()
    sset = style_set(args.per_style, store=args.store)
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
    import time
    from musicar.checkpoint import load_model
    model, it = load_model(args.checkpoint, 'cuda', torch.bfloat16)
    t = time.time()
    res = evaluate(model, sset, cuda_graph=not args.no_cuda_graph)
    print(format_line(it, sum(len(v) for v in sset.values()), time.time() - t, res))


if __name__ == '__main__':
    main()
