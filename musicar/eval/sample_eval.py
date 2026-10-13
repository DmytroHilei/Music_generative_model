"""
Generative evaluation: sample from fixed prompts and measure what the val loss can't see (instrument drift, one
instrument taking over, runaway density, ...). Used by train.py at every evaluation (sample_eval_rows > 0) and as a
command for any checkpoint:

    python -m musicar.eval.sample_eval --checkpoint checkpoints/long_450m_snap          # prints the metrics per mode

Prompts: the same rows every time (the first files of the clean GigaMIDI val store with >= min_instruments
instruments and >= prompt_notes notes, at evenly spaced indices), the same seed, so changes reflect the model.
Modes: 'none' (no conditioning), for models trained with cond_inst 'band' (instrument set of the prompt), and with
n_density too 'band_dens' (band + the prompt's density level: the way the model would be used).

Per row, then averaged: instruments (distinct programs in the new notes), out_of_band (share of new notes on
instruments the prompt doesn't use), notes_per_s, top_share (share of the most used instrument), runaway (notes/s
> 40), takeover (top_share > 0.8), pc_entropy (pitch-class entropy, bits; 0 = one pitch class), same_onset (share
of notes with delta_time 0) and, with BOS/EOS, ended (share of rows that produced EOS).
"""

import argparse

import numpy as np
import torch

from musicar.data_loader import BOS_PITCH, EOS_PITCH, DRUM_PROGRAM
from musicar.sampling import generate, prompt_inputs

STORE = 'data/cache/gigamidi_clean_validation'


def fixed_prompts(n_rows=8, prompt_notes=128, min_instruments=3, store=STORE, candidates=4000):
    """(tokens (n_rows, T, 4), programs (n_rows, T)) int64, deterministic."""
    offsets = np.load(f'{store}/offsets.npy')
    n = int(offsets[-1])
    tokens = np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    programs = np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(n,))
    rows, progs = [], []
    for i in np.linspace(0, len(offsets) - 2, candidates).astype(int):
        a, b = offsets[i], offsets[i + 1]
        if b - a < prompt_notes:
            continue
        p = np.asarray(programs[a:a + prompt_notes], dtype=np.int64)
        if len(set(p.tolist())) < min_instruments:
            continue
        rows.append(np.asarray(tokens[a:a + prompt_notes], dtype=np.int64))
        progs.append(p)
        if len(rows) == n_rows:
            break
    return torch.from_numpy(np.stack(rows)), torch.from_numpy(np.stack(progs))


def reference(n_rows=8, prompt_notes=128, min_instruments=3, new_notes=1000, store=STORE, candidates=4000):
    """The same metrics on the REAL continuations of the fixed prompts (the next new_notes notes of each file): what
    good looks like. Real songs bring in new instruments and can be dense, so out_of_band / runaway aren't 0."""
    offsets = np.load(f'{store}/offsets.npy')
    n = int(offsets[-1])
    tokens = np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    programs = np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(n,))
    rows = []
    for i in np.linspace(0, len(offsets) - 2, candidates).astype(int):  # same selection as fixed_prompts
        a, b = offsets[i], offsets[i + 1]
        if b - a < prompt_notes:
            continue
        p = np.asarray(programs[a:a + prompt_notes], dtype=np.int64)
        if len(set(p.tolist())) < min_instruments:
            continue
        cont = slice(a + prompt_notes, min(b, a + prompt_notes + new_notes))
        t = torch.from_numpy(np.asarray(tokens[cont], dtype=np.int64))
        g = torch.from_numpy(np.asarray(programs[cont], dtype=np.int64))
        r = row_metrics(t[:, 0], t[:, 3], g, set(p.tolist())) if len(t) else None
        if r:
            rows.append(r)
        if len(rows) == n_rows:
            break
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out['sd'] = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in rows[0]}
    return out


# Distance to real music: D = sum_i w_i * |model_i - real_i| / s_i, s_i = the spread (sd) of metric i across the
# real continuations, floored so a metric that real songs never show (takeover) can't divide by ~0. Units: "real
# song-to-song spreads"; 0 = like real music, < ~0.3 close. Weights favour the audible failures seen so far.
DIST_WEIGHTS = {'takeover': 0.20, 'top_share': 0.20, 'notes_per_s': 0.20, 'out_of_band': 0.15, 'instruments': 0.15,
                'runaway': 0.10}
SD_FLOOR = {'takeover': 0.22, 'runaway': 0.22, 'top_share': 0.05, 'out_of_band': 0.05, 'notes_per_s': 5.0,
            'instruments': 1.0}  # binary rates: the per-row sd of a 5% rate


def distance(metrics, ref):
    """D for one mode's metrics against the reference dict (needs ref['sd'])."""
    sd = ref.get('sd', {})
    return float(sum(w * abs(metrics[k] - ref[k]) / max(sd.get(k, 0.0), SD_FLOOR[k]) for k, w in DIST_WEIGHTS.items()))


def row_metrics(pitch, dt, prog, prompt_prog, dt_seconds=0.02):
    """Metrics of one generated row (new notes only, cut at the first EOS)."""
    pitch, dt, prog = pitch.tolist(), dt.tolist(), prog.tolist()
    ended = EOS_PITCH in pitch
    if ended:
        cut = pitch.index(EOS_PITCH)
        pitch, dt, prog = pitch[:cut], dt[:cut], prog[:cut]
    keep = [i for i, p in enumerate(pitch) if p < BOS_PITCH]
    pitch, dt, prog = [pitch[i] for i in keep], [dt[i] for i in keep], [prog[i] for i in keep]
    if not pitch:
        return None
    counts = np.bincount(prog, minlength=DRUM_PROGRAM + 1)
    band = set(prompt_prog)
    secs = max(sum(dt) * dt_seconds, 0.5)
    pitched = [p for p, g in zip(pitch, prog) if g != DRUM_PROGRAM]
    pc = np.bincount([p % 12 for p in pitched], minlength=12) / max(len(pitched), 1)
    top = counts.max() / len(pitch)
    return dict(instruments=int((counts > 0).sum()),
                out_of_band=sum(1 for g in prog if g not in band) / len(prog),
                notes_per_s=len(pitch) / secs, top_share=top, runaway=float(len(pitch) / secs > 40),
                takeover=float(top > 0.8),
                pc_entropy=float(-(pc[pc > 0] * np.log2(pc[pc > 0])).sum()) if pitched else 0.0,
                same_onset=sum(1 for d in dt if d == 0) / len(dt), ended=float(ended))


def consistency_metrics(pitch, vel, dt, prog, prompt_prog, window_s=4.0, dt_seconds=0.02):
    """Consistency over time of one row (new notes only, cut at the first EOS), from whole window_s windows of the
    onset timeline (the incomplete tail dropped, empty windows kept); None with fewer than 3 windows. thirds = the
    first / last floor(K / 3) of the K windows. prompt_prog: the instrument of every prompt note (a sequence, not a
    set: the 5 % rule counts them).
      density_var      std / mean of the windows' notes/s
      density_drift    log((last-third notes/s + 0.25) / (first-third notes/s + 0.25))
      velocity_var     std of the mean velocity bin of the non-empty windows
      mix_change       mean Jensen-Shannon distance (base 2) between the instrument shares of neighbouring windows,
                       over pairs where both have notes
      prompt_kept      share of the prompt's main instruments (>= 5 % of its notes) with a note in the last third
      dominance_drift  mean top-instrument share of the last third minus that of the first third (non-empty windows)
    """
    pitch, vel, dt, prog = pitch.tolist(), vel.tolist(), dt.tolist(), prog.tolist()
    if EOS_PITCH in pitch:
        cut = pitch.index(EOS_PITCH)
        pitch, vel, dt, prog = pitch[:cut], vel[:cut], dt[:cut], prog[:cut]
    onset = np.cumsum(dt) * dt_seconds  # the BOS/EOS rows' delta_time still moves the clock, like in the MIDI
    keep = [i for i, p in enumerate(pitch) if p < BOS_PITCH]
    if not keep:
        return None
    onset, vel, prog = onset[keep], np.asarray(vel)[keep], np.asarray(prog)[keep]
    n_win = int(onset[-1] // window_s)
    if n_win < 3:
        return None
    win = (onset // window_s).astype(int)
    inside = win < n_win
    win, vel, prog = win[inside], vel[inside], prog[inside]
    counts = np.zeros((n_win, DRUM_PROGRAM + 1))
    np.add.at(counts, (win, prog), 1)
    n = counts.sum(1)
    nps = n / window_s
    full = n > 0
    third = n_win // 3
    first, last = slice(0, third), slice(n_win - third, n_win)

    def js(p, q):
        m = (p + q) / 2
        kl = lambda a: float((a[a > 0] * np.log2(a[a > 0] / m[a > 0])).sum())
        return np.sqrt(max((kl(p) + kl(q)) / 2, 0.0))

    pairs = [js(counts[k] / n[k], counts[k + 1] / n[k + 1]) for k in range(n_win - 1) if full[k] and full[k + 1]]
    vel_win = np.array([vel[win == k].mean() for k in range(n_win) if full[k]])
    top = np.array([counts[k].max() / n[k] if full[k] else np.nan for k in range(n_win)])
    pp = np.bincount(np.asarray(list(prompt_prog), dtype=np.int64), minlength=DRUM_PROGRAM + 1)
    main = np.flatnonzero(pp >= 0.05 * pp.sum())
    played_last = counts[last].sum(0)
    return dict(density_var=float(nps.std() / max(nps.mean(), 1e-6)),
                density_drift=float(np.log((nps[last].mean() + 0.25) / (nps[first].mean() + 0.25))),
                velocity_var=float(vel_win.std()) if len(vel_win) else 0.0,
                mix_change=float(np.mean(pairs)) if pairs else 0.0,
                prompt_kept=float((played_last[main] > 0).mean()) if len(main) else 1.0,
                dominance_drift=float(np.nan_to_num(np.nanmean(top[last]) - np.nanmean(top[first]))
                                      if full[last].any() and full[first].any() else 0.0))


@torch.no_grad()
def evaluate(model, prompts, new_notes=1000, temperature=0.9, seed=1234, device='cuda'):
    """{mode: {metric: mean over rows}} for 'none' and, for cond_inst models, 'band' (the prompt's instrument set) and
    with n_density also 'band_dens' (band + the prompt's density level: how the model would be used). The model is
    put back into its previous train/eval mode even after an error."""
    cfg = model.config
    modes = {'none': (False, False)}
    if cfg.cond_inst:
        modes['band'] = (True, False)
        if cfg.n_density:
            modes['band_dens'] = (True, True)
    tokens, programs = (t.to(device) for t in prompts)
    was_training = model.training
    model.eval()
    try:
        out = {}
        for mode, (band, density) in modes.items():
            streams, progs, cond = prompt_inputs(model, tokens, programs, band, density)
            torch.manual_seed(seed)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                res = generate(model, *streams, max_new_tokens=new_notes, temperature=temperature,
                               program=None if cfg.n_programs else 0, programs=progs if cfg.n_programs else None,
                               return_programs=True, cond=cond, cuda_graph=False)
            n0 = streams[0].size(1)
            # the prompt's own instruments (without BOS's program 0) define the band
            rows = [row_metrics(res[0][r, n0:].cpu(), res[3][r, n0:].cpu(), res[4][r, n0:].cpu(),
                                set(programs[r].tolist())) for r in range(len(tokens))]
            rows = [r for r in rows if r]
            out[mode] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
    finally:
        model.train(was_training)  # also after a failure: training must continue in train mode
    ref = _load_reference()
    for mode, m in out.items():
        if m and ref and len(tokens) == ref.get('rows'):  # same prompt set as the reference
            m['distance'] = distance(m, ref)
    return out


def format_line(step, rows, secs, res):
    """One log line per evaluation."""
    return f"samples @{step} ({rows} rows, {secs:.0f} s): " + " | ".join(
        f"{mode}: instr {m['instruments']:.1f} oob {m['out_of_band']:.0%} nps {m['notes_per_s']:.0f} "
        f"top {m['top_share']:.0%} runaway {m['runaway']:.0%} takeover {m['takeover']:.0%}"
        for mode, m in res.items() if m)

def _load_reference():
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cloud', 'sample_reference.json')
    try:
        return json.loads(open(path).read())
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--new-notes', type=int, default=1000)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--reference', default=None, help="write the real-continuation metrics to this JSON (no model) "
                   "and exit; cloud/status_tables.py shows them as the 'real' row")
    args = p.parse_args()
    if args.reference:
        import json
        ref = reference(args.rows, new_notes=args.new_notes)
        open(args.reference, 'w').write(json.dumps({'rows': args.rows, 'new_notes': args.new_notes, **ref}, indent=1))
        print(json.dumps(ref, indent=1))
        return
    if not args.checkpoint:
        p.error('--checkpoint is required (or --reference to write the real-music reference)')
    from musicar.checkpoint import load_model
    model, _ = load_model(args.checkpoint, 'cuda', torch.bfloat16)
    res = evaluate(model, fixed_prompts(args.rows), args.new_notes, args.temperature)
    for mode, m in res.items():
        print(f'{mode:5s} ' + ' | '.join(f'{k} {v:.3g}' for k, v in m.items()))


if __name__ == '__main__':
    main()
