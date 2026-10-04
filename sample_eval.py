"""
Generative evaluation: sample from fixed prompts and measure what the val loss can't see (instrument drift, one
instrument taking over, runaway density, ...). Used by train.py at every evaluation (sample_eval_rows > 0) and as a
command for any checkpoint:

    python sample_eval.py --checkpoint checkpoints/long_450m_snap          # prints the metrics per mode

Prompts: the same rows every time (the first files of the clean GigaMIDI val store with >= min_instruments
instruments and >= prompt_notes notes, at evenly spaced indices), the same seed, so changes reflect the model.
Modes: 'none' (no conditioning) and, for models trained with cond_inst, 'band' (instrument set of the prompt).

Per row, then averaged: instruments (distinct programs in the new notes), out_of_band (share of new notes on
instruments the prompt doesn't use), notes_per_s, top_share (share of the most used instrument), runaway (notes/s
> 40), takeover (top_share > 0.8), pc_entropy (pitch-class entropy, bits; 0 = one pitch class), same_onset (share
of notes with delta_time 0) and, with BOS/EOS, ended (share of rows that produced EOS).
"""

import argparse

import numpy as np
import torch

from model import BOS_PITCH, EOS_PITCH, DRUM_PROGRAM, window_conditions

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


@torch.no_grad()
def evaluate(model, prompts, new_notes=1000, temperature=0.9, seed=1234, device='cuda'):
    """{mode: {metric: mean over rows}} for 'none' and (cond_inst models) 'band'."""
    cfg = model.config
    tokens, programs = (t.to(device) for t in prompts)
    streams = [tokens[:, :, j] for j in range(4)]
    if cfg.pitch_size > BOS_PITCH:  # models with BOS/EOS: prompts start at the piece's beginning
        bos = [torch.full_like(streams[0][:, :1], BOS_PITCH if j == 0 else 0) for j in range(4)]
        streams = [torch.cat([b, s], 1) for b, s in zip(bos, streams)]
        programs = torch.cat([torch.zeros_like(programs[:, :1]), programs], 1)
    modes = {'none': None}
    if cfg.cond_inst or cfg.n_density:
        none = (torch.zeros(len(tokens), cfg.n_programs, device=device),
                torch.zeros(len(tokens), dtype=torch.long, device=device))
        modes = {'none': none}
        if cfg.cond_inst:
            inst, _ = window_conditions(streams[0], streams[3], programs, cfg.n_programs, cfg.n_density or 16)
            modes['band'] = (inst, none[1])
    was_training = model.training
    model.eval()
    try:
        return _sample_modes(model, cfg, streams, programs, tokens, modes, new_notes, temperature, seed, device)
    finally:
        model.train(was_training)  # also after a failure: training must continue in train mode


def _sample_modes(model, cfg, streams, programs, tokens, modes, new_notes, temperature, seed, device):
    out = {}
    for mode, cond in modes.items():
        torch.manual_seed(seed)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            res = model.generate(*streams, max_new_tokens=new_notes, temperature=temperature,
                                 program=None if cfg.n_programs else 0, programs=programs if cfg.n_programs else None,
                                 return_programs=True, cond=cond, cuda_graph=False)
        n0 = streams[0].size(1)
        rows = [row_metrics(res[0][r, n0:].cpu(), res[3][r, n0:].cpu(), res[4][r, n0:].cpu(),
                            set(programs[r].tolist())) for r in range(len(tokens))]
        rows = [r for r in rows if r]
        out[mode] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
    return out


def main():
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--new-notes', type=int, default=1000)
    p.add_argument('--temperature', type=float, default=0.9)
    args = p.parse_args()
    from generate import resolve_checkpoint
    from model import GPT, MusicConfig
    ckpt = torch.load(resolve_checkpoint(args.checkpoint), map_location='cuda')
    fields = set(MusicConfig.__dataclass_fields__)
    cfg = MusicConfig(**{k: v for k, v in ckpt['model_args'].items() if k in fields}, dropout=0.0)
    model = GPT(cfg)
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    model = model.cuda().to(torch.bfloat16)
    res = evaluate(model, fixed_prompts(args.rows), args.new_notes, args.temperature)
    for mode, m in res.items():
        print(f'{mode:5s} ' + ' | '.join(f'{k} {v:.3g}' for k, v in m.items()))


if __name__ == '__main__':
    main()
