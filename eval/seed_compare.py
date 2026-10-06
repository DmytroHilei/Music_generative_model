"""
Prompted comparison of piano models on Ukrainian songs: every model continues the same openings of the Ukrainian
covers val songs, and the continuations are scored against what the real cover does next.

    python eval/seed_compare.py --checkpoints checkpoints/ua_2048_v2 checkpoints/long_450m_final \\
        --out-dir samples/seed_compare --out logs/seed_compare.txt

Seeds: one cover per distinct (artist, title) among the validation rows of --csv, Ukrainian-language songs only
(NON_UA_ARTISTS / NON_UA_TITLES drop the Russian-, English- and Spanish-language ones). Prompt = BOS + the first
--prompt-notes notes of the song. Each model gets its native setting: its trained style (structure_eval.pick_style),
and models with window conditions get instruments = piano, density none. EOS allowed after --min-notes new notes.
Table (structure_eval metrics over the continuations; reference = the real continuations, cropped to --max-new):
the structure_eval columns plus
  keySim   pitch-class histogram similarity of each continuation to its own song's real continuation (1 = same)
  dNps     |log notes/s ratio| to its own song's real continuation (0 = same density)
--out-dir writes per song <model>/<nn>_<song>.mid/.mp3 (prompt + continuation, the prompt is the same in every
model) and real/<nn>_<song>.mid/.mp3 (the real cover, the same length as the longest continuation) for listening.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from musicar.data_loader import tokenize_midi
from musicar.midi_io import render_mp3, tokens_to_midi
from musicar.jobstatus import JobStatus
from musicar.model import BOS_PITCH, EOS_PITCH
from musicar.eval.structure_eval import load_model, metrics, pick_style, summarize

# validation covers that are not Ukrainian-language songs (2026-10-07 v4 val set)
NON_UA_ARTISTS = {'Мішель Андраде', 'Пошлая Молли', 'Луна', 'Витас', 'Ундервуд', 'Pianoбой', 'Ані Лорак'}
NON_UA_TITLES = {('The Hardkiss', 'stones')}
SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'


def seed_songs(csv, need):
    """[(name, tokens (n, 4))] of the Ukrainian-language val songs with at least `need` notes, one per song."""
    df = pd.read_csv(csv)
    df = df[df['split'] == 'validation']
    songs, seen = [], set()
    for _, r in df.iterrows():
        key = (r['artist'], r['title'])
        if r['artist'] in NON_UA_ARTISTS or key in NON_UA_TITLES or key in seen:
            continue
        toks = tokenize_midi(r['midi_filename'])
        if toks is None or len(toks[0]) < need:
            continue
        seen.add(key)
        name = re.sub(r'[^\w-]+', '_', f"{r['artist']} - {r['title']}").strip('_')[:60]
        songs.append((name, np.stack([t.numpy() for t in toks], 1)))
    return songs


def continue_songs(model, cfg, args, prompts, style, device, job, base):
    """Continuations (new notes only, no BOS/EOS) and whether each ended with EOS, batched."""
    out_all = []
    for b in range(0, len(prompts), args.batch_size):
        P = np.stack(prompts[b:b + args.batch_size])  # (B, n, 4)
        B = len(P)
        x = [torch.from_numpy(P[:, :, j]).long().to(device) for j in range(4)]
        bos = [torch.full((B, 1), BOS_PITCH if j == 0 else 0, dtype=torch.long, device=device) for j in range(4)]
        x = [torch.cat([s, t], 1) for s, t in zip(bos, x)]
        cond = None
        if cfg.cond_inst or cfg.n_density:
            inst = torch.zeros(B, cfg.n_programs or 1, device=device)
            inst[:, 0] = 1.0  # piano
            cond = (inst, torch.zeros(B, dtype=torch.long, device=device))  # density: none
        torch.manual_seed(args.seed + b)
        with torch.no_grad():
            out = model.generate(*x, max_new_tokens=args.max_new, temperature=args.temperature, top_p=args.top_p,
                                 style=style,
                                 min_new=args.min_notes, cuda_graph=False,
                                 programs=torch.zeros_like(x[0]) if cfg.n_programs else None, cond=cond,
                                 progress=lambda n, b=b, B=B: job.update(base + (b + B * n / args.max_new)
                                                                        * args.max_new))
        n0 = x[0].size(1)
        for r in range(B):
            rows = np.stack([o[r, n0:].float().cpu().numpy().astype(np.int64) for o in out[:4]], 1)
            ended = EOS_PITCH in rows[:, 0]
            if ended:
                rows = rows[:list(rows[:, 0]).index(EOS_PITCH)]
            out_all.append((rows[rows[:, 0] < BOS_PITCH], ended))
    return out_all


def pc_hist(p):
    return np.bincount(p % 12, minlength=12) / max(len(p), 1)


def nps(t):
    return len(t) / max(t[1:, 3].sum() * 0.02, 1e-3)


def write(path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(*[tokens[:, j].tolist() for j in range(4)]).write(str(path))
    render_mp3(path, SOUNDFONT)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoints', nargs='+', default=['checkpoints/ua_2048_v2', 'checkpoints/long_450m_final'])
    ap.add_argument('--csv', default='data/finetune/ukrainian_covers_v4.csv')
    ap.add_argument('--style', default='cover,reduction', help='style preference list (structure_eval.pick_style)')
    ap.add_argument('--prompt-notes', type=int, default=128)
    ap.add_argument('--max-new', type=int, default=1500)
    ap.add_argument('--min-notes', type=int, default=200, help='no EOS before this many new notes')
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', type=float, default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out-dir', default=None, help='write MIDI + mp3 per song here')
    ap.add_argument('--out', default=None, help='also write the table here')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    songs = seed_songs(args.csv, args.prompt_notes + args.min_notes)
    prompts = [t[:args.prompt_notes] for _, t in songs]
    reals = [t[args.prompt_notes:args.prompt_notes + args.max_new] for _, t in songs]
    print(f"{len(songs)} Ukrainian seed songs: " + ', '.join(n for n, _ in songs), flush=True)

    header = (f"{'':40s}{'n':>4s}{'sec':>7s}{'nps':>7s}{'chord%':>7s}{'ends%':>6s}{'key%':>6s}{'melRep':>8s}"
              f"{'pitRep':>8s}{'recur':>7s}{'dDens':>7s}{'dReg':>7s}{'OA':>7s}{'dist':>7s}{'keySim':>8s}{'dNps':>7s}")
    lines = [f"seed_compare: {len(songs)} Ukrainian val covers ({Path(args.csv).stem}), prompt BOS + "
             f"{args.prompt_notes} notes, T={args.temperature}, top_p={args.top_p}, seed={args.seed}, EOS after "
             f"{args.min_notes}, max {args.max_new} new notes; reference = the real continuations", header]
    row, ref, _ = summarize('real continuation', [metrics(p, True) for p in reals], reals)
    lines.append(row)
    print('\n'.join(lines), flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else None
    job = JobStatus('seed_compare', len(args.checkpoints) * len(songs) * args.max_new,
                    f'{len(args.checkpoints)} checkpoints x {len(songs)} songs')
    longest = [0] * len(songs)
    ranking = []
    for ci, path in enumerate(args.checkpoints):
        model, cfg, it = load_model(path, args.device)
        if args.device == 'cuda':
            model = model.to(torch.bfloat16)
        style, style_name = pick_style(model, cfg, [s.strip() for s in args.style.split(',') if s.strip()])
        conts = continue_songs(model, cfg, args, prompts, style, args.device, job, ci * len(songs) * args.max_new)
        del model
        torch.cuda.empty_cache()
        key_sim = np.mean([1 - 0.5 * np.abs(pc_hist(c[:, 0]) - pc_hist(r[:, 0])).sum()
                           for (c, _), r in zip(conts, reals)])
        d_nps = np.mean([abs(np.log(nps(c) / nps(r))) for (c, _), r in zip(conts, reals)])
        name = f"{Path(path).name} @{it} [{style_name}]"
        row, _, dist = summarize(name, [metrics(c, e) for c, e in conts], [c for c, _ in conts], ref)
        row += f"{key_sim:8.3f}{d_nps:7.2f}"
        lines.append(row)
        ranking.append((dist, name))
        print(row, flush=True)
        if out_dir:
            for i, ((c, _), (song, _), p) in enumerate(zip(conts, songs, prompts)):
                longest[i] = max(longest[i], len(c))
                write(out_dir / Path(path).name / f'{i + 1:02d}_{song}.mid', np.concatenate([p, c]))
    if out_dir:
        for i, (song, t) in enumerate(songs):
            write(out_dir / 'real' / f'{i + 1:02d}_{song}.mid', t[:args.prompt_notes + longest[i]])
    lines += ['', 'closest to the real continuations: ' + ' < '.join(f"{n} ({d:.3f})" for d, n in sorted(ranking))]
    print(lines[-1])
    job.finish('done')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
