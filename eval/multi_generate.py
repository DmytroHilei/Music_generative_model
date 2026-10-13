"""
Multi-instrument samples for listening, each scored against its own song's real continuation: a multi-instrument
checkpoint continues real band songs, either GigaMIDI songs of style_eval's five styles (every style with its own
best settings from the settings sweep) or the validation songs of a multi-track CSV (e.g. the Ukrainian set).

    python eval/multi_generate.py --checkpoint checkpoints/long_450m_final --out-dir samples/multi_450m
    python eval/multi_generate.py --checkpoint checkpoints/ua_450_multi --csv data/finetune/ukrainian_multi_v1.csv \\
        --songs 12 --out-dir samples/multi_ua

Songs: --per-style per style from style_eval.style_set on --store, default the held-out 10 % of the clean GigaMIDI
test split (tools/split_store.py; the other 90 % is fine-tune replay, and the val store is not on the laptop). With
--csv: --songs validation songs of that CSV (evenly spaced over the ones long enough), with the settings of
--csv-style; --match picks songs by name instead. Prompt = the first 128 notes with their instruments, then
--new-notes notes; the instrument of every new note is sampled. Settings per style from --settings
(eval/style_search.py's best_settings.json): temperature, top-p, instrument temperature, delta-time bias,
conditioning 'band' (the prompt's instrument set) or 'none'; --set overrides them (e.g. 'dt_bias=0,temperature=1').
--min-new: no end-of-piece before that many new notes. --density prompt: density condition = the prompt's level
(default none). --target-nps prompt: the density controller tracks the prompt's notes/s.

Score per sample = style_eval's row distance to its own song's real next --new-notes notes (instruments, out-of-band
share, notes/s, top-instrument share, takeover, runaway; spreads from the real continuations of the same group),
lower = closer; the mean per group is printed and written to <out-dir>/scores.json.
Writes <out-dir>/<group>_<song>.mid/.mp3 (prompt + continuation) and real/<group>_<song>.mid/.mp3 (the real song,
same length) and prints the instruments each sample used.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch

from musicar.data_loader import DRUM_PROGRAM, MaestroDataset
from musicar.eval.sample_eval import row_metrics
from musicar.eval.structure_eval import load_model
from musicar.eval.style_eval import CLIP, STYLES, TRACK, WEIGHTS, row_distance, style_scales, style_set
from musicar.midi_io import render_mp3, tokens_to_midi
from musicar.model import BOS_PITCH, EOS_PITCH, window_conditions

SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
STORE = 'data/cache/gigamidi_heldout_validation'


def names(programs):
    return ', '.join('Drums' if p == DRUM_PROGRAM else pretty_midi.program_to_instrument_name(p)
                     for p in sorted(set(programs)))


def write(path, tokens, programs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(*[tokens[:, j].tolist() for j in range(4)], programs=programs.tolist()).write(str(path))
    render_mp3(path, SOUNDFONT)


def store_arrays(store):
    off = np.load(f'{store}/offsets.npy')
    n = int(off[-1])
    return (off, np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4)),
            np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(n,)))


def csv_songs(csv, n_songs, prompt_notes, new_notes, match=''):
    """Validation songs of a multi-track CSV, in style_set's format (+ store, name)."""
    ds = MaestroDataset(csv, root_dir='.', split='validation', block_size=prompt_notes + new_notes,
                        programs=True, eval_stride=prompt_notes + new_notes)
    store = ds.store_dirs[0]
    df = pd.read_csv(csv)
    df = df[df['split'] == 'validation'].reset_index(drop=True)
    df = df[[Path(p).exists() for p in df['midi_filename']]].reset_index(drop=True)  # = the store's file order
    off, tokens, programs = store_arrays(store)
    assert len(off) - 1 == len(df), f'{store}: {len(off) - 1} files vs {len(df)} CSV rows'
    long_enough = [i for i in range(len(df)) if off[i + 1] - off[i] >= prompt_notes + new_notes]
    picked = [long_enough[int(k)] for k in np.linspace(0, len(long_enough) - 1, min(n_songs, len(long_enough)))]
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
        name = re.sub(r'[^\w-]+', '_', f"{df['artist'][i]} - {df['title'][i]}").strip('_')[:60]
        songs.append(dict(prompt=t[:prompt_notes], prompt_prog=p[:prompt_notes], real=real, file=i, store=str(store),
                          name=name))
    return songs


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoint', default='checkpoints/long_450m_final')
    ap.add_argument('--settings', default='results/long_450m_sweep/best_settings.json')
    ap.add_argument('--styles', default=','.join(STYLES))
    ap.add_argument('--per-style', type=int, default=3)
    ap.add_argument('--store', default=STORE)
    ap.add_argument('--skip', type=int, default=0, help='leave out the first songs of each style')
    ap.add_argument('--csv', default='', help='multi-track CSV: continue its validation songs instead of the styles')
    ap.add_argument('--songs', type=int, default=12, help='with --csv: number of songs')
    ap.add_argument('--match', default='', help="with --csv: comma-separated parts of 'artist - title' (any case) "
                                                "to continue these songs instead of --songs")
    ap.add_argument('--csv-style', default='rock_pop', help="with --csv: whose settings to use ('rock_pop' = band songs)")
    ap.add_argument('--new-notes', type=int, default=1000)
    ap.add_argument('--set', default='', help="settings overrides, e.g. 'dt_bias=0,temperature=1,top_p=none'")
    ap.add_argument('--min-new', type=int, default=0, help='no end-of-piece before this many new notes')
    ap.add_argument('--density', default='none', choices=['none', 'prompt'], help='density condition')
    ap.add_argument('--target-nps', default='none', choices=['none', 'prompt'],
                    help="density controller target: the prompt's notes/s")
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out-dir', default='samples/multi_450m')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    best = json.loads(Path(args.settings).read_text()) if Path(args.settings).exists() else {}
    if args.csv:
        groups = {Path(args.csv).stem: (args.csv_style,
                                        csv_songs(args.csv, args.songs, 128, args.new_notes, args.match))}
    else:
        sset = style_set(per_style=args.per_style, new_notes=args.new_notes, store=args.store, skip=args.skip)
        groups = {s.strip(): (s.strip(), [dict(song, store=args.store, name=str(song['file'])) for song in sset[s.strip()]])
                  for s in args.styles.split(',')}
    dev = args.device
    model, cfg, it = load_model(args.checkpoint, dev)
    model = model.to(torch.bfloat16) if dev == 'cuda' else model
    out = Path(args.out_dir)
    print(f"{args.checkpoint} @{it}: {', '.join(f'{g} {len(v[1])}' for g, v in groups.items())} songs, "
          f"{args.new_notes} new notes", flush=True)

    scores = {'checkpoint': args.checkpoint, 'iter': it, 'groups': {}}
    for group, (style, songs) in groups.items():
        st = dict(best.get(style, {}).get('settings', dict(TRACK, cond='band')))
        for kv in filter(None, args.set.split(',')):
            k, v = kv.split('=')
            st[k] = v if k == 'cond' else None if v == 'none' else float(v)
        st.update(min_new=args.min_new, density=args.density, target_nps=args.target_nps)
        scale = style_scales(songs)
        tok = torch.stack([torch.from_numpy(s['prompt']) for s in songs]).to(dev)
        prog = torch.stack([torch.from_numpy(s['prompt_prog']) for s in songs]).to(dev)
        streams = [tok[:, :, j] for j in range(4)]
        has_bos = cfg.pitch_size > BOS_PITCH  # models with BOS/EOS: the prompt starts at the piece's beginning
        if has_bos:
            bos = [torch.full_like(streams[0][:, :1], BOS_PITCH if j == 0 else 0) for j in range(4)]
            streams = [torch.cat([b, x], 1) for b, x in zip(bos, streams)]
            prog = torch.cat([torch.zeros_like(prog[:, :1]), prog], 1)
        cond = None
        if cfg.cond_inst or cfg.n_density:
            inst, density = window_conditions(streams[0], streams[3], prog, cfg.n_programs, cfg.n_density or 16)
            cond = (inst if st['cond'] == 'band' else torch.zeros_like(inst),
                    density if args.density == 'prompt' else torch.zeros_like(density))
        target = None
        if args.target_nps == 'prompt':  # same notes/s as window_conditions (BOS rows ignored, 0.02 s per bin)
            real = streams[0] < BOS_PITCH
            target = (real.sum(1) / ((streams[3] * real).sum(1) * 0.02).clamp(min=0.5)).tolist()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            res = model.generate(*streams, max_new_tokens=args.new_notes, temperature=st['temperature'],
                                 top_p=st['top_p'], dt_bias=st['dt_bias'], program=None, programs=prog,
                                 return_programs=True, cond=cond, cuda_graph=False,
                                 program_temperature=st.get('instrument_temperature'), min_new=args.min_new,
                                 target_nps=target)
        print(f"\n{group}: settings of {style}: T {st['temperature']}, top-p {st['top_p']}, instrument T "
              f"{st.get('instrument_temperature')}, dt bias {st['dt_bias']}, cond {st['cond']}, min new "
              f"{args.min_new}, density {args.density}, target nps {args.target_nps}", flush=True)
        n0 = streams[0].size(1)
        rows_out = []
        for r, song in enumerate(songs):
            m = row_metrics(res[0][r, n0:].cpu(), res[3][r, n0:].cpu(), res[4][r, n0:].cpu(),
                            set(song['prompt_prog'].tolist()))
            d = CLIP * sum(WEIGHTS.values()) if m is None else row_distance(m, song['real'], scale)
            rows = np.stack([x[r].float().cpu().numpy().astype(np.int64) for x in res[:4]], 1)[int(has_bos):]
            progs = res[4][r].cpu().numpy().astype(np.int64)[int(has_bos):]
            p0 = n0 - int(has_bos)  # first new note in rows
            new = rows[p0:, 0]
            if EOS_PITCH in new:
                cut = p0 + list(new).index(EOS_PITCH)
                rows, progs = rows[:cut], progs[:cut]
            keep = rows[:, 0] < BOS_PITCH
            rows, progs = rows[keep], progs[keep]
            name = f"{group}_{song['name']}"
            write(out / f'{name}.mid', rows, progs)
            print(f"  {name}: d {d:.2f} | {m['notes_per_s'] if m else 0:.1f} notes/s (real "
                  f"{song['real']['notes_per_s']:.1f}), {m['instruments'] if m else 0} instr (real "
                  f"{song['real']['instruments']}), out-of-band {m['out_of_band'] if m else 0:.0%} | prompt "
                  f"[{names(song['prompt_prog'])}] -> sample [{names(progs[p0:])}]", flush=True)
            rows_out.append(dict(song=song['name'], file=song['file'], dist=d, metrics=m, real=song['real']))
            # the real song, as long as the sample
            off, tokens, programs = store_arrays(song['store'])
            a = off[song['file']]
            write(out / 'real' / f'{name}.mid', np.asarray(tokens[a:a + len(rows)], dtype=np.int64),
                  np.asarray(programs[a:a + len(rows)], dtype=np.int64))
        mean = float(np.mean([x['dist'] for x in rows_out]))
        print(f"  {group} mean distance {mean:.3f} over {len(rows_out)} songs", flush=True)
        scores['groups'][group] = dict(settings=st, mean_dist=mean, songs=rows_out)
    if len(scores['groups']) > 1:
        scores['mean_dist'] = float(np.mean([g['mean_dist'] for g in scores['groups'].values()]))
        print(f"\nmean over groups {scores['mean_dist']:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scores.json').write_text(json.dumps(scores, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
