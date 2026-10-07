"""
Multi-instrument samples for listening: a multi-instrument checkpoint continues real band songs of each style
(style_eval's five styles), every style with its own best settings from the settings sweep.

    python eval/multi_generate.py --checkpoint checkpoints/long_450m_final --out-dir samples/multi_450m

Songs: --per-style per style from style_eval.style_set on --store, default the clean GigaMIDI **test** split (never
trained on, and not the val songs eval/style_search.py picked the settings on; the val store is not on the laptop). Prompt = the first 128 notes with their instruments, then --new-notes notes; the instrument of every
new note is sampled. Settings per style from --settings (eval/style_search.py's best_settings.json): temperature,
top-p, instrument temperature, delta-time bias, conditioning 'band' (the prompt's instrument set) or 'none'.
Writes <out-dir>/<style>_<file>.mid/.mp3 (prompt + continuation) and real/<style>_<file>.mid/.mp3 (the real song,
same length) and prints the instruments each sample used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from musicar.data_loader import DRUM_PROGRAM
from musicar.eval.structure_eval import load_model
from musicar.eval.style_eval import STYLES, TRACK, style_set
from musicar.midi_io import render_mp3, tokens_to_midi
from musicar.model import BOS_PITCH, EOS_PITCH, window_conditions

SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'


def names(programs):
    return ', '.join('Drums' if p == DRUM_PROGRAM else pretty_midi.program_to_instrument_name(p)
                     for p in sorted(set(programs)))


def write(path, tokens, programs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(*[tokens[:, j].tolist() for j in range(4)], programs=programs.tolist()).write(str(path))
    render_mp3(path, SOUNDFONT)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoint', default='checkpoints/long_450m_final')
    ap.add_argument('--settings', default='results/long_450m_sweep/best_settings.json')
    ap.add_argument('--styles', default=','.join(STYLES))
    ap.add_argument('--per-style', type=int, default=3)
    ap.add_argument('--store', default='data/cache/gigamidi_clean_test')
    ap.add_argument('--skip', type=int, default=0, help='leave out the first songs of each style')
    ap.add_argument('--new-notes', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out-dir', default='samples/multi_450m')
    args = ap.parse_args()

    best = json.loads(Path(args.settings).read_text()) if Path(args.settings).exists() else {}
    styles = [s.strip() for s in args.styles.split(',')]
    sset = style_set(per_style=args.per_style, new_notes=args.new_notes, store=args.store, skip=args.skip)
    model, cfg, it = load_model(args.checkpoint, 'cuda')
    model = model.to(torch.bfloat16)
    out = Path(args.out_dir)
    print(f"{args.checkpoint} @{it}: {args.per_style} songs/style, {args.new_notes} new notes", flush=True)

    for style in styles:
        songs = sset[style]
        st = best.get(style, {}).get('settings', dict(TRACK, cond='band'))
        tok = torch.stack([torch.from_numpy(s['prompt']) for s in songs]).cuda()
        prog = torch.stack([torch.from_numpy(s['prompt_prog']) for s in songs]).cuda()
        streams = [tok[:, :, j] for j in range(4)]
        bos = [torch.full_like(streams[0][:, :1], BOS_PITCH if j == 0 else 0) for j in range(4)]
        streams = [torch.cat([b, x], 1) for b, x in zip(bos, streams)]
        prog = torch.cat([torch.zeros_like(prog[:, :1]), prog], 1)
        cond = None
        if cfg.cond_inst or cfg.n_density:
            inst, _ = window_conditions(streams[0], streams[3], prog, cfg.n_programs, cfg.n_density or 16)
            cond = (inst if st['cond'] == 'band' else torch.zeros_like(inst),
                    torch.zeros(len(songs), dtype=torch.long, device='cuda'))
        torch.manual_seed(args.seed)
        with torch.no_grad():
            res = model.generate(*streams, max_new_tokens=args.new_notes, temperature=st['temperature'],
                                 top_p=st['top_p'], dt_bias=st['dt_bias'], program=None, programs=prog,
                                 return_programs=True, cond=cond, cuda_graph=False,
                                 program_temperature=st.get('instrument_temperature'))
        print(f"\n{style}: T {st['temperature']}, top-p {st['top_p']}, instrument T {st.get('instrument_temperature')}, "
              f"dt bias {st['dt_bias']}, cond {st['cond']}", flush=True)
        n0 = streams[0].size(1)
        for r, song in enumerate(songs):
            rows = np.stack([x[r].float().cpu().numpy().astype(np.int64) for x in res[:4]], 1)[1:]  # drop BOS
            progs = res[4][r].cpu().numpy().astype(np.int64)[1:]
            new = rows[n0 - 1:, 0]
            if EOS_PITCH in new:
                cut = n0 - 1 + list(new).index(EOS_PITCH)
                rows, progs = rows[:cut], progs[:cut]
            keep = rows[:, 0] < BOS_PITCH
            rows, progs = rows[keep], progs[keep]
            name = f"{style}_{song['file']}"
            write(out / f'{name}.mid', rows, progs)
            print(f"  {name}: prompt [{names(song['prompt_prog'])}] -> sample [{names(progs[n0 - 1:])}]", flush=True)
            # the real song, as long as the sample
            store = args.store
            off = np.load(f'{store}/offsets.npy')
            n = int(off[-1])
            a = off[song['file']]
            t = np.asarray(np.memmap(f'{store}/tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))[a:a + len(rows)],
                           dtype=np.int64)
            p = np.asarray(np.memmap(f'{store}/programs.u8', dtype=np.uint8, mode='r', shape=(n,))[a:a + len(rows)],
                           dtype=np.int64)
            write(out / 'real' / f'{name}.mid', t, p)


if __name__ == '__main__':
    main()
