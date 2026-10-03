import argparse
import random
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch

from jobstatus import JobStatus
from model import GPT, MusicConfig
from data_loader import tokenize_midi, load_styles, BOS_PITCH, EOS_PITCH, DRUM_PROGRAM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate piano MIDI from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt",
                        help="checkpoint file, or a run directory (uses model_bf16.pt, else best.pt, else ckpt.pt)")
    parser.add_argument("--output", type=str, default="generated.mid",
                        help="output MIDI; with --num-samples > 1 a _<i> suffix is added")
    parser.add_argument("--num-samples", type=int, default=1, help="samples to generate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="samples generated together in one batch (default: all of them). A batch costs about "
                             "as much as one sample on GPU. Batch k uses seed + k, so results depend on the batching")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto",
                        help="auto = bf16 on CUDA, fp32 on CPU")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="decode eagerly on CUDA instead of replaying a captured CUDA graph (same notes)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="'cpu' keeps the GPU free for a running training job")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default = all cores)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soundfont", type=str,
                        default="/usr/share/sounds/sf2/FluidR3_GM.sf2",
                        help="Path to .sf2 soundfont for fluidsynth rendering")
    parser.add_argument("--no-mp3", action="store_true",
                        help="Skip MP3 rendering, save MIDI only")
    parser.add_argument("--prompt", type=str, default=None,
                        help="MIDI file whose first --prompt-notes notes seed the generation "
                             "(default: a single artificial seed note, which is out of distribution)")
    parser.add_argument("--val-prompt", choices=["combined", "aria", "gigamidi"], default=None,
                        help="take the prompt from a validation piece (never trained on): 'combined' = MAESTRO + "
                             "GiantMIDI val MIDI files, 'aria' = Aria val token store, 'gigamidi' = clean GigaMIDI "
                             "val store (multi-instrument, with its instruments). Random piece per sample "
                             "unless --val-index is given")
    parser.add_argument("--sample-instruments", action="store_true",
                        help="multi-instrument models: sample the instrument of every note (default: all piano)")
    parser.add_argument("--instruments", type=str, default=None,
                        help="with --sample-instruments: allowed GM programs, e.g. '0,33,25,128' (128 = drums), or "
                             "'prompt' = only the instruments of each row's prompt")
    parser.add_argument("--instrument-temperature", type=float, default=None,
                        help="with --sample-instruments: temperature of the instrument choice only (default: "
                             "--temperature); lower = fewer jumps to new instruments")
    parser.add_argument("--cond-instruments", type=str, default=None,
                        help="models trained with cond_inst: the band to condition on: 'prompt' (default when the "
                             "prompt has instruments), a GM program list like '0,33,48,128', or 'none'")
    parser.add_argument("--cond-density", type=str, default=None,
                        help="models trained with n_density: notes per second to condition on (e.g. 8), 'prompt', "
                             "or 'none' (default)")
    parser.add_argument("--min-instruments", type=int, default=1,
                        help="--val-prompt gigamidi: only prompts with at least this many instruments")
    parser.add_argument("--val-index", type=int, default=None, help="which validation piece (see --list-val)")
    parser.add_argument("--val-genre", type=str, default=None, help="aria only: restrict to a genre, e.g. pop")
    parser.add_argument("--list-val", action="store_true", help="print the --val-prompt pieces with indices and exit")
    parser.add_argument("--prompt-start", type=int, default=0, help="first prompt note inside the piece")
    parser.add_argument("--prompt-notes", type=int, default=64)
    parser.add_argument("--keep-prompt", action="store_true",
                        help="Include the prompt notes in the output MIDI")
    parser.add_argument("--anchor", type=int, default=0,
                        help="optional, off by default: pin the first N notes (e.g. the prompt's theme) at the start "
                             "of the context once generation runs past the 512-note window")
    parser.add_argument("--style", type=str, default=None,
                        help="models trained with styles: a name from data/styles.json (genre or artist, e.g. pop, "
                             "Скрябін) or 'none' (default: none = unconditional)")
    parser.add_argument("--min-notes", type=int, default=0,
                        help="models with BOS/EOS: don't allow the end token before this many new notes")
    parser.add_argument("--density", type=str, default=None,
                        help="optional: steer notes per second without retraining. A number (e.g. 5) or 'prompt' = "
                             "match the prompt's own density. Only the delta_time head is biased")
    parser.add_argument("--dt-bias", type=float, default=0.0,
                        help="fixed bias toward longer gaps (> 0 = calmer); with --density it's the starting value")
    parser.add_argument("--slide", type=int, default=None,
                        help="notes dropped per KV-cache rebuild past the window (default block_size // 4)")
    parser.add_argument("--max-polyphony", type=int, default=None,
                        help="Post-processing: max simultaneous notes per time slot (default: off, raw model output)")
    parser.add_argument("--max-delta", type=float, default=None,
                        help="Post-processing: cap on seconds per time step (default: off, raw model output)")
    return parser.parse_args()


def tokens_to_midi(pitches, velocities, durations, delta_times,
                   velocity_bins=32, time_resolution=0.02,
                   max_polyphony=None, max_delta=None, programs=None):
    """programs (optional, one per note, 128 = drums): one MIDI track per instrument, drums on channel 10 with a
    fixed short length (the model's drum durations are untrained: their loss is masked)."""
    midi = pretty_midi.PrettyMIDI()
    tracks = {}

    def track(program):
        if program not in tracks:
            drum = program == DRUM_PROGRAM
            name = 'Drums' if drum else pretty_midi.program_to_instrument_name(program)
            tracks[program] = pretty_midi.Instrument(program=0 if drum else program, is_drum=drum, name=name)
        return tracks[program]
    programs = programs if programs is not None else [0] * len(pitches)

    current_time = 0.0
    pending = []  # notes at the current time slot, flushed when time advances

    def flush(pending):
        pending.sort(key=lambda n: n[1].velocity, reverse=True)
        for prog, note in pending[:max_polyphony] if max_polyphony else pending:
            track(prog).notes.append(note)

    for p, v, d, dt, prog in zip(pitches, velocities, durations, delta_times, programs):
        delta = dt * time_resolution
        if max_delta is not None:
            delta = min(delta, max_delta)  # cap runaway gaps

        if delta > 0:
            flush(pending)
            pending = []
            current_time += delta

        velocity = min(127, int(v * 128 / velocity_bins) + 2)
        duration = 0.1 if prog == DRUM_PROGRAM else max(0.08, d * time_resolution)
        pending.append((int(prog), pretty_midi.Note(
            velocity=velocity,
            pitch=int(p),
            start=current_time,
            end=current_time + duration,
        )))

    flush(pending)
    midi.instruments.extend(tracks[k] for k in sorted(tracks))
    return midi


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        for name in ("model_bf16.pt", "best.pt", "ckpt.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"no model_bf16.pt / best.pt / ckpt.pt in {path}")
    return path


def val_pieces(source, genre=None):
    """Validation pieces as a list of (label, loader); loader() returns an (n, 4) int array of tokens, or for
    'gigamidi' a tuple (tokens, programs (n,))."""
    if source == "gigamidi":
        store = Path("data/cache/gigamidi_clean_validation")
        offsets = np.load(store / "offsets.npy")
        n = int(offsets[-1])
        tokens = np.memmap(store / "tokens.u16", dtype=np.uint16, mode="r", shape=(n, 4))
        programs = np.memmap(store / "programs.u8", dtype=np.uint8, mode="r", shape=(n,))
        return [(f"gigamidi_clean val file {i}",
                 lambda a=offsets[i], b=offsets[i + 1]: (np.asarray(tokens[a:b], dtype=np.int64),
                                                         np.asarray(programs[a:b], dtype=np.int64)))
                for i in range(len(offsets) - 1)]
    if source == "combined":
        df = pd.read_csv("data/combined.csv")
        paths = [Path(f) for f in df.loc[df["split"] == "validation", "midi_filename"] if Path(f).exists()]
        return [(p.stem, lambda p=p: np.stack([t.numpy() for t in tokenize_midi(p)], axis=1)) for p in paths]
    # Aria: store files are in the order of the validation rows of data/aria/aria.csv
    store = Path("data/cache/aria_validation")
    offsets = np.load(store / "offsets.npy")
    tokens = np.memmap(store / "tokens.u16", dtype=np.uint16, mode="r", shape=(int(offsets[-1]), 4))
    df = pd.read_csv("data/aria/aria.csv", dtype=str, keep_default_na=False)
    df = df[df["split"] == "validation"].reset_index(drop=True)
    assert len(df) == len(offsets) - 1, "aria.csv and the validation store are out of sync"
    pieces = []
    for i, row in df.iterrows():
        if genre and row["genre"] != genre:
            continue
        label = f"aria {row['file_id']}_{row['segment']} [{row['genre'] or 'no genre'}" + \
                (f", {row['composer']}]" if row["composer"] else "]")
        pieces.append((label, lambda a=offsets[i], b=offsets[i + 1]: np.asarray(tokens[a:b], dtype=np.int64)))
    return pieces


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)

    pieces = val_pieces(args.val_prompt, args.val_genre) if args.val_prompt else None
    if args.list_val:
        if not pieces:
            raise SystemExit("--list-val needs --val-prompt")
        for i, (label, _) in enumerate(pieces):
            print(f"{i:5d}  {label}")
        return

    ckpt = torch.load(resolve_checkpoint(args.checkpoint), map_location=device)
    ma = ckpt.get("model_args", {})
    # every architecture field the checkpoint knows; keys it predates fall back to what old checkpoints used
    fields = set(MusicConfig.__dataclass_fields__)
    legacy = dict(cascade_heads=False, cascade_residual=False, n_embd=256)
    config = MusicConfig(**{**legacy, **{k: v for k, v in ma.items() if k in fields}, 'dropout': 0.0})

    model = GPT(config)
    state_dict = ckpt["model"]
    # strip torch.compile prefix if present
    for k in list(state_dict):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(args.dtype, torch.bfloat16 if device == "cuda"
                                                                 else torch.float32)
    model.eval().to(device=device, dtype=dtype)

    print(f"Checkpoint: {resolve_checkpoint(args.checkpoint)} (iter {ckpt.get('iter_num', '?')}), "
          f"device {device}, {str(dtype).replace('torch.', '')}")
    out_base = Path(args.output)
    names = [out_base if args.num_samples == 1 else out_base.with_name(f"{out_base.stem}_{i + 1}{out_base.suffix}")
             for i in range(args.num_samples)]
    bs = args.batch_size or args.num_samples
    job = JobStatus('generate', args.num_samples * args.max_new_tokens, f'{out_base.name} on {device}')
    for k, first in enumerate(range(0, args.num_samples, bs)):
        rows = names[first:first + bs]
        seed = args.seed + k
        progress = lambda n, first=first, rows=rows: job.update(
            first * args.max_new_tokens + n * len(rows),
            f'{rows[0].name}..{rows[-1].name} (batch of {len(rows)}) on {device}')
        generate_batch(model, config, args, pieces, seed, rows, device, progress)
    job.finish(f'{args.num_samples} sample(s) → {out_base.parent}')


def prompt_rows(config, args, pieces, seed, n_rows, device):
    """(pitch, velocity, duration, delta_time) each (n_rows, T), one prompt per row, and a label per row.
    Models with BOS/EOS get BOS in front of a prompt that starts at the beginning of its piece, and BOS as the seed."""
    streams, labels, programs = _prompt_rows(config, args, pieces, seed, n_rows, device)
    if programs is None:
        programs = torch.zeros_like(streams[0])
    if config.pitch_size > BOS_PITCH and (args.prompt_start == 0 or streams[0].size(1) == 0):
        bos = [torch.full((n_rows, 1), BOS_PITCH if j == 0 else 0, dtype=torch.long, device=device) for j in range(4)]
        streams = tuple(torch.cat([b, s], dim=1) for b, s in zip(bos, streams))
        programs = torch.cat([torch.zeros_like(bos[0]), programs], dim=1)
    return streams, labels, programs


def instrument_names(programs):
    names = sorted({int(p) for p in programs})
    return ', '.join('Drums' if p == DRUM_PROGRAM else pretty_midi.program_to_instrument_name(p) for p in names)


def _prompt_rows(config, args, pieces, seed, n_rows, device):
    a, b = args.prompt_start, args.prompt_start + args.prompt_notes
    if pieces:
        rng = random.Random(seed)
        rows, progs, labels = [], [], []
        for _ in range(n_rows):
            while True:
                idx = args.val_index if args.val_index is not None else rng.randrange(len(pieces))
                label, load = pieces[idx]
                toks = load()
                prog = None
                if isinstance(toks, tuple):
                    toks, prog = toks
                enough = len(toks) >= b and (prog is None or len(set(prog[a:b].tolist())) >= args.min_instruments)
                if enough or args.val_index is not None:
                    break  # random pick: retry pieces shorter than the prompt (or with too few instruments)
            rows.append(torch.from_numpy(toks[a:b]).long())
            progs.append(None if prog is None else torch.from_numpy(prog[a:b]).long())
            labels.append(f"notes {a}-{a + len(rows[-1])} of val piece #{idx}: {label}"
                          + (f" [{instrument_names(prog[a:b])}]" if prog is not None else ""))
        n = min(len(r) for r in rows)  # rows must share a length; only a fixed short --val-index can differ
        toks = torch.stack([r[:n] for r in rows]).to(device)
        programs = torch.stack([p[:n] for p in progs]).to(device) if progs[0] is not None else None
        return tuple(toks[:, :, j] for j in range(4)), labels, programs
    if args.prompt:
        tokens = tokenize_midi(args.prompt, velocity_bins=config.velocity_size,
                               max_duration_bin=config.duration_size - 1,
                               max_delta_bin=config.delta_time_size - 1)
        streams = tuple(t[a:b].unsqueeze(0).expand(n_rows, -1).contiguous().to(device) for t in tokens)
        return streams, [f"notes {a}-{a + streams[0].size(1)} of {args.prompt}"] * n_rows, None
    if config.pitch_size > BOS_PITCH:
        return (tuple(torch.zeros((n_rows, 0), dtype=torch.long, device=device) for _ in range(4)), [None] * n_rows,
                None)
    # single-token seed (silent note) for models without BOS
    z = lambda fill: torch.full((n_rows, 1), fill, dtype=torch.long, device=device)
    return (z(0), z(0), z(10), z(0)), [None] * n_rows, None


def generate_batch(model, config, args, pieces, seed, outs, device, progress=None):
    torch.manual_seed(seed)
    (pitch, velocity, duration, delta_time), labels, prompt_programs = prompt_rows(config, args, pieces, seed,
                                                                                  len(outs), device)
    if args.sample_instruments and not config.n_programs:
        raise SystemExit("--sample-instruments needs a multi-instrument checkpoint (n_programs > 0)")
    allowed = None
    if args.instruments == 'prompt':
        if len({tuple(sorted(set(r.tolist()))) for r in prompt_programs}) > 1:
            raise SystemExit("--instruments prompt needs rows with the same instrument set: use --batch-size 1")
        allowed = sorted(set(prompt_programs[0].tolist()))
    elif args.instruments:
        allowed = [int(x) for x in args.instruments.split(',')]
    for out, label in zip(outs, labels):
        if label:
            print(f"Prompt for {out.name}: {label}")
    has_prompt = bool(pieces or args.prompt)
    style = None
    if config.n_styles:
        names = load_styles()
        name = args.style or 'none'
        if name not in names:
            raise SystemExit(f"unknown style {name!r}; choose from: {', '.join(names)}")
        style = names.index(name)
        print(f"Style: {name} (id {style})")
    elif args.style:
        raise SystemExit("this checkpoint was trained without styles")
    target_nps = None
    if args.density == "prompt":
        if not has_prompt or pitch.size(1) < 2:
            raise SystemExit("--density prompt needs a prompt")
        secs = delta_time[:, 1:].sum(dim=1).float().clamp(min=1) * 0.02  # (a BOS row adds 0 time)
        target_nps = (pitch.size(1) / secs).tolist()
        print("Density target (from the prompts): " + ", ".join(f"{t:.1f}" for t in target_nps) + " notes/s")
    elif args.density:
        target_nps = float(args.density)
    n_prompt = 0 if args.keep_prompt and has_prompt else pitch.size(1)

    cond = None
    if config.cond_inst or config.n_density:
        from model import window_conditions, density_level
        from_prompt = window_conditions(pitch, delta_time, prompt_programs if config.n_programs else None,
                                        config.n_programs or 1, config.n_density or 16)
        inst = torch.zeros(len(outs), config.n_programs or 1, device=device)
        how = args.cond_instruments or ('prompt' if prompt_programs is not None and has_prompt else 'none')
        if how == 'prompt':
            inst = from_prompt[0]
        elif how != 'none':
            inst[:, [int(x) for x in how.split(',')]] = 1.0
        density = torch.zeros(len(outs), dtype=torch.long, device=device)
        dens = args.cond_density or 'none'
        if dens == 'prompt':
            density = from_prompt[1]
        elif dens != 'none':
            density = density_level(float(dens), config.n_density).expand(len(outs)).to(device)
        cond = (inst, density)
        print(f"Conditioning: instruments {how} ({instrument_names(inst[0].nonzero().flatten().tolist()) or 'none'}), "
              f"density {dens} (levels {density.tolist()})")
    print(f"Generating {args.max_new_tokens} notes x {len(outs)} rows on {device} (seed {seed})...")
    t0 = time.time()
    with torch.no_grad():
        pitch, velocity, duration, delta_time, programs = model.generate(
            pitch, velocity, duration, delta_time,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            anchor=args.anchor,
            slide=args.slide,
            progress=progress,
            dt_bias=args.dt_bias,
            target_nps=target_nps,
            cuda_graph=not args.no_cuda_graph,
            style=style,
            min_new=args.min_notes,
            program=None if args.sample_instruments else 0,
            programs=prompt_programs if config.n_programs else None,
            allowed_programs=allowed,
            return_programs=True,
            program_temperature=args.instrument_temperature,
            cond=cond,
        )
    streams = [t.cpu() for t in (pitch, velocity, duration, delta_time, programs)]
    print(f"  {len(outs) * args.max_new_tokens / (time.time() - t0):.0f} notes/s")

    for r, out in enumerate(outs):
        # drop the seed / prompt unless --keep-prompt; cut at the first EOS; drop BOS/EOS rows
        rows = [s[r, n_prompt:].tolist() for s in streams]
        new = streams[0][r, streams[0].size(1) - args.max_new_tokens:].tolist()
        if EOS_PITCH in new:
            cut = len(rows[0]) - (len(new) - new.index(EOS_PITCH))
            rows = [x[:cut] for x in rows]
            print(f"  {out.name}: ended with EOS after {new.index(EOS_PITCH)} new notes")
        keep = [i for i, p in enumerate(rows[0]) if p < BOS_PITCH]
        rows = [[x[i] for i in keep] for x in rows]
        midi = tokens_to_midi(*rows[:4], programs=rows[4], max_polyphony=args.max_polyphony, max_delta=args.max_delta)
        out.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(out))
        print(f"Saved MIDI → {out}" + (f"  [{instrument_names(rows[4])}]" if config.n_programs else ""))
        if not args.no_mp3:
            _render_mp3(out, args.soundfont)


def _render_mp3(midi_path: Path, soundfont: str) -> None:
    if not shutil.which("fluidsynth"):
        print("fluidsynth not found — skipping MP3 render (sudo apt install fluidsynth)")
        return
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found — skipping MP3 render (sudo apt install ffmpeg)")
        return
    if not Path(soundfont).exists():
        print(f"Soundfont not found: {soundfont}")
        print("Install with: sudo apt install fluid-soundfont-gm")
        return

    raw_path = midi_path.with_suffix(".raw")
    mp3_path = midi_path.with_suffix(".mp3")
    # bounded render: fluidsynth's fast render once kept going on a 16 s multi-instrument MIDI until it had written
    # 14.5 GB (2026-10-03). Raw 16-bit stereo PCM has no header to break, so the watchdog can stop it at the
    # MIDI's length + 5 s of release tail and ffmpeg encodes exactly that much.
    seconds = pretty_midi.PrettyMIDI(str(midi_path)).get_end_time() + 5.0
    rate = 44100
    limit = int(seconds * rate * 4)  # 2 channels x 2 bytes

    print("Rendering MP3 with fluidsynth...")
    proc = subprocess.Popen(
        ["fluidsynth", "-ni", "-r", str(rate), "-O", "s16", "-T", "raw", "-F", str(raw_path), soundfont,
         str(midi_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if raw_path.exists() and raw_path.stat().st_size >= limit:
            proc.kill()
            break
        time.sleep(0.05)
    proc.wait()
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(rate), "-ac", "2", "-i", str(raw_path), "-t", f"{seconds:.2f}",
         "-q:a", "2", str(mp3_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    raw_path.unlink()
    print(f"Saved MP3  → {mp3_path}")


if __name__ == "__main__":
    main()
