import argparse
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch

from model import GPT, MusicConfig
from data_loader import tokenize_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate piano MIDI from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt",
                        help="checkpoint file, or a run directory (uses model_bf16.pt, else best.pt, else ckpt.pt)")
    parser.add_argument("--output", type=str, default="generated.mid",
                        help="output MIDI; with --num-samples > 1 a _<i> suffix is added")
    parser.add_argument("--num-samples", type=int, default=1, help="samples to generate, seeds seed, seed+1, ...")
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
    parser.add_argument("--val-prompt", choices=["combined", "aria"], default=None,
                        help="take the prompt from a validation piece (never trained on): 'combined' = MAESTRO + "
                             "GiantMIDI val MIDI files, 'aria' = Aria val token store. Random piece per sample "
                             "unless --val-index is given")
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
    parser.add_argument("--slide", type=int, default=None,
                        help="notes dropped per KV-cache rebuild past the window (default block_size // 4)")
    parser.add_argument("--max-polyphony", type=int, default=None,
                        help="Post-processing: max simultaneous notes per time slot (default: off, raw model output)")
    parser.add_argument("--max-delta", type=float, default=None,
                        help="Post-processing: cap on seconds per time step (default: off, raw model output)")
    return parser.parse_args()


def tokens_to_midi(pitches, velocities, durations, delta_times,
                   velocity_bins=32, time_resolution=0.02,
                   max_polyphony=None, max_delta=None):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    current_time = 0.0
    pending = []  # notes at the current time slot, flushed when time advances

    def flush(pending):
        pending.sort(key=lambda n: n.velocity, reverse=True)
        for note in pending[:max_polyphony] if max_polyphony else pending:
            piano.notes.append(note)

    for p, v, d, dt in zip(pitches, velocities, durations, delta_times):
        delta = dt * time_resolution
        if max_delta is not None:
            delta = min(delta, max_delta)  # cap runaway gaps

        if delta > 0:
            flush(pending)
            pending = []
            current_time += delta

        velocity = min(127, int(v * 128 / velocity_bins) + 2)
        duration = max(0.08, d * time_resolution)
        pending.append(pretty_midi.Note(
            velocity=velocity,
            pitch=int(p),
            start=current_time,
            end=current_time + duration,
        ))

    flush(pending)
    midi.instruments.append(piano)
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
    """Validation pieces as a list of (label, loader); loader() returns an (n, 4) int array of tokens."""
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
    model.eval().to(device)

    print(f"Checkpoint: {resolve_checkpoint(args.checkpoint)} (iter {ckpt.get('iter_num', '?')}), device {device}")
    out_base = Path(args.output)
    for i in range(args.num_samples):
        seed = args.seed + i
        torch.manual_seed(seed)
        out = out_base if args.num_samples == 1 else out_base.with_name(f"{out_base.stem}_{i + 1}{out_base.suffix}")
        generate_one(model, config, args, pieces, seed, out, device)


def generate_one(model, config, args, pieces, seed, out, device):
    a, b = args.prompt_start, args.prompt_start + args.prompt_notes
    if pieces:
        rng = random.Random(seed)
        while True:
            idx = args.val_index if args.val_index is not None else rng.randrange(len(pieces))
            label, load = pieces[idx]
            toks = load()
            if len(toks) >= b or args.val_index is not None:
                break  # random pick: retry pieces shorter than the prompt
        toks = torch.from_numpy(toks[a:b]).long().to(device)
        pitch, velocity, duration, delta_time = (toks[:, j].unsqueeze(0) for j in range(4))
        print(f"Prompt: notes {a}-{a + pitch.size(1)} of val piece #{idx}: {label}")
    elif args.prompt:
        tokens = tokenize_midi(args.prompt, velocity_bins=config.velocity_size,
                               max_duration_bin=config.duration_size - 1,
                               max_delta_bin=config.delta_time_size - 1)
        pitch, velocity, duration, delta_time = (
            t[a:b].unsqueeze(0).to(device) for t in tokens
        )
        print(f"Prompt: notes {a}-{a + pitch.size(1)} of {args.prompt}")
    else:
        # single-token seed (silent note)
        pitch      = torch.zeros((1, 1), dtype=torch.long, device=device)
        velocity   = torch.zeros((1, 1), dtype=torch.long, device=device)
        duration   = torch.full((1, 1), 10, dtype=torch.long, device=device)
        delta_time = torch.zeros((1, 1), dtype=torch.long, device=device)
    has_prompt = bool(pieces or args.prompt)
    n_prompt = 0 if args.keep_prompt and has_prompt else pitch.size(1)

    print(f"Generating {args.max_new_tokens} tokens on {device} (seed {seed})...")
    with torch.no_grad():
        pitch, velocity, duration, delta_time = model.generate(
            pitch, velocity, duration, delta_time,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            anchor=args.anchor,
            slide=args.slide,
        )

    # drop the seed / prompt unless --keep-prompt
    midi = tokens_to_midi(
        pitch[0, n_prompt:].cpu().tolist(),
        velocity[0, n_prompt:].cpu().tolist(),
        duration[0, n_prompt:].cpu().tolist(),
        delta_time[0, n_prompt:].cpu().tolist(),
        max_polyphony=args.max_polyphony,
        max_delta=args.max_delta,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out))
    print(f"Saved MIDI → {out}")

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

    wav_path = midi_path.with_suffix(".wav")
    mp3_path = midi_path.with_suffix(".mp3")

    print("Rendering MP3 with fluidsynth...")
    subprocess.run(
        ["fluidsynth", "-a", "file", "-F", str(wav_path), soundfont, str(midi_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wav_path.unlink()
    print(f"Saved MP3  → {mp3_path}")


if __name__ == "__main__":
    main()
