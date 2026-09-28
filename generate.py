import argparse
import shutil
import subprocess
from pathlib import Path

import pretty_midi
import torch

from model import GPT, MusicConfig
from data_loader import tokenize_midi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate piano MIDI from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt.pt")
    parser.add_argument("--output", type=str, default="generated.mid")
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
    parser.add_argument("--prompt-notes", type=int, default=64)
    parser.add_argument("--keep-prompt", action="store_true",
                        help="Include the prompt notes in the output MIDI")
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    ma = ckpt.get("model_args", {})
    config = MusicConfig(
        block_size=ma.get("block_size", 512),
        pitch_size=ma.get("pitch_size", 128),
        velocity_size=ma.get("velocity_size", 32),
        duration_size=ma.get("duration_size", 512),
        delta_time_size=ma.get("delta_time_size", 512),
        n_layer=ma.get("n_layer", 6),
        n_head=ma.get("n_head", 8),
        n_embd=ma.get("n_embd", 256),
        dropout=0.0,
        bias=ma.get("bias", False),
        # checkpoints from before cascade heads have no such key and use independent heads
        cascade_heads=ma.get("cascade_heads", False),
    )

    model = GPT(config)
    state_dict = ckpt["model"]
    # strip torch.compile prefix if present
    for k in list(state_dict):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    if args.prompt:
        tokens = tokenize_midi(args.prompt, velocity_bins=config.velocity_size,
                               max_duration_bin=config.duration_size - 1,
                               max_delta_bin=config.delta_time_size - 1)
        pitch, velocity, duration, delta_time = (
            t[:args.prompt_notes].unsqueeze(0).to(device) for t in tokens
        )
        print(f"Prompt: first {pitch.size(1)} notes of {args.prompt}")
    else:
        # single-token seed (silent note)
        pitch      = torch.zeros((1, 1), dtype=torch.long, device=device)
        velocity   = torch.zeros((1, 1), dtype=torch.long, device=device)
        duration   = torch.full((1, 1), 10, dtype=torch.long, device=device)
        delta_time = torch.zeros((1, 1), dtype=torch.long, device=device)
    n_prompt = 0 if args.keep_prompt and args.prompt else pitch.size(1)

    print(f"Generating {args.max_new_tokens} tokens on {device}...")
    pitch, velocity, duration, delta_time = model.generate(
        pitch, velocity, duration, delta_time,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
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

    out = Path(args.output)
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
