"""Generate MIDI from a checkpoint: the command line over musicar/generation/piece.py (options: --help)."""

from pathlib import Path

import torch

from musicar.data.midi_io import render_mp3
from musicar.generation.piece import (PRESETS, GenerationError, Generator, build_parser, instrument_names,  # noqa: F401
                                      preset_defaults, score_line, val_pieces)
from musicar.jobstatus import JobStatus


def parse_args(argv=None):
    parser = build_parser()
    presets, _ = parser.parse_known_args(argv)
    try:
        parser.set_defaults(**preset_defaults(parser, presets.config))
    except GenerationError as e:
        parser.error(str(e))
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        parser.error("--checkpoint is required (on the command line or in a --config preset)")
    return args


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.list_val:
        if not args.val_prompt:
            raise SystemExit("--list-val needs --val-prompt")
        for i, (label, _) in enumerate(val_pieces(args.val_prompt, args.val_genre)):
            print(f"{i:5d}  {label}")
        return

    gen = Generator(args.checkpoint, args.device, args.dtype)
    out_base = Path(args.output)
    names = [out_base if args.num_samples == 1 else out_base.with_name(f"{out_base.stem}_{i + 1}{out_base.suffix}")
             for i in range(args.num_samples)]
    job = JobStatus('generate', args.num_samples * args.best_of * args.max_new_tokens,
                    f'{out_base.name} on {gen.device}')
    try:
        pieces = gen.run(args, progress=lambda done, total, detail: job.update(done, detail))
    except GenerationError as e:
        raise SystemExit(str(e))
    for out, piece in zip(names, pieces):
        if piece.label:
            print(f"Prompt for {out.name}: {piece.label}")
        save(out, piece, args, gen.config)
        if piece.score:
            print(f"  {out.name}: {score_line(piece.score)}")
    job.finish(f'{args.num_samples} sample(s) → {out_base.parent}')


def save(out, piece, args, config):
    """The piece as a MIDI (and MP3)."""
    if piece.eos:
        print(f"  {out.name}: ended with EOS after {piece.new_notes} new notes, at {piece.seconds:.1f} s"
              + (" (--length-stop" + (f", faded over {args.fade_seconds:g} s)" if args.fade_seconds else ")")
                 if piece.stopped else ""))
    midi = piece.midi(args.max_polyphony, args.max_delta)
    out.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out))
    print(f"Saved MIDI → {out}" + (f"  [{instrument_names(piece.tokens[4])}]" if config.n_programs else ""))
    if not args.no_mp3:
        render_mp3(out, args.soundfont)


if __name__ == "__main__":
    main()
