"""
musicar-generate (installed with the package; python generate.py in the repo is the same): MIDI from a checkpoint or a
released model, the command line over musicar/generation/piece.py (options: --help).

    musicar-generate --model musicar-S --max-new-tokens 1000 --output x.mid
    musicar-generate --model musicar-L-caption --config f1 --text "a sad slow piano ballad" --length-seconds 60

--model (= --checkpoint) takes a checkpoint file or run directory, a released model (musicar-L, musicar-S,
musicar-S-piano, musicar-L-caption) or a Hub id (user/repo). Next to each MIDI an MP3 is rendered when fluidsynth,
ffmpeg and the --soundfont are installed (skipped with a note when one is missing); --no-mp3 skips it.
"""

import sys
from pathlib import Path

import torch

from musicar.data.midi_io import render_mp3
from musicar.generation.piece import (GenerationError, build_parser, instrument_names, preset_defaults, score_line,
                                      val_pieces)
from musicar.hub import MODELS, OWNER, from_pretrained
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
        parser.error("--model / --checkpoint is required (a checkpoint, a released model such as musicar-S, or a "
                     "--config preset with a checkpoint)")
    args.checkpoint_from_preset = presets.checkpoint is None  # checked in main(), when the model is loaded
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # e.g. a Windows console code page without '→'
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.list_val:
        if not args.val_prompt:
            raise SystemExit("--list-val needs --val-prompt")
        try:
            pieces = val_pieces(args.val_prompt, args.val_genre)
        except GenerationError as e:
            raise SystemExit(str(e))
        for i, (label, _) in enumerate(pieces):
            print(f"{i:5d}  {label}")
        return

    local = args.checkpoint not in MODELS and not args.checkpoint.startswith(OWNER + '/')
    if getattr(args, 'checkpoint_from_preset', False) and local and not Path(args.checkpoint).exists():
        last = Path(args.config[-1]).stem if args.config else ''
        released = dict(piano='musicar-S-piano').get(last, 'musicar-L-caption')
        raise SystemExit(f"--config {' '.join(args.config)} names the checkpoint {args.checkpoint}, which is not here "
                         f"(it lives in the training repo): pass a model, e.g. --model {released}")
    try:
        gen = from_pretrained(args.checkpoint, args.device, args.dtype)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
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
