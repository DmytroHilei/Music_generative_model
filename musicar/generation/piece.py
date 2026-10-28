"""
One generation request -> finished pieces: everything generate.py does between its options and the files it writes
(prompt rows, caption input, retrieval, instrument masks, guidance, best-of ranking, EOS cut, length-stop fade-out),
as a library, so the command line and the app engine run the same code.

    gen = Generator('checkpoints/f1')                    # loads the model once
    args = settings(['s50'], text='a sad slow piano ballad', max_new_tokens=600)
    pieces = gen.run(args, on_notes=lambda c, notes: print(c, len(notes)))   # best first
    pieces[0].midi().write('x.mid')

Options are generate.py's (build_parser) in an argparse.Namespace: settings() fills one from the parser defaults,
presets in config/generate/ and keyword overrides. Options the checkpoint can't follow raise GenerationError.
"""

import argparse
import functools
import random
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch

from musicar.model.checkpoint import load_model, resolve_checkpoint
from musicar.data.loader import (BOS_PITCH, DRUM_PROGRAM, EOS_PITCH, density_level, load_styles, tokenize_midi,
                                 tokenize_tracks, window_conditions)
from musicar.data.midi_io import note_events, tokens_to_midi
from musicar.generation.caption_match import allowed_programs, describe, match, note_attributes, parse_caption
from musicar.generation.repetition import copy_share
from musicar.generation.sampling import generate, with_bos
from musicar.text.captions import hold_caption


COPY_MIN, COPY_FREE = 16, 0.25  # best-of: copies of >= 16 chords; real GigaMIDI band songs: 0-74% of notes, median ~13%
PRESETS = Path(__file__).resolve().parents[2] / "config" / "generate"
NOTES_EVERY = 16  # on_notes: notes per call
SECTION_TOKENS = 48  # encoder tokens per section caption (train.py section_max_length)


class GenerationError(ValueError):
    """Options this checkpoint or these prompts can't follow (generate.py exits with the message)."""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate MIDI (piano or multi-instrument) from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", action="append", default=[],
                        help="generation preset (repeatable): a TOML file of options, or its name in config/generate/ "
                             "(e.g. --config caption); options given on the command line override it")
    parser.add_argument("--checkpoint", type=str, default=None,
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
    parser.add_argument("--top-p", type=float, default=None,
                        help="nucleus sampling: keep the smallest set of values reaching this probability (e.g. 0.9; "
                             "the 2026-10-04 settings search found 0.9 helps in every style)")
    parser.add_argument("--min-p", type=float, default=None,
                        help="min-p truncation: drop values below this share of the most likely one (e.g. 0.035-0.1; "
                             "after --top-k, before --top-p)")
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
    parser.add_argument("--prompt-notes", type=int, default=None,
                        help="prompt length (default 64; 128 with --retrieve)")
    parser.add_argument("--keep-prompt", action="store_true",
                        help="Include the prompt notes in the output MIDI")
    parser.add_argument("--anchor", type=int, default=0,
                        help="optional, off by default: pin the first N notes (e.g. the prompt's theme) at the start "
                             "of the context once generation runs past the model's context window")
    parser.add_argument("--style", type=str, default=None,
                        help="models trained with styles: a name from data/styles.json (genre or artist, e.g. pop, "
                             "Скрябін) or 'none' (default: none = unconditional)")
    parser.add_argument("--text", type=str, default=None,
                        help="models trained with text (text_dim): a prompt in English or Ukrainian, e.g. "
                             "'a sad slow piano ballad' (default: none = unconditional)")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="classifier-free guidance strength over --text / --style (1 = off; e.g. 1.5-3)")
    parser.add_argument("--cfg-notes", type=int, default=0,
                        help="guidance schedule: --cfg-scale for this many new notes, then --cfg-end (0 = constant)")
    parser.add_argument("--cfg-end", type=float, default=1.0, help="the scale after --cfg-notes notes (1 = off)")
    parser.add_argument("--cfg-linear", action="store_true",
                        help="a straight line from --cfg-scale to --cfg-end over --cfg-notes notes instead of a step")
    parser.add_argument("--retrieve", type=str, default=None,
                        help="caption retrieval (musicar/generation/retrieval.py): each row's prompt is the opening "
                             "(--prompt-notes) of a file of this captioned store (a name under data/cache, e.g. "
                             "gigamidi_replay_train, or a store directory) whose captions best match --text. Works "
                             "with any checkpoint; models without text use the caption only to pick the openings. "
                             "The index is built once (LaBSE) and cached in <labels-dir>/<store>.retrieval.npz")
    parser.add_argument("--retrieve-top", type=int, default=32,
                        help="--retrieve: rows take distinct openings from this many best matches")
    parser.add_argument("--retrieve-files", type=int, default=60000,
                        help="--retrieve: files indexed at most (a fixed random subset of a bigger store)")
    parser.add_argument("--retrieve-min-notes", type=int, default=512,
                        help="--retrieve: only files with at least this many notes are indexed")
    parser.add_argument("--labels-dir", type=str, default="data/labels", help="--retrieve: the store's label tables")
    parser.add_argument("--retrieve-strict", type=str, default="",
                        help="--retrieve: caption_match categories an opening must fully meet when --text asks for "
                             "them, e.g. 'genre,mood' (its labels name the asked genre / mood); default: soft ranking")
    parser.add_argument("--best-of", type=int, default=1,
                        help="generate this many candidates per sample and keep the --num-samples best by caption "
                             "match (what --text asks for that the notes show: instruments, drums, solo, key, pace, "
                             "dynamics, register; musicar/generation/caption_match.py) minus their runaway share "
                             "(10-s windows over 40 notes/s) and their verbatim repetition (share of the new pitched "
                             f"notes inside copies of >= {COPY_MIN} chords, over {COPY_FREE * 100:.0f}%%)")
    parser.add_argument("--loose-instruments", action="store_true",
                        help="don't turn --text's instrument limits into a hard mask. By default 'only' / 'just' / "
                             "'лише' / 'тільки' with named instruments, or 'solo' with one ('solo piano', 'фортепіано "
                             "соло'), allow just those instruments, and 'without X' / 'без X' bans X: the instrument "
                             "head can't sample anything else (multi-instrument checkpoints sample instruments then), "
                             "and --retrieve takes openings only from files within them")
    parser.add_argument("--dry-multiplier", type=float, default=0.0,
                        help="repetition penalty (musicar/generation/repetition.py, the DRY sampler on chords): when "
                             "the last >= --dry-allowed chords (the pitched notes of one onset, as a set, with their "
                             "timing; drums left out) repeat an earlier passage, the pitches that would continue it "
                             "lose multiplier x base^(match - allowed) logits. 0 = off; e.g. 1")
    parser.add_argument("--dry-allowed", type=int, default=16,
                        help="chords a passage may repeat verbatim before the penalty starts")
    parser.add_argument("--dry-base", type=float, default=1.15, help="growth of the penalty per further copied chord")
    parser.add_argument("--dry-key", choices=["notes", "pitch_class"], default="notes",
                        help="what makes chords equal: their instruments and pitches, or only their pitch classes "
                             "(a repeat with another voicing or instrument counts too)")
    parser.add_argument("--eos-ramp", type=float, default=0.0,
                        help="with --length-seconds: + this x (seconds past the length) on the EOS logit, so a row "
                             "that runs over ends soon after (any model with BOS/EOS)")
    parser.add_argument("--length-stop", type=float, default=0.0,
                        help="with --length-seconds: rows still going at this x the length end there (hard limit, "
                             "e.g. 1.2; any model with BOS/EOS)")
    parser.add_argument("--eos-floor", type=float, default=0.0,
                        help="with --length-seconds: no end before this x the length (e.g. 0.85), so a row can't end "
                             "far too early (any model with BOS/EOS)")
    parser.add_argument("--fade-seconds", type=float, default=3.0,
                        help="rows ended by --length-stop: velocities faded out over their last this many seconds")
    parser.add_argument("--length-seconds", type=float, default=None,
                        help="models with the length input (n_remain): the piece's length in seconds from the prompt's "
                             "first note, so it ends there with EOS (default: none)")
    parser.add_argument("--length-after-prompt", action="store_true",
                        help="with a prompt left out of the result (no --keep-prompt): --length-seconds counts from "
                             "the prompt's end, where the result starts (per row: the length + that row's prompt "
                             "seconds)")
    parser.add_argument("--min-notes", type=int, default=0,
                        help="models with BOS/EOS: don't allow the end token before this many new notes")
    parser.add_argument("--density", type=str, default=None,
                        help="optional: steer notes per second without retraining. A number (e.g. 5) or 'prompt' = "
                             "match the prompt's own density. Only the delta_time head is biased")
    parser.add_argument("--dt-bias", type=float, default=0.0,
                        help="fixed bias toward longer gaps (> 0 = calmer); with --density it's the starting value")
    parser.add_argument("--block-size", type=int, default=None,
                        help="the generation window in notes (default: the checkpoint's block_size, which is its last "
                             "training phase's window: 512 for f1_cap); RoPE models take any value")
    parser.add_argument("--keep-style", action="store_true",
                        help="section-caption models: once the first section (section_len notes from the start) is "
                             "written, its own description (pace, loudness, register, texture, drums, up to 3 "
                             "instruments) becomes the section caption of every later section, so the piece keeps "
                             "the opening's character instead of drifting")
    parser.add_argument("--lock-instruments", action="store_true",
                        help="with --sample-instruments: after the first section, only its instruments (>= 4 notes "
                             "and 2%% of them) and those the --text names")
    parser.add_argument("--slide", type=int, default=None,
                        help="notes dropped per KV-cache rebuild past the window (default block_size // 4)")
    parser.add_argument("--max-polyphony", type=int, default=None,
                        help="Post-processing: max simultaneous notes per time slot (default: off, raw model output)")
    parser.add_argument("--max-delta", type=float, default=None,
                        help="Post-processing: cap on seconds per time step (default: off, raw model output)")
    return parser


def option_name(parser, key, value, where):
    """The parser's dest for an option given as `key` (dashes or underscores) with `value`, checked."""
    action = {a.dest: a for a in parser._actions}.get(key.replace("-", "_"))
    if action is None or action.dest in ("help", "config"):
        raise GenerationError(f"{where}: {key} is not a generate.py option")
    if action.choices and value not in action.choices:
        raise GenerationError(f"{where}: {key} = {value!r} is not one of {', '.join(map(str, action.choices))}")
    return action.dest


def preset_defaults(parser, names):
    """Option defaults from generation presets: TOML files of option names (dashes or underscores) and values, as a
    path or a name in config/generate/; later presets override earlier ones."""
    out = {}
    for name in names:
        path = Path(name) if Path(name).is_file() else PRESETS / f"{name.removesuffix('.toml')}.toml"
        if not path.is_file():
            raise GenerationError(f"--config {name}: no such file, nor {path} (presets: "
                                  f"{', '.join(sorted(f.stem for f in PRESETS.glob('*.toml')))})")
        for key, value in tomllib.loads(path.read_text()).items():
            out[option_name(parser, key, value, path)] = value
    return out


def settings(presets=(), **options):
    """generate.py's options as a Namespace: the parser defaults, then the presets, then `options` (names with
    underscores or dashes)."""
    parser = build_parser()
    values = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    values.update(preset_defaults(parser, presets), config=list(presets))
    for key, value in options.items():
        values[option_name(parser, key, value, "settings")] = value
    return argparse.Namespace(**values)


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
        return [(p.stem, lambda p=p: tokenize_midi(p).astype(np.int64)) for p in paths]
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


@dataclass
class Piece:
    """One finished candidate. tokens: [pitch, velocity, duration, delta_time, program] lists as they go into its
    MIDI: the prompt left out unless --keep-prompt, cut at the first EOS, BOS / EOS removed, velocities faded out if
    --length-stop ended it."""
    index: int  # candidate number in generation order (on_notes' candidate)
    seed: int  # the seed of its batch (row `row` of that batch)
    row: int
    tokens: list
    label: str | None  # where its prompt came from
    score: dict | None  # best-of / caption score (score_line), None without --text and --best-of
    new_notes: int  # notes generated before its EOS
    eos: bool  # ended with EOS (else it ran to --max-new-tokens)
    seconds: float  # from the first prompt note to the end, as --length-seconds counts
    stopped: bool  # ended by --length-stop

    def midi(self, max_polyphony=None, max_delta=None):
        return tokens_to_midi(*self.tokens[:4], programs=self.tokens[4], max_polyphony=max_polyphony,
                              max_delta=max_delta)

    def notes(self, max_delta=None):
        """[onset s, program, pitch, duration s, velocity] of every note, as in its MIDI (before --max-polyphony)."""
        return [list(e) for e in note_events(*self.tokens[:4], programs=self.tokens[4], max_delta=max_delta)]


class NoteStream:
    """sampling.generate's on_notes for one batch -> callback(candidate, notes) per row with new notes, as
    [onset s, program, pitch, duration s, velocity] lists timed as in the row's MIDI: from its first kept note
    (the prompt's with --keep-prompt), BOS left out, nothing from the first EOS on. The length-stop fade-out and
    --max-polyphony are applied only to the finished Piece. Returns True once every row has ended, which stops the
    generation early (sampling.generate)."""

    def __init__(self, callback, first, n_prompt, max_delta=None):
        self.callback, self.first, self.n_prompt, self.max_delta = callback, first, n_prompt, max_delta
        self.onset, self.ended = {}, set()

    def __call__(self, a, notes):
        skip = max(0, self.n_prompt - a)
        for r in range(notes[0].size(0)):
            if r in self.ended:
                continue
            rows = [t[r, skip:].tolist() for t in notes]
            if EOS_PITCH in rows[0]:
                rows = [x[:rows[0].index(EOS_PITCH)] for x in rows]
                self.ended.add(r)
            keep = [i for i, p in enumerate(rows[0]) if p < BOS_PITCH]
            rows = [[x[i] for i in keep] for x in rows]
            out = [list(e) for e in note_events(*rows[:4], programs=rows[4], max_delta=self.max_delta,
                                                start=self.onset.get(r, 0.0))]
            if out:
                self.onset[r] = out[-1][0]
                self.callback(self.first + r, out)
        return len(self.ended) == notes[0].size(0)


class Generator:
    """A checkpoint loaded once; run() is one generate.py call on it (the options' checkpoint, device and dtype are
    ignored: the constructor's count)."""

    def __init__(self, checkpoint, device="auto", dtype="auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(
            dtype, torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.model, self.iter = load_model(checkpoint, self.device, self.dtype)
        self.config = self.model.config
        self.block_size = self.config.block_size  # the checkpoint's; --block-size sets it per run
        self.checkpoint = resolve_checkpoint(checkpoint)
        print(f"Checkpoint: {self.checkpoint} (iter {self.iter}), "
              f"device {self.device}, {str(self.dtype).replace('torch.', '')}")

    def run(self, args, progress=None, on_notes=None):
        """All --num-samples x --best-of candidates as Pieces, best first (generation order without --text and
        --best-of); generate.py saves the first --num-samples.
        progress(done, total, detail): notes generated so far over all batches (rows x notes), after every note.
        on_notes(candidate, notes): every NOTES_EVERY notes, each candidate's notes since its last call (NoteStream)."""
        args = argparse.Namespace(**vars(args))
        config, device = self.config, self.device
        if args.prompt_notes is None:
            args.prompt_notes = 128 if args.retrieve else 64
        if args.best_of < 1:
            raise GenerationError("--best-of must be at least 1")
        block = args.block_size or self.block_size
        if block > self.block_size and config.pos_emb != 'rope':
            raise GenerationError(f"--block-size {block}: learned positions stop at {self.block_size}")
        config.block_size = block
        if args.retrieve and (args.prompt or args.val_prompt):
            raise GenerationError("--retrieve takes the prompts from its store: drop --prompt / --val-prompt")
        if args.retrieve and not args.text:
            raise GenerationError("--retrieve needs --text (the caption to find openings for)")
        pieces = val_pieces(args.val_prompt, args.val_genre) if args.val_prompt else None

        wants = parse_caption(args.text) if args.text else None
        if wants is not None and (args.retrieve or args.best_of > 1):
            print(f"Caption checks: {describe(wants)}")
        explicit = [int(x) for x in args.instruments.split(',')] if args.instruments and args.instruments != 'prompt' \
            else None
        args.caption_allowed = None if args.loose_instruments else allowed_programs(wants)
        if args.caption_allowed is not None:
            if explicit is not None:
                args.caption_allowed = sorted(set(args.caption_allowed) & set(explicit))
            if not args.caption_allowed:
                raise GenerationError("--text and --instruments leave no instrument to play")
            allowed = args.caption_allowed
            print("Caption instruments (hard mask, --loose-instruments to drop it): "
                  + (instrument_names(allowed) if len(allowed) < 20 else
                     f"all but {instrument_names(set(range(129)) - set(allowed))}"))
            if not config.n_programs and 0 not in args.caption_allowed:
                raise GenerationError("this checkpoint plays piano only: it can't follow the caption's instruments")
        text = caption_input(config, args, device)
        if args.retrieve:
            pieces = retrieved_pieces(args, device)

        n_cand = args.num_samples * args.best_of
        bs = args.batch_size or n_cand
        total = n_cand * args.max_new_tokens
        cands = []  # (batch result, row)
        for k, first in enumerate(range(0, n_cand, bs)):
            n_rows = min(bs, n_cand - first)
            step = None if progress is None else lambda n, first=first, n_rows=n_rows: progress(
                first * args.max_new_tokens + n * n_rows, total, f'rows {first + 1}..{first + n_rows} of {n_cand} on '
                                                                f'{device}')
            res = generate_batch(self.model, config, args, pieces, args.seed + k, n_rows, device, step, text, first,
                                 on_notes)
            cands += [(res, r) for r in range(n_rows)]
        order, scores = rank(cands, args, wants)
        return [finish_row(*cands[c], c, scores[c] if scores else None, args) for c in order]


def caption_input(config, args, device):
    """--text as the model's input (caption vector / token states), or None: no --text, or a checkpoint without text
    input whose caption only picks the openings (--retrieve) or ranks the candidates (--best-of)."""
    if args.text and not config.text_dim and (args.retrieve or args.best_of > 1 or args.caption_allowed is not None):
        print("Caption: this checkpoint has no text input; --text only "
              + ("picks the openings" if args.retrieve else "ranks the candidates" if args.best_of > 1
                 else "sets the instruments"))
        return None
    return text_vector(config, args.text, device)


def retrieved_pieces(args, device):
    """--retrieve: the openings of the --retrieve-top files best matching --text, as (label, loader) pieces in a
    seeded random order (row i of the run takes piece i, so the rows get distinct files)."""
    store = Path(args.retrieve) if Path(args.retrieve).is_dir() else Path("data/cache") / args.retrieve
    index = caption_index(str(store), args.labels_dir, args.retrieve_files, args.retrieve_min_notes, device)
    strict = tuple(c.strip() for c in args.retrieve_strict.split(',') if c.strip())
    hits = index.search(args.text, k=args.retrieve_top, only=args.caption_allowed, strict=strict)
    if not hits:
        raise GenerationError(f"no file of {store.name} uses only the caption's instruments")
    print(f"Retrieval from {store.name} ({len(index.files):,} files indexed"
          + (f", strict {'/'.join(strict)}: {'met' if hits[0]['strict'] else 'no file meets it, soft'}" if strict
             else "") + "), best matches:")
    rng = random.Random(0)
    for h in hits[:8]:
        print(f"  #{h['file']:<7d} similarity {h['sim']:.2f}, match {h['match']:.2f}: "
              f"{index.caps.caption(h['file'], rng, 'en')}")
    n_cand = args.num_samples * args.best_of
    if n_cand > len(hits):
        print(f"  ({n_cand} rows from {len(hits)} openings: some openings are used more than once)")
    random.Random(args.seed).shuffle(hits)
    end = args.prompt_start + args.prompt_notes
    return [(f"{store.name} file #{h['file']} (similarity {h['sim']:.2f}, match {h['match']:.2f})",
             lambda h=h: index.opening(h['file'], end)) for h in hits]


def new_notes(streams, r, args):
    """(tokens (n, 4), programs (n,)) of row r's generated notes, cut at the first EOS, without BOS / EOS."""
    a = streams[0].size(1) - args.max_new_tokens
    pitch = streams[0][r, a:].tolist()
    n = pitch.index(EOS_PITCH) if EOS_PITCH in pitch else len(pitch)
    tok = np.stack([s[r, a:a + n].numpy() for s in streams[:4]], 1).astype(np.int64)
    keep = tok[:, 0] < BOS_PITCH
    return tok[keep], streams[4][r, a:a + n].numpy().astype(np.int64)[keep]


def rank(cands, args, wants):
    """(candidate order, scores): generation order without --best-of; else best first by caption match minus
    runaway share minus the share of verbatim copies beyond COPY_FREE. scores (one dict per candidate) when there is
    --text or --best-of, else []."""
    if wants is None and args.best_of == 1:
        return list(range(len(cands))), []
    scores = []
    for res, r in cands:
        a = note_attributes(*new_notes(res['streams'], r, args))
        m, checks = match(wants, a) if wants is not None else (1.0, {})
        streams, start = res['streams'], res['streams'][0].size(1) - args.max_new_tokens
        row = streams[0][r].tolist()
        end = start + row[start:].index(EOS_PITCH) if EOS_PITCH in row[start:] else len(row)
        copies = copy_share(row[:end], streams[3][r, :end].numpy(), streams[4][r, :end].numpy(), COPY_MIN, start)
        scores.append(dict(score=m - a['runaway'] - max(0.0, copies - COPY_FREE), match=m, checks=checks,
                           runaway=a['runaway'], copies=copies, nps=a['nps'], label=res['labels'][r]))
    if args.best_of == 1:
        return list(range(len(cands))), scores
    order = sorted(range(len(cands)), key=lambda c: -scores[c]['score'])
    print(f"Best {args.num_samples} of {len(cands)} (score = caption match - runaway share - copies over "
          f"{COPY_FREE:.0%}):")
    for rank_, c in enumerate(order):
        mark = '*' if rank_ < args.num_samples else ' '
        print(f"  {mark} candidate {c + 1:3d}: {score_line(scores[c])}"
              + (f" | {scores[c]['label'].split(' (')[0]}" if scores[c]['label'] else ''))
    return order, scores


def score_line(s):
    missing = ', '.join(f"{k} {v:.1f}" if v else k for k, v in s['checks'].items() if v < 1)
    return (f"score {s['score']:.2f} = match {s['match']:.2f} - runaway {s['runaway']:.2f} - copies "
            f"{max(0.0, s['copies'] - COPY_FREE):.2f} ({s['copies']:.0%} of the new pitched notes) | "
            f"{s['nps']:.1f} notes/s" + (f" | short of: {missing}" if missing else ""))


def prompt_rows(config, args, pieces, seed, n_rows, device, first=0):
    """(pitch, velocity, duration, delta_time) each (n_rows, T), one prompt per row, and a label per row.
    Models with BOS/EOS get BOS in front of a prompt that starts at the beginning of its piece, and BOS as the seed.
    first: the run's index of the first row (--retrieve: row i takes piece i)."""
    streams, labels, programs = _prompt_rows(config, args, pieces, seed, n_rows, device, first)
    if programs is None:
        programs = torch.zeros_like(streams[0])
    if config.pitch_size > BOS_PITCH and (args.prompt_start == 0 or streams[0].size(1) == 0):
        streams, programs = with_bos(streams, programs)
    return streams, labels, programs


def instrument_names(programs):
    names = sorted({int(p) for p in programs})
    return ', '.join('Drums' if p == DRUM_PROGRAM else pretty_midi.program_to_instrument_name(p) for p in names)


def _prompt_rows(config, args, pieces, seed, n_rows, device, first=0):
    a, b = args.prompt_start, args.prompt_start + args.prompt_notes
    if pieces:
        rng = random.Random(seed)
        rows, progs, labels = [], [], []
        for r in range(n_rows):
            while True:
                idx = (args.val_index if args.val_index is not None else (first + r) % len(pieces) if args.retrieve
                       else rng.randrange(len(pieces)))
                label, load = pieces[idx]
                toks = load()
                prog = None
                if isinstance(toks, tuple):
                    toks, prog = toks
                enough = len(toks) >= b and (prog is None or len(set(prog[a:b].tolist())) >= args.min_instruments)
                if enough or args.val_index is not None or args.retrieve:
                    break  # random pick: retry pieces shorter than the prompt (or with too few instruments)
            rows.append(torch.from_numpy(toks[a:b]).long())
            progs.append(None if prog is None else torch.from_numpy(prog[a:b]).long())
            source = label if args.retrieve else f"val piece #{idx}: {label}"
            labels.append(f"notes {a}-{a + len(rows[-1])} of {source}"
                          + (f" [{instrument_names(prog[a:b])}]" if prog is not None else ""))
        n = min(len(r) for r in rows)  # rows must share a length; only a fixed short --val-index can differ
        toks = torch.stack([r[:n] for r in rows]).to(device)
        programs = torch.stack([p[:n] for p in progs]).to(device) if progs[0] is not None else None
        return tuple(toks[:, :, j] for j in range(4)), labels, programs
    if args.prompt:
        kw = dict(velocity_bins=config.velocity_size, max_duration_bin=config.duration_size - 1,
                  max_delta_bin=config.delta_time_size - 1)
        # multi-instrument models read every track with its instrument, drums included (as their token stores);
        # tokenize_midi would merge the tracks into piano and drop the drums
        res = tokenize_tracks(args.prompt, **kw) if config.n_programs else tokenize_midi(args.prompt, **kw)
        if res is None:
            raise GenerationError(f"no notes in {args.prompt} (empty or unreadable MIDI)")
        tokens, prog = res if config.n_programs else (res, None)
        tokens = torch.from_numpy(tokens[a:b].astype(np.int64))
        streams = tuple(tokens[:, j].unsqueeze(0).expand(n_rows, -1).contiguous().to(device) for j in range(4))
        programs = None
        label = f"notes {a}-{a + streams[0].size(1)} of {args.prompt}"
        if prog is not None:
            prog = prog[a:b].astype(np.int64)
            programs = torch.from_numpy(prog).unsqueeze(0).expand(n_rows, -1).contiguous().to(device)
            label += f" [{instrument_names(prog)}]"
        return streams, [label] * n_rows, programs
    if config.pitch_size > BOS_PITCH:
        return (tuple(torch.zeros((n_rows, 0), dtype=torch.long, device=device) for _ in range(4)), [None] * n_rows,
                None)
    # single-token seed (silent note) for models without BOS
    z = lambda fill: torch.full((n_rows, 1), fill, dtype=torch.long, device=device)
    return (z(0), z(0), z(10), z(0)), [None] * n_rows, None


def generate_batch(model, config, args, pieces, seed, n_rows, device, progress=None, text=None, first=0,
                   on_notes=None):
    """One batch of n_rows rows: dict(streams = (pitch, velocity, duration, delta_time, programs) on the CPU, prompt
    included; labels = the prompt of each row; n_prompt = prompt notes left out of the saved MIDI; seed; length = the
    length the rows were given: --length-seconds, or one per row with --length-after-prompt)."""
    torch.manual_seed(seed)
    (pitch, velocity, duration, delta_time), labels, prompt_programs = prompt_rows(config, args, pieces, seed,
                                                                                  n_rows, device, first)
    if args.sample_instruments and not config.n_programs:
        raise GenerationError("--sample-instruments needs a multi-instrument checkpoint (n_programs > 0)")
    allowed = None
    if args.instruments == 'prompt':
        if len({tuple(sorted(set(r.tolist()))) for r in prompt_programs}) > 1:
            raise GenerationError("--instruments prompt needs rows with the same instrument set: use --batch-size 1")
        allowed = sorted(set(prompt_programs[0].tolist()))
    elif args.instruments:
        allowed = [int(x) for x in args.instruments.split(',')]
    sample_instruments = args.sample_instruments
    if args.caption_allowed is not None and config.n_programs:  # the caption's limits as a hard mask
        allowed = sorted(set(args.caption_allowed) & set(allowed)) if allowed is not None else args.caption_allowed
        if not allowed:
            raise GenerationError("the prompt's instruments and the caption's leave no instrument to play")
        sample_instruments = sample_instruments or allowed != [0]
    has_prompt = bool(pieces or args.prompt)
    style = style_id(config, args.style)
    if args.length_seconds is not None and not config.n_remain and not (args.eos_ramp or args.length_stop):
        raise GenerationError("this checkpoint was trained without the length input (n_remain): --length-seconds "
                              "only works as a time limit here (--eos-ramp / --length-stop)")
    if args.eos_floor and args.length_stop and args.eos_floor >= args.length_stop:
        raise GenerationError(f"--eos-floor {args.eos_floor} must stay below --length-stop {args.length_stop}")
    target_nps = None
    if args.density == "prompt":
        if not has_prompt or pitch.size(1) < 2:
            raise GenerationError("--density prompt needs a prompt")
        secs = delta_time[:, 1:].sum(dim=1).float().clamp(min=1) * 0.02  # (a BOS row adds 0 time)
        target_nps = (pitch.size(1) / secs).tolist()
        print("Density target (from the prompts): " + ", ".join(f"{t:.1f}" for t in target_nps) + " notes/s")
    elif args.density:
        target_nps = float(args.density)
    n_prompt = 0 if args.keep_prompt and has_prompt else pitch.size(1)
    length = args.length_seconds
    if length is not None and args.length_after_prompt and has_prompt and n_prompt:
        # the length counts from the first prompt note, the result starts after the prompt: add each row's prompt
        length = [length + float(s) * 0.02 for s in delta_time.sum(dim=1).tolist()]
    cond = window_cond(config, args, pitch, delta_time, prompt_programs, has_prompt)
    stream = NoteStream(on_notes, first, n_prompt, args.max_delta) if on_notes is not None else None
    steps = [0]  # notes generated per row (fewer than --max-new-tokens when the stream stops early)

    def step(n):
        steps[0] = n
        if progress is not None:
            progress(n)
    print(f"Generating {args.max_new_tokens} notes x {n_rows} rows on {device} (seed {seed})...")
    t0 = time.time()
    with torch.no_grad():
        pitch, velocity, duration, delta_time, programs = generate(
            model, pitch, velocity, duration, delta_time,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            anchor=args.anchor,
            slide=args.slide,
            progress=step,
            dt_bias=args.dt_bias,
            target_nps=target_nps,
            cuda_graph=not args.no_cuda_graph,
            style=style,
            min_new=args.min_notes,
            program=None if sample_instruments else 0,
            programs=prompt_programs if config.n_programs else None,
            allowed_programs=allowed,
            return_programs=True,
            program_temperature=args.instrument_temperature,
            cond=cond,
            text=text,
            cfg_scale=args.cfg_scale,
            cfg_notes=args.cfg_notes, cfg_end=args.cfg_end, cfg_linear=args.cfg_linear, min_p=args.min_p,
            length_seconds=length,
            eos_ramp=args.eos_ramp,
            length_stop=args.length_stop,
            eos_floor=args.eos_floor,
            dry_multiplier=args.dry_multiplier,
            dry_base=args.dry_base,
            dry_allowed=args.dry_allowed,
            dry_key=args.dry_key,
            on_notes=stream,
            notes_every=NOTES_EVERY,
            hold=style_hold(config, args, device),
        )
    print(f"  {n_rows * steps[0] / (time.time() - t0):.0f} notes/s"
          + (f" (stopped after {steps[0]} notes: every row ended)" if steps[0] < args.max_new_tokens else ""))
    return dict(streams=[t.cpu() for t in (pitch, velocity, duration, delta_time, programs)], labels=labels,
                n_prompt=n_prompt, seed=seed, length=length)


def text_vector(config, prompt, device):
    """The caption vector of --text (LaBSE, frozen) for models trained with text, else None."""
    if not prompt:
        return None
    if not config.text_dim:
        raise GenerationError("this checkpoint was trained without text")
    return text_encoder(device).encode([prompt], tokens=config.text_prefix > 0)


@functools.cache
def caption_index(store, labels_dir, max_files, min_notes, device):
    """The retrieval index of a store, loaded once per process; built (when its cache file is stale) and searched
    with the cached full-precision LaBSE (text_encoder)."""
    from musicar.generation.retrieval import CaptionIndex
    return CaptionIndex(store, labels_dir=labels_dir, max_files=max_files, min_notes=min_notes, device=device,
                        encoder=text_encoder(device))


def style_hold(config, args, device):
    """generate's hold for --keep-style / --lock-instruments (the first section's description as the section caption
    of the rest, its instruments + the caption's as the only ones left), or None."""
    keep = args.keep_style and bool(config.section_len)
    lock = args.lock_instruments and bool(config.n_programs) and args.sample_instruments
    if args.keep_style and not keep:
        raise GenerationError("--keep-style needs a checkpoint trained with section captions (section_len)")
    if not (keep or lock):
        return None
    named = set()
    wants = parse_caption(args.text) if args.text else None
    if wants is not None:
        for _, progs in wants['instruments']:
            named |= set(progs)
        if wants['drums']:
            named.add(DRUM_PROGRAM)

    def hold(notes):
        pitch, velocity, duration, delta_time, programs = (t.numpy() for t in notes)
        caps, allowed = [], []
        for r in range(pitch.shape[0]):
            ends = np.flatnonzero(pitch[r] == EOS_PITCH)
            ix = np.flatnonzero(pitch[r][:ends[0] if len(ends) else None] < BOS_PITCH)  # real notes before the EOS
            tok = np.stack([pitch[r], velocity[r], duration[r], delta_time[r]], 1)[ix]
            prog = programs[r][ix]
            caps.append(hold_caption(tok, prog) if keep else '')
            counts = np.bincount(prog, minlength=129)
            present = set(np.flatnonzero((counts >= 4) & (counts >= 0.02 * len(prog))).tolist())
            allowed.append(sorted(present | named) if lock and present else None)
            print(f"  style hold, row {r}: " + (f"{caps[-1]!r}" if keep else "")
                  + (f" instruments {instrument_names(allowed[-1])}" if allowed[-1] else ""))
        states = text_encoder(device).encode(caps, tokens=True, max_length=SECTION_TOKENS) if keep else None
        return states, allowed if lock else None
    return hold


@functools.cache
def text_encoder(device):
    """LaBSE on `device`, loaded once per process (a server reuses it for every request)."""
    from musicar.text.encoder import TextEncoder
    return TextEncoder(device=device)


def style_id(config, name):
    """The style id of --style for models trained with styles (default 'none' = 0), else None."""
    if not config.n_styles:
        if name:
            raise GenerationError("this checkpoint was trained without styles")
        return None
    names = load_styles()
    name = name or 'none'
    if name not in names:
        raise GenerationError(f"unknown style {name!r}; choose from: {', '.join(names)}")
    print(f"Style: {name} (id {names.index(name)})")
    return names.index(name)


def window_cond(config, args, pitch, delta_time, prompt_programs, has_prompt):
    """(instrument multi-hot, density level) per row for models with cond_inst / n_density, else None:
    --cond-instruments (default: the prompt's set when it has instruments) and --cond-density (default none)."""
    if not (config.cond_inst or config.n_density):
        return None
    n_rows, device = pitch.size(0), pitch.device
    from_prompt = window_conditions(pitch, delta_time, prompt_programs if config.n_programs else None,
                                    config.n_programs or 1, config.n_density or 16)
    inst = torch.zeros(n_rows, config.n_programs or 1, device=device)
    how = args.cond_instruments or ('prompt' if prompt_programs is not None and has_prompt else 'none')
    if how == 'prompt':
        inst = from_prompt[0]
    elif how != 'none':
        inst[:, [int(x) for x in how.split(',')]] = 1.0
    density = torch.zeros(n_rows, dtype=torch.long, device=device)
    dens = args.cond_density or 'none'
    if dens == 'prompt':
        density = from_prompt[1]
    elif dens != 'none':
        density = density_level(float(dens), config.n_density).expand(n_rows).to(device)
    print(f"Conditioning: instruments {how} ({instrument_names(inst[0].nonzero().flatten().tolist()) or 'none'}), "
          f"density {dens} (levels {density.tolist()})")
    return inst, density


def fade_out(velocity, delta_time, seconds, floor=0.3, dt_seconds=0.02):
    """Velocity bins of a row scaled down linearly over its last `seconds` (to floor x at the last onset), for rows
    cut by --length-stop."""
    onsets, t = [], 0.0
    for d in delta_time:
        t += d * dt_seconds
        onsets.append(t)
    end = onsets[-1] if onsets else 0.0
    return [max(1, round(v * (floor + (1 - floor) * min(1.0, (end - o) / seconds)))) if v else v
            for v, o in zip(velocity, onsets)]


def finish_row(res, r, index, score, args):
    """Row r of a batch result as a Piece: the prompt dropped unless --keep-prompt, cut at the first EOS, BOS / EOS
    removed, faded out if --length-stop ended it."""
    streams = res['streams']
    rows = [s[r, res['n_prompt']:].tolist() for s in streams]
    start = streams[0].size(1) - args.max_new_tokens
    new = streams[0][r, start:].tolist()
    eos = EOS_PITCH in new
    n_new = new.index(EOS_PITCH) if eos else len(new)
    rows = [x[:len(x) - (len(new) - n_new)] for x in rows]
    seconds = sum(streams[3][r, :start + n_new].tolist()) * 0.02  # from the first prompt note, as --length-seconds
    length = res['length'][r] if isinstance(res.get('length'), list) else args.length_seconds
    stopped = eos and bool(args.length_stop and length) and seconds >= args.length_stop * length
    keep = [i for i, p in enumerate(rows[0]) if p < BOS_PITCH]
    rows = [[x[i] for i in keep] for x in rows]
    if stopped and args.fade_seconds:
        rows[1] = fade_out(rows[1], rows[3], args.fade_seconds)
    return Piece(index=index, seed=res['seed'], row=r, tokens=rows, label=res['labels'][r], score=score,
                 new_notes=n_new, eos=eos, seconds=seconds, stopped=stopped)
