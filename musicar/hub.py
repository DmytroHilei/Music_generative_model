"""
The released models without the repo: load one from the Hugging Face Hub (or a local checkpoint), generate MIDI.

    import musicar
    model = musicar.from_pretrained('musicar-S')         # downloaded once (Hub cache); CUDA when available
    midi = musicar.generate(model, n_notes=500, seed=0)  # a pretty_midi.PrettyMIDI
    midi.write('x.mid')

generate() maps its arguments onto generate.py's options (musicar.generation.piece.settings) and runs them with
Generator.run, so the same options and seed give the same notes as `musicar-generate` / generate.py. Its own defaults
differ in three places: a random seed (generate.py: 42), 1000 notes (generate.py: 512) and instruments='sample'.
"""

import random
import re
from pathlib import Path

OWNER = 'Dimitri-206265'
MODELS = ('musicar-L', 'musicar-S', 'musicar-S-piano', 'musicar-L-caption')  # short names of the released models


# a Hub id: user/repo, ASCII letters, digits, '-', '_', '.'; no '--' or '..'; no part starting or ending in '-' / '.'
HUB_ID = re.compile(r'(?!.*(--|\.\.))[A-Za-z0-9](?:[\w.-]*[A-Za-z0-9])?/[A-Za-z0-9](?:[\w.-]*[A-Za-z0-9])?', re.ASCII)


def repo_id(name):
    """The Hub repo of a model name: a short name in MODELS -> 'Dimitri-206265/<name>', 'user/repo' as it is, else
    None."""
    if name in MODELS:
        return f'{OWNER}/{name}'
    return name if HUB_ID.fullmatch(name) and len(name) <= 96 else None


def looks_local(name):
    """A name meant as a local path, never sent to the Hub: './x', '~/x', a checkpoint file name, or 'dir/x' where dir
    exists here (e.g. checkpoints/f1_cap in the training repo)."""
    return (name.startswith(('.', '~', '/')) or name.endswith(('.pt', '.safetensors', '.json'))
            or Path(name.split('/')[0]).is_dir())


def resolve(name_or_path, revision=None):
    """A local checkpoint file or run directory as it is, else the Hub repo's model.safetensors (bf16, no pickle) with
    its config.json next to it, downloaded once into huggingface_hub's cache."""
    name = str(name_or_path)
    if Path(name).expanduser().exists():
        return Path(name).expanduser()
    repo = None if looks_local(name) and name not in MODELS else repo_id(name)
    if repo is None:
        raise FileNotFoundError(f"{name}: no such checkpoint file or directory, nor a Hub model name "
                                f"({', '.join(MODELS)} or user/repo)")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (EntryNotFoundError, HFValidationError, LocalEntryNotFoundError,
                                       RepositoryNotFoundError, RevisionNotFoundError)
    try:
        hf_hub_download(repo, 'config.json', revision=revision)
        return Path(hf_hub_download(repo, 'model.safetensors', revision=revision))
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, LocalEntryNotFoundError,
            HFValidationError) as e:
        reason = str(e).strip().split('\n')[0]
        raise FileNotFoundError(f"{name}: no local checkpoint of that name, and the Hub repo {repo} could not be "
                                f"fetched ({type(e).__name__}: {reason})") from e


# generate.py options that are not generate() keywords
NOT_HERE = dict(checkpoint='is set by musicar.from_pretrained(name_or_path)', model='is set by from_pretrained',
                device='is set by musicar.from_pretrained(..., device=)',
                dtype='is set by musicar.from_pretrained(..., dtype=)', config='is preset= here',
                output='is not needed: generate() returns the MIDI, write it with .write(path)',
                no_mp3='is command-line only: generate() returns the MIDI and renders no audio',
                soundfont='is command-line only (MP3 rendering)', threads='is torch.set_num_threads(n) in Python',
                list_val='is command-line only')


def from_pretrained(name_or_path, device='auto', dtype='auto', revision=None):
    """A model ready to generate (Generator: eval mode, dropout 0). name_or_path: a released model (musicar-L,
    musicar-S, musicar-S-piano, musicar-L-caption), any 'user/repo' Hub id, or a local checkpoint file / run
    directory. device 'auto' = CUDA when available, else the CPU; dtype 'auto' = bf16 on CUDA GPUs with compute
    capability 8.0+ (Ampere and newer), float32 on older GPUs (e.g. Colab's T4), the CPU and MPS; or 'fp32' / 'bf16'
    / 'fp16' / a torch dtype. revision: a Hub branch, tag or commit."""
    from musicar.generation.piece import Generator
    return Generator(resolve(name_or_path, revision), device, dtype)


def instrument_options(instruments, config, prompt):
    """generate.py's instrument options for generate(instruments=...)."""
    from musicar.generation.piece import GenerationError
    if instruments in (None, 'piano'):
        return dict(sample_instruments=False)
    if instruments == 'sample':
        return dict(sample_instruments=bool(config.n_programs))  # piano models play piano
    if not config.n_programs:
        raise GenerationError(f"instruments={instruments!r}: this model plays piano only ('sample' or 'piano')")
    if instruments == 'prompt':
        if prompt is None:
            raise GenerationError("instruments='prompt' needs a prompt")
        return dict(sample_instruments=True, instruments='prompt')
    programs = [int(p) for p in (instruments.split(',') if isinstance(instruments, str) else instruments)]
    if not programs or not all(0 <= p <= 128 for p in programs):
        raise GenerationError(f"instruments={instruments!r}: 'sample', 'piano', 'prompt' or GM programs 0-127 "
                              "(128 = drums)")
    return dict(sample_instruments=True, instruments=','.join(map(str, programs)))


def generate(model, n_notes=None, prompt=None, prompt_notes=None, instruments='sample', text=None, temperature=None,
             top_p=None, top_k=None, seed=None, num_samples=1, preset=None, progress=None, **options):
    """MIDI from a model (from_pretrained, or a name / path to load first): a pretty_midi.PrettyMIDI, or a list of
    num_samples of them.
    n_notes: notes to generate at most (fewer when the model ends the piece); None = the preset's max-new-tokens (f1:
    2048), else 1000. prompt: a MIDI file whose first prompt_notes notes the model continues (not part of the result);
    None = from the start; prompt_notes None = 64 (128 with retrieve). instruments (multi-instrument models): 'sample' =
    the model picks them, 'piano' / None = all piano, a list of GM programs (128 = drums) = only those, 'prompt' = the
    prompt's own; piano models play piano with 'sample'. text: a caption (English or Ukrainian) for models with text
    input (musicar-L-caption; needs pip install 'musicar[text]'). seed: None = a random seed (printed with the run;
    pass it to get the same piece again). preset: name(s) of generation presets (musicar/generation/presets, e.g. 'f1'
    for musicar-L-caption). progress: called as progress(done, total, detail) after every generated note (notes
    over all rows and candidates). options: any other generate.py option, dashes or underscores (e.g.
    cfg_scale=2.5,
    length_seconds=60; python generate.py --help). Other None arguments = the preset's value, else generate.py's
    default."""
    from musicar.generation.piece import GenerationError, build_parser, preset_defaults, settings
    if isinstance(model, (str, Path)):
        model = from_pretrained(model)
    presets = [str(preset)] if isinstance(preset, (str, Path)) else [str(p) for p in preset or ()]
    for key in options:
        hint = NOT_HERE.get(key.replace('-', '_'))
        if hint:
            raise GenerationError(f"generate(): {key} {hint}")
    from_presets = preset_defaults(build_parser(), presets) if presets else {}
    if n_notes is None and 'max_new_tokens' not in from_presets:
        n_notes = 1000
    if seed is None and 'seed' not in from_presets:
        seed = random.randrange(2 ** 31)
    opts = dict(num_samples=num_samples)
    opts.update({k: v for k, v in dict(max_new_tokens=n_notes, prompt_notes=prompt_notes,
                                       prompt=None if prompt is None else str(prompt), text=text,
                                       temperature=temperature, top_p=top_p, top_k=top_k, seed=seed).items()
                 if v is not None})
    opts.update(instrument_options(instruments, model.config, prompt))
    opts.update(options)
    args = settings(presets, **opts)
    if args.text and not model.config.text_dim and not (args.retrieve or args.best_of > 1):
        raise GenerationError("this model was trained without text: text= needs a model with text input "
                              "(musicar-L-caption)")
    # a note stream lets sampling stop once every row has ended with EOS (the same notes, no idle steps after it)
    pieces = model.run(args, progress=progress, on_notes=lambda candidate, notes: None)[:args.num_samples]
    midis = [p.midi(args.max_polyphony, args.max_delta) for p in pieces]
    return midis[0] if args.num_samples == 1 else midis
