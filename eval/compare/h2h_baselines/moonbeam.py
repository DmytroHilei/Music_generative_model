"""
Moonbeam 839M (guozixunnicolas/moonbeam-midi-foundation-model) on the head-to-head continuation prompts. Runs in
checkpoints/baselines/env_moonbeam with the authors' transformers fork (src/llama_recipes/transformers_minimal) put
first on the path; torch from the main .venv through a .pth file.

The authors' sampler (MusicLlama.generate of recipes/inference/custom_music_generation: one trunk step per note,
then the GRU decoder samples its 6 attributes, top-p) at the README's unconditional settings (T 0.9, top-p 0.95).
The context is 1,024 positions (SOS + 1,023 notes), shorter than the 128 prompt notes + 1,000 new ones: past it the
window restarts on the last --keep notes, onsets shifted to 0 as data_preprocess.chunk_compounds cuts training
chunks, until --notes new notes or an EOS.

    checkpoints/baselines/env_moonbeam/bin/python -I eval/compare/h2h_baselines/moonbeam.py [--prompts 1 --notes 50
        --name moonbeam_smoke]
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import BASE, Progress, out_dir, prompts, write_continuation  # noqa: E402

REPO = BASE / 'moonbeam' / 'repo_moonbeam-midi-foundation-model'
WEIGHTS = BASE / 'moonbeam' / 'moonbeam_839M.pt'
CONFIG = REPO / 'src' / 'llama_recipes' / 'configs' / 'model_config.json'
sys.path.insert(0, str(REPO / 'src' / 'llama_recipes' / 'transformers_minimal' / 'src'))  # before the .venv's
sys.path.insert(0, str(REPO / 'recipes' / 'inference' / 'custom_music_generation'))

# llama_recipes.datasets/__init__ imports every training dataset (and their packages); generation.py needs only the
# tokenizer module, loaded from its file under the name it imports
for name in ('llama_recipes', 'llama_recipes.datasets'):
    sys.modules.setdefault(name, types.ModuleType(name))
_spec = importlib.util.spec_from_file_location('llama_recipes.datasets.music_tokenizer',
                                               REPO / 'src' / 'llama_recipes' / 'datasets' / 'music_tokenizer.py')
music_tokenizer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = music_tokenizer
_spec.loader.exec_module(music_tokenizer)
# the fork's modeling_llama.py imports music21's chord / harmony at the top and uses neither
music21 = types.ModuleType('music21')
music21.chord = music21.harmony = None
sys.modules.setdefault('music21', music21)

import torch  # noqa: E402
from generation import MusicLlama  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

MusicTokenizer = music_tokenizer.MusicTokenizer


def load():
    """MusicLlama.build without its torch.load of a full pickle: the weights only, bf16 on the GPU."""
    config = LlamaConfig.from_pretrained(str(CONFIG))
    model = LlamaForCausalLM(config)
    sd = torch.load(WEIGHTS, map_location='cpu', weights_only=True)['model_state_dict']
    res = model.load_state_dict({k.removeprefix('module.'): v for k, v in sd.items()}, strict=False)
    print(f'{WEIGHTS.name}: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}')
    model = model.to('cuda', dtype=torch.bfloat16).eval()
    torch.set_default_dtype(torch.bfloat16)  # build() sets the cuda bf16 default tensor type; generate relies on it
    torch.set_default_device('cuda')
    tok = MusicTokenizer(timeshift_vocab_size=config.onset_vocab_size, dur_vocab_size=config.dur_vocab_size,
                         octave_vocab_size=config.octave_vocab_size,
                         pitch_class_vocab_size=config.pitch_class_vocab_size,
                         instrument_vocab_size=config.instrument_vocab_size,
                         velocity_vocab_size=config.velocity_vocab_size)
    return MusicLlama(model, tok, config)


def to_midi(notes):
    """(PrettyMIDI of compound notes (10-ms ticks), one track per program, 128 = drums; notes left out: pitch > 127
    or velocity 0). Not MusicTokenizer.compound_to_midi: it gives every instrument its own channel and fails past 15."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    tracks, dropped = {}, 0
    for onset, dur, octave, pitch_class, instrument, velocity in notes:
        if octave * 12 + pitch_class > 127 or velocity <= 0:
            dropped += 1
            continue
        if instrument not in tracks:
            tracks[instrument] = pretty_midi.Instrument(program=0 if instrument == 128 else instrument,
                                                        is_drum=instrument == 128)
            pm.instruments.append(tracks[instrument])
        tracks[instrument].notes.append(pretty_midi.Note(velocity=int(velocity), pitch=octave * 12 + pitch_class,
                                                         start=onset / 100, end=(onset + max(dur, 1)) / 100))
    return pm, dropped


def continue_notes(gen, notes, new, keep, temperature, top_p, context=0):
    """`new` notes after `notes` ([onset, dur, octave, pitch class, instrument, velocity], onsets in 10-ms ticks)
    in windows of at most max_len - 1 notes; returns the new notes (absolute onsets), fewer if an EOS came."""
    max_notes = (context or gen.config.max_len) - 1
    out = []
    while len(out) < new:
        window = (notes + out)[-(keep if out else max_notes):]
        t0 = window[0][0]
        shifted = [[n[0] - t0, *n[1:]] for n in window]
        want = min(new - len(out), max_notes - len(shifted))
        toks, _ = gen.generate([gen.tokenizer.encode_series(shifted, if_add_sos=True, if_add_eos=False)],
                               max_gen_len=want, temperature=temperature, top_p=top_p, echo=True)
        # echo: generate drops SOS but echo=False would also skip the first new note (its start counts SOS)
        got = [[n[0] + t0, *n[1:]] for n in toks[0][len(shifted):]]
        out += got
        if len(got) < want:  # EOS
            break
    return out[:new]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default='moonbeam')
    p.add_argument('--prompts', type=int, default=0)
    p.add_argument('--notes', type=int, default=1000)
    p.add_argument('--keep', type=int, default=768, help='notes kept when the window restarts')
    p.add_argument('--context', type=int, default=0, help="positions per window (default the model's 1,024; smaller "
                   'only to test the window restart)')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    gen = load()
    rows = prompts('cont', args.prompts)
    d = out_dir('cont', args.name, dict(model='guozixunnicolas/moonbeam-midi-foundation-model', weights=str(WEIGHTS),
                                        dtype='bf16', notes=args.notes, keep=args.keep, context=args.context,
                                        temperature=args.temperature,
                                        top_p=args.top_p, seed=args.seed, sampler='generation.MusicLlama.generate'))
    prog = Progress(len(rows), args.name)
    for i, r in enumerate(rows):
        path = d / f"{r['id']}_0.mid"
        if path.exists():
            continue
        prompt = d.parent / 'prompts' / f"{r['id']}.mid"
        torch.manual_seed(args.seed + i)
        notes = MusicTokenizer.midi_to_compound(str(prompt))
        new = continue_notes(gen, notes, args.notes, args.keep, args.temperature, args.top_p, args.context)
        pm, dropped = to_midi(new)
        n = write_continuation(prompt, pm, path)
        prog.step(f"{r['id']}: {n} new notes, {dropped} left out")


if __name__ == '__main__':
    main()
