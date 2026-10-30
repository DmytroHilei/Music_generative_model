"""
SkyTNT midi-model (MidiTokenizerV2) on the head-to-head continuation prompts: tv2o-large (asigalov61/Music-Llama,
the 'large' of the SkyTNT Space) or tv2o-medium (skytnt/midi-model-tv2o-medium). The prompt is tokenized as the
Space's 'midi' tab does (cc / tempo eps 4, remap tracks and channels, default instruments), the trailing EOS
dropped, then MIDIModel.generate with the Space's defaults (T 1.0, top-p 0.94, top-k 20) for --events new events.

    .venv/bin/python eval/compare/h2h_baselines/skytnt.py --model large [--prompts 1 --events 50 --name sky_smoke]
"""

import argparse
import sys
import types
from collections import Counter

import numpy as np
import torch
# before the PIL stub below: transformers checks at import time whether PIL is installed
from transformers import DynamicCache, LlamaConfig, LlamaModel, PreTrainedModel, PretrainedConfig  # noqa: F401

from common import BASE, Progress, out_dir, prompts, write_continuation

REPO = BASE / 'skytnt_large' / 'repo_midi-model'
MODELS = {'large': ('asigalov61/Music-Llama', BASE / 'skytnt_large' / 'pytorch_model.bin'),
          'medium': ('skytnt/midi-model-tv2o-medium', BASE / 'skytnt_medium' / 'model.safetensors')}

# midi_model.py imports peft for LoRA merging only and midi_tokenizer.py PIL for piano-roll images only, neither of
# which this script uses
peft = types.ModuleType('peft')
peft.PeftConfig = peft.LoraModel = peft.load_peft_weights = peft.set_peft_model_state_dict = None
sys.modules.setdefault('peft', peft)
pil = types.ModuleType('PIL')
pil.Image = types.ModuleType('PIL.Image')
sys.modules.setdefault('PIL', pil)
sys.modules.setdefault('PIL.Image', pil.Image)
sys.path.insert(0, str(REPO))
import MIDI  # noqa: E402
from midi_model import MIDIModel, MIDIModelConfig  # noqa: E402


def new_notes_score(full, own):
    """The score (MIDI.py: [ticks per beat, track, ...], notes as ['note', start, duration, channel, pitch, velocity])
    of `full` without the notes of `own`, the prompt's decoding; tempo and patch events kept. Notes match without
    their duration: detokenize cuts a note where the next note of its channel and pitch starts, so a new note can
    shorten a prompt note."""
    def key(e):
        return e[1], *e[3:6]
    old = Counter(key(e) for track in own[1:] for e in track if e[0] == 'note')
    out = [full[0]]
    for track in full[1:]:
        keep = []
        for e in track:
            if e[0] == 'note' and old[key(e)] > 0:
                old[key(e)] -= 1
                continue
            keep.append(e)
        out.append(keep)
    return out


def load(weights):
    model = MIDIModel(config=MIDIModelConfig.from_json_file(weights.parent / 'config.json'))
    if weights.suffix == '.safetensors':
        from safetensors.torch import load_file
        sd = load_file(weights)
    else:
        sd = torch.load(weights, map_location='cpu', weights_only=True)
        sd = sd.get('state_dict', sd)
    res = model.load_state_dict(sd, strict=False)
    print(f'{weights}: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}')
    return model.to('cuda', dtype=torch.bfloat16).eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=sorted(MODELS), required=True)
    p.add_argument('--name', default=None, help='default skytnt_<model>')
    p.add_argument('--prompts', type=int, default=0)
    p.add_argument('--events', type=int, default=1200, help='new MIDI events (notes + control / tempo changes)')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.94)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    repo, weights = MODELS[args.model]
    name = args.name or f'skytnt_{args.model}'
    model = load(weights)
    tokenizer = model.tokenizer
    rows = prompts('cont', args.prompts)
    d = out_dir('cont', name, dict(model=repo, weights=str(weights), dtype='bf16', events=args.events,
                                   temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed))
    prog = Progress(len(rows), name)
    for i, r in enumerate(rows):
        path = d / f"{r['id']}_0.mid"
        if path.exists():
            continue
        prompt = d.parent / 'prompts' / f"{r['id']}.mid"
        mid = tokenizer.tokenize(MIDI.midi2score(prompt.read_bytes()), cc_eps=4, tempo_eps=4,
                                 remap_track_channel=True, add_default_instr=True, remove_empty_channels=False)
        if mid[-1][0] == tokenizer.eos_id:
            mid = mid[:-1]
        mid = np.asarray(mid, dtype=np.int64)
        gen = torch.Generator('cuda').manual_seed(args.seed + i)
        seq = model.generate(mid, batch_size=1, max_len=len(mid) + args.events, temp=args.temperature,
                             top_p=args.top_p, top_k=args.top_k, generator=gen)[0]
        new = new_notes_score(tokenizer.detokenize(seq.tolist()), tokenizer.detokenize(mid.tolist()))
        n = write_continuation(prompt, bytes(MIDI.score2midi(new)), path)
        prog.step(f"{r['id']}: {n} new notes")


if __name__ == '__main__':
    main()
