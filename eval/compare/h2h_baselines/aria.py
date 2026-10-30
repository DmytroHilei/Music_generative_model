"""
Aria (loubb/aria-medium-base, model-gen: the continuation fine-tune, 659M) on the head-to-head piano prompts. The
authors' CUDA sampler (aria.inference.sample_cuda.sample_batch, bf16 weights + autocast) at the README's settings
(T 0.98, min-p 0.035); the whole prompt file is the prompt (get_inference_prompt), then --tokens new tokens (~3.4 per
note with the time tokens: 4,000 for 1,000 notes), cut after --notes new notes.

    .venv/bin/python eval/compare/h2h_baselines/aria.py [--task piano96]
        [--prompts 1 --notes 50 --tokens 200 --name aria_smoke]
"""

import argparse
import sys

import torch

from common import BASE, Progress, out_dir, prompts, write_continuation

sys.path.insert(0, str(BASE / 'aria' / 'repo_aria'))
sys.path.insert(0, str(BASE / 'aria' / 'repo_aria-utils'))
from aria.config import load_model_config  # noqa: E402
from aria.inference import get_inference_prompt  # noqa: E402
from aria.inference.model_cuda import TransformerLM  # noqa: E402
from aria.inference.sample_cuda import sample_batch  # noqa: E402
from aria.model import ModelConfig  # noqa: E402
from ariautils.midi import MidiDict  # noqa: E402
from ariautils.tokenizer import AbsTokenizer  # noqa: E402

WEIGHTS = BASE / 'aria' / 'model-gen.safetensors'


def load(tokenizer):
    """run._load_inference_model_torch('medium', strict=False): the embedding adapter is not in the gen weights."""
    from safetensors.torch import load_file
    config = ModelConfig(**load_model_config(name='medium'))
    config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerLM(config)
    res = model.load_state_dict(load_file(str(WEIGHTS)), strict=False)
    print(f'{WEIGHTS.name}: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}')
    return model.to('cuda', dtype=torch.bfloat16).eval()  # the stored weights are bf16; 1.3 GB instead of 2.6


def is_note(tok):
    """The first token of a note: (instrument, pitch, velocity), or ('drum', pitch)."""
    return isinstance(tok, tuple) and tok[0] not in ('prefix', 'onset', 'dur')


def cut_after(seq, start, notes):
    """seq up to the end of its `notes`-th note at or after `start`."""
    count = 0
    for j in range(start, len(seq)):
        if is_note(seq[j]):
            count += 1
            if count > notes:
                return seq[:j]
    return seq


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['piano', 'piano96'], default='piano')
    p.add_argument('--name', default='aria')
    p.add_argument('--prompts', type=int, default=0)
    p.add_argument('--notes', type=int, default=1000)
    p.add_argument('--tokens', type=int, default=4000)
    p.add_argument('--temperature', type=float, default=0.98)
    p.add_argument('--min-p', type=float, default=0.035)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    tokenizer = AbsTokenizer()
    model = load(tokenizer)
    rows = prompts(args.task, args.prompts)
    d = out_dir(args.task, args.name, dict(model='loubb/aria-medium-base (model-gen)', weights=str(WEIGHTS),
                                           dtype='bf16, bf16 autocast', notes=args.notes, tokens=args.tokens,
                                           temperature=args.temperature, min_p=args.min_p, seed=args.seed,
                                           sampler='aria.inference.sample_cuda.sample_batch'))
    prog = Progress(len(rows), args.name)
    for i, r in enumerate(rows):
        path = d / f"{r['id']}_0.mid"
        if path.exists():
            continue
        prompt_path = d.parent / 'prompts' / f"{r['id']}.mid"
        prompt = get_inference_prompt(MidiDict.from_midi(str(prompt_path)), tokenizer, prompt_len_ms=1e9)
        torch.manual_seed(args.seed + i)
        seq = sample_batch(model, tokenizer, prompt, num_variations=1, max_new_tokens=args.tokens,
                           temp=args.temperature, min_p=args.min_p)[0]
        seq = [t for t in cut_after(seq, len(prompt), args.notes) if t != tokenizer.eos_tok]
        full = tokenizer.detokenize(seq).to_midi()
        own = tokenizer.detokenize(prompt).to_midi()
        n = write_continuation(prompt_path, full, path, minus=own)
        prog.step(f"{r['id']}: {n} new notes, {len(seq) - len(prompt)} tokens")


if __name__ == '__main__':
    main()
