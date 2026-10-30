"""
MIDI-LLM (slseanwu/MIDI-LLM_Llama-3.2-1B) on the head-to-head text prompts, as the authors' generate_transformers.py
runs it: their system prompt + caption + ' ', the AMT MIDI BOS, T 1.0, top-p 0.98, 2,046 new tokens (682 AMT
events), no guidance (the released code has none). Every output is kept: the authors' check for > 64 notes at one
time only counts here (config.json 'excessive'), the same as our runaway rows are kept.

    .venv/bin/python eval/compare/h2h_baselines/midi_llm.py [--prompts 2 --samples 2 --name midi_llm_smoke]
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE, Progress, out_dir, prompts

sys.path.insert(0, str(BASE / 'amt' / 'repo_anticipation'))
from anticipation.convert import events_to_midi  # noqa: E402

AMT_BOS = 55026  # midi_llm/utils.py AMT_GPT2_BOS_ID: the AMT vocabulary sits after Llama's 128,256 ids
LLAMA_VOCAB = 128256
SYSTEM = 'You are a world-class composer. Please compose some music according to the following description: '


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default='midi_llm')
    p.add_argument('--prompts', type=int, default=0)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--max-tokens', type=int, default=2046)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.98)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    weights = BASE / 'midi_llm'
    tok = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16).cuda().eval()
    rows = prompts('text', args.prompts)
    config = dict(model='slseanwu/MIDI-LLM_Llama-3.2-1B', weights=str(weights), dtype='bf16', samples=args.samples,
                  max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                  system_prompt=SYSTEM)
    d = out_dir('text', args.name, config)
    excessive, failed = 0, []
    prog = Progress(len(rows), args.name)
    for i, r in enumerate(rows):
        names = [d / f"{r['id']}_{k}.mid" for k in range(args.samples)]
        if all(n.exists() for n in names):
            continue
        ids = tok(SYSTEM + r['caption'] + ' ', return_tensors='pt').input_ids
        ids = torch.cat([ids, torch.tensor([[AMT_BOS + LLAMA_VOCAB]])], dim=1).cuda()
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            out = model.generate(input_ids=ids, do_sample=True, max_new_tokens=args.max_tokens,
                                 temperature=args.temperature, top_p=args.top_p, num_return_sequences=args.samples,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for name, seq in zip(names, (out[:, ids.shape[1]:] - LLAMA_VOCAB).tolist()):
            end = next((j for j, t in enumerate(seq) if not 0 <= t < AMT_BOS), len(seq))
            seq = seq[:end - end % 3]
            excessive += bool(seq) and max(map(seq[::3].count, set(seq[::3]))) > 64
            try:
                events_to_midi(seq).save(str(name))
            except Exception as e:  # noqa: BLE001  (an undecodable output: no file, listed in config.json)
                failed.append(f'{name.name}: {type(e).__name__} {e}')
        prog.step(f"{r['id']}")
    config.update(excessive=excessive, failed=failed)
    (d / 'config.json').write_text(json.dumps(dict(system=args.name, task='text', **config), indent=1))


if __name__ == '__main__':
    main()
