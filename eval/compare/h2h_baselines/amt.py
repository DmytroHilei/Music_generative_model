"""
Anticipatory Music Transformer large (stanford-crfm/music-large-800k, 780M) on the head-to-head continuation
prompts. The authors' sampler (anticipation.sample.add_token: top-p, a 1,017-token sliding window, no KV cache) in
autoregressive mode, as sample.generate runs it, until --notes new notes or --max-seconds of new music.

    .venv/bin/python eval/compare/h2h_baselines/amt.py [--prompts 1 --notes 50 --name amt_smoke]
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM

from common import BASE, Progress, out_dir, prompts, write_continuation

sys.path.insert(0, str(BASE / 'amt' / 'repo_anticipation'))
from anticipation import ops  # noqa: E402
from anticipation.config import TIME_RESOLUTION  # noqa: E402
from anticipation.convert import events_to_midi, midi_to_events  # noqa: E402
from anticipation.sample import add_token  # noqa: E402
from anticipation.vocab import AUTOREGRESS, REST, TIME_OFFSET  # noqa: E402

TOP_P = 0.98  # the README's examples


def continue_events(model, events, notes, top_p, max_seconds):
    """New events (rests removed) after `events`, sample.generate's loop with a note-count stop."""
    start = ops.max_time(events, seconds=False)
    tokens, _ = ops.anticipate(ops.pad(events, start), [])
    stop = start + max_seconds * TIME_RESOLUTION
    current, new, count = start, [], 0
    while count < notes:
        tok = add_token(model, [AUTOREGRESS], tokens, top_p, current)
        t = tok[0] - TIME_OFFSET
        if t >= stop:
            break
        tokens.extend(tok)
        new.extend(tok)
        current = t
        count += tok[2] != REST
    return ops.unpad(new)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default='amt')
    p.add_argument('--prompts', type=int, default=0)
    p.add_argument('--notes', type=int, default=1000)
    p.add_argument('--max-seconds', type=float, default=240)
    p.add_argument('--top-p', type=float, default=TOP_P)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    weights = BASE / 'amt'
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16).cuda().eval()
    rows = prompts('cont', args.prompts)
    d = out_dir('cont', args.name, dict(model='stanford-crfm/music-large-800k', weights=str(weights), dtype='bf16',
                                        top_p=args.top_p, notes=args.notes, max_seconds=args.max_seconds,
                                        seed=args.seed, sampler='anticipation.sample.add_token, AUTOREGRESS'))
    prog = Progress(len(rows), args.name)
    for i, r in enumerate(rows):
        path = d / f"{r['id']}_0.mid"
        if path.exists():
            continue
        prompt = d.parent / 'prompts' / f"{r['id']}.mid"
        torch.manual_seed(args.seed + i)
        new = continue_events(model, midi_to_events(str(prompt)), args.notes, args.top_p, args.max_seconds)
        n = write_continuation(prompt, events_to_midi(new), path)
        prog.step(f"{r['id']}: {n} new notes")


if __name__ == '__main__':
    main()
