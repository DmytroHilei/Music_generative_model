"""
Text2midi (amaai-lab/text2midi) on the head-to-head text prompts. Runs in checkpoints/baselines/env_text2midi (its
pinned transformers 4.47 / miditok 3.0.3 / symusic 0.5.5; torch from the main .venv through a .pth file).

The README's model (Transformer(vocab, 768, 8, 2048, 18, 1024, False, 8)), FLAN-T5-base caption tokens, T 1.0 without
truncation, 2,000 REMI tokens, decoded by the authors' pickled REMI tokenizer. The authors' generate() samples one
row only (it takes the last logit of the flattened batch) and re-runs the T5 encoder at every step; the loop below
is the same step for a batch of rows with the encoder output computed once, under bf16 autocast.

    checkpoints/baselines/env_text2midi/bin/python eval/compare/h2h_baselines/text2midi.py [--prompts 1 --samples 2
        --max-len 100 --name text2midi_smoke]
"""

import argparse
import math
import os
import pickle
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')  # google/flan-t5-base is in the HF cache

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

from common import BASE, Progress, out_dir, prompts  # noqa: E402

sys.path.insert(0, str(BASE / 'text2midi' / 'repo_Text2midi' / 'model'))
from transformer_model import Transformer  # noqa: E402


@torch.no_grad()
def sample(model, ids, mask, rows, max_len, temperature):
    """(rows, max_len) token ids: Transformer.forward step by step with the encoder memory kept."""
    src, src_mask = ids.repeat(rows, 1), mask.repeat(rows, 1)
    tgt = torch.ones((rows, 1), dtype=torch.long, device=ids.device)  # generate()'s start token
    with torch.autocast('cuda', dtype=torch.bfloat16):
        memory = model.encoder(src, attention_mask=src_mask).last_hidden_state
        for _ in range(max_len):
            x = model.pos_encoder(model.input_emb(tgt) * math.sqrt(model.d_model))
            out = model.decoder(x, memory, memory_mask=None, memory_key_padding_mask=None, tgt_is_causal=True,
                                memory_is_causal=False)
            logits = model.projection(out[:, -1:])[:, -1].float()
            tgt = torch.cat([tgt, torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)], dim=1)
    return tgt[:, 1:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default='text2midi')
    p.add_argument('--prompts', type=int, default=0)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--max-len', type=int, default=2000)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    folder = BASE / 'text2midi'
    with open(folder / 'vocab_remi.pkl', 'rb') as f:  # the authors' miditok REMI tokenizer object
        r_tokenizer = pickle.load(f)
    model = Transformer(len(r_tokenizer), 768, 8, 2048, 18, 1024, False, 8, device='cuda')
    model.load_state_dict(torch.load(folder / 'pytorch_model.bin', map_location='cuda', weights_only=True))
    model.eval()
    tok = T5Tokenizer.from_pretrained('google/flan-t5-base')
    rows = prompts('text', args.prompts)
    d = out_dir('text', args.name, dict(model='amaai-lab/text2midi', weights=str(folder), dtype='bf16 autocast',
                                        samples=args.samples, max_len=args.max_len, temperature=args.temperature,
                                        seed=args.seed))
    prog = Progress(len(rows), args.name)
    for i, r in enumerate(rows):
        names = [d / f"{r['id']}_{k}.mid" for k in range(args.samples)]
        if all(n.exists() for n in names):
            continue
        enc = tok(r['caption'], return_tensors='pt', padding=True, truncation=True)
        torch.manual_seed(args.seed + i)
        out = sample(model, enc.input_ids.cuda(), enc.attention_mask.cuda(), args.samples, args.max_len,
                     args.temperature)
        for name, seq in zip(names, out.tolist()):
            r_tokenizer.decode(seq).dump_midi(str(name))
        prog.step(r['id'])


if __name__ == '__main__':
    main()
