"""
GPU-side training throughput benchmark on synthetic data (no data loader): forward + backward + AdamW step.

    python bench.py                                   # default sweep for the L model
    python bench.py --n_layer 6 --n_embd 256 --n_head 8 --micro 6 24 48 --compile 0 1

Reports tokens/s, ms per optimizer step (at a fixed tokens-per-step budget) and peak memory.
"""

import argparse
import time
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import GPT, MusicConfig

BACKENDS = {
    'default': None,
    'flash': SDPBackend.FLASH_ATTENTION,
    'efficient': SDPBackend.EFFICIENT_ATTENTION,
    'cudnn': SDPBackend.CUDNN_ATTENTION,
    'math': SDPBackend.MATH,
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n_layer', type=int, default=12)
    p.add_argument('--n_embd', type=int, default=512)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--block', type=int, default=512)
    p.add_argument('--pos_emb', default='learned', choices=['learned', 'rope'])
    p.add_argument('--tokens_per_step', type=int, default=30720)
    p.add_argument('--micro', type=int, nargs='+', default=[6, 12, 30, 60])
    p.add_argument('--compile', type=int, nargs='+', default=[0, 1])
    p.add_argument('--attn', nargs='+', default=['default'], choices=list(BACKENDS))
    p.add_argument('--fp8', type=int, nargs='+', default=[0])
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--profile', action='store_true', help='profile one config (first of each list): top CUDA kernels')
    return p.parse_args()


def bench(args, micro, compiled, attn, fp8, profile=False):
    torch.manual_seed(0)
    cfg = MusicConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, block_size=args.block,
                      pos_emb=args.pos_emb, dropout=0.0, label_smoothing=0.0)
    model = GPT(cfg).cuda()
    if fp8:
        from torchao.float8 import convert_to_float8_training
        # only transformer matmuls; heads/embeddings stay bf16
        convert_to_float8_training(model.transformer.h)
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), 'cuda')
    fwd = torch.compile(model) if compiled else model
    accum = max(1, args.tokens_per_step // (micro * args.block))
    sizes = (cfg.pitch_size, cfg.velocity_size, cfg.duration_size, cfg.delta_time_size)
    X = tuple(torch.randint(0, s, (micro, args.block), device='cuda') for s in sizes)
    Y = tuple(torch.randint(0, s, (micro, args.block), device='cuda') for s in sizes)
    backend = sdpa_kernel(BACKENDS[attn]) if BACKENDS[attn] is not None else nullcontext()

    def step():
        for _ in range(accum):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = fwd(*X, targets=Y)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    with backend:
        for _ in range(3):  # warmup (+ compilation)
            step()
        torch.cuda.synchronize()
        if profile:
            from torch.profiler import profile as prof, ProfilerActivity
            with prof(activities=[ProfilerActivity.CUDA]) as pr:
                step()
                torch.cuda.synchronize()
            events = pr.key_averages()
            total = sum(e.device_time_total for e in events)
            groups = {}
            for e in events:
                n = e.key.lower()
                g = ('gemm' if any(k in n for k in ('gemm', 'cutlass', 'sm90', 'sm100', 'sm120', 'xmma', 'cublas'))
                     else 'attention' if any(k in n for k in ('flash', 'fmha', 'attention', 'cudnn'))
                     else 'cross-entropy/softmax' if any(k in n for k in ('softmax', 'nll', 'log_softmax', 'cross'))
                     else 'layernorm' if 'norm' in n else 'optimizer' if 'adam' in n or 'foreach' in n
                     else 'elementwise/other')
                groups[g] = groups.get(g, 0) + e.device_time_total
            print('  time by kernel group:')
            for g, t_ in sorted(groups.items(), key=lambda x: -x[1]):
                print(f'    {g:<24}{100 * t_ / total:6.1f}%')
            print('  top kernels:')
            for e in sorted(events, key=lambda e: -e.device_time_total)[:12]:
                print(f'    {100 * e.device_time_total / total:5.1f}%  {e.count:>5}x  {e.key[:95]}')
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / args.steps
    tokens = accum * micro * args.block
    return tokens / dt, dt * 1000, torch.cuda.max_memory_allocated() / 2**30


def main():
    args = parse_args()
    n = sum(p.numel() for p in GPT(MusicConfig(n_layer=args.n_layer, n_embd=args.n_embd,
                                               n_head=args.n_head)).parameters())
    print(f"model {args.n_layer}L x {args.n_embd} ({n / 1e6:.1f}M), block {args.block} {args.pos_emb}, "
          f"{args.tokens_per_step:,} tokens/step, {torch.cuda.get_device_name()}")
    print(f"{'micro':>6}{'compile':>8}{'attn':>10}{'fp8':>5}{'tok/s':>11}{'ms/step':>9}{'peak GB':>9}")
    if args.profile:
        print(bench(args, args.micro[0], args.compile[0], args.attn[0], args.fp8[0], profile=True))
        return
    for fp8 in args.fp8:
        for attn in args.attn:
            for compiled in args.compile:
                for micro in args.micro:
                    try:
                        tps, ms, mem = bench(args, micro, compiled, attn, fp8)
                        print(f"{micro:>6}{compiled:>8}{attn:>10}{fp8:>5}{tps:>11,.0f}{ms:>9.0f}{mem:>9.2f}")
                    except Exception as e:  # OOM, unsupported backend, ...
                        print(f"{micro:>6}{compiled:>8}{attn:>10}{fp8:>5}   failed: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:70]}")
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
