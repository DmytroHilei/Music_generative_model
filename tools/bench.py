"""
GPU-side training throughput benchmark on synthetic data (no data loader): forward + backward + AdamW step.

    python tools/bench.py                                   # default sweep for the L model
    python tools/bench.py --n_layer 6 --n_embd 256 --n_head 8 --micro 6 24 48 --compile 0 1

Reports tokens/s, ms per optimizer step (at a fixed tokens-per-step budget) and peak memory.
--profile: time per phase (forward / backward incl. checkpoint recompute / clip / optimizer) and the top CUDA kernels.
Steps carry NVTX ranges; for a timeline in Nsight Systems:
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi -o logs/prof/x python tools/bench.py ... --profile
"""

import argparse
import os
import time
from contextlib import nullcontext

# same allocator setting as train.py (fewer fragmentation OOMs); PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False to compare
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from musicar.model import GPT, MusicConfig, convert_fp8, refresh_fp8_weights

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
    p.add_argument('--optim', default='adamw', choices=['adamw', 'muon', 'muon_bf16'])
    p.add_argument('--act_ckpt', type=int, nargs='+', default=[0], help='checkpointed blocks (-1 = all)')
    p.add_argument('--n_programs', type=int, default=0, help='129 = the multi-instrument model')
    p.add_argument('--norm', default='layernorm', choices=['layernorm', 'rmsnorm'])
    p.add_argument('--mlp', default='gelu', choices=['gelu', 'swiglu'])
    p.add_argument('--qk_norm', type=int, default=0)
    p.add_argument('--ckpt_save', default='', help="selective checkpointing: 'attn' | 'attn,mm' (model.act_ckpt_save)")
    p.add_argument('--fp8_recipe', default='tensorwise', choices=['tensorwise', 'rowwise', 'rowwise_with_gw_hp'])
    p.add_argument('--fp8_skip', default='', help="Linear name suffixes kept in bf16, e.g. 'attn.c_proj'")
    p.add_argument('--fp8_cache', type=int, default=0, help='cast each fp8 weight once per step (tensorwise)')
    p.add_argument('--compile_ns', type=int, default=0, help='Muon: compile the batched Newton-Schulz')
    p.add_argument('--profile', action='store_true', help='profile one config (first of each list): top CUDA kernels')
    p.add_argument('--profile_steps', type=int, default=2, help='steps timed per phase in --profile')
    return p.parse_args()


def bench(args, micro, compiled, attn, fp8, profile=False, act_ckpt=0):
    torch.manual_seed(0)
    cfg = MusicConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, block_size=args.block,
                      pos_emb=args.pos_emb, dropout=0.0, label_smoothing=0.0, n_programs=args.n_programs,
                      norm=args.norm, mlp=args.mlp, qk_norm=bool(args.qk_norm))
    model = GPT(cfg).cuda()
    model.act_ckpt = act_ckpt
    model.act_ckpt_save = args.ckpt_save
    if fp8:  # only transformer matmuls; heads/embeddings stay bf16
        convert_fp8(model.transformer.h, args.fp8_recipe, args.fp8_skip, cache_weights=bool(args.fp8_cache))
        refresh_fp8_weights(model)
    if args.optim == 'adamw':
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), 'cuda')
    else:
        from musicar.optim import build_muon_optimizer
        opt = build_muon_optimizer(model, 0.1, 1e-3, (0.9, 0.95), 'cuda',
                                   momentum_dtype=torch.bfloat16 if args.optim == 'muon_bf16' else None,
                                   compile_ns=bool(args.compile_ns))
    fwd = torch.compile(model) if compiled else model
    accum = max(1, args.tokens_per_step // (micro * args.block))
    sizes = (cfg.pitch_size, cfg.velocity_size, cfg.duration_size, cfg.delta_time_size)
    X = tuple(torch.randint(0, s, (micro, args.block), device='cuda') for s in sizes)
    Y = tuple(torch.randint(0, s, (micro, args.block), device='cuda') for s in sizes)
    kw = {}
    if args.n_programs:
        kw['program'] = torch.randint(0, args.n_programs, (micro, args.block), device='cuda')
        Y = Y + (torch.randint(0, args.n_programs, (micro, args.block), device='cuda'),)
    backend = sdpa_kernel(BACKENDS[attn]) if BACKENDS[attn] is not None else nullcontext()

    phases = {}

    def phase(name, timed):
        """NVTX range; with timed=True also synchronize and add the wall time to phases[name]."""
        class Range:
            def __enter__(self):
                torch.cuda.nvtx.range_push(name)
                if timed:
                    torch.cuda.synchronize()
                    self.t = time.perf_counter()

            def __exit__(self, *exc):
                if timed:
                    torch.cuda.synchronize()
                    phases[name] = phases.get(name, 0.0) + time.perf_counter() - self.t
                torch.cuda.nvtx.range_pop()
        return Range()

    def step(timed=False):
        for _ in range(accum):
            with phase('forward', timed), torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = fwd(*X, targets=Y, **kw)
            with phase('backward', timed):
                (loss / accum).backward()
        with phase('clip', timed):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with phase('optimizer', timed):
            opt.step()
            opt.zero_grad(set_to_none=True)
            refresh_fp8_weights(model)

    with backend:
        for _ in range(3):  # warmup (+ compilation)
            step()
        torch.cuda.synchronize()
        if profile:
            for _ in range(args.profile_steps):
                step(timed=True)
            total = sum(phases.values())
            print(f'  time per step by phase ({args.profile_steps} steps, synchronized):')
            for name, t_ in phases.items():
                print(f'    {name:<12}{1000 * t_ / args.profile_steps:9.0f} ms {100 * t_ / total:6.1f}%')
            from torch.profiler import profile as prof, ProfilerActivity
            torch.cuda.profiler.start()  # capture range for nsys --capture-range=cudaProfilerApi
            with prof(activities=[ProfilerActivity.CUDA]) as pr:
                step()
                torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            events = pr.key_averages()
            total = sum(e.device_time_total for e in events)
            groups = {}
            for e in events:
                n = e.key.lower()
                g = ('gemm' if any(k in n for k in ('gemm', 'cutlass', 'sm90', 'sm100', 'sm120', 'xmma', 'cublas'))
                     else 'attention' if any(k in n for k in ('flash', 'fmha', 'attention', 'cudnn'))
                     else 'cross-entropy/softmax' if any(k in n for k in ('softmax', 'nll', 'log_softmax', 'cross'))
                     else 'fp8 scaling/casts' if any(k in n for k in ('amax', 'float8', 'fp8', 'e4m3', 'to_copy'))
                     else 'layernorm' if 'norm' in n else 'optimizer' if 'adam' in n or 'foreach' in n
                     else 'elementwise/other')
                groups[g] = groups.get(g, 0) + e.device_time_total
            print('  time by kernel group:')
            for g, t_ in sorted(groups.items(), key=lambda x: -x[1]):
                print(f'    {g:<24}{100 * t_ / total:6.1f}%')
            print('  top kernels:')
            for e in sorted(events, key=lambda e: -e.device_time_total)[:20]:
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
    n = sum(p.numel() for p in GPT(MusicConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                                               mlp=args.mlp, qk_norm=bool(args.qk_norm))).parameters())
    print(f"model {args.n_layer}L x {args.n_embd} ({n / 1e6:.1f}M), block {args.block} {args.pos_emb}, "
          f"{args.tokens_per_step:,} tokens/step, {torch.cuda.get_device_name()}")
    print(f"optimizer {args.optim}, n_programs {args.n_programs}")
    print(f"{'micro':>6}{'compile':>8}{'attn':>10}{'fp8':>5}{'ckpt':>5}{'tok/s':>11}{'ms/step':>9}{'peak GB':>9}")
    if args.profile:
        print(bench(args, args.micro[0], args.compile[0], args.attn[0], args.fp8[0], profile=True,
                    act_ckpt=args.act_ckpt[0]))
        return
    for fp8 in args.fp8:
        for attn in args.attn:
            for compiled in args.compile:
                for micro in args.micro:
                    for ck in args.act_ckpt:
                        try:
                            tps, ms, mem = bench(args, micro, compiled, attn, fp8, act_ckpt=ck)
                            print(f"{micro:>6}{compiled:>8}{attn:>10}{fp8:>5}{ck:>5}{tps:>11,.0f}{ms:>9.0f}{mem:>9.2f}",
                                  flush=True)
                        except Exception as e:  # OOM, unsupported backend, ...
                            print(f"{micro:>6}{compiled:>8}{attn:>10}{fp8:>5}{ck:>5}   failed: {type(e).__name__}: "
                                  f"{str(e).splitlines()[0][:70]}", flush=True)
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
