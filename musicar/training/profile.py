"""Step profiling for train.py (--profile_steps) and tools/bench/bench.py (--profile): phase timers with NVTX ranges,
and the CUDA kernel report of a torch.profiler window."""

import time
from contextlib import contextmanager

import torch
from torch.autograd import DeviceType


class Phases:
    """Wall time per phase of a training step. mode None: the ranges do nothing; 'time': NVTX range + synchronized
    timer; 'trace': NVTX range only (the profiler / nsys window, where a synchronize would change the overlap)."""

    def __init__(self):
        self.mode, self.ms = None, {}

    @contextmanager
    def __call__(self, name):
        if self.mode is None:
            yield
            return
        timed = self.mode == 'time'
        if timed:
            torch.cuda.synchronize()
            t = time.perf_counter()
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            if timed:
                torch.cuda.synchronize()
                self.ms[name] = self.ms.get(name, 0.0) + 1000 * (time.perf_counter() - t)

    def table(self, steps):
        total = sum(self.ms.values())
        return '\n'.join(f'    {name:<12}{ms / steps:9.0f} ms {100 * ms / total:6.1f}%' for name, ms in self.ms.items())


def kernel_group(name):
    """Attention first: flash kernels carry 'cutlass' in their template arguments."""
    n = name.lower()
    return ('attention' if any(k in n for k in ('flash', 'fmha', 'attention', 'cudnn'))
            else 'gemm' if any(k in n for k in ('gemm', 'cutlass', 'sm90', 'sm100', 'sm120', 'xmma', 'cublas', 'nvjet'))
            else 'cross-entropy/softmax' if any(k in n for k in ('softmax', 'nll', 'log_softmax', 'cross'))
            else 'fp8 scaling/casts' if any(k in n for k in ('amax', 'float8', 'fp8', 'e4m3', 'to_copy'))
            else 'layernorm' if 'norm' in n else 'optimizer' if 'adam' in n or 'foreach' in n
            else 'elementwise/other')


def kernel_report(events, top=20):
    """Share of the CUDA kernel time by kernel group and the top kernels of a profiler's key_averages() (kernels only:
    with CPU activity the ops also carry their kernels' time)."""
    events = [e for e in events if e.device_type == DeviceType.CUDA and e.device_time_total > 0]
    total = sum(e.device_time_total for e in events)
    groups = {}
    for e in events:
        groups[kernel_group(e.key)] = groups.get(kernel_group(e.key), 0) + e.device_time_total
    lines = [f'  CUDA kernel time {total / 1e3:,.0f} ms, by kernel group:']
    lines += [f'    {g:<24}{100 * t / total:6.1f}%' for g, t in sorted(groups.items(), key=lambda x: -x[1])]
    lines.append('  top kernels:')
    lines += [f'    {100 * e.device_time_total / total:5.1f}%  {e.count:>5}x  {e.key[:95]}'
              for e in sorted(events, key=lambda e: -e.device_time_total)[:top]]
    return '\n'.join(lines)
