"""Sampling from a GPT: batched generation with a KV cache and a CUDA-graph one-note step."""

import torch
from torch.nn import functional as F

from musicar.data_loader import BOS_PITCH, EOS_PITCH
from musicar.model import KVCache, STREAM_ORDER

# one side stream per device for the CUDA-graph warm-up and capture: cuBLAS keeps a workspace for every
# stream it has run on (~32 MB each, never freed), so a new stream per call grew the memory of every evaluation
_SIDE_STREAMS = {}


@torch.no_grad()
def generate(model, pitch, velocity, duration, delta_time, max_new_tokens, temperature=0.85, top_k=None,
             anchor=0, slide=None, progress=None, dt_bias=0.0, target_nps=None, dt_seconds=0.02,
             cuda_graph=True, style=None, min_new=0, program=0, top_p=None, cfg_scale=1.0,
             eos_t1=False, programs=None, allowed_programs=None, return_programs=False, program_temperature=None,
             cond=None):
    """
    Batched sampling (B rows, same prompt length) with a preallocated KV cache: each new note costs one position
    through the model, not a full re-run. On CUDA the one-note transformer step is captured once as a CUDA graph
    and replayed (cuda_graph=True), which removes the per-kernel launch cost that dominates small-batch decode.
    Positions are learned up to block_size, so the window can't roll one note at a time with a cache (every
    cached position would shift). When the cache is full it is refilled from a shorter window, making room
    for `slide` new notes (default block_size // 4, so the context stays between 3/4 and all of block_size).
    anchor > 0 (optional, off by default): the first `anchor` notes of the sequence (e.g. the prompt's theme)
    stay pinned at the start of every refilled window, in front of the most recent notes.
    progress: optional callback(n_generated) after every note (e.g. a jobstatus.JobStatus update).
    Density control (no retraining; only the delta_time head is touched, every other choice stays the model's):
      dt_bias: fixed logit bias b * log(max(bin, 1)) on delta_time; b > 0 favours longer gaps = fewer notes/s.
      target_nps: notes per second to track (a float, or one per row). A controller nudges each row's bias after
        every note from the density of its last 64 notes, starting at dt_bias. dt_seconds = seconds per bin.
    style: (B,) style ids (models with n_styles; 0 = none/unconditional).
    program: models with n_programs: an int = every new note is played by this instrument (0 = piano) and its
      head isn't sampled; None = the instrument head is sampled per note (dt -> instrument -> pitch -> ...).
    programs: (B, T) instruments of the prompt notes (default: all `program`, or 0 when sampling).
    allowed_programs: with program=None, only these instruments can be sampled (e.g. [0, 33, 128]).
    return_programs: also return the (B, T) instrument of every note as a fifth tensor.
    cond: models with cond_inst / n_density: (instrument multi-hot (B, n_programs), density level (B,)), see
      window_conditions; None = 'none' for both.
    program_temperature: temperature of the instrument head only (default: temperature). Lower = fewer switches to
      new instruments; at 0.9 the instrument choice drifted to dozens of rare instruments.
    top_p: nucleus sampling, keep the smallest set of values whose probability reaches top_p (after top_k).
    cfg_scale != 1 (needs a style): classifier-free guidance. The batch runs twice, with the style and with
      style 0 (none, which style dropout trained), and every head samples from uncond + s * (cond - uncond); both
      halves are fed the same note. Not combined with target_nps.
    eos_t1: EOS keeps its probability at temperature 1 (and outside top-k/top-p); the other pitches use the
      tempered distribution, rescaled to the remaining mass. Lower temperatures otherwise starve the rare EOS.
    Special tokens (pitch vocab 130): BOS is never sampled; EOS (end of piece) not before min_new notes. Rows keep
    running after their EOS (batch shapes stay fixed); cut them at the first EOS when writing MIDI.
    Nothing here syncs the GPU with the CPU inside the loop.
    """
    L = model.config.block_size
    slide = slide or L // 4
    assert 0 <= anchor and anchor + slide < L, "need anchor + slide < block_size"
    device = pitch.device
    Bo = pitch.size(0)  # rows asked for; with guidance the model runs 2 * Bo (conditional, then unconditional)
    guided = cfg_scale != 1.0 and style is not None
    assert not (guided and target_nps is not None), "guidance with a density target isn't supported"
    if guided:
        pitch, velocity, duration, delta_time = (torch.cat([t, t]) for t in (pitch, velocity, duration, delta_time))
    B, n0 = pitch.shape
    total = n0 + max_new_tokens
    # output buffers, filled in place (no growing torch.cat)
    out = [torch.zeros(B, total, dtype=torch.long, device=device) for _ in range(4)]
    for o, s in zip(out, (pitch, velocity, duration, delta_time)):
        o[:, :n0] = s

    # log(bin), with bins 0 and 1 unbiased: longer gaps get likelier, but "same onset" (bin 0 = chord notes)
    # keeps its odds against the shortest gap, so chords aren't broken up into arpeggios
    dt_shape = torch.log(torch.arange(model.config.delta_time_size, dtype=torch.float32,
                                      device=device).clamp(min=1))
    bias = torch.full((Bo, 1), float(dt_bias), device=device)
    target = None
    if target_nps is not None:
        target = torch.as_tensor(target_nps, dtype=torch.float32, device=device).reshape(-1, 1).expand(B, 1)

    specials = model.config.pitch_size > BOS_PITCH
    step_i = [0]
    if style is not None:
        style = torch.as_tensor(style, dtype=torch.long, device=device).reshape(-1).expand(Bo).contiguous()
        if guided:
            style = torch.cat([style, torch.zeros_like(style)])
    if cond is not None and guided:
        cond = tuple(torch.cat([c, c]) for c in cond)
    n_prog = model.config.n_programs
    sample_programs = bool(n_prog) and program is None
    fixed_program = torch.full((B, 1), int(program), dtype=torch.long, device=device) \
        if n_prog and not sample_programs else None
    prog_out = None
    if n_prog:  # the instrument of every note, like the 4 note streams
        prog_out = torch.zeros(B, total, dtype=torch.long, device=device)
        if programs is not None:
            prog_out[:, :n0] = torch.cat([programs, programs]) if guided else programs
        elif fixed_program is not None:
            prog_out[:, :n0] = fixed_program
    prog_mask = None
    if sample_programs and allowed_programs is not None:
        prog_mask = torch.full((n_prog,), -float('Inf'), device=device)
        prog_mask[torch.as_tensor(list(allowed_programs), device=device)] = 0.0

    def sample(logits, name=None):  # (B, 1, vocab) → (B, 1)
        if name == 'program' and not sample_programs:
            return fixed_program
        logits = logits[:, -1, :].float()
        if name == 'program' and prog_mask is not None:
            logits = logits + prog_mask
        if guided:
            logits = logits[Bo:] + cfg_scale * (logits[:Bo] - logits[Bo:])
        p_eos = None
        if name == 'pitch' and specials and eos_t1 and step_i[0] >= min_new:
            p_eos = F.softmax(logits, dim=-1)[:, [EOS_PITCH]]  # the model's own chance of ending here
        logits = logits / (program_temperature if name == 'program' and program_temperature else temperature)
        if name == 'pitch' and specials:
            logits[:, BOS_PITCH] = -float('Inf')
            if step_i[0] < min_new:
                logits[:, EOS_PITCH] = -float('Inf')
        if name == 'delta_time' and (dt_bias != 0.0 or target is not None):
            logits = logits + bias * dt_shape
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if top_p is not None:
            sorted_p, order = torch.sort(probs, dim=-1, descending=True)
            drop = sorted_p.cumsum(-1) - sorted_p > top_p  # mass before this value already reaches top_p
            probs = probs.scatter(-1, order, sorted_p.masked_fill(drop, 0.0))
        if p_eos is not None:
            probs[:, EOS_PITCH] = 0.0
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-12) * (1 - p_eos)
            probs[:, EOS_PITCH] = p_eos.squeeze(-1)
        nxt = torch.multinomial(probs, num_samples=1)
        return torch.cat([nxt, nxt]) if guided else nxt

    dtype = next(model.parameters()).dtype
    if device.type == 'cuda' and torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype()  # fp32 master weights under autocast: a bf16 cache is enough
    C = model.config.n_embd
    # a sequence that fits the context never refills the cache, so it only needs prompt + new slots (less memory,
    # and the one-note step attends over fewer slots)
    cache = KVCache(model.config.n_layer, B, model.config.n_head, min(L, total), C // model.config.n_head, device,
                    dtype)

    def prefill(n):
        """Refill the cache from the window ending at note n; returns (last hidden state, cache fill)."""
        if n <= L:
            window = [o[:, :n] for o in out]
        else:
            recent = L - slide - anchor
            window = [torch.cat([o[:, :anchor], o[:, n - recent:n]], dim=1) for o in out]
        w = window[0].size(1)
        if w == 1:
            cache.pos.fill_(0)  # a 1-note window goes through the single-step path at slot 0
        prog = None
        if prog_out is not None:
            prog = prog_out[:, :n] if n <= L else torch.cat([prog_out[:, :anchor],
                                                             prog_out[:, n - (L - slide - anchor):n]], dim=1)
        return model.encode(*window, style=style, program=prog, cache=cache, cond=cond)[:, [-1], :], w

    # the one-note step: static inputs -> static output, optionally captured as a CUDA graph
    step_in = [torch.zeros(B, 1, dtype=torch.long, device=device) for _ in range(4)]
    step_prog = torch.zeros(B, 1, dtype=torch.long, device=device) if n_prog else None  # the new note's instrument
    step_style = style  # constant during generation, so the graph can read it directly
    graph = None
    if cuda_graph and device.type == 'cuda':
        if str(device) not in _SIDE_STREAMS:
            _SIDE_STREAMS[str(device)] = torch.cuda.Stream(device)
        stream = _SIDE_STREAMS[str(device)]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm-up outside the graph (allocator, kernel selection)
            for _ in range(2):
                model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step_out = model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond)
        # the warm-up and capture wrote junk into slot 0; prefill below overwrites every slot it uses

    def step(new, pos, prog=None):
        for buf, t in zip(step_in, new):
            buf.copy_(t)
        if step_prog is not None:
            step_prog.copy_(prog)
        cache.pos.fill_(pos)
        if graph is not None:
            graph.replay()
            return step_out
        return model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond)

    h, filled = prefill(n0)
    for i in range(max_new_tokens):
        nxt = model.cascade.sample(h, sample)
        new = [nxt[name] for name in STREAM_ORDER]
        n = n0 + i
        for o, t in zip(out, new):
            o[:, n:n + 1] = t
        if prog_out is not None:
            prog_out[:, n:n + 1] = nxt['program'] if 'program' in nxt else fixed_program
        n += 1
        if target is not None:
            # per-row integral controller on log density, bias kept in [-2, 6]; stays on the GPU
            w = min(n, 64)
            secs = out[3][:, n - w:n].sum(dim=1, keepdim=True).float() * dt_seconds
            nps = torch.where(secs > 0, w / secs.clamp(min=1e-6), torch.full_like(secs, 1e3))
            bias.add_(0.05 * torch.log(nps.clamp(min=1e-3) / target)).clamp_(-2.0, 6.0)
        step_i[0] = i + 1
        if progress is not None:
            progress(i + 1)
        if i == max_new_tokens - 1:
            break
        if filled < L:
            h = step(new, filled, None if prog_out is None else prog_out[:, n - 1:n])
            filled += 1
        else:
            h, filled = prefill(n)

    result = tuple(o[:Bo] for o in out)
    if return_programs:
        result += (prog_out[:Bo] if prog_out is not None else torch.zeros_like(result[0]),)
    return result
