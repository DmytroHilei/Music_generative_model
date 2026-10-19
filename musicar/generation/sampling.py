"""Sampling from a GPT: batched generation with a KV cache and a CUDA-graph one-note step."""

import torch
from torch.nn import functional as F

from musicar.data.loader import BOS_PITCH, EOS_PITCH, window_conditions
from musicar.model.gpt import KVCache, STREAM_ORDER

# one side stream per device for the CUDA-graph warm-up and capture: cuBLAS keeps a workspace for every
# stream it has run on (~32 MB each, never freed), so a new stream per call grew the memory of every evaluation
_SIDE_STREAMS = {}



def with_bos(streams, programs=None):
    """A BOS note (pitch BOS_PITCH, other attributes 0, instrument 0) in front of every row of 4 prompt streams
    (B, T), T may be 0, and of their programs (B, T) if given. Returns (streams, programs)."""
    rows = streams[0].size(0)
    bos = [streams[0].new_full((rows, 1), BOS_PITCH if j == 0 else 0) for j in range(4)]
    streams = tuple(torch.cat([b, s], 1) for b, s in zip(bos, streams))
    return streams, None if programs is None else torch.cat([programs.new_zeros(rows, 1), programs], 1)


def prompt_inputs(model, tokens, programs, band=False, density=False):
    """generate()'s inputs for prompt rows: tokens (B, T, 4) and programs (B, T) on the model's device -> (4 streams,
    programs, cond). Models with BOS/EOS get BOS in front (the prompts start at their piece's beginning). Conditioned
    models (cond_inst / n_density) get the prompt's own instrument set if `band` and its density level if `density`,
    else 'none'; cond is None for unconditioned models."""
    cfg = model.config
    streams = tuple(tokens[:, :, j] for j in range(4))
    if cfg.pitch_size > BOS_PITCH:
        streams, programs = with_bos(streams, programs)
    if not (cfg.cond_inst or cfg.n_density):
        return streams, programs, None
    inst, dens = window_conditions(streams[0], streams[3], programs, cfg.n_programs, cfg.n_density or 16)
    return streams, programs, (inst if band else torch.zeros_like(inst), dens if density else torch.zeros_like(dens))


@torch.no_grad()
def generate(model, pitch, velocity, duration, delta_time, max_new_tokens, temperature=0.85, top_k=None,
             anchor=0, slide=None, progress=None, dt_bias=0.0, target_nps=None, dt_seconds=0.02,
             cuda_graph=True, style=None, min_new=0, program=0, top_p=None, cfg_scale=1.0,
             eos_t1=False, programs=None, allowed_programs=None, return_programs=False, program_temperature=None,
             cond=None, text=None, sections=None):
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
    text: models with text_dim: caption vectors (text_dim,) or (B, text_dim) from musicar.text.encoder.TextEncoder; text_prefix
      models: token states (L, text_dim) or (B, L, text_dim) (encode(..., tokens=True)), whose prefix occupies the
      first text_prefix cache slots and is rebuilt with every refill.
    sections: section_len models: token states of the section captions, (n, L, text_dim) or (n, B, L, text_dim): one
      per section of section_len notes counted from the first prompt note (BOS), the last one repeated for the rest;
      all-zero = the null caption, None = all null. Section k's caption comes before note max(0, k * section_len -
      section_lead) like in training; refilled windows start with the caption in force. Guidance drops them too.
    cfg_scale != 1 (needs a style, a text or sections): classifier-free guidance. The batch runs twice, with the style / text
      and with style 0 / no text (none, which style and caption dropout trained), and every head samples from
      uncond + s * (cond - uncond); both halves are fed the same note. Not combined with target_nps.
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
    guided = cfg_scale != 1.0 and (style is not None or text is not None or sections is not None)
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
    k_pre = model.config.text_prefix  # cache slots of the T2 prefix in front of the notes
    if text is not None:
        text = torch.as_tensor(text, dtype=torch.float32, device=device)
        text = text.reshape((-1,) + text.shape[-2 if k_pre else -1:])
        text = text.expand(Bo, *text.shape[1:]).contiguous()
        if guided:
            text = torch.cat([text, torch.zeros_like(text)])
    S = model.config.section_len
    sec_vecs = None
    if S:
        assert anchor == 0, 'section captions: no anchor'
        if sections is None:
            sections = torch.zeros(1, 1, model.config.text_dim)
        sections = torch.as_tensor(sections, dtype=torch.float32, device=device)
        sections = sections.reshape(sections.size(0), -1, *sections.shape[-2:])
        sections = sections.expand(-1, Bo, -1, -1).contiguous()
        if guided:
            sections = torch.cat([sections, torch.zeros_like(sections)], 1)
        sec_vecs = model.section_vectors(sections)  # (n, B, slots, C)
        sec_at = lambda k: max(0, k * S - model.config.section_lead)  # noqa: E731
        sec_vec = lambda k: sec_vecs[min(k, sec_vecs.size(0) - 1)]  # noqa: E731
        sec_starts = {}  # note index -> section whose caption comes before it
        for k in range(total // S + 2):
            sec_starts[sec_at(k)] = k  # k * S - lead clamps to 0 for the first sections: the latest wins

        def sec_now(n):
            """The section whose caption is in force for predicting note n."""
            return max(k for k in range(total // S + 2) if sec_at(k) <= n)
    prefix_sec = S and model.config.section_mode == 'prefix'
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
    sec_slots = model.config.section_slots * (2 + -(-L // S)) if prefix_sec else 0  # inserted section vectors
    cache = KVCache(model.config.n_layer, B, model.config.n_head, min(L, total) + k_pre + sec_slots,
                    C // model.config.n_head, device, dtype)
    extra = [0]  # cache slots taken by inserted section vectors

    def prefill(n):
        """Refill the cache from the window ending at note n; returns (last hidden state, cache fill)."""
        if n <= L:
            window = [o[:, :n] for o in out]
        else:
            recent = L - slide - anchor
            window = [torch.cat([o[:, :anchor], o[:, n - recent:n]], dim=1) for o in out]
        w = window[0].size(1)
        if w == 1 and not k_pre:
            cache.pos.fill_(0)  # a 1-note window goes through the single-step path at slot 0
        prog = None
        if prog_out is not None:
            prog = prog_out[:, :n] if n <= L else torch.cat([prog_out[:, :anchor],
                                                             prog_out[:, n - (L - slide - anchor):n]], dim=1)
        kw = dict(step=False) if k_pre else {}  # prefix + window: a prefill even for a 1-note window
        if S:  # the caption in force at the window start, then those arriving up to note n (incl. before note n)
            a = n - w
            ks = [sec_now(a)] + [k for k in range(total // S + 2) if a < sec_at(k) <= n]
            pos = torch.tensor([0] + [sec_at(k) - a for k in ks[1:]], device=device).expand(B, -1)
            kw['sections'] = (torch.stack([sec_vec(k) for k in ks], 1), pos)
            extra[0] = model.config.section_slots * len(ks) if prefix_sec else 0
        return model.encode(*window, style=style, program=prog, cache=cache, cond=cond, text=text,
                            **kw)[:, [-1], :], w

    # the one-note step: static inputs -> static output, optionally captured as a CUDA graph
    step_in = [torch.zeros(B, 1, dtype=torch.long, device=device) for _ in range(4)]
    step_prog = torch.zeros(B, 1, dtype=torch.long, device=device) if n_prog else None  # the new note's instrument
    step_style = style  # constant during generation, so the graph can read it directly
    # 'xattn' sections: the caption of the note being predicted, copied in before every step
    step_sec = (sec_vecs[0][:, None].clone(), None) if S and not prefix_sec else None
    graph = None
    if cuda_graph and device.type == 'cuda':
        if str(device) not in _SIDE_STREAMS:
            _SIDE_STREAMS[str(device)] = torch.cuda.Stream(device)
        stream = _SIDE_STREAMS[str(device)]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm-up outside the graph (allocator, kernel selection)
            for _ in range(2):
                model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond, text=text,
                             sections=step_sec)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step_out = model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond, text=text,
                                    sections=step_sec)
        # the warm-up and capture wrote junk into slot 0; prefill below overwrites every slot it uses

    def step(new, pos, prog=None, n=None):
        """The note at window index pos (cache slot pos + prefix + inserted section vectors); n = the index of the
        note it predicts (picks the 'xattn' section caption)."""
        for buf, t in zip(step_in, new):
            buf.copy_(t)
        if step_prog is not None:
            step_prog.copy_(prog)
        if step_sec is not None:
            step_sec[0].copy_(sec_vec(sec_now(n))[:, None])
        cache.pos.fill_(pos + k_pre + extra[0])
        if graph is not None:
            graph.replay()
            return step_out
        return model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond, text=text,
                            sections=step_sec)

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
            h = step(new, filled, None if prog_out is None else prog_out[:, n - 1:n], n)
            filled += 1
            if prefix_sec and n in sec_starts and sec_starts[n] == sec_now(n) and n > 0:
                h = model.encode_chunk(sec_vec(sec_starts[n]), cache, k_pre + filled + extra[0])
                extra[0] += model.config.section_slots
        else:
            h, filled = prefill(n)

    result = tuple(o[:Bo] for o in out)
    if return_programs:
        result += (prog_out[:Bo] if prog_out is not None else torch.zeros_like(result[0]),)
    return result
