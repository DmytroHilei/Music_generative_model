"""Sampling from a GPT: batched generation with a KV cache and a CUDA-graph one-note step."""

import torch
from torch.nn import functional as F

from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM, EOS_PITCH, progress_level, remain_level, window_conditions
from musicar.generation.repetition import RepeatPenalty
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


def guidance_scale(i, cfg_scale, cfg_notes=0, cfg_end=1.0, cfg_linear=False):
    """The guidance scale for new note i (0 = the first generated note) under the schedule: cfg_notes 0 = cfg_scale
    throughout; else cfg_scale for the first cfg_notes notes and cfg_end after (a step: guidance on the opening only,
    TokenSynth), or with cfg_linear a straight line from cfg_scale at note 0 to cfg_end at note cfg_notes (an anneal
    like MAGNeT's, or a warm-up like Aria's when cfg_end > cfg_scale)."""
    if not cfg_notes:
        return cfg_scale
    if cfg_linear:
        return cfg_scale + (cfg_end - cfg_scale) * min(1.0, i / cfg_notes)
    return cfg_scale if i < cfg_notes else cfg_end


def min_p_filter(logits, min_p):
    """min-p truncation (Nguyen et al. 2024): values whose probability is below min_p x the largest one are dropped
    (-inf), on tempered logits before the softmax."""
    probs = F.softmax(logits, dim=-1)
    keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(~keep, -float('Inf'))


@torch.no_grad()
def generate(model, pitch, velocity, duration, delta_time, max_new_tokens, temperature=0.85, top_k=None,
             anchor=0, slide=None, progress=None, dt_bias=0.0, target_nps=None, dt_seconds=0.02,
             cuda_graph=True, style=None, min_new=0, program=0, top_p=None, cfg_scale=1.0,
             eos_t1=False, programs=None, allowed_programs=None, return_programs=False, program_temperature=None,
             cond=None, text=None, sections=None, text_neg=None, sections_neg=None, cfg_section=None,
             cfg_boost=0.0, cfg_boost_notes=32, length_seconds=None, eos_ramp=0.0, length_stop=0.0,
             dry_multiplier=0.0, dry_base=1.15, dry_allowed=16, dry_key='notes', cfg_notes=0, cfg_end=1.0,
             cfg_linear=False, min_p=None):
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
    text_neg / sections_neg: with guidance, what the unconditional half gets instead of no caption (shaped like text /
      sections, same number of sections; all-zero entries = null): e.g. the file caption in both halves so only the
      section captions are guided, or the previous section caption after a switch (contrastive guidance: the new
      caption against the old one instead of against none).
    cfg_section (section_len models, with text and sections): separate scales for the file and the section captions
      (InstructPix2Pix-style double guidance). The batch runs three times: file + section captions (c), file caption +
      null sections (f; sections_neg instead of null when given), no captions (u; text_neg instead of none when
      given), and every head samples from u + cfg_scale * (f - u) + cfg_section * (c - f).
    cfg_boost: a guidance schedule (YuE: higher at a section start): the section scale (cfg_section, else cfg_scale)
      is raised by cfg_boost at each section's first note, falling linearly to its base over cfg_boost_notes notes.
    cfg_notes / cfg_end / cfg_linear: a guidance schedule over the generated notes (guidance_scale): the (section)
      scale is cfg_scale for the first cfg_notes new notes and cfg_end (default 1 = off) after; cfg_linear: a line
      from cfg_scale to cfg_end over cfg_notes notes (anneal, or warm-up with cfg_end > cfg_scale). Guidance runs
      when either end of the schedule is not 1. cfg_boost adds to the scheduled scale.
    min_p: values below min_p x the most likely one are dropped (after temperature and top_k, before top_p; Aria uses
      0.035). A truncation that adapts to how peaked the distribution is, unlike top_p.
    length_seconds: the length of the piece in seconds (a float, or one per row), counted from the first prompt note.
      n_remain models: every note gets the level of length_seconds minus its onset (0 s past the end), as the
      seconds-to-the-end input was trained (n_progress: and the level of onset / length_seconds). None = 'none'.
      Any model with BOS/EOS: the time limit for eos_ramp / length_stop.
    eos_ramp: models with EOS and length_seconds: + eos_ramp x (seconds past length_seconds at the latest note) on the
      EOS logit (before temperature), so a row that runs over ends soon after its length.
    length_stop: > 0: rows still going at length_stop x length_seconds (the latest note's onset) get EOS (a hard limit;
      generate.py fades such rows out).
    eos_t1: EOS keeps its probability at temperature 1 (and outside top-k/top-p); the other pitches use the
      tempered distribution, rescaled to the remaining mass. Lower temperatures otherwise starve the rare EOS.
    dry_multiplier > 0: repetition penalty on verbatim copies of pitched passages (musicar/generation/repetition.py,
      the DRY sampler on chords = the pitched notes of one onset as a set, dry_key 'notes' (instrument + pitch) or
      'pitch_class'): when the last m >= dry_allowed chords repeat an earlier passage, the pitches that would continue
      it lose dry_multiplier * dry_base^(m - dry_allowed) logits (after guidance, before temperature). Prompt notes
      count as earlier passages.
    Special tokens (pitch vocab 130): BOS is never sampled; EOS (end of piece) not before min_new notes. Rows keep
    running after their EOS (batch shapes stay fixed); cut them at the first EOS when writing MIDI.
    Nothing here syncs the GPU with the CPU inside the loop.
    """
    L = model.config.block_size
    slide = slide or L // 4
    assert 0 <= anchor and anchor + slide < L, "need anchor + slide < block_size"
    device = pitch.device
    Bo = pitch.size(0)  # rows asked for; with guidance the model runs 2 * Bo (conditional, then unconditional)
    double = cfg_section is not None
    assert not double or (text is not None and model.config.section_len), 'cfg_section needs text and sections'
    guided = double or ((cfg_scale != 1.0 or cfg_boost or (cfg_notes and cfg_end != 1.0)) and
                        (style is not None or text is not None or sections is not None))
    n_br = 3 if double else 2 if guided else 1  # batch copies: conditional, (file only,) unconditional
    assert not (guided and target_nps is not None), "guidance with a density target isn't supported"
    if programs is not None:  # prompt drum notes as in training: duration 0 (MIDI-tokenized prompts keep theirs)
        duration = torch.where(programs == DRUM_PROGRAM, 0, duration)
    if guided:
        pitch, velocity, duration, delta_time = (t.repeat(n_br, 1) for t in (pitch, velocity, duration, delta_time))
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
            style = torch.cat([style] * (n_br - 1) + [torch.zeros_like(style)])
    if cond is not None and guided:
        cond = tuple(c.repeat(n_br, *[1] * (c.dim() - 1)) for c in cond)
    k_pre = model.config.text_prefix  # cache slots of the T2 prefix in front of the notes
    if text is not None:
        text = torch.as_tensor(text, dtype=torch.float32, device=device)
        text = text.reshape((-1,) + text.shape[-2 if k_pre else -1:])
        text = text.expand(Bo, *text.shape[1:]).contiguous()
        if guided:
            neg = torch.zeros_like(text)
            if text_neg is not None:
                neg = torch.as_tensor(text_neg, dtype=torch.float32, device=device)
                neg = neg.reshape(-1, *text.shape[1:]).expand_as(text)
            text = torch.cat([text] * (n_br - 1) + [neg])
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
            neg = torch.zeros_like(sections)
            if sections_neg is not None:
                neg = torch.as_tensor(sections_neg, dtype=torch.float32, device=device)
                neg = neg.reshape(neg.size(0), -1, *neg.shape[-2:]).expand(-1, Bo, -1, -1)
                assert neg.shape == sections.shape, (neg.shape, sections.shape)
            sections = torch.cat([sections, neg] + ([torch.zeros_like(neg)] if double else []), 1)
        sec_vecs = model.section_vectors(sections)  # (n, B, slots, C)
        sec_at = lambda k: max(0, k * S - model.config.section_lead)  # noqa: E731
        sec_vec = lambda k: sec_vecs[min(k, sec_vecs.size(0) - 1)]  # noqa: E731
        R = model.config.section_repeat or S
        n_sec = total // S + 2
        # (note index, section) of every caption arrival: each section's first, then every R notes until the next
        # section's (section_repeat), as the loader lays them out
        sec_arrive = sorted((sec_at(k) + j * R, k) for k in range(n_sec) for j in range(-(-S // R))
                            if j == 0 or k + 1 >= n_sec or sec_at(k) + j * R < sec_at(k + 1))
        sec_starts = {}  # note index -> section whose caption comes before it
        for p, k in sec_arrive:
            sec_starts[p] = k  # k * S - lead clamps to 0 for the first sections: the latest wins

        def sec_now(n):
            """The section whose caption is in force for predicting note n."""
            return max((p, k) for p, k in sec_arrive if p <= n)[1]
    prefix_sec = S and model.config.section_mode == 'prefix'
    n_remain, n_progress = model.config.n_remain, model.config.n_progress
    length = None  # (B, 1) requested seconds; onsets come from the cumulative delta_time of out[3]
    if length_seconds is not None:
        length = torch.as_tensor(length_seconds, dtype=torch.float32, device=device).reshape(-1, 1).expand(Bo, 1)
        length = length.repeat(n_br, 1)
    length_levels = bool(n_remain) and length is not None

    def remain_at(onset):
        """length levels of notes at these onsets (seconds, (B, n)): remain levels; (B, n, 2) with the progress
        level for n_progress models."""
        level = remain_level((length - onset).clamp(min=0), n_remain)
        if n_progress:
            level = torch.stack([level, progress_level(onset, length.expand_as(onset), n_progress)], -1)
        return level
    n_prog = model.config.n_programs
    sample_programs = bool(n_prog) and program is None
    fixed_program = torch.full((B, 1), int(program), dtype=torch.long, device=device) \
        if n_prog and not sample_programs else None
    prog_out = None
    if n_prog:  # the instrument of every note, like the 4 note streams
        prog_out = torch.zeros(B, total, dtype=torch.long, device=device)
        if programs is not None:
            prog_out[:, :n0] = programs.repeat(n_br, 1)
        elif fixed_program is not None:
            prog_out[:, :n0] = fixed_program
    rep = None
    if dry_multiplier > 0:
        rep = RepeatPenalty(Bo, total, dry_multiplier, dry_base, dry_allowed, device, key=dry_key)
        rep.prime(out[0][:Bo, :n0], out[3][:Bo, :n0],
                  prog_out[:Bo, :n0] if prog_out is not None else torch.zeros_like(out[0][:Bo, :n0]))
        if fixed_program is not None:
            rep.prog.fill_(int(program))
    prog_mask = None
    if sample_programs and allowed_programs is not None:
        prog_mask = torch.full((n_prog,), -float('Inf'), device=device)
        prog_mask[torch.as_tensor(list(allowed_programs), device=device)] = 0.0

    note_program = [None]  # the instrument just sampled for the note being decided (the cascade samples it first)
    base_scale = cfg_section if double else cfg_scale
    scale = [base_scale]  # the section (or only) guidance scale for the note being decided (cfg_boost schedule)

    def set_scale(n):
        """The guidance scale for predicting note n: the schedule's scale for new note n - n0 (guidance_scale),
        + cfg_boost at a section's first note, linearly back to the base over cfg_boost_notes notes."""
        base = guidance_scale(n - n0, base_scale, cfg_notes, cfg_end, cfg_linear)
        if cfg_boost and S:
            since = n - sec_at(sec_now(n))
            base += cfg_boost * max(0.0, 1.0 - since / cfg_boost_notes)
        scale[0] = base

    def sample(logits, name=None):  # (B, 1, vocab) → (B, 1)
        if name == 'program' and not sample_programs:
            note_program[0] = fixed_program
            return fixed_program
        logits = logits[:, -1, :].float()
        if double:
            c, f, u = logits.split(Bo)
            logits = u + cfg_scale * (f - u) + scale[0] * (c - f)
        elif guided:
            logits = logits[Bo:] + scale[0] * (logits[:Bo] - logits[Bo:])
        if name == 'program' and prog_mask is not None:  # after the guidance mix: -inf - -inf there would be NaN
            logits = logits + prog_mask
        if name == 'pitch' and rep is not None:
            logits = rep.penalize(logits)
        if name == 'pitch' and specials and eos_ramp and length is not None:  # running over: EOS ever more likely
            over = (step_onset[:Bo] - length[:Bo]).clamp(min=0)  # (Bo, 1) seconds past the length
            logits = logits + eos_ramp * over * F.one_hot(torch.tensor(EOS_PITCH, device=device), logits.size(-1))
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
        if min_p:
            logits = min_p_filter(logits, min_p)
        probs = F.softmax(logits, dim=-1)
        if top_p is not None:
            sorted_p, order = torch.sort(probs, dim=-1, descending=True)
            drop = sorted_p.cumsum(-1) - sorted_p > top_p  # mass before this value already reaches top_p
            probs = probs.scatter(-1, order, sorted_p.masked_fill(drop, 0.0))
        if p_eos is not None:
            probs[:, EOS_PITCH] = 0.0
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-12) * (1 - p_eos)
            probs[:, EOS_PITCH] = p_eos.squeeze(-1)
        if name == 'pitch' and specials and length_stop and length is not None:  # the hard limit
            stop = step_onset[:Bo] >= length_stop * length[:Bo]
            probs = torch.where(stop, F.one_hot(torch.tensor(EOS_PITCH, device=device), probs.size(-1)).to(probs.dtype),
                                probs)
        nxt = torch.multinomial(probs, num_samples=1)
        if rep is not None and name in ('delta_time', 'program'):  # what the pitch penalty compares with
            (rep.dt if name == 'delta_time' else rep.prog).copy_(nxt[:, 0])
        nxt = nxt.repeat(n_br, 1)
        if name == 'program':
            note_program[0] = nxt
        elif name == 'duration' and note_program[0] is not None:
            # drum notes are trained with duration 0 as input (and as the cascade condition: masked target -> 0);
            # the duration head never learned them, so a sampled value would feed the model unseen inputs
            nxt = torch.where(note_program[0] == DRUM_PROGRAM, 0, nxt)
        return nxt

    dtype = next(model.parameters()).dtype
    if device.type == 'cuda' and torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype()  # fp32 master weights under autocast: a bf16 cache is enough
    C = model.config.n_embd
    # a sequence that fits the context never refills the cache, so it only needs prompt + new slots (less memory,
    # and the one-note step attends over fewer slots)
    # inserted section vectors: the caption in force + the arrivals in the sections a window touches
    sec_slots = model.config.section_slots * (1 + (1 + -(-L // S)) * -(-S // R)) if prefix_sec else 0
    cache = KVCache(model.config.n_layer, B, model.config.kv_heads, min(L, total) + k_pre + sec_slots,
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
        if length is not None:  # the onsets of the window's notes, counted from the first note of the sequence
            onset = out[3][:, :n].cumsum(1).float() * dt_seconds
            idx = torch.arange(n, device=device) if n <= L else torch.cat(
                [torch.arange(anchor, device=device), torch.arange(n - (L - slide - anchor), n, device=device)])
            if length_levels:
                kw['remain'] = remain_at(onset[:, idx])
            step_onset.copy_(onset[:, n - 1:n])
        if S:  # the caption in force at the window start, then those arriving up to note n (incl. before note n)
            a = n - w
            later = [(p, k) for p, k in sec_arrive if a < p <= n and sec_starts[p] == k]
            ks = [sec_now(a)] + [k for _, k in later]
            pos = torch.tensor([0] + [p - a for p, _ in later], device=device).expand(B, -1)
            kw['sections'] = (torch.stack([sec_vec(k) for k in ks], 1), pos)
            extra[0] = model.config.section_slots * len(ks) if prefix_sec else 0
        return model.encode(*window, style=style, program=prog, cache=cache, cond=cond, text=text,
                            **kw)[:, [-1], :], w

    # the one-note step: static inputs -> static output, optionally captured as a CUDA graph
    step_in = [torch.zeros(B, 1, dtype=torch.long, device=device) for _ in range(4)]
    step_prog = torch.zeros(B, 1, dtype=torch.long, device=device) if n_prog else None  # the new note's instrument
    step_style = style  # constant during generation, so the graph can read it directly
    # length control: the onset of the latest note (seconds) and the step's remain level, a static buffer
    step_onset = torch.zeros(B, 1, device=device)
    step_remain = torch.zeros(B, 1, *((2,) if n_progress else ()), dtype=torch.long, device=device) \
        if length_levels else None
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
                             sections=step_sec, remain=step_remain)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step_out = model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond, text=text,
                                    sections=step_sec, remain=step_remain)
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
        if length is not None:  # the new note's onset = the previous one's + its delta_time
            step_onset.add_(new[3].float() * dt_seconds)
        if step_remain is not None:
            step_remain.copy_(remain_at(step_onset))
        cache.pos.fill_(pos + k_pre + extra[0])
        if graph is not None:
            graph.replay()
            return step_out
        return model.encode(*step_in, style=step_style, program=step_prog, cache=cache, cond=cond, text=text,
                            sections=step_sec, remain=step_remain)

    h, filled = prefill(n0)
    for i in range(max_new_tokens):
        set_scale(n0 + i)
        nxt = model.cascade.sample(h, sample)
        new = [nxt[name] for name in STREAM_ORDER]
        n = n0 + i
        for o, t in zip(out, new):
            o[:, n:n + 1] = t
        if prog_out is not None:
            prog_out[:, n:n + 1] = nxt['program'] if 'program' in nxt else fixed_program
        if rep is not None:
            rep.push(new[0][:Bo, 0], new[3][:Bo, 0], prog_out[:Bo, n] if prog_out is not None else rep.prog)
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
