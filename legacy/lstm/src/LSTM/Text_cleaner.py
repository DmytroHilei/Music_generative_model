import re
from collections import defaultdict

SECTION_MAP = {
    "verse": "<VERSE>",
    "chorus": "<CHORUS>",
    "interlude": "<INTERLUDE>",
    "bridge": "<BRIDGE>",
    "intro": "<INTRO>",
    "outro": "<OUTRO>"
}
SECTION_TOKENS = set(SECTION_MAP.values())

def is_chord(token: str) -> bool:
    return re.match(r"^[A-G][#b]?(m|maj|min|sus|dim|aug|7|m7|maj7)?$", token) is not None

def tokens_to_bars(tokens):
    """Convert token stream into list of bars; each bar is a tuple of chords."""
    bars = []
    cur = []
    for t in tokens:
        if t in SECTION_TOKENS:
            # ignore existing sections here; this function is for raw tokens
            continue
        if t == "<BAR>":
            if cur:
                bars.append(tuple(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        bars.append(tuple(cur))
    return bars

def bars_to_tokens(bars):
    """Flatten bars back to tokens + <BAR>."""
    out = []
    for bar in bars:
        out.extend(list(bar))
        out.append("<BAR>")
    return out

def find_repeating_block(bars, min_len=4, max_len=8):
    """
    Find a repeating block of bars (length 4..8).
    Returns (best_len, best_block_tuple, occurrences_positions).
    """
    best = None  # (score, L, block, positions)
    n = len(bars)
    for L in range(min_len, max_len + 1):
        seen = defaultdict(list)
        for i in range(0, n - L + 1):
            block = tuple(bars[i:i+L])
            seen[block].append(i)
        for block, pos in seen.items():
            if len(pos) >= 2:
                # score: occurrences * length, small penalty if overlaps are massive
                score = len(pos) * L
                if best is None or score > best[0]:
                    best = (score, L, block, pos)
    if best is None:
        return None
    _, L, block, pos = best
    return L, block, pos

def mark_sections_from_bars(bars):
    """
    Heuristic sectioning:
    - detect main repeating block as CHORUS
    - INTRO = bars before first chorus (clamped to 2..4)
    - OUTRO = last 2..4 bars if similar to INTRO or CHORUS tail
    - everything else = VERSE, and short unique chunks between chorus occurrences = BRIDGE
    """
    n = len(bars)
    if n < 8:
        # just INTRO + VERSE
        intro_len = min(2, n)
        return (
            [("<INTRO>", bars[:intro_len])] +
            ([("<VERSE>", bars[intro_len:])] if intro_len < n else [])
        )

    rep = find_repeating_block(bars, min_len=4, max_len=8)
    if rep is None:
        intro_len = min(4, n)
        return [("<INTRO>", bars[:intro_len]), ("<VERSE>", bars[intro_len:])]

    L, chorus_block, pos = rep

    # take first occurrence as "chorus anchor"
    first_ch = pos[0]

    # INTRO heuristics: 2..4 bars before first chorus
    intro_len = min(4, max(2, first_ch))
    intro = bars[:intro_len]

    # build a set of chorus-start positions, but avoid overlapping duplicates
    chorus_starts = []
    last = -10**9
    for p in pos:
        if p >= last + L:
            chorus_starts.append(p)
            last = p

    # Now segment timeline
    segments = []
    cur_i = 0

    # add INTRO
    segments.append(("<INTRO>", intro))
    cur_i = intro_len

    def add_chunk_as(chunk_bars, default="<VERSE>"):
        if not chunk_bars:
            return
        # small unique chunks between choruses -> BRIDGE
        if len(chunk_bars) <= 4:
            segments.append(("<BRIDGE>", chunk_bars))
        else:
            segments.append((default, chunk_bars))

    for cs in chorus_starts:
        # add verse/bridge chunk before chorus (from cur_i to cs)
        if cs > cur_i:
            add_chunk_as(bars[cur_i:cs], default="<VERSE>")
        # add chorus itself
        segments.append(("<CHORUS>", bars[cs:cs+L]))
        cur_i = cs + L

    # tail after last chorus
    tail = bars[cur_i:]

    # OUTRO heuristics: if tail exists, mark last 2..4 bars as OUTRO
    if tail:
        outro_len = min(4, max(2, len(tail)))
        main_tail = tail[:-outro_len]
        outro = tail[-outro_len:]
        if main_tail:
            add_chunk_as(main_tail, default="<VERSE>")
        segments.append(("<OUTRO>", outro))

    return segments

def apply_sectioning(tokens):
    """If there are no explicit sections, infer them from chord bars."""
    has_sections = any(t in SECTION_TOKENS for t in tokens)
    if has_sections:
        return tokens  # trust source

    bars = tokens_to_bars(tokens)
    segments = mark_sections_from_bars(bars)

    out = []
    for tag, seg_bars in segments:
        if seg_bars:
            out.append(tag)
            out.extend(bars_to_tokens(seg_bars))
    return out

def parse_song(filepath):
    output_tokens = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Detect section if present
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1].lower()
                for key in SECTION_MAP:
                    if key in name:
                        output_tokens.append(SECTION_MAP[key])
                        break
                continue

            tokens = line.split()
            chord_line = [t for t in tokens if is_chord(t)]
            if chord_line:
                output_tokens.extend(chord_line)
                output_tokens.append("<BAR>")

    # If no explicit tags in source, infer them
    output_tokens = apply_sectioning(output_tokens)
    return output_tokens


tokens = parse_song("Individual_Songs/FewGoodAmericanOrBritish/Zombie.txt")

with open("Dataset for LSTM.txt", "a") as f:
    f.write(" ".join(tokens))