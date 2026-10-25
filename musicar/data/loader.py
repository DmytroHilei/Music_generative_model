from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import os
import random

import numpy as np
import pandas as pd
import pretty_midi
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


BOS_PITCH, EOS_PITCH = 128, 129       # special "notes" when special_tokens=True (pitch vocab 130)
DRUM_PROGRAM = 128                    # programs.u8 value of drum notes (GM programs are 0-127)
DEFAULT_CSV_STYLE = {'combined': 'classical'}   # style of CSV sources without a style/artist column
DEFAULT_STORE_STYLE = 'none'


def load_styles(path='data/styles.json'):
    return json.loads(Path(path).read_text())


def density_level(notes_per_s, n_density=16):
    """Notes per second -> level 1..n_density-1 (two per octave from 0.5 to 64 notes/s; 0 = none)."""
    nps = torch.as_tensor(notes_per_s, dtype=torch.float32)
    return (1 + (2 * torch.log2(nps.clamp(min=1e-3)) + 2).floor().clamp(0, n_density - 2)).long()


REMAIN_END, REMAIN_MAX = 0.5, 1200.0  # seconds: the 'at the end' level, the start of the last level


def remain_level(seconds, n_remain=33):
    """Seconds to the end of the piece -> level 1..n_remain-1 (MusicConfig.n_remain): 1 = under REMAIN_END (the
    last notes), then n_remain - 2 log steps from REMAIN_END to REMAIN_MAX (~28% each at 33 levels; longer = the last
    level); 0 = none."""
    s = torch.as_tensor(seconds, dtype=torch.float32)
    n = n_remain - 2
    step = torch.log(s.clamp(min=REMAIN_END) / REMAIN_END) / math.log(REMAIN_MAX / REMAIN_END)
    return torch.where(s < REMAIN_END, 1, 2 + (step * n).floor().clamp(0, n - 1)).long()


def progress_level(elapsed, total, n_progress=17):
    """Share of the piece played at each note's onset (elapsed / total seconds) -> level 1..n_progress-1 in equal steps
    (MusicConfig.n_progress; at or past the end = the last level); 0 = none."""
    n = n_progress - 1
    r = torch.as_tensor(elapsed, dtype=torch.float32) / torch.as_tensor(total, dtype=torch.float32).clamp(min=REMAIN_END)
    return 1 + (r * n).floor().clamp(0, n - 1).long()


def elapsed_seconds(pitch, delta_time, head, dt_seconds=0.02):
    """Seconds from the start of each row's piece to its onset, for the rows (R,) of a window: `head` (the onset of
    the window's first row in its piece) plus the delta_times since, restarting at 0 on a BOS (packed windows). On
    every row of a piece elapsed + remaining_seconds = the piece's length."""
    steps = delta_time.double() * dt_seconds
    steps[0] = head
    cum = steps.cumsum(0)
    bos = pitch == BOS_PITCH
    latest = torch.where(bos, torch.arange(len(pitch)), -1).cummax(0).values  # the last BOS at or before each row
    return (cum - torch.where(latest >= 0, cum[latest.clamp(min=0)], 0.0)).float()


def remaining_seconds(pitch, delta_time, tail, dt_seconds=0.02):
    """Seconds from each row's onset to the end of its piece, for the rows (R,) of a window: the delta_times of the
    rows after it up to the next EOS (0 at an EOS, and in the padding after it), plus `tail` (the seconds of the piece
    after the window's last row) where no EOS follows. Packed windows restart after each EOS."""
    R = len(pitch)
    c = torch.zeros(R + 1, dtype=torch.float64)
    c[:R - 1] = delta_time[1:].double() * dt_seconds  # c[r] = the step from row r to row r + 1
    c[R - 1] = tail
    S = c.flip(0).cumsum(0).flip(0)  # S[r] = sum of c[r:], S[R] = 0
    idx = torch.full((R,), R, dtype=torch.long)
    eos = torch.nonzero(pitch == EOS_PITCH).flatten()
    idx[eos] = eos
    nxt = idx.flip(0).cummin(0).values.flip(0)  # the first EOS at or after each row (R = none)
    return (S[:R] - S[nxt]).float()


def section_change_scores(store, section_len, pad, version='v1'):
    """musicar.text.captions.section_change of every section of every file of a token store vs the section before it
    (int8, files in store order, ceil((notes + 2 * pad) / section_len) sections each; 0 for the first one and those
    with < section_len / 2 notes); sections as in
    NoteWindowDataset._section_notes. Cached in <store>/section_changes_<section_len>_<pad>[_<version>].npy (~1 min
    for 168M notes); version: of section_attributes ('v1' keeps the old cache name)."""
    from musicar.text.captions import section_attributes, section_change
    cache = Path(store) / f"section_changes_{section_len}_{pad}{'' if version == 'v1' else '_' + version}.npy"
    if cache.exists():
        return np.load(cache)
    offsets = np.load(Path(store) / "offsets.npy")
    n_total = int(offsets[-1])
    tokens = np.memmap(Path(store) / "tokens.u16", dtype=np.uint16, mode="r", shape=(n_total, 4))
    progs = np.memmap(Path(store) / "programs.u8", dtype=np.uint8, mode="r", shape=(n_total,)) \
        if (Path(store) / "programs.u8").exists() else None
    scores = []
    for f in tqdm(range(len(offsets) - 1), desc=f"Section changes of {Path(store).name}"):
        start, n = int(offsets[f]), int(offsets[f + 1] - offsets[f])
        tok = np.asarray(tokens[start:start + n], dtype=np.int64)
        prog = np.asarray(progs[start:start + n], dtype=np.int64) if progs is not None else np.zeros(n, np.int64)
        prev = None
        for k in range(-(-(n + 2 * pad) // section_len)):
            lo = max(k * section_len - pad, 0)
            hi = max(min((k + 1) * section_len - pad, n), lo)
            attrs = section_attributes(tok[lo:hi], prog[lo:hi], version)
            # a file's short last section (or a short file) has too few notes for its attributes to mean much
            scores.append(0 if prev is None or hi - lo < section_len // 2 else section_change(prev, attrs))
            prev = attrs
    scores = np.array(scores, dtype=np.int8)
    np.save(cache, scores)
    return scores


def window_conditions(pitch, delta_time, program, n_programs, n_density, drop_p=0.0, dt_seconds=0.02):
    """Conditioning of whole windows (B, T): instrument multi-hot (B, n_programs) and density level (B,), from the
    notes themselves (BOS/EOS rows ignored). drop_p: per row, both conditions are replaced by 'none' (zeros / 0);
    a float, or a (B,) tensor of one probability per row (captioned_windows)."""
    real = pitch < BOS_PITCH
    inst = torch.zeros(pitch.size(0), n_programs, device=pitch.device)
    if program is not None:
        inst.scatter_add_(1, program.masked_fill(~real, 0), real.float()).clamp_(max=1.0)
    else:
        inst[:, 0] = 1.0  # piano-only data
    n = real.sum(1).float()
    secs = (delta_time * real).sum(1).float() * dt_seconds
    density = density_level(n / secs.clamp(min=0.5), n_density).to(pitch.device) if n_density else \
        torch.zeros(pitch.size(0), dtype=torch.long, device=pitch.device)
    if torch.is_tensor(drop_p) or drop_p > 0:
        keep = torch.rand(pitch.size(0), device=pitch.device) >= drop_p
        inst = inst * keep[:, None]
        density = density * keep
    return inst, density


def captioned_windows(text, sec_text=None):
    """Which windows of a batch carry a caption: a file caption (text, B strings, '' = none) or any section caption
    (sec_text, B lists of strings, '' = unused or dropped). Returns a (B,) bool tensor."""
    return torch.tensor([bool(t) or bool(sec_text is not None and any(sec_text[i])) for i, t in enumerate(text)])


class NoteWindowDataset(Dataset):
    """
    Windows of note tokens from MIDI files listed in CSVs (columns: split, midi_filename) and/or prebuilt token
    stores (MAESTRO, GiantMIDI, Aria, GigaMIDI, the transcribed songs).

    Each note is represented by 4 token streams:
        pitch:      MIDI pitch, 0..127
        velocity:   quantized velocity, 0..velocity_bins-1
        duration:   quantized note duration
        delta_time: quantized time shift from previous note start

    Tokenized notes of all files are stored once in a flat on-disk array (cache_dir/<name>/tokens.u16,
    shape (total_notes, 4) uint16) plus per-file offsets, and memory-mapped, so datasets much larger than RAM work.

    Train mode: every item is a random window, with the start drawn uniformly over all valid positions of all
    files (long files are sampled proportionally more often than short fragments).
    Eval mode (eval_stride): deterministic windows every eval_stride notes of every file.

    Output:
        x = (pitch, velocity, duration, delta_time) [+ program] [+ style]
        y = (pitch, velocity, duration, delta_time) [+ program], shifted by 1 token
    """

    def __init__(
        self,
        csv_path,
        root_dir,
        split,
        block_size,
        velocity_bins=32,
        time_resolution=0.02,
        max_duration_bin=511,
        max_delta_bin=511,
        augment=False,
        cache_dir="data/cache",
        eval_stride=None,
        source_weights=None,
        special_tokens=False,
        styles=None,
        style_dropout=0.0,
        boundary_frac=0.0,
        aug_tempo=0.0,
        aug_stores=None,
        programs=False,
        pad_short=False,
        pack_short=False,
        min_notes=64,
        captions=None,
        caption_dropout=0.0,
        caption_mode='true',
        caption_uk=0.3,
        section_len=0,
        section_lead=0,
        section_dropout=0.0,
        section_caption_mode='true',
        section_style='plain',
        section_attrs='v1',
        change_frac=0.0,
        change_min=2,
        section_repeat=0,
        counterfactual=False,
        splice_frac=0.0,
        remain=False,
        elapsed=False,
        length_words='v1',
    ):
        """
        csv_path:    one or more sources separated by ',': a CSV (paths inside relative to root_dir or absolute),
                     or 'store:<prefix>' = a prebuilt token store <prefix>_<split>/ (e.g. from
                     data_prep/datasets/prepare_aria.py).
                     A CSV with a 'programs' column is multi-track: all tracks incl. drums are kept with their
                     programs (tokenize_tracks, as GigaMIDI), which needs programs=True; otherwise piano (tokenize_midi)
        eval_stride: if set, the dataset is deterministic (no randomness, no augmentation): use it for validation
        source_weights: optional sampling weight per store (each CSV is its own store, CSVs first in order; then
                     each 'store:' source in order). Default: uniform over all note positions, i.e. by size. Use it
                     to mix a small fine-tune set with replay, e.g. [0.8, 0.2]
        special_tokens: every file is read as [BOS] + notes + [EOS] (pitch BOS_PITCH / EOS_PITCH, other attributes 0),
                     without touching the token stores
        styles:      a style vocabulary (list of names, index 0 = 'none'): each window then also returns the style id of
                     its file (CSV column 'style', else 'artist', else DEFAULT_CSV_STYLE; store metadata 'genre').
                     style_dropout: probability (training only) of replacing it with 0 = unconditional
        boundary_frac: training only, with special_tokens: this fraction of windows is placed exactly at a piece's
                     start (half) or end (half), so openings and EOS are seen often (otherwise ~0.2% of windows)
        aug_tempo:   training only: stretch time (duration and delta_time) by a log-uniform factor in
                     [1/(1+aug_tempo), 1+aug_tempo] per window, with stochastic rounding (0 = off)
        aug_stores:  optional 0/1 per store (same order as source_weights): which stores get the tempo
                     augmentation (e.g. [1, 0] = only the fine-tune CSVs, the replay store stays real). Default: all.
                     Transposition applies to every store regardless
        pad_short:   keep files shorter than the window (>= min_notes notes) as one whole-file window, zero-padded at
                     the end with masked (-1) targets; a short file is sampled as often per note as a long one. Without
                     it they are dropped (at block_size 2048: 99% of the Ukrainian songs, 57% of Aria's notes)
        pack_short:  training only (pretraining): keep short files too, but fill their window with further random
                     short files of the same store, each read whole as [BOS] + notes + [EOS], until block_size + 1 rows
                     (the last one is cut). No padding, so no wasted compute (GigaMIDI's median file is 160 notes).
                     Attention is not masked at the joins: BOS marks a new piece. Files are sampled per note, short
                     and long alike. The window's style is its first file's. Validation (eval_stride) still pads
        programs:    multi-instrument: also return the instrument stream (x and y), from the store's programs.u8
                     (data_prep/datasets/prepare_gigamidi.py); stores without one are piano = program 0. Drum notes are never
                     transposed and their duration is set to 0 in x and masked (-1) in y. False = piano-only: stores
                     with programs.u8 are refused, since their drum notes would be read as pitched notes
        captions:    text control: a labels dir (data/labels): each window also returns a caption string of its file
                     (musicar.text.captions.StoreCaptions of the store; '' for stores without labels), last in x.
                     caption_dropout (training): share of windows whose caption is ''. caption_mode (validation):
                     'true' = the file's own caption, 'shuffled' = another file's, 'none' = '' (for val/text_gain);
                     validation captions are fixed per file. caption_uk: share of Ukrainian captions.
                     A store source may end in '#labeled' / '#unlabeled':
                     only its files with / without MidiCaps or XMIDI labels (tier 1 vs the rest)
        section_len: section captions (MusicConfig.section_len, needs captions): the virtual sequence of a file is cut
                     into sections of section_len rows (from BOS), each captioned from its own notes
                     (musicar.text.captions.section_caption); the caption of section k comes before row
                     max(0, k * section_len - section_lead). Each window then also returns, after its caption, the
                     captions of its sections (a list of section_events strings: the one in force at the window start,
                     then those arriving inside it, '' = unused or dropped) and their positions (a (section_events,)
                     tensor of the window index each comes before; block_size + 1 = unused). section_dropout
                     (training): share of section captions replaced by ''. section_caption_mode (validation): 'true' /
                     'shuffled' (another file's sections) / 'none' / 'previous' (section k gets the caption of
                     section k - 1 of its own file, the first section its own), fixed per file like caption_mode.
                     section_repeat: > 0: each section's caption also arrives again every section_repeat rows after its first
                     arrival, up to the next section's (MusicConfig.section_repeat; the same string each time)
                     section_style: 'plain' = each section described on its own; 'change' = what changed vs the
                     section before is always named (musicar.text.captions.section_changed), e.g. the instrument that
                     entered or 'loud' after a soft section (the first section of a file stays plain)
                     section_attrs: musicar.text.captions.section_attributes version of the section captions and the
                     change scores ('v2': section-level thresholds, rhythm + texture instead of pace)
        change_frac: training only, with section_len: this share of windows is placed around a section boundary where
                     the section differs from the one before by >= change_min points (section_change_scores: dynamics,
                     pace, register, drums, instruments), uniform over such boundaries of the drawn store, the
                     boundary at window index lead + margin .. block_size - margin (margin = min(64, block_size // 8)),
                     so the new section's caption arrives inside the window after notes of the old section. Before
                     boundary_frac, which applies to the rest
        counterfactual: training only, with section_len: each window also returns (cf_text, cf_event, cf_start): the
                     caption of the section before the changed one (change_frac / splice windows), the index of the
                     changed section's event in its section captions, and the window index of that section's first
                     note; ('', 0, -1) = none (other windows, a dropped caption, the same caption). For the
                     counterfactual loss (GPT.forward cf): the new caption must beat the previous one on its first notes
        remain:      length control (MusicConfig.n_remain): each window also returns, last in x, the seconds from each
                     input row's onset to the end of its piece (remaining_seconds; float (block_size,), after the tempo
                     stretch); bucketed and dropped in train.py (remain_level)
        elapsed:     with remain: also, after it, the seconds from the start of each row's piece to its onset
                     (elapsed_seconds), for the progress input (MusicConfig.n_progress) and the length jitter in train.py
        length_words: file captions' length attribute ('v1' short / long at the extremes, 'v2' five buckets for every
                     file; musicar.text.captions.LENGTH_V2_Q)
        splice_frac: training only, with section_len: this share of windows (drawn before change_frac) is spliced:
                     notes of one file up to a section boundary (at window index lead + margin .. block_size - margin,
                     as change windows), then a section of another long file of the same store from its own section
                     boundary on, with that file's section captions (the first one, change-aware, against the notes
                     before the splice) and no file caption: a caption that contradicts the context
        """
        sources = [s.strip() for s in str(csv_path).split(',') if s.strip()]
        self.csv_paths = [Path(s) for s in sources if not s.startswith('store:')]
        store_sources = [s[len('store:'):].partition('#') for s in sources if s.startswith('store:')]
        prebuilt = [Path(f"{prefix}_{split}") for prefix, _, _ in store_sources]
        store_tiers = [None] * len(self.csv_paths) + [tier or None for _, _, tier in store_sources]
        assert all(t in (None, 'labeled', 'unlabeled') for t in store_tiers), store_tiers
        self.root_dir = Path(root_dir)
        self.split = split
        self.block_size = block_size

        self.velocity_bins = velocity_bins
        self.time_resolution = time_resolution
        self.max_duration_bin = max_duration_bin
        self.max_delta_bin = max_delta_bin
        self.augment = augment
        self.aug_tempo = aug_tempo
        self.aug_stores = aug_stores
        self.cache_dir = Path(cache_dir)
        self.eval_stride = eval_stride
        self.programs = programs

        self.pad = 1 if special_tokens else 0
        self.pad_short = pad_short
        self.pack_short = pack_short and not eval_stride
        assert not self.pack_short or special_tokens, "pack_short needs special_tokens (BOS/EOS separate the pieces)"
        self.min_notes = min_notes
        self.style_dropout = style_dropout
        self.boundary_frac = boundary_frac if special_tokens else 0.0
        self.style_ids = {name: i for i, name in enumerate(styles)} if styles else None
        self.store_dirs, file_styles = [], []
        for csv in self.csv_paths:  # one store per CSV, so source_weights can weight each CSV
            paths, csv_styles, multi = self._load_midi_paths(csv)
            file_styles.append(csv_styles)
            self.store_dirs.append(self._build_store(csv, paths, multi))
        for store in prebuilt:
            if not (store / "meta.json").exists():
                raise FileNotFoundError(f"Prebuilt token store {store} missing or incomplete")
            meta = json.loads((store / "meta.json").read_text())
            print(f"Loaded tokenized {self.split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes from {store}")
            self.store_dirs.append(store)
            file_styles.append(self._store_styles(store, meta['n_files']))
        self._tokens = None  # memmaps, opened lazily in each DataLoader worker
        self._programs = None
        self.caption_dropout = caption_dropout
        self.caption_mode = caption_mode
        self.captioners = None
        if captions or any(store_tiers):
            from musicar.text.captions import StoreCaptions
            self.captioners = [StoreCaptions(store.name, captions or 'data/labels', uk_share=caption_uk,
                                             length_words=length_words) for store in self.store_dirs]
        self.with_captions = bool(captions)
        self.caption_uk = caption_uk
        self.section_len, self.section_lead = section_len, section_lead
        self.section_dropout, self.section_caption_mode = section_dropout, section_caption_mode
        assert section_style in ('plain', 'change'), section_style
        self.section_style = section_style
        self.section_attrs = section_attrs
        assert not section_len or (captions and not self.pack_short), 'section captions need captions, no pack_short'
        assert 0 <= section_lead <= section_len
        # the caption in force at the window start + at most one per section_len rows inside it (+ the one before
        # the note after the window)
        self.section_repeat = section_repeat
        assert 0 <= section_repeat <= section_len
        # (with repeats: up to ceil(section_len / section_repeat) arrivals in each section the window touches)
        self.section_events = 1 + (1 + -(-block_size // section_len)) * -(-section_len // (section_repeat or section_len)) \
            if section_len else 0
        self.change_frac = change_frac if section_len and not eval_stride else 0.0
        self.counterfactual = bool(counterfactual and section_len and not eval_stride)
        self.splice_frac = splice_frac if section_len and not eval_stride else 0.0
        self.remain = remain
        self.elapsed = elapsed and remain

        store_ids, starts, lengths, styles_kept, file_ids = [], [], [], [], []
        for i, store in enumerate(self.store_dirs):
            offsets = np.load(store / "offsets.npy")
            n = np.diff(offsets) + 2 * self.pad  # virtual length incl. BOS/EOS
            keep = n > self.block_size  # need block_size + 1 notes because targets are shifted by one
            if self.pad_short or self.pack_short:
                keep = n >= self.min_notes + 2 * self.pad
            if store_tiers[i]:
                labeled = np.zeros(len(n), bool)
                labeled[[f for f in self.captioners[i].labels if self.captioners[i].labeled(f)]] = True
                keep &= labeled if store_tiers[i] == 'labeled' else ~labeled
                print(f"{store.name}#{store_tiers[i]}: {int(keep.sum()):,} files")
            file_ids.append(np.flatnonzero(keep))
            store_ids.append(np.full(int(keep.sum()), i, dtype=np.int32))
            starts.append(offsets[:-1][keep])
            lengths.append(n[keep])
            if self.style_ids is not None:
                ids = np.array([self.style_ids.get(x, 0) for x in file_styles[i]], dtype=np.int64)
                assert len(ids) == len(n), f"{store}: {len(ids)} style entries for {len(n)} files"
                styles_kept.append(ids[keep])
        self.store_ids, self.starts, self.lengths = map(np.concatenate, (store_ids, starts, lengths))
        self.file_ids = np.concatenate(file_ids)  # each kept file's index in its own store (label tables)
        self.file_style = np.concatenate(styles_kept) if self.style_ids is not None else None
        if len(self.lengths) == 0:
            raise ValueError("No MIDI sequences longer than block_size. Reduce block_size or check the data.")
        self.n_files = len(self.lengths)
        self.source_weights = None
        if source_weights is not None and not eval_stride:
            assert len(source_weights) == len(self.store_dirs), \
                f"{len(source_weights)} source_weights for {len(self.store_dirs)} stores"
            ids = np.arange(len(self.store_dirs))
            # files of one store are contiguous (stores are concatenated in order)
            self.store_file_range = [(int(np.searchsorted(self.store_ids, i, 'left')),
                                      int(np.searchsorted(self.store_ids, i, 'right'))) for i in ids]
            self.source_weights = [float(w) for w in source_weights]
        if aug_stores is not None:
            assert len(aug_stores) == len(self.store_dirs), \
                f"{len(aug_stores)} aug_stores for {len(self.store_dirs)} stores"
        # number of valid window starts per file, cumulative (for uniform sampling over all positions)
        # a short (padded) file counts as block_size starts: the per-note rate a long file approaches.
        # Packed, it counts as its length: its notes fill only part of a window, the rest are other short files
        self.short = self.lengths <= self.block_size
        self.cum_valid = np.cumsum(np.where(self.short, self.lengths if self.pack_short else self.block_size,
                                            self.lengths - self.block_size))
        if self.pack_short:
            # per store: the short files and their cumulative lengths, to draw fillers per note
            self.short_files = []
            for i in range(len(self.store_dirs)):
                idx = np.flatnonzero((self.store_ids == i) & self.short)
                self.short_files.append((idx, np.cumsum(self.lengths[idx])))

        if self.change_frac:
            self._find_changes(change_min)

        if eval_stride:
            # (file index, offset in the file's virtual sequence); order = sources in order, files in store order
            self.windows = np.concatenate([
                np.stack([np.full(len(r), fi), r], axis=1)
                for fi, n in enumerate(self.lengths)
                for r in [np.arange(0, n - self.block_size, eval_stride) if n > self.block_size
                          else np.zeros(1, np.int64)]
            ])

    def _find_changes(self, change_min):
        """self.changes: (file index, virtual row of the section start) of the boundaries with a section change of
        >= change_min points in the long files (short ones are one window anyway), stores in order;
        self.change_range: (first, end) of each store's."""
        S, found = self.section_len, []
        for i, store in enumerate(self.store_dirs):
            scores = section_change_scores(store, S, self.pad, self.section_attrs)
            n_sec = -(-(np.diff(np.load(store / "offsets.npy")) + 2 * self.pad) // S)
            assert len(scores) == n_sec.sum(), f"{store}: stale section_changes cache"
            fid = np.repeat(np.arange(len(n_sec)), n_sec)  # store file of every section
            k = np.arange(len(scores)) - np.repeat(np.cumsum(n_sec) - n_sec, n_sec)
            file_idx = np.full(len(n_sec), -1)
            mine = np.flatnonzero((self.store_ids == i) & ~self.short)
            file_idx[self.file_ids[mine]] = mine
            hit = (scores >= change_min) & (k > 0) & (file_idx[fid] >= 0)
            found.append(np.stack([file_idx[fid[hit]], k[hit] * S], 1))
            print(f"{store.name}: {int(hit.sum()):,} section changes >= {change_min} "
                  f"({hit.sum() / max(1, ((k > 0) & (file_idx[fid] >= 0)).sum()):.1%} of the boundaries)")
        self.changes = np.concatenate(found)
        ends = np.cumsum([len(c) for c in found])
        self.change_range = list(zip(ends - [len(c) for c in found], ends))
        assert len(self.changes), f"no section changes >= {change_min}"

    def _change_window(self, sid):
        """(file index, window start) around a random section change of store sid (None = any store); None if the
        store has none."""
        lo, hi = self.change_range[sid] if sid is not None else (0, len(self.changes))
        if hi == lo:
            return None
        file_idx, row = (int(x) for x in self.changes[random.randrange(lo, hi)])
        T = self.block_size
        margin = min(64, T // 8)
        offset = random.randint(min(self.section_lead + margin, T - margin), T - margin)
        return file_idx, min(max(row - offset, 0), int(self.lengths[file_idx]) - T - 1), row

    def _random_file(self, sid):
        """A file of store sid (None = any) drawn per note like the training windows."""
        if sid is None:
            pos = random.randrange(int(self.cum_valid[-1]))
        else:
            lo, hi = self.store_file_range[sid]
            base = int(self.cum_valid[lo - 1]) if lo > 0 else 0
            pos = base + random.randrange(int(self.cum_valid[hi - 1]) - base)
        return int(np.searchsorted(self.cum_valid, pos, side="right"))

    def _splice_window(self, sid):
        """(file, window start, boundary index b in the window, second file, its window start) of a spliced window
        (see splice_frac): window rows < b from the first file, rows >= b from the second, both at section
        boundaries (k * section_len), k >= 1. None after 20 failed draws (short files)."""
        S, T = self.section_len, self.block_size
        margin = min(64, T // 8)
        for _ in range(20):
            f1, f2 = self._random_file(sid), self._random_file(sid)
            if f1 == f2 or self.short[f1] or self.short[f2]:
                continue
            b = random.randint(min(self.section_lead + margin, T - margin), T - margin)
            ks = [[k for k in range(1, -(-int(self.lengths[f]) // S)) if 0 <= k * S - b <= int(self.lengths[f]) - T - 1]
                  for f in (f1, f2)]
            if ks[0] and ks[1]:
                return f1, random.choice(ks[0]) * S - b, b, f2, random.choice(ks[1]) * S - b
        return None

    # ------------------------------------------------------------------ storage

    def _store_styles(self, store, n_files):
        """Per-file style names of a prebuilt store: 'genre' of the <split> rows of data/aria/<name>.csv."""
        if self.style_ids is None:
            return None
        name = store.name[:-len(f"_{self.split}")]
        meta_csv = Path("data/aria") / f"{name}.csv"
        if not meta_csv.exists():
            return [DEFAULT_STORE_STYLE] * n_files
        df = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
        return list(df.loc[df["split"] == self.split, "genre"])

    def _load_midi_paths(self, csv_path):
        """(MIDI paths, styles, multi-track?) of one CSV's rows in this split."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "split" not in df.columns or "midi_filename" not in df.columns:
            raise ValueError(f"{csv_path} must contain 'split' and 'midi_filename' columns.")
        multi = "programs" in df.columns
        df = df[df["split"] == self.split]
        paths = [self.root_dir / f for f in df["midi_filename"]]
        col = "style" if "style" in df.columns else "artist" if "artist" in df.columns else None
        styles = list(df[col]) if col else [DEFAULT_CSV_STYLE.get(csv_path.stem, "none")] * len(df)

        if len(paths) == 0:
            raise ValueError(f"No MIDI files found in {csv_path} for split='{self.split}'.")

        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"WARNING [{self.split}]: {len(missing)}/{len(paths)} MIDI files are missing "
                  f"and will be skipped (first: {missing[0]})")
            styles = [st for p, st in zip(paths, styles) if p.exists()]
            paths = [p for p in paths if p.exists()]
            if not paths:
                raise FileNotFoundError(f"All MIDI files for split='{self.split}' are missing.")
        return paths, styles, multi

    def _build_store(self, csv_path, midi_paths, multi=False):
        key = (
            [str(p) for p in midi_paths],
            self.velocity_bins, self.time_resolution, self.max_duration_bin, self.max_delta_bin,
        ) + (('programs',) if multi else ())
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
        name = csv_path.stem  # same name and key as before for a single CSV -> existing stores still match
        store = self.cache_dir / f"{name}_{self.split}_{digest}"
        if (store / "meta.json").exists():
            meta = json.loads((store / "meta.json").read_text())
            print(f"Loaded tokenized {self.split}: {meta['n_files']:,} files, {meta['n_notes']:,} notes from {store}")
            return store

        store.mkdir(parents=True, exist_ok=True)
        parse = partial(tokenize_tracks if multi else tokenize_midi, velocity_bins=self.velocity_bins,
                        time_resolution=self.time_resolution,
                        max_duration_bin=self.max_duration_bin, max_delta_bin=self.max_delta_bin)
        workers = min(len(midi_paths), os.cpu_count() or 1)
        offsets, n_notes, n_failed = [0], 0, 0
        # streamed to disk in order, so memory stays flat no matter how large the dataset is
        with open(store / "tokens.u16", "wb") as f, \
                open(store / "programs.u8", "wb") if multi else contextlib.nullcontext() as fp, \
                ProcessPoolExecutor(max_workers=workers) as pool:
            for arr in tqdm(pool.map(parse, midi_paths, chunksize=16), total=len(midi_paths),
                            desc=f"Tokenizing {self.split} ({workers} workers)"):
                if arr is None:
                    n_failed += 1
                else:
                    if multi:
                        arr, programs = arr
                        fp.write(programs.tobytes())
                    f.write(arr.tobytes())
                    n_notes += len(arr)
                offsets.append(n_notes)
        np.save(store / "offsets.npy", np.array(offsets, dtype=np.int64))
        meta = dict(n_files=len(midi_paths), n_notes=n_notes, n_empty_or_failed=n_failed,
                    **({'programs': True} if multi else {}))
        (store / "meta.json").write_text(json.dumps(meta))  # written last = store is complete
        print(f"Tokenized {self.split}: {n_notes:,} notes, {n_failed} empty/broken files skipped -> {store}")
        return store

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens, self._programs = [], []
            for store in self.store_dirs:
                has_programs = (store / "programs.u8").exists()
                # multi-instrument stores (data_prep/datasets/prepare_gigamidi.py) keep drums: not for the piano model
                assert self.programs or not has_programs, f"{store} has instrument programs: pass programs=True"
                n = int(np.load(store / "offsets.npy")[-1])
                self._tokens.append(np.memmap(store / "tokens.u16", dtype=np.uint16, mode="r", shape=(n, 4)))
                self._programs.append(np.memmap(store / "programs.u8", dtype=np.uint8, mode="r", shape=(n,))
                                      if has_programs else None)  # None = piano, program 0
        return self._tokens

    def __getstate__(self):
        # never pickle the memmap into DataLoader workers, each worker opens its own
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_programs"] = None
        return state

    # ------------------------------------------------------------------ sampling

    def num_notes(self):
        return int(self.lengths.sum()) - 2 * self.pad * len(self.lengths)

    def __len__(self):
        if self.eval_stride:
            return len(self.windows)
        # one "epoch" = as many windows as fit without overlap; items are random anyway
        return max(1, int(self.lengths.sum()) // self.block_size)

    def _read(self, file_idx, v):
        """block_size + 1 rows of the file's virtual sequence ([BOS] + notes + [EOS] with special tokens) from v:
        (rows, 4) tokens and, with programs, (rows,) instruments (BOS/EOS and piano stores = 0), else None."""
        store_id, start = int(self.store_ids[file_idx]), int(self.starts[file_idx])
        n = int(self.lengths[file_idx]) - 2 * self.pad
        tokens = self.tokens[store_id]
        lo, hi = v - self.pad, v - self.pad + self.block_size + 1  # real note range
        a, b = start + max(lo, 0), start + min(hi, n)
        rows = tokens[a:b].astype(np.int64)
        programs = None
        if self.programs:
            store_programs = self._programs[store_id]
            programs = store_programs[a:b].astype(np.int64) if store_programs is not None else np.zeros(b - a, np.int64)
        if lo < 0:
            rows = np.concatenate([np.array([[BOS_PITCH, 0, 0, 0]]), rows])
            programs = None if programs is None else np.concatenate([[0], programs])
        if hi > n:
            rows = np.concatenate([rows, np.array([[EOS_PITCH, 0, 0, 0]])])
            programs = None if programs is None else np.concatenate([programs, [0]])
        return torch.from_numpy(rows), None if programs is None else torch.from_numpy(programs)

    def __getitem__(self, idx):
        change = None
        if self.eval_stride:
            file_idx, v = (int(x) for x in self.windows[idx])
        else:
            sid = None if self.source_weights is None else \
                random.choices(range(len(self.source_weights)), weights=self.source_weights)[0]
            splice = self._splice_window(sid) if self.splice_frac and random.random() < self.splice_frac else None
            change = None
            if splice is None and self.change_frac and random.random() < self.change_frac:
                change = self._change_window(sid)
            if splice is not None:
                file_idx, v = splice[:2]
                change = splice
            elif change is not None:
                file_idx, v, _ = change
            elif sid is None:
                pos = random.randrange(int(self.cum_valid[-1]))
            else:
                lo, hi = self.store_file_range[sid]
                base = int(self.cum_valid[lo - 1]) if lo > 0 else 0
                pos = base + random.randrange(int(self.cum_valid[hi - 1]) - base)
            if change is None:
                file_idx = int(np.searchsorted(self.cum_valid, pos, side="right"))
                prev = int(self.cum_valid[file_idx - 1]) if file_idx > 0 else 0
                v = 0 if self.short[file_idx] else pos - prev
            if change is None and self.boundary_frac and not self.short[file_idx]:
                r = random.random()
                if r < self.boundary_frac / 2:
                    v = 0                                                   # starts with BOS
                elif r < self.boundary_frac:
                    v = int(self.lengths[file_idx]) - self.block_size - 1  # ends with EOS

        splice = change if not self.eval_stride and change is not None and len(change) == 5 else None
        tail = 0.0  # seconds of the (last) piece after the window, for remain
        head = 0.0  # onset of the window's first row in its piece, for elapsed
        if splice is not None:
            rows, program = self._read(file_idx, v)
            rows2, program2 = self._read(splice[3], splice[4])
            b = splice[2]
            rows = torch.cat([rows[:b], rows2[b:]])
            program = None if program is None else torch.cat([program[:b], program2[b:]])
            tail = self._tail_seconds(splice[3], splice[4] + self.block_size) if self.remain else 0.0
            head = self._head_seconds(file_idx, v) if self.elapsed else 0.0
        elif self.pack_short and self.short[file_idx]:
            rows, program, tail = self._read_packed(file_idx)  # starts at the file's BOS: head 0
        else:
            rows, program = self._read(file_idx, v)
            tail = self._tail_seconds(file_idx, v + self.block_size) if self.remain else 0.0
            head = self._head_seconds(file_idx, v) if self.elapsed else 0.0
        pitch, velocity, duration, delta_time = rows.unbind(1)
        drums = program == DRUM_PROGRAM if program is not None else None
        if drums is not None:
            duration = torch.where(drums, 0, duration)  # drum hits have no meaningful length

        if self.augment and not self.eval_stride:
            # transpose by up to ±5 semitones, limited so no note leaves 0..127 (clamping would bend them);
            # BOS/EOS (pitch >= 128) and drums (pitch = drum sound) stay as they are
            notes = pitch < BOS_PITCH
            pitched = notes if drums is None else notes & ~drums
            if pitched.any():
                lo, hi = int(pitch[pitched].min()), int(pitch[pitched].max())
                shift = random.randint(max(-5, -lo), min(5, 127 - hi))
                pitch = torch.where(pitched, pitch + shift, pitch)
            store_augmented = self.aug_stores is None or bool(self.aug_stores[int(self.store_ids[file_idx])])
            if self.aug_tempo and store_augmented:
                s = (1 + self.aug_tempo) ** random.uniform(-1, 1)
                duration = self._stretch(duration, s, self.max_duration_bin)
                delta_time = self._stretch(delta_time, s, self.max_delta_bin)
                tail *= s
                head *= s
        remain = remaining_seconds(pitch, delta_time, tail, self.time_resolution) if self.remain else None
        elapsed = elapsed_seconds(pitch, delta_time, head, self.time_resolution) if self.elapsed else None

        streams = (pitch, velocity, duration, delta_time) + ((program,) if program is not None else ())
        n_rows = len(pitch)
        if n_rows < self.block_size + 1:  # a short file: pad after EOS (causal, so nothing before it changes)
            streams = tuple(F.pad(t, (0, self.block_size + 1 - n_rows)) for t in streams)
            if drums is not None:
                drums = F.pad(drums, (0, self.block_size + 1 - n_rows))
        x, y = tuple(t[:-1] for t in streams), tuple(t[1:] for t in streams)  # y = next note
        if drums is not None:
            y = (y[0], y[1], torch.where(drums[1:], -1, y[2])) + y[3:]  # no duration loss on drum notes
        if n_rows < self.block_size + 1:
            y = tuple(torch.cat([t[:n_rows - 1], torch.full_like(t[n_rows - 1:], -1)]) for t in y)
        if self.file_style is not None:
            style = int(self.file_style[file_idx])
            if self.augment and not self.eval_stride and random.random() < self.style_dropout:
                style = 0
            x = x + (torch.tensor(style),)
        if self.with_captions:
            x = x + ('' if splice is not None else self._caption(file_idx),)
        if self.section_len:
            caps, pos, events = self._sections(file_idx, v, splice)
            x = x + (caps, pos)
            if self.counterfactual:
                x = x + self._counterfactual(file_idx, v, change, caps, events)
        if remain is not None:
            if len(remain) < self.block_size + 1:
                remain = F.pad(remain, (0, self.block_size + 1 - len(remain)))
            x = x + (remain[:-1],)
        if elapsed is not None:
            if len(elapsed) < self.block_size + 1:
                elapsed = F.pad(elapsed, (0, self.block_size + 1 - len(elapsed)))
            x = x + (elapsed[:-1],)
        return x, y

    def _head_seconds(self, file_idx, first_row):
        """Onset of file_idx's virtual row first_row from the piece's start (the sum of the delta_times up to and
        including it; 0 at BOS)."""
        a = first_row - self.pad
        if a < 0:
            return 0.0
        store_id, start = int(self.store_ids[file_idx]), int(self.starts[file_idx])
        a = min(a, int(self.lengths[file_idx]) - 2 * self.pad - 1)
        dt = np.asarray(self.tokens[store_id][start:start + a + 1, 3], dtype=np.int64)
        return float(dt.sum()) * self.time_resolution

    def _tail_seconds(self, file_idx, last_row):
        """Seconds of file_idx's notes after virtual row last_row (the sum of their delta_times; 0 from the last note
        or EOS on)."""
        store_id, start = int(self.store_ids[file_idx]), int(self.starts[file_idx])
        n = int(self.lengths[file_idx]) - 2 * self.pad
        a = max(last_row - self.pad + 1, 0)
        if a >= n:
            return 0.0
        dt = np.asarray(self.tokens[store_id][start + a:start + n, 3], dtype=np.int64)
        return float(dt.sum()) * self.time_resolution

    def _counterfactual(self, file_idx, v, change, caps, events):
        """(cf_text, cf_event, cf_start) of a window (see counterfactual): the caption of the section before the
        changed one, the changed section's event index and the window index of its first note."""
        none = ('', torch.tensor(0), torch.tensor(-1))
        if change is None:
            return none
        S = self.section_len
        if len(change) == 5:  # splice: the second file's section at the boundary b
            b, f2, v2 = change[2:]
            k, start, key = (v + b) // S, b, ((v2 + b) // S, f2)
        else:
            k = change[2] // S
            start, key = k * S - v, (k, file_idx)
        e = next((i for i, (_, kk, f) in enumerate(events) if (kk, f) == key), None)
        if not e or not 0 < start < self.block_size or not caps[e]:
            return none
        cf = self._section_caption(file_idx, k - 1)
        if not cf or cf == caps[e]:
            return none
        return cf, torch.tensor(e), torch.tensor(start)

    def _caption(self, file_idx):
        """The caption string of a window of file_idx (see captions / caption_mode in __init__)."""
        if self.eval_stride:
            if self.caption_mode == 'none':
                return ''
            if self.caption_mode == 'shuffled':  # a fixed other file: half the file list away
                file_idx = (file_idx + self.n_files // 2) % self.n_files
            store_id, fid = int(self.store_ids[file_idx]), int(self.file_ids[file_idx])
            return self.captioners[store_id].caption(fid, random.Random(store_id * 10_000_000 + fid))
        if self.augment and random.random() < self.caption_dropout:
            return ''
        return self.captioners[int(self.store_ids[file_idx])].caption(int(self.file_ids[file_idx]))

    def _sections(self, file_idx, v, splice=None):
        """(captions, positions, events) of the sections of the window at virtual row v of file_idx (see
        section_len); events = (window index, section, file) of each caption. A spliced window (splice = the
        _splice_window tuple): the first file's sections before the boundary b, then the second file's from its
        section at the boundary on (the first of them change-aware against the first file's section before b)."""
        S, T, lead = self.section_len, self.block_size, self.section_lead
        R = self.section_repeat or S
        at = [max(0, k * S - lead) for k in range(-(-int(self.lengths[file_idx]) // S))]
        now = max(k for k, p in enumerate(at) if p <= v)
        # arrivals: every section's first one, then every R rows until the next section's caption
        arrive = [(p + j * R, k) for k, p in enumerate(at) for j in range(-(-S // R))
                  if j == 0 or k + 1 >= len(at) or p + j * R < at[k + 1]]
        events = [(0, now, file_idx)] + [(p - v, k, file_idx) for p, k in arrive if v < p <= v + T]
        prev_notes = {}
        if splice is not None:
            b, f2, v2 = splice[2:]
            k1, k2 = (v + b) // S, (v2 + b) // S  # the sections that start at the boundary
            events = [e for e in events if e[1] < k1]
            n2 = -(-int(self.lengths[f2]) // S)
            events += [(b + m * S - lead + j * R, k2 + m, f2) for m in range(n2 - k2) for j in range(-(-S // R))
                       if 0 < b + m * S - lead + j * R <= T]
            prev_notes[(f2, k2)] = self._section_notes(file_idx, k1 - 1)
        drawn = {}  # one caption per section, repeated as is (one dropout draw per section)
        for _, k, f in events:
            if (f, k) not in drawn:
                drawn[(f, k)] = self._section_caption(f, k, prev_notes.get((f, k)))
        caps = [drawn[(f, k)] for _, k, f in events]
        pad = self.section_events - len(events)
        assert pad >= 0, (len(events), self.section_events)
        return caps + [''] * pad, torch.tensor([p for p, _, _ in events] + [T + 1] * pad), events

    def _section_caption(self, file_idx, k, prev_notes=None):
        """Caption of section k of file_idx; prev_notes: (tokens, programs) of the notes before it when they aren't
        its own file's section k - 1 (a splice), for change-aware captions."""
        S = self.section_len
        rng = random
        if self.eval_stride:
            if self.section_caption_mode == 'none':
                return ''
            if self.section_caption_mode == 'previous':  # the caption that was in force before this section
                k = max(k - 1, 0)
            if self.section_caption_mode == 'shuffled':  # the same fixed other file as caption_mode 'shuffled'
                file_idx = (file_idx + self.n_files // 2) % self.n_files
                k %= -(-int(self.lengths[file_idx]) // S)
            store_id, fid = int(self.store_ids[file_idx]), int(self.file_ids[file_idx])
            rng = random.Random((store_id * 10_000_000 + fid) * 1_000 + k)
        elif self.augment and random.random() < self.section_dropout:
            return ''
        from musicar.text.captions import section_attributes, section_caption
        prev = None
        if self.section_style == 'change' and (k > 0 or prev_notes is not None):
            prev = section_attributes(*(prev_notes if prev_notes is not None else self._section_notes(file_idx, k - 1)),
                                      version=self.section_attrs)
        return section_caption(*self._section_notes(file_idx, k), rng, uk_share=self.caption_uk, prev=prev,
                               version=self.section_attrs)

    def _section_notes(self, file_idx, k):
        """Tokens (n, 4) and programs (n,) of the real notes in rows k * section_len .. (k + 1) * section_len - 1 of
        a file's virtual sequence (BOS / EOS left out; programs 0 for piano stores)."""
        S = self.section_len
        store_id, start = int(self.store_ids[file_idx]), int(self.starts[file_idx])
        n = int(self.lengths[file_idx]) - 2 * self.pad
        lo = max(k * S - self.pad, 0)
        hi = max(min((k + 1) * S - self.pad, n), lo)
        tok = np.asarray(self.tokens[store_id][start + lo:start + hi], dtype=np.int64)
        progs = self._programs[store_id]
        prog = np.asarray(progs[start + lo:start + hi], dtype=np.int64) if progs is not None else \
            np.zeros(hi - lo, np.int64)
        return tok, prog

    def _read_packed(self, file_idx):
        """A short file followed by random short files of the same store (drawn per note), whole, cut at
        block_size + 1 rows; (rows, programs, seconds of the cut file after the cut)."""
        idx, cum = self.short_files[int(self.store_ids[file_idx])]
        parts = [self._read(file_idx, 0)]
        n_rows = len(parts[0][0])
        while n_rows < self.block_size + 1:
            j = int(idx[np.searchsorted(cum, random.randrange(int(cum[-1])), side="right")])
            parts.append(self._read(j, 0))
            n_rows += len(parts[-1][0])
        rows = torch.cat([r for r, _ in parts])[:self.block_size + 1]
        program = torch.cat([p for _, p in parts])[:self.block_size + 1] if self.programs else None
        tail = 0.0
        if self.remain:  # the last file's last row in the window: virtual row len(last) - 1 - (rows cut off)
            last = j if len(parts) > 1 else file_idx
            tail = self._tail_seconds(last, len(parts[-1][0]) - 1 - (n_rows - self.block_size - 1))
        return rows, program, tail

    @staticmethod
    def _stretch(bins, s, max_bin):
        # stochastic rounding keeps the stretched bins unbiased and avoids a lattice of empty bins; 0 (chords) stays 0
        scaled = bins.double() * s
        low = scaled.floor()
        out = low + (torch.rand_like(scaled) < scaled - low).double()
        return out.long().clamp(max=max_bin)


def _read_midi(midi):
    """pretty_midi.PrettyMIDI from a path, bytes or a file-like object (e.g. a member read from an archive); None if it
    can't be parsed (large crawled datasets always contain a few broken files)."""
    if isinstance(midi, (bytes, bytearray)):
        midi = io.BytesIO(midi)
    try:
        return pretty_midi.PrettyMIDI(str(midi) if isinstance(midi, (str, Path)) else midi)
    except Exception:
        return None


def _note_tokens(notes, velocity_bins, time_resolution, max_duration_bin, max_delta_bin):
    """Sorted (onset s, pitch, velocity, length s) notes -> (n, 4) uint16 tokens (the store layout): pitch, quantized
    velocity, duration, and delta_time = onset difference to the previous note (0 for the first)."""
    tokens = np.empty((len(notes), 4), dtype=np.uint16)
    prev = notes[0][0]
    for i, (start, pitch, velocity, length) in enumerate(notes):
        tokens[i] = (pitch, quantize_velocity(velocity, velocity_bins),
                     quantize_time(max(0.0, length), time_resolution, max_duration_bin),
                     quantize_time(max(0.0, start - prev), time_resolution, max_delta_bin))
        prev = start
    return tokens


def tokenize_midi(midi, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """MIDI (path, bytes or file-like) -> (n, 4) uint16 tokens, one row per note: all non-drum tracks merged into one
    piano stream, sorted by (onset, pitch). None if empty or broken."""
    midi = _read_midi(midi)
    if midi is None:
        return None
    notes = [(n.start, n.pitch, n.velocity, n.end - n.start)
             for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return None
    notes.sort(key=lambda n: n[:2])
    return _note_tokens(notes, velocity_bins, time_resolution, max_duration_bin, max_delta_bin)


def tokenize_tracks(midi, velocity_bins=32, time_resolution=0.02, max_duration_bin=511, max_delta_bin=511):
    """MIDI (path, bytes or file-like) -> ((n, 4) uint16 tokens, (n,) uint8 programs), all tracks incl. drums
    (program DRUM_PROGRAM), sorted by (onset, program, pitch); None if empty or broken. The multi-instrument stores
    (data_prep/datasets/prepare_gigamidi.py, prepare_discover.py) and multi-track CSVs are tokenized with it."""
    midi = _read_midi(midi)
    if midi is None:
        return None
    notes = [(n.start, DRUM_PROGRAM if inst.is_drum else inst.program, n.pitch, n.velocity, n.end - n.start)
             for inst in midi.instruments for n in inst.notes]
    if not notes:
        return None
    notes.sort(key=lambda n: n[:3])
    tokens = _note_tokens([(start, pitch, velocity, length) for start, _, pitch, velocity, length in notes],
                          velocity_bins, time_resolution, max_duration_bin, max_delta_bin)
    return tokens, np.array([n[1] for n in notes], dtype=np.uint8)


def quantize_time(time_sec, time_resolution, max_bin):
    token = round(time_sec / time_resolution)
    return min(max_bin, max(0, token))


def quantize_velocity(velocity, velocity_bins):
    token = velocity * velocity_bins // 128
    return min(velocity_bins - 1, max(0, token))


def collate_windows(batch, extra_names=()):
    """Dataset items -> ((4 note streams, {name: tensor} of the optional inputs `extra_names`, in item order after the
    4 streams, e.g. 'program', 'style', 'text'), the target streams). Strings (captions) and lists of them (section
    captions) stay lists."""
    xs, ys = zip(*batch)
    x = [[x[i] for x in xs] if isinstance(xs[0][i], (str, list)) else torch.stack([x[i] for x in xs])
         for i in range(len(xs[0]))]
    assert len(x) == 4 + len(extra_names), f"{len(x)} input streams, expected 4 + {list(extra_names)}"
    y = tuple(torch.stack([y[i] for y in ys]) for i in range(len(ys[0])))
    return (tuple(x[:4]), dict(zip(extra_names, x[4:]))), y


def val_loader(sources, batch_size, max_batches, num_workers=0, extra_names=(), block_size=512, **dataset_kw):
    """Fixed non-overlapping validation windows (eval_stride = block_size), evenly thinned to at most max_batches
    batches: identical at every evaluation and in every run."""
    ds = NoteWindowDataset(sources, split='validation', block_size=block_size, eval_stride=block_size, **dataset_kw)
    max_windows = max_batches * batch_size
    if len(ds) > max_windows:
        step = len(ds) / max_windows
        ds = Subset(ds, [int(i * step) for i in range(max_windows)])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=partial(collate_windows, extra_names=tuple(extra_names)))


def source_name(sources):
    """'store:data/cache/gigamidi_clean,data/finetune/x.csv' -> 'gigamidi_clean+x'"""
    return '+'.join(os.path.splitext(os.path.basename(s.removeprefix('store:')))[0] for s in sources.split(',') if s)
