"""
A token store of musicar-L continuations of real openings from different styles: synthetic training data (census doc
"Data augmentation ideas", idea 12; tools/sweeps/aug_sweep.py arm synth25; later more of it for bigger models).

Prompts: the first --prompt-notes notes of real pieces of --prompt-store, only its --prompt-frac share of files
(musicar.data.loader.file_subset: the same quarter tools/sweeps/aug_sweep.py trains on) and only pieces at least
--prompt-notes + 64 notes long. A piece belongs to each of its genres (StoreCaptions.genre_mood on data/labels: the
MidiCaps / GigaMIDI / Aria / XMIDI labels, else the confident pseudo-labels) that has at least --min-group pieces;
pieces whose genres are all smaller go to 'other' (GigaMIDI's decade tags end up there), pieces without any to
'untagged'. Every row draws a genre with probability proportional to (pieces of that genre) ** --style-alpha (0 = the
same for every style, 1 = by size), then a piece of it uniformly; the draw is a function of (--seed, row number), so
a run is reproducible and a restart continues it exactly. --list-styles prints the groups and how many continuations
each opening gets at --notes.
Sampling: generate_batch of musicar/generation/piece.py with the base model's tracking setting (musicar/eval/
style_eval.py TRACK: temperature 0.9, top-p 0.9, dt bias 0.1), instruments sampled, the prompt's instrument set as
the condition, at most --new-notes notes after the prompt, no EOS in the first --min-new.
Output: shards of about --shard-notes notes (<out>_shards/shard_00000/ ...), each written whole, so a restart skips
the finished ones; when the shards reach --notes (or with --merge alone) they are joined into the store <out>_train/
(read as store:<out>). Each file keeps its prompt first (a piece starts at its real opening, after BOS); files.jsonl
has one row per file: the prompt's store, file and genre, prompt notes, new notes, whether it ended with EOS.

    .venv/bin/python data_prep/datasets/synth_store.py --notes 40e6 --out data/cache/synth_musicar_l
    .venv/bin/python data_prep/datasets/synth_store.py --list-styles      # genre groups and their shares, then exit
"""

import argparse
import contextlib
import io
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from musicar.data.loader import file_subset
from musicar.generation.piece import Generator, finish_row, generate_batch, settings
from musicar.text.captions import StoreCaptions

MIN_EXTRA = 64  # a prompt piece has at least this many notes after the prompt


def prompt_groups(store, frac, prompt_notes, labels_dir, min_group=50):
    """{genre: [file index]} of the eligible prompt pieces of a store: a piece is in each of its genres that has at
    least min_group pieces; a piece whose genres are all smaller (GigaMIDI's decade tags, one-off styles) is in
    'other', one without any in 'untagged'."""
    lengths = np.diff(np.load(store / 'offsets.npy'))
    ok = file_subset(store.name, len(lengths), frac) if frac < 1 else np.ones(len(lengths), bool)
    ok &= lengths >= prompt_notes + MIN_EXTRA
    captions = StoreCaptions(store.name, labels_dir)
    genres = {int(i): sorted(captions.genre_mood(i)[0]) for i in np.flatnonzero(ok)}
    counts = defaultdict(int)
    for gs in genres.values():
        for g in gs:
            counts[g] += 1
    groups = defaultdict(list)
    for i, gs in genres.items():
        for g in [g for g in gs if counts[g] >= min_group] or (['other'] if gs else ['untagged']):
            groups[g].append(i)
    return dict(sorted(groups.items()))


def group_weights(groups, alpha):
    w = {g: len(m) ** alpha for g, m in groups.items()}
    total = sum(w.values())
    return {g: v / total for g, v in w.items()}


def draw(seed, row, names, cum, groups):
    """(genre, file index) of a global row: a function of (seed, row) only."""
    rng = random.Random(f'{seed}:{row}')
    g = rng.choices(names, cum_weights=cum)[0]
    return g, rng.choice(groups[g])


def write_store(path, tokens, programs, records, meta):
    """A token store at path (tokens (n, 4) uint16, programs (n,) uint8 lists per file), written to a temporary
    directory first and renamed, so a store directory is always complete."""
    tmp = path.with_name(path.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.concatenate(tokens).astype(np.uint16).tofile(tmp / 'tokens.u16')
    np.concatenate(programs).astype(np.uint8).tofile(tmp / 'programs.u8')
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in tokens])]).astype(np.int64)
    np.save(tmp / 'offsets.npy', offsets)
    (tmp / 'files.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (tmp / 'meta.json').write_text(json.dumps(dict(n_files=len(tokens), n_notes=int(offsets[-1]), programs=True,
                                                   **meta), indent=1))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def finished_shards(shards_dir):
    return sorted(p for p in shards_dir.glob('shard_*') if p.is_dir() and not p.name.endswith('.tmp'))


def merge(shards_dir, out_store):
    """The finished shards, in order, as one store (streamed: no shard is held in memory)."""
    shards = finished_shards(shards_dir)
    assert shards, f'no finished shards in {shards_dir}'
    metas = [json.loads((s / 'meta.json').read_text()) for s in shards]
    assert all(m['plan'] == metas[0]['plan'] for m in metas), 'shards from different plans'
    tmp = out_store.with_name(out_store.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    offsets, base, groups = [np.zeros(1, np.int64)], 0, defaultdict(int)
    with open(tmp / 'tokens.u16', 'wb') as tok, open(tmp / 'programs.u8', 'wb') as prog, \
            open(tmp / 'files.jsonl', 'w') as files:
        for s in shards:
            with open(s / 'tokens.u16', 'rb') as f:
                shutil.copyfileobj(f, tok)
            with open(s / 'programs.u8', 'rb') as f:
                shutil.copyfileobj(f, prog)
            offsets.append(np.load(s / 'offsets.npy')[1:] + base)
            base = int(offsets[-1][-1])
            for line in (s / 'files.jsonl').read_text().splitlines():
                files.write(line + '\n')
                groups[json.loads(line)['group']] += 1
    offsets = np.concatenate(offsets)
    np.save(tmp / 'offsets.npy', offsets)
    meta = dict(n_files=len(offsets) - 1, n_notes=int(offsets[-1]), programs=True, plan=metas[0]['plan'],
                shards=len(shards), files_per_genre=dict(sorted(groups.items())),
                seconds=round(sum(m['seconds'] for m in metas)))
    (tmp / 'meta.json').write_text(json.dumps(meta, indent=1))
    shutil.rmtree(out_store, ignore_errors=True)
    tmp.rename(out_store)
    print(f'{out_store}: {meta["n_files"]:,} files, {meta["n_notes"]:,} notes from {len(shards)} shards')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', default='checkpoints/musicar-L')
    p.add_argument('--prompt-store', default='data/cache/gigamidi_replay_train')
    p.add_argument('--prompt-frac', type=float, default=0.25, help='the share of its files the prompts come from')
    p.add_argument('--labels-dir', default='data/labels')
    p.add_argument('--style-alpha', type=float, default=0.3, help='genre weights (pieces of the genre) ** alpha')
    p.add_argument('--min-group', type=int, default=50, help='genres with fewer pieces are pooled as "other"')
    p.add_argument('--prompt-notes', type=int, default=128)
    p.add_argument('--new-notes', type=int, default=1024, help='at most this many notes after the prompt')
    p.add_argument('--min-new', type=int, default=128, help='no EOS before this many new notes')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--dt-bias', type=float, default=0.1)
    p.add_argument('--rows', type=int, default=64, help='rows per batch')
    p.add_argument('--notes', type=float, default=40e6, help='generate shards until they hold this many notes')
    p.add_argument('--shard-notes', type=float, default=2e6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='data/cache/synth_musicar_l', help='prefix: <out>_shards/, <out>_train/')
    p.add_argument('--device', default='auto')
    p.add_argument('--merge', action='store_true', help='only join the finished shards into <out>_train/')
    p.add_argument('--list-styles', action='store_true')
    a = p.parse_args()
    shards_dir, out_store = Path(a.out + '_shards'), Path(a.out + '_train')
    if a.merge:
        return merge(shards_dir, out_store)

    store = Path(a.prompt_store)
    groups = prompt_groups(store, a.prompt_frac, a.prompt_notes, a.labels_dir, a.min_group)
    weights = group_weights(groups, a.style_alpha)
    n_pieces = len({i for m in groups.values() for i in m})
    rows_needed = a.notes / (a.prompt_notes + a.new_notes)  # at least this many rows (EOS makes rows shorter)
    print(f'{store.name}: {n_pieces:,} prompt pieces in {len(groups)} genre groups (alpha {a.style_alpha:g}); '
          f'{a.notes:,.0f} notes = at least {rows_needed:,.0f} rows:')
    for g, m in groups.items():
        print(f'  {g:16s} {len(m):7,} pieces  {weights[g]:6.1%} of the rows  '
              f'~{rows_needed * weights[g] / len(m):5.1f} continuations per piece')
    if a.list_styles:
        return
    names = list(groups)
    cum = np.cumsum([weights[g] for g in names]).tolist()

    gen = Generator(a.checkpoint, a.device)
    model, config = gen.model, gen.config
    assert 1 + a.prompt_notes + a.new_notes <= config.block_size, 'BOS + prompt + new notes must fit the block'
    gargs = settings(sample_instruments=True, temperature=a.temperature, top_p=a.top_p, dt_bias=a.dt_bias,
                     prompt_notes=a.prompt_notes, prompt_start=0, max_new_tokens=a.new_notes, min_notes=a.min_new,
                     keep_prompt=True, batch_size=a.rows, no_cuda_graph=gen.device != 'cuda',
                     retrieve='prompt plan')  # retrieve: row r takes the batch's piece r (_prompt_rows)
    gargs.caption_allowed = None  # set by Generator.run otherwise
    batches = max(1, round(a.shard_notes / (a.rows * (a.prompt_notes + a.new_notes))))
    plan = dict(checkpoint=str(gen.checkpoint), iter=gen.iter, prompt_store=store.name, prompt_frac=a.prompt_frac,
                style_alpha=a.style_alpha, prompt_notes=a.prompt_notes, new_notes=a.new_notes, min_new=a.min_new,
                temperature=a.temperature, top_p=a.top_p, dt_bias=a.dt_bias, rows=a.rows, batches_per_shard=batches,
                seed=a.seed)

    n = int(json.loads((store / 'meta.json').read_text())['n_notes'])
    offsets = np.load(store / 'offsets.npy')
    tokens = np.memmap(store / 'tokens.u16', dtype=np.uint16, mode='r', shape=(n, 4))
    programs = np.memmap(store / 'programs.u8', dtype=np.uint8, mode='r', shape=(n,)) \
        if (store / 'programs.u8').exists() else None

    def piece(g, i):
        a0 = int(offsets[i])
        load = lambda: (np.asarray(tokens[a0:a0 + a.prompt_notes], dtype=np.int64),  # noqa: E731
                        np.zeros(a.prompt_notes, np.int64) if programs is None else
                        np.asarray(programs[a0:a0 + a.prompt_notes], dtype=np.int64))
        return f'{store.name} file {i} [{g}]', load

    shards_dir.mkdir(parents=True, exist_ok=True)
    done = finished_shards(shards_dir)
    for s in done:
        if json.loads((s / 'meta.json').read_text())['plan'] != plan:
            raise SystemExit(f'{s} was made with other settings: delete {shards_dir} or run with its settings')
    assert [s.name for s in done] == [f'shard_{k:05d}' for k in range(len(done))], 'shards must be contiguous'
    total = sum(json.loads((s / 'meta.json').read_text())['n_notes'] for s in done)
    print(f'{len(done)} shards with {total:,} notes already; target {a.notes:,.0f}; {batches} batches x {a.rows} '
          f'rows per shard')
    k = len(done)
    while total < a.notes:
        t0 = time.time()
        toks, progs, records = [], [], []
        for b in range(batches):
            batch = k * batches + b
            rows = [batch * a.rows + r for r in range(a.rows)]
            drawn = [draw(a.seed, row, names, cum, groups) for row in rows]
            tb = time.time()
            with contextlib.redirect_stdout(io.StringIO()):  # generate_batch's per-batch lines
                res = generate_batch(model, config, gargs, [piece(g, i) for g, i in drawn], a.seed * 1_000_003 + batch,
                                     a.rows, gen.device)
            made = 0
            for r, (row, (g, i)) in enumerate(zip(rows, drawn)):
                out = finish_row(res, r, row, None, gargs)
                toks.append(np.array(out.tokens[:4], dtype=np.int64).T)
                progs.append(np.array(out.tokens[4], dtype=np.int64))
                records.append(dict(row=row, prompt_store=store.name, prompt_file=i, group=g,
                                    prompt_notes=len(out.tokens[0]) - out.new_notes, new_notes=out.new_notes,
                                    eos=out.eos))
                made += len(out.tokens[0])
            print(f'shard {k} batch {b + 1}/{batches}: {made:,} notes, {made / (time.time() - tb):,.0f} notes/s',
                  flush=True)
        seconds = time.time() - t0
        write_store(shards_dir / f'shard_{k:05d}', toks, progs, records, dict(plan=plan, seconds=round(seconds, 1)))
        notes = sum(len(t) for t in toks)
        total += notes
        print(f'shard {k}: {len(toks)} files, {notes:,} notes in {seconds / 60:.1f} min; {total:,} of '
              f'{a.notes:,.0f}', flush=True)
        k += 1
    merge(shards_dir, out_store)


if __name__ == '__main__':
    main()
