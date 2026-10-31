"""
CLaMP 3 score of the head-to-head (eval/compare/head2head.py): how near a system's files lie to their captions, and on
midicaps how near its files lie to real ones, in the space of CLaMP 3's C2 checkpoint (the symbolic music + text one).

    .venv/bin/python eval/compare/h2h_clamp3.py normalize --task text
    checkpoints/baselines/env_clamp3/bin/python eval/compare/h2h_clamp3.py embed --task text
    checkpoints/baselines/env_clamp3/bin/python eval/compare/h2h_clamp3.py score --task text

The authors' pipeline, written out here (checkpoints/baselines/clamp3/repo_clamp3 is the reference and is not run):
midi_to_mtf = batch_midi2mtf.load_midi with m3_compatible, mtf_patches = M3Patchilizer.encode, Clamp3 = the text and
symbolic encoders of CLaMP3Model with extract_feature's global features.

normalize  MTF holds raw MIDI messages (tempo, control changes, tick resolution), so files written by different systems
  differ in ways that reveal the writer, not the music. Every file, real ones too, goes read_midi -> tokens_to_midi
  (head2head's read-back) first, into <task>/clamp3_tmp/<system>/. Needs the main .venv.
embed  every normalised file and caption -> <task>/clamp3_emb/<system>.npz (ids, vectors, first, segments, patches) and
  _captions.npz. Existing ids are skipped, so a run resumes; clamp3_tmp goes when its files are embedded.
score  <task>/clamp3.json + clamp3.md, per caption language: mean own-caption cosine, own minus the other captions
  (the language's on text; --others random ones per prompt on midicaps, also the field of the rank) and the reciprocal
  rank of the own caption, which controls for systems that score high on every caption. Bootstrap over prompts: one
  resampling for all rows, 95% percentile interval, share of draws where a row beats another. With a real/ folder:
  real = the ceiling, real_shuffled (each real file against another prompt's caption) = the floor.
  Frechet distance (midicaps) of each row to the real_all pool without the files of real/, on the raw features, real/
  included as the floor; all rows on the prompts they share, so one N. It is computed here on CLaMP 3 C2 embeddings
  instead of the frechet-music-distance package and its CLaMP 2 weights, to save disk.

Length. The authors cut a piece into segments of 512 patches (the last one = the last 512, overlapping) and take the
mean of the segment features weighted by the patches each adds. So a longer file is averaged over more segments, which
alone raises its cosine to every caption and its rank: on 441 real songs with their MidiCaps captions, first segment
-> whole file moved own 0.117 -> 0.139, contrast 0.136 -> 0.146, MRR 0.392 -> 0.426, and the Frechet between two halves
of the songs on the raw features rose 13% when only one half was averaged. Generated files run under a minute, real
songs for minutes. So every score comes twice: first = the file's first segment alone, the same input size for all
rows (the headline), global = the authors' feature of the whole file. A file under 512 patches stays shorter, which
cannot be avoided without cutting every file to the shortest: the same songs at 256 instead of 512 patches lost 0.004
contrast and 0.017 MRR. clamp3.md gives the patches per row.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'head2head'
CLAMP3 = ROOT / 'checkpoints' / 'baselines' / 'clamp3'
WEIGHTS = ('weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12'
           '_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth')
TEXT_MODEL = 'FacebookAI/xlm-roberta-base'
HIDDEN, PATCH_SIZE, PATCH_LENGTH, TEXT_LENGTH = 768, 64, 512, 128  # code/config.py
M3_SKIPPED = ('text', 'copyright', 'track_name', 'instrument_name', 'lyrics', 'marker', 'cue_marker', 'device_name')
POOL = 'real_all'  # midicaps: the real reference pool, a folder of .mid files but no row of its own
FIELDS = ('vectors', 'first', 'segments', 'patches')
METRICS = ('own', 'contrast', 'mrr')
WINDOWS = {'first': 'first', 'global': 'vectors'}  # report name -> embedding field, the headline first
LABELS = dict(first='first segment', own='own cosine', contrast='contrast', mrr='MRR',
              **{'global': 'whole file (the authors\' feature)'})


def prompts_of(task, n=0):
    """The prompts with a caption ('' = unconditioned has none to match), first n if n > 0; not the `excluded` ones."""
    rows = json.loads((OUT / task / 'prompts.json').read_text())['prompts']
    return [p for p in rows if p['caption'] and not p.get('excluded')][:n or None]


def systems_of(task, only=()):
    """The folders of <task>/ that hold .mid files."""
    found = sorted(d.name for d in (OUT / task).iterdir() if d.is_dir() and next(d.glob('*.mid'), None))
    return [s for s in found if s in only] if only else found


def prompt_id(stem):
    """<prompt id>_<k> -> <prompt id>."""
    return stem.rpartition('_')[0]


def read_emb(path):
    """The arrays of an embedding file, or {} if it does not exist yet."""
    if not path.exists():
        return {}
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def write_emb(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def pool_excluded(task_dir):
    """The file stems of the real pool that may not enter it (h2h_midicaps eligible)."""
    path = task_dir / 'eligible.json'
    return set(json.loads(path.read_text())['excluded']) if path.exists() else set()


def normalize(args):
    sys.path[:0] = [str(ROOT), str(ROOT / 'eval' / 'compare')]
    from head2head import read_midi, write_midi

    task_dir = OUT / args.task
    ids = {p['id'] for p in prompts_of(args.task, args.prompts)}
    left_out = pool_excluded(task_dir)
    for system in systems_of(args.task, args.systems):
        done = set(read_emb(task_dir / 'clamp3_emb' / f'{system}.npz').get('ids', np.array([])).tolist())
        tmp = task_dir / 'clamp3_tmp' / system
        files = [f for f in sorted((task_dir / system).glob('*.mid'))
                 if (f.stem not in left_out if system == POOL else prompt_id(f.stem) in ids)]
        todo = [f for f in files if f.stem not in done and not (tmp / f.name).exists()]
        t0, bad = time.time(), []
        for f in todo:
            res = read_midi(f)
            if res is None:
                bad.append(f.name)
                continue
            write_midi(tmp / f.name, *res)
        print(f'{system}: {len(todo) - len(bad)} of {len(files)} files normalised in {time.time() - t0:.0f} s'
              + (f', unreadable: {bad}' if bad else ''), flush=True)


def midi_to_mtf(path):
    """batch_midi2mtf.load_midi with m3_compatible."""
    import mido

    mid = mido.MidiFile(path)
    lines = [f'ticks_per_beat {mid.ticks_per_beat}']
    for msg in mid.merged_track:
        if not (msg.is_meta and msg.type in M3_SKIPPED):
            lines.append(' '.join(str(v) for v in msg.dict().values()).strip().encode('unicode_escape').decode())
    return '\n'.join(lines)


def mtf_patches(mtf):
    """M3Patchilizer.encode(mtf, add_special_patches=True): a run of messages of one type shares a patch while it fits,
    BOS and EOS patches around. Its unidecode is left out: MTF is ASCII after unicode_escape."""
    patches, patch = [], ''
    for line in filter(None, re.findall(r'.*?\n|.*$', mtf)):
        head, _, rest = line.partition(' ')
        if patch.startswith(head) and len(patch) + len(rest) <= PATCH_SIZE - 2:
            patch = patch[:-1] + '\t' + rest
        else:
            if patch:
                patches.append(patch)
            patch = line
    if patch:
        patches.append(patch)
    rows = [[1, *map(ord, p), 2][:PATCH_SIZE] for p in [chr(1) * PATCH_SIZE, *patches, chr(2) * PATCH_SIZE]]
    return [r + [0] * (PATCH_SIZE - len(r)) for r in rows]


def segment_spans(n, length):
    """extract_feature's cut of n items: (spans, weights), consecutive segments of `length` whose last one is the last
    `length` items, overlapping the one before it, and the items each segment adds, which weigh its feature."""
    starts = list(range(0, n, length))
    spans = [(s, s + length) for s in starts[:-1]] + [(max(0, n - length), n)]
    return spans, [length] * (n // length) + ([n % length] if n % length else [])


class Clamp3:
    """The text and symbolic encoders of the C2 checkpoint on the state dict's own names (the file holds the whole text
    encoder, so xlm-roberta-base gives only its config and tokenizer). The load is strict: every text and symbolic
    weight of the file finds its module."""

    def __init__(self, device):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel

        bert = BertConfig(vocab_size=1, hidden_size=HIDDEN, num_hidden_layers=12, num_attention_heads=HIDDEN // 64,
                          intermediate_size=HIDDEN * 4, max_position_embeddings=PATCH_LENGTH)
        self.net = torch.nn.ModuleDict(dict(
            text_model=AutoModel.from_config(AutoConfig.from_pretrained(TEXT_MODEL)),
            text_proj=torch.nn.Linear(HIDDEN, HIDDEN),
            symbolic_model=torch.nn.ModuleDict(dict(patch_embedding=torch.nn.Linear(PATCH_SIZE * 128, HIDDEN),
                                                    base=BertModel(bert))),
            symbolic_proj=torch.nn.Linear(HIDDEN, HIDDEN)))
        state = torch.load(CLAMP3 / WEIGHTS, map_location='cpu', weights_only=True, mmap=True)['model']
        self.net.load_state_dict({k: v for k, v in state.items() if not k.startswith('audio_')})
        self.net.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.torch, self.device = torch, device

    def segments(self, ids, length, pad, hidden, proj):
        """One projected feature per segment_spans, masked mean over the segment's items, and the weighted mean of them:
        (global feature, per-segment features)."""
        torch = self.torch
        spans, weights = segment_spans(len(ids), length)
        feats = []
        for a, b in spans:
            part = ids[a:b]
            mask = torch.zeros(length, device=self.device)
            mask[:len(part)] = 1
            filler = torch.full((length - len(part), *part.shape[1:]), pad, device=self.device, dtype=part.dtype)
            state = hidden(torch.cat([part, filler])[None], mask[None])
            feats.append(proj((state * mask[None, :, None]).sum(1) / mask.sum()))
        feats = torch.cat(feats)
        w = torch.tensor(weights, device=self.device, dtype=feats.dtype)[:, None]
        return (feats * w).sum(0) / w.sum(), feats

    def text(self, caption):
        """The global feature of a caption; its lines are joined by the tokenizer's separator."""
        item = self.tokenizer.sep_token.join(dict.fromkeys(filter(None, caption.split('\n'))))
        ids = self.tokenizer(item, return_tensors='pt')['input_ids'][0].to(self.device)
        with self.torch.no_grad():
            return self.segments(ids, TEXT_LENGTH, self.tokenizer.pad_token_id,
                                 lambda x, m: self.net.text_model(x, attention_mask=m)['last_hidden_state'],
                                 self.net.text_proj)[0].cpu().numpy()

    def music(self, midi):
        """The FIELDS of a MIDI file."""
        def hidden(x, m):
            one_hot = self.torch.nn.functional.one_hot(x, num_classes=128).reshape(1, -1, PATCH_SIZE * 128).float()
            model = self.net.symbolic_model
            return model.base(inputs_embeds=model.patch_embedding(one_hot), attention_mask=m)['last_hidden_state']

        patches = self.torch.tensor(mtf_patches(midi_to_mtf(midi)), device=self.device)
        with self.torch.no_grad():
            whole, feats = self.segments(patches, PATCH_LENGTH, 0, hidden, self.net.symbolic_proj)
        return dict(vectors=whole.cpu().numpy(), first=feats[0].cpu().numpy(), segments=len(feats),
                    patches=len(patches))


def embed(args):
    task_dir = OUT / args.task
    tmp, emb = task_dir / 'clamp3_tmp', task_dir / 'clamp3_emb'
    emb.mkdir(exist_ok=True)
    model = Clamp3(args.device)

    old = read_emb(emb / '_captions.npz')
    done = dict(zip(old.get('ids', np.array([])).tolist(), old.get('vectors', [])))
    t0 = time.time()
    for p in prompts_of(args.task):
        if p['id'] not in done:
            done[p['id']] = model.text(p['caption'])
    write_emb(emb / '_captions.npz', ids=np.array(list(done)), vectors=np.stack(list(done.values())))
    print(f'{len(done)} captions embedded in {time.time() - t0:.1f} s', flush=True)

    for folder in sorted(d for d in tmp.iterdir() if d.is_dir()) if tmp.exists() else []:
        old = read_emb(emb / f'{folder.name}.npz')
        rows = {i: {k: old[k][n] for k in FIELDS} for n, i in enumerate(old.get('ids', np.array([])).tolist())}
        files = [f for f in sorted(folder.glob('*.mid')) if f.stem not in rows]
        t0 = time.time()
        for f in files:
            rows[f.stem] = model.music(f)
        write_emb(emb / f'{folder.name}.npz', ids=np.array(list(rows)),
                  **{k: np.stack([r[k] for r in rows.values()]) for k in FIELDS})
        shutil.rmtree(folder)
        print(f'{folder.name}: {len(files)} files embedded in {time.time() - t0:.1f} s', flush=True)
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def comparators(n, k, seed):
    """(n, min(k, n - 1)) indices: for each of n captions, a fixed random k of the other captions."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(np.delete(np.arange(n), i))[:k] for i in range(n)])


def deranged(n, seed):
    """A random permutation of range(n) without a fixed point: one cycle through a shuffled order."""
    order = np.random.default_rng(seed).permutation(n)
    out = np.empty(n, int)
    out[order] = np.roll(order, 1)
    return out


def file_scores(music, captions, own, others):
    """Per file (own-caption cosine, own minus the mean of the comparators, reciprocal rank of the own caption among
    them, ties against it). music (n, d) and captions (m, d) unit vectors, own (n,) caption index, others (m, k)."""
    cos = music @ captions.T
    mine = cos[np.arange(len(music)), own]
    rest = np.take_along_axis(cos, others[own], 1)
    return dict(own=mine, contrast=mine - rest.mean(1), mrr=1 / (1 + (rest >= mine[:, None]).sum(1)))


def per_prompt(values, own, n):
    """Mean of the files of each of n prompts, nan where a prompt has no file."""
    with np.errstate(invalid='ignore'):
        return np.bincount(own, weights=values, minlength=n) / np.bincount(own, minlength=n)


def wins(boots, sign=1):
    """{row: {other: share of draws where sign * row > sign * other}}."""
    return {a: {b: float((sign * boots[a] > sign * boots[b]).mean()) for b in boots if b != a} for a in boots}


def bootstrap(table, draws, seed):
    """table {row: (units,) values, nan = missing}, resampled with replacement, one resampling for all rows; units no
    row has are left out. dict(rows={row: dict(mean, lo, hi)}, wins=wins())."""
    values = np.stack(list(table.values()))
    values = values[:, ~np.isnan(values).all(0)]
    idx = np.random.default_rng(seed).integers(0, values.shape[1], (draws, values.shape[1]))
    boots, rows = {}, {}
    for name, v in zip(table, values):
        ok = ~np.isnan(v)
        with np.errstate(invalid='ignore'):
            boots[name] = np.where(ok, v, 0)[idx].sum(1) / ok[idx].sum(1)
        lo, hi = np.nanpercentile(boots[name], [2.5, 97.5])
        rows[name] = dict(mean=float(np.nanmean(v)), lo=float(lo), hi=float(hi))
    return dict(rows=rows, wins=wins(boots))


def frechet_to(reference):
    """distance(x): Frechet distance between the Gaussians of the rows of x and of reference, ||m1 - m2||^2 +
    Tr(C1 + C2 - 2 (C1 C2)^1/2); the root's trace through the symmetric C2^1/2 C1 C2^1/2."""
    mu, cov = reference.mean(0), np.cov(reference, rowvar=False)
    w, v = np.linalg.eigh(cov)
    half = (v * np.sqrt(w.clip(0))) @ v.T

    def distance(x):
        c = np.cov(x, rowvar=False)
        root = np.sqrt(np.linalg.eigvalsh(half @ c @ half).clip(0)).sum()
        return float(((x.mean(0) - mu) ** 2).sum() + np.trace(c) + np.trace(cov) - 2 * root)
    return distance


def frechet_boot(table, reference, draws, seed):
    """table {row: (n, d) files paired by index}, resampled with one resampling for all rows, the reference fixed.
    A resample repeats files, which alone raises the distance (real songs, N = 380: 45.6 -> a mean of 63.1), so each
    row's draws are shifted to centre on its value. dict(rows={row: dict(mean, lo, hi)}, wins=wins(): lower beats)."""
    distance = frechet_to(reference)
    n = len(next(iter(table.values())))
    idx = np.random.default_rng(seed).integers(0, n, (draws, n))
    boots, rows = {}, {}
    for name, x in table.items():
        value = distance(x)
        d = np.array([distance(x[i]) for i in idx])
        boots[name] = d + value - d.mean()
        lo, hi = np.percentile(boots[name], [2.5, 97.5])
        rows[name] = dict(mean=value, lo=float(lo), hi=float(hi))
    return dict(rows=rows, wins=wins(boots, -1))


def disjoint(pool, real):
    """Mask of the pool's rows that are no file of real, by embedding: a same normalised file has cosine 1."""
    return (unit(pool) @ unit(real).T).max(1) < 1 - 1e-6


def caption_scores(rows, ids, captions, others, swap, field):
    """{row: {metric: per-prompt means over the prompts `ids`}}, real_shuffled after real."""
    index = {p: i for i, p in enumerate(ids)}
    table = {}
    for name, emb in rows.items():
        keep = [n for n, s in enumerate(emb['ids'].tolist()) if prompt_id(s) in index]
        own = np.array([index[prompt_id(s)] for s in emb['ids'][keep].tolist()], int)
        music = unit(emb[field][keep].astype(np.float64))
        for row, target in [(name, own)] + ([('real_shuffled', swap[own])] if name == 'real' else []):
            scores = file_scores(music, captions, target, others)
            table[row] = {m: per_prompt(scores[m], target, len(ids)) for m in METRICS}
    return table


def frechet_rows(rows, pool, field, draws, seed):
    """frechet_boot of every row on the files all rows have, against the pool without the files of real; + n and the
    reference size."""
    stems = sorted(set.intersection(*[set(r['ids'].tolist()) for r in rows.values()]))
    table = {}
    for name, emb in rows.items():
        place = {s: i for i, s in enumerate(emb['ids'].tolist())}
        table[name] = emb[field][[place[s] for s in stems]].astype(np.float64)
    reference = pool[field][disjoint(pool['vectors'].astype(np.float64), rows['real']['vectors'].astype(np.float64))]
    return dict(frechet_boot(table, reference.astype(np.float64), draws, seed), n=len(stems), reference=len(reference))


def stat(s):
    return f"{s['mean']:.3f} ({s['lo']:.3f} to {s['hi']:.3f})"


def win_table(title, result):
    names = list(result['rows'])
    lines = [f'{title}: share of draws where the row beats the column', '', '| | ' + ' | '.join(names) + ' |',
             '|---' * (len(names) + 1) + '|']
    for a in names:
        lines.append(f'| {a} | ' + ' | '.join('' if a == b else f"{result['wins'][a][b]:.3f}" for b in names) + ' |')
    return lines + ['']


def report(res):
    """clamp3.md of a clamp3.json."""
    lines = [f"# CLaMP 3 scores, {res['task']} ({res['date']})", '',
             f"C2 checkpoint, every file through read_midi -> tokens_to_midi. {res['draws']} bootstrap draws over "
             f"prompts (seed {res['seed']}), 95% percentile interval. Up to {res['others']} other captions per prompt. "
             "first = the file's first segment of 512 patches, the same input size for all rows; global = the "
             "authors' whole-file feature, which favours longer files (module docstring).", '',
             '## Length', '', '| system | files | patches | segments | files over one segment |',
             '|---|---|---|---|---|']
    lines += [f"| {k} | {v['files']} | {v['patches']:.0f} | {v['segments']:.2f} | {v['over']:.3f} |"
              for k, v in res['length'].items()]
    lines.append('')
    for lang, windows in res['caption'].items():
        for window, metrics in windows.items():
            names = list(metrics['own']['rows'])
            lines += [f'## {lang}, {LABELS[window]}', '',
                      '| system | ' + ' | '.join(LABELS[m] for m in METRICS) + ' |',
                      '|---' * (len(METRICS) + 1) + '|']
            lines += [f'| {n} | ' + ' | '.join(stat(metrics[m]['rows'][n]) for m in METRICS) + ' |' for n in names]
            lines.append('')
            if window == 'first':
                for m in METRICS:
                    lines += win_table(f'{lang}, {LABELS[window]}, {LABELS[m]}', metrics[m])
    for window, f in res.get('frechet', {}).items():
        lines += [f"## Frechet to the real pool, {LABELS[window]} (lower is better)", '',
                  f"N = {f['n']} files per row, {f['reference']} reference files outside real/, on the raw features. "
                  f"{res['frechet_draws']} draws over files, each row's draws centred on its value (a resample repeats "
                  'files, which alone raises the distance). real = the floor at this N.', '',
                  '| system | Frechet |', '|---|---|']
        lines += [f'| {n} | {stat(s)} |' for n, s in f['rows'].items()]
        lines.append('')
        lines += win_table(f'Frechet, {LABELS[window]}', f)
    return '\n'.join(lines)


def score(args):
    task_dir, emb = OUT / args.task, OUT / args.task / 'clamp3_emb'
    stored = read_emb(emb / '_captions.npz')
    vectors = dict(zip(stored['ids'].tolist(), unit(stored['vectors'].astype(np.float64))))
    rows = {p.stem: read_emb(p) for p in sorted(emb.glob('*.npz')) if not p.stem.startswith('_')}
    pool = rows.pop(POOL, None)
    groups = {}
    for p in prompts_of(args.task):
        groups.setdefault('uk' if p['id'].startswith('uk_') else 'en', []).append(p['id'])

    res = dict(task=args.task, date=time.strftime('%Y-%m-%d %H:%M'), draws=args.draws, seed=args.seed,
               others=args.others,
               length={k: dict(files=len(r['ids']), patches=float(r['patches'].mean()),
                               segments=float(r['segments'].mean()), over=float((r['segments'] > 1).mean()))
                       for k, r in rows.items()}, caption={})
    have = {prompt_id(s) for r in rows.values() for s in r['ids'].tolist()}
    for lang, ids in groups.items():
        if not have.intersection(ids):
            continue
        captions = np.stack([vectors[p] for p in ids])
        others = comparators(len(ids), args.others, args.seed)
        swap = deranged(len(ids), args.seed)
        res['caption'][lang] = {}
        for window, field in WINDOWS.items():
            table = caption_scores(rows, ids, captions, others, swap, field)
            res['caption'][lang][window] = {m: bootstrap({r: t[m] for r, t in table.items()}, args.draws, args.seed)
                                            for m in METRICS}
    if pool is not None and 'real' in rows:
        res['frechet_draws'] = args.frechet_draws
        res['frechet'] = {w: frechet_rows(rows, pool, f, args.frechet_draws, args.seed) for w, f in WINDOWS.items()}
    (task_dir / 'clamp3.json').write_text(json.dumps(res, indent=1))
    (task_dir / 'clamp3.md').write_text(report(res))
    print(report(res))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('normalize', help='one MIDI writer for every file (main .venv)')
    n.add_argument('--task', default='text')
    n.add_argument('--systems', nargs='*', default=[], help='only these folders (default: all with .mid files)')
    n.add_argument('--prompts', type=int, default=0, help='> 0: only the first N captions (smoke tests)')
    e = sub.add_parser('embed', help='normalised files and captions -> clamp3_emb (env_clamp3)')
    e.add_argument('--task', default='text')
    e.add_argument('--device', default='cuda')
    s = sub.add_parser('score', help='clamp3_emb -> clamp3.json / clamp3.md (env_clamp3)')
    s.add_argument('--task', default='text')
    s.add_argument('--draws', type=int, default=10000, help='bootstrap draws over prompts')
    s.add_argument('--frechet-draws', type=int, default=1000, help='bootstrap draws over files (33 ms each per row)')
    s.add_argument('--others', type=int, default=100, help='other captions per prompt, at most the language\'s rest')
    s.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    {'normalize': normalize, 'embed': embed, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    main()
