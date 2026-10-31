"""Data for the charts of findings.tex: parses the run logs and results into small .dat tables in curves/.

    python docs/figures/curves_data.py

block_{base,newblock}_s{1,2,3}.dat  step, val CE (clean GigaMIDI) of the 42M block A/B (logs/abl_*.log)
block_{base,newblock}_mean.dat      the mean over the 3 seeds
base_450m.dat                       step, notes (B), clean GigaMIDI CE of the long 450M run
section_arms.dat                    arm, switch mean, held-out CE of the section-caption arms (logs/sections/forget.md)
gqa_{full,mean,pca,lw}.dat          step, held-out CE of the 107M GQA conversion arms (logs/sections/): full attention
                                    (forget_ctrl), mean-pooled K / V (forget_gqa4), the merge (gqa_pca), the merge +
                                    per-layer transfer (gqa_lw; its 300 transfer steps at steps -300..-1)
heads_cascade_s{1,2,3}.dat          step, held-out GigaMIDI CE of the 42M heads A/B (logs/muon_*.log): the cascade
heads_cascade_mean.dat              (muon_base, seeds 1337 / 2 / 3) and its mean, the same residual heads without the
heads_{indep,linear}.dat            chain (muon_indep) and independent linear heads (muon_linear), seed 1337
f1_run.dat                          step, notes (B), held-out captioned GigaMIDI CE, clean GigaMIDI CE of the F1 run
                                    (logs/f1/f1_evals.log, from the f1-results Hub repo)
h2h_cont.dat                        system, D of the head-to-head continuations (results/head2head/scores.json: the first
                                    1,000 new notes) and D200 (scores_first200.json: the first 200)
h2h_text.dat                        system, F over the EN / UK pairs of the head-to-head text task (scores.json)
h2h_piano.dat                       system, D and D200 of the head-to-head piano continuations (as h2h_cont.dat)
h2h_cont96.dat                      system, D and D200 of the 96-prompt continuations (scores_96.json,
                                    scores_96_first200.json), also over the published 32, the new 64 and the two folders
h2h_piano96.dat                     system, D and D200 of the 96-prompt piano continuations
h2h_midicaps.dat                    one row per score, one column per system: MidiCaps test set (midicaps/scores.json,
                                    midicaps/clamp3.json, first segment)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / 'logs'
OUT = Path(__file__).resolve().parent / 'curves'
NOTES_PER_STEP = 65_536


def evals(path, key):
    """(step, CE) pairs from 'step N: ... | <key> CE x' lines; the last value per step wins (resumes repeat steps)."""
    text = path.read_text(errors='replace')
    out = {}
    for m in re.finditer(rf'step (\d+): [^\n\r]*?\| {key} CE ([0-9.]+)', text):
        out[int(m.group(1))] = float(m.group(2))
    return sorted(out.items())


def write(name, header, rows):
    OUT.mkdir(exist_ok=True)
    lines = [' '.join(header)] + [' '.join(str(v) for v in r) for r in rows]
    (OUT / name).write_text('\n'.join(lines) + '\n')
    print(f'{name}: {len(rows)} rows')


def f1_run():
    clean = dict(evals(LOGS / 'f1/f1_evals.log', 'clean'))
    write('f1_run.dat', ('step', 'notes', 'ce', 'clean'),
          [(s, round(s * NOTES_PER_STEP / 1e9, 4), ce, clean[s]) for s, ce in
           evals(LOGS / 'f1/f1_evals.log', 'heldout_labeled') if s in clean])


H2H_CONT = ('real', 'long_450m', 'f1', 'amt', 'skytnt_medium', 'skytnt_large', 'moonbeam', 'moonbeam_t06')
H2H_TEXT = ('f1', 'f1_retrieve', 's50', 'caption', 'midi_llm', 'text2midi')
H2H_PIANO = ('real', 'ua_2048_v2', 'long_450m', 'aria', 'piano')


def head2head():
    scores = json.loads((ROOT / 'results/head2head/scores.json').read_text())
    first200 = json.loads((ROOT / 'results/head2head/scores_first200.json').read_text())
    cont, cont200 = scores.get('cont', {}), first200.get('cont', {})
    write('h2h_cont.dat', ('system', 'D', 'D200'),
          [(s, round(cont[s]['D'], 3), round(cont200[s]['D'], 3)) for s in H2H_CONT if s in cont])
    text = scores.get('text', {})
    rows = [(s, round(text[s]['F_en'], 3), round(text[s]['F_uk'], 3)) for s in H2H_TEXT
            if s in text and text[s].get('F_en') == text[s].get('F_en')]  # nan = not scored yet
    write('h2h_text.dat', ('system', 'F_en', 'F_uk'), rows)
    piano, piano200 = scores.get('piano', {}), first200.get('piano', {})
    write('h2h_piano.dat', ('system', 'D', 'D200'),
          [(s, round(piano[s]['D'], 3), round(piano200[s]['D'], 3)) for s in H2H_PIANO if s in piano])


H2H_GROUPS = dict(pub='published', new='new', nd='no_drums', wd='with_drums')
H2H_MIDICAPS = ('f1', 'midi_llm', 'text2midi', 'real', 'real_shuffled')
H2H_MIDICAPS_ROWS = dict(tb='TB', ck='CK', recall='recall', match='match')


def head2head_96():
    scores = json.loads((ROOT / 'results/head2head/scores_96.json').read_text())
    first200 = json.loads((ROOT / 'results/head2head/scores_96_first200.json').read_text())
    keys = [f'{k}{n}' for k in H2H_GROUPS for n in ('', '200')]
    rows = []
    for s in H2H_CONT:
        if s in scores['cont96']:
            row = [s, round(scores['cont96'][s]['D'], 3), round(first200['cont96'][s]['D'], 3)]
            for group in H2H_GROUPS.values():
                row += [round(table[f'cont96 {group}'][s]['D'], 3) for table in (scores, first200)]
            rows.append(row)
    write('h2h_cont96.dat', ('system', 'D', 'D200', *keys), rows)
    write('h2h_piano96.dat', ('system', 'D', 'D200'), [(s, round(scores['piano96'][s]['D'], 3),
                                                        round(first200['piano96'][s]['D'], 3)) for s in H2H_PIANO])


def head2head_midicaps():
    scores = json.loads((ROOT / 'results/head2head/midicaps/scores.json').read_text())['scores']
    clamp = json.loads((ROOT / 'results/head2head/midicaps/clamp3.json').read_text())['caption']['en']['first']
    rows = [[name, *(round(scores[k]['rows'][s]['mean'], 3) for s in H2H_MIDICAPS)]
            for k, name in H2H_MIDICAPS_ROWS.items()]
    rows.append(['MRR', *(round(clamp['mrr']['rows'][s]['mean'], 3) for s in H2H_MIDICAPS)])
    write('h2h_midicaps.dat', ('score', *H2H_MIDICAPS), rows)


def main():
    for arm in ('base', 'newblock'):
        seeds = []
        for i, suffix in enumerate(('', '_s2', '_s3'), 1):
            # the seed-3 runs name the same validation set 'gigamidi_clean' instead of 'val'
            rows = [(s, ce) for s, ce in evals(LOGS / f'abl_{arm}{suffix}.log', '(?:val|gigamidi_clean)') if s > 0]
            write(f'block_{arm}_s{i}.dat', ('step', 'ce'), rows)
            seeds.append(dict(rows))
        steps = sorted(set.intersection(*(set(r) for r in seeds)))
        write(f'block_{arm}_mean.dat', ('step', 'ce'), [(t, round(sum(r[t] for r in seeds) / 3, 4)) for t in steps])

    rows = evals(LOGS / 'long_450m_instance/hub/instance_logs/long_450m.log', 'gigamidi_clean')
    write('base_450m.dat', ('step', 'notes', 'ce'),
          [(s, round(s * NOTES_PER_STEP / 1e9, 4), ce) for s, ce in rows if s > 0])

    table = (LOGS / 'sections/forget.md').read_text()
    rows = []
    for arm in ('ctrl', 'repeat', 'cf', 'splice', 'all', 'all_v2', 's50_v2'):
        cells = next(l for l in table.splitlines() if l.startswith(f'| {arm} |')).strip('|').split('|')
        cells = [c.strip() for c in cells]
        rows.append((arm, float(cells[-1]), float(cells[2])))  # switch mean, held-out CE
    write('section_arms.dat', ('arm', 'switch', 'ce'), rows)

    sec = LOGS / 'sections'
    for name, log in (('full', 'forget_ctrl'), ('mean', 'forget_gqa4'), ('pca', 'gqa_pca')):
        write(f'gqa_{name}.dat', ('step', 'ce'), evals(sec / f'{log}.log', 'heldout_labeled'))
    transfer = [(s - 300, ce) for s, ce in evals(sec / 'gqa_lw_s1.log', 'heldout_labeled')]
    write('gqa_lw.dat', ('step', 'ce'), transfer + evals(sec / 'gqa_lw.log', 'heldout_labeled'))

    seeds = []
    for i, seed in enumerate((1337, 2, 3), 1):
        rows = [(t, ce) for t, ce in evals(LOGS / f'muon_base_s{seed}.log', 'gigamidi_heldout') if t > 0]
        write(f'heads_cascade_s{i}.dat', ('step', 'ce'), rows)
        seeds.append(dict(rows))
    steps = sorted(set.intersection(*(set(r) for r in seeds)))
    write('heads_cascade_mean.dat', ('step', 'ce'), [(t, round(sum(r[t] for r in seeds) / 3, 4)) for t in steps])
    for arm in ('indep', 'linear'):
        write(f'heads_{arm}.dat', ('step', 'ce'),
              [(t, ce) for t, ce in evals(LOGS / f'muon_{arm}_s1337.log', 'gigamidi_heldout') if t > 0])
    f1_run()
    head2head()
    head2head_96()
    head2head_midicaps()


if __name__ == '__main__':
    main()
