"""Data for the training-curves row of architecture.tex: parses the run logs into small .dat tables in curves/.

    python docs/figures/curves_data.py

block_{base,newblock}_s{1,2,3}.dat  step, val CE (clean GigaMIDI) of the 42M block A/B (logs/abl_*.log)
block_{base,newblock}_mean.dat      the mean over the 3 seeds
base_450m.dat                       step, notes (B), clean GigaMIDI CE of the long 450M run
section_arms.dat                    arm, switch mean, held-out CE of the section-caption arms (logs/sections/forget.md)
gqa_{full,mean,pca,lw}.dat          step, held-out CE of the 107M GQA conversion arms (logs/sections/): full attention
                                    (forget_ctrl), mean-pooled K / V (forget_gqa4), the merge (gqa_pca), the merge +
                                    per-layer transfer (gqa_lw; its 300 transfer steps at steps -300..-1)
"""
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


if __name__ == '__main__':
    main()
