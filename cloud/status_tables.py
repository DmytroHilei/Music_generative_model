"""The last N evaluations and sample evaluations of a training log as two small tables (used by cloud/status.sh).
    python3 cloud/status_tables.py logs/long_450m.log [N]
Only the standard library: runs with the system python3."""

import json
import re
import sys
from pathlib import Path


# same definition as sample_eval.distance (kept here so this file needs only the standard library)
DIST_WEIGHTS = {'takeover': 0.20, 'top_share': 0.20, 'notes_per_s': 0.20, 'out_of_band': 0.15, 'instruments': 0.15,
                'runaway': 0.10}
SD_FLOOR = {'takeover': 0.22, 'runaway': 0.22, 'top_share': 0.05, 'out_of_band': 0.05, 'notes_per_s': 5.0,
            'instruments': 1.0}


def distance(m, ref):
    return sum(w * abs(m[k] - ref[k]) / max(ref['sd'].get(k, 0.0), SD_FLOOR[k]) for k, w in DIST_WEIGHTS.items())


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    try:
        text = open(path, errors='ignore').read().replace('\r', '\n')
    except OSError:
        return
    lines = text.splitlines()
    evals = [l for l in lines if re.match(r'step \d+:', l)][-n:]
    all_samples = [l for l in lines if l.startswith('samples @')]
    samples = all_samples[-n:]

    print('--- last evals (val CE)')
    cols = None
    for line in evals:
        step = re.match(r'step (\d+):', line).group(1)
        named = re.findall(r'\| (\S+) CE ([\d.]+)', line)
        if cols is None:
            cols = [name for name, _ in named]
            print(f'{"step":>7} ' + ' '.join(f'{c[:14]:>14}' for c in cols))
        values = dict(named)
        print(f'{step:>7} ' + ' '.join(f'{values.get(c, ""):>14}' for c in cols))

    print('--- last sample evals (samples/* on wandb)')
    print(f'{"step":>7} {"mode":>5} {"instr":>6} {"oob":>5} {"nps":>5} {"top":>5} {"runaway":>8} {"takeover":>9} '
          f'{"D":>6}  time')
    ref_path = Path(__file__).with_name('sample_reference.json')  # real continuations of the same prompts
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
    if ref:
        r = ref
        print(f'{"real":>7} {"":>5} {r["instruments"]:>6.1f} {100 * r["out_of_band"]:>4.0f}% {r["notes_per_s"]:>5.0f} '
              f'{100 * r["top_share"]:>4.0f}% {100 * r["runaway"]:>7.0f}% {100 * r["takeover"]:>8.0f}% {0:>6.2f}  (target)')
    for line in samples:
        m = SAMPLE.match(line)
        if not m:  # e.g. "samples @38000: FAILED (...), training continues"
            print('  ' + line[:110])
            continue
        for part in m.group(4).split(' | '):
            f = re.match(r'(\w+): instr ([\d.]+) oob (\d+)% nps (\d+) top (\d+)% runaway (\d+)% takeover (\d+)%', part)
            if f:
                d = ''
                if ref and 'sd' in ref:
                    vals = dict(instruments=float(f[2]), out_of_band=int(f[3]) / 100, notes_per_s=float(f[4]),
                                top_share=int(f[5]) / 100, runaway=int(f[6]) / 100, takeover=int(f[7]) / 100)
                    d = f'{distance(vals, ref):.2f}'
                print(f'{m.group(1):>7} {f[1]:>5} {f[2]:>6} {f[3]:>4}% {f[4]:>5} {f[5]:>4}% {f[6]:>7}% {f[7]:>8}% '
                      f'{d:>6}  {m.group(3)} s')
    trend(all_samples, ref)
    style_table(lines)



SAMPLE = re.compile(r'samples @(\d+) \((?:(\d+) rows, )?(\d+) s\): (.*)')
PART = re.compile(r'(\w+): instr ([\d.]+) oob (\d+)% nps (\d+) top (\d+)% runaway (\d+)% takeover (\d+)%')
EIGHT_ROW_RATES = {0, 12, 25, 38, 50, 62, 75, 88, 100}  # k/8 rounded: what 8-row evals can print


def rows_of(match, parts):
    """Rows of a sample eval: printed since the 2026-10-04 change, else inferred (8-row evals can only print k/8)."""
    if match.group(2):
        return int(match.group(2))
    rates = {int(p[6]) for p in parts} | {int(p[7]) for p in parts}
    return 8 if rates <= EIGHT_ROW_RATES else 32


def trend(all_samples, ref, k=8, se_point=0.07):
    """Slope of D over the last <= k evals with the current row count, per mode, per 10k steps."""
    if not ref or 'sd' not in ref:
        return
    evals = []
    for line in all_samples:
        m = SAMPLE.match(line)
        if not m:
            continue
        parts = [PART.match(p) for p in m.group(4).split(' | ')]
        parts = [p for p in parts if p]
        if parts:
            evals.append((int(m.group(1)), rows_of(m, parts), parts))
    if not evals:
        return
    rows = evals[-1][1]
    same = [e for e in evals if e[1] == rows][-k:]
    print(f'--- D trend over the last {len(same)} evals with {rows} rows (steps {same[0][0]}-{same[-1][0]})')
    if len(same) < 3:
        print('  need >= 3 evals for a slope')
        return
    for mode in [p[1] for p in same[-1][2]]:
        pts = []
        for step, _, parts in same:
            for p in parts:
                if p[1] == mode:
                    vals = dict(instruments=float(p[2]), out_of_band=int(p[3]) / 100, notes_per_s=float(p[4]),
                                top_share=int(p[5]) / 100, runaway=int(p[6]) / 100, takeover=int(p[7]) / 100)
                    pts.append((step / 1e4, distance(vals, ref)))
        if len(pts) < 3:
            continue
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        resid = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)
        # the residual estimate is shaky with few points: never below the known per-eval noise of D
        se = max(resid, se_point ** 2) ** 0.5 / sxx ** 0.5
        verdict = ('RISING' if slope > 2 * se else 'falling' if slope < -2 * se else 'flat (not significant)')
        print(f'  {mode:5s} D {ys[0]:.2f} -> {ys[-1]:.2f}, slope {slope:+.3f} +- {se:.3f} per 10k steps: {verdict}')



STYLE_LINE = re.compile(r'styles @(\d+) \((\d+) rows, (\d+) s\): (.*)')
STYLE_PART = re.compile(r'(\w+): P ([\d.]+) \[(.*?)\] nps (\d+) top (\d+)% takeover (\d+)%')


def style_table(lines, n=4, k=8):
    """Per-style evaluation (style_eval.py): last n evals as a table, slope of P over the last k."""
    evals = []
    for line in lines:
        m = STYLE_LINE.match(line)
        if m:
            parts = {p[1]: (float(p[2]), dict((a, float(b)) for a, b in re.findall(r'(\w+) ([\d.]+)', p[3])),
                            int(p[4]), int(p[5]), int(p[6])) for p in STYLE_PART.finditer(m.group(4))}
            evals.append((int(m.group(1)), int(m.group(3)), parts))
        elif line.startswith('styles @'):
            evals.append((None, line, None))
    if not evals:
        return
    styles = ['rock_pop', 'orchestral', 'piano_keys', 'electronic', 'acoustic']
    print('--- per-style distance to real songs (styles/* on wandb; 0 = like the real continuations)')
    print(f'{"step":>7} {"mode":>5} {"all":>6} ' + ' '.join(f'{s[:9]:>9}' for s in styles) +
          f' {"nps":>5} {"top":>5} {"takeovr":>7}  time')
    for step, secs, parts in evals[-n:]:
        if parts is None:
            print('  ' + secs[:110])
            continue
        for mode, (p_all, per, nps, top, tk) in parts.items():
            print(f'{step:>7} {mode:>5} {p_all:>6.2f} ' + ' '.join(f'{per.get(s, float("nan")):>9.2f}' for s in styles)
                  + f' {nps:>5} {top:>4}% {tk:>6}%  {secs} s')
    pts = [(step, parts) for step, _, parts in evals if parts][-k:]
    if len(pts) < 3:
        print(f'  P trend: need >= 3 evals (have {len(pts)})')
        return
    for mode in pts[-1][1]:
        xs = [s / 1e4 for s, p in pts if mode in p]
        ys = [p[mode][0] for s, p in pts if mode in p]
        n_ = len(xs)
        mx, my = sum(xs) / n_, sum(ys) / n_
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        resid = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys)) / max(n_ - 2, 1)
        se = max(resid, 0.03 ** 2) ** 0.5 / sxx ** 0.5
        verdict = 'RISING (worse)' if slope > 2 * se else 'falling (better)' if slope < -2 * se else 'flat (n.s.)'
        print(f'  P trend {mode:5s} {ys[0]:.2f} -> {ys[-1]:.2f}, slope {slope:+.3f} +- {se:.3f} per 10k steps: {verdict}')


if __name__ == '__main__':
    main()
