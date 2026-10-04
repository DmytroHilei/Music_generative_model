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
    samples = [l for l in lines if l.startswith('samples @')][-n:]

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
        m = re.match(r'samples @(\d+) \((\d+) s\): (.*)', line)
        if not m:  # e.g. "samples @38000: FAILED (...), training continues"
            print('  ' + line[:110])
            continue
        for part in m.group(3).split(' | '):
            f = re.match(r'(\w+): instr ([\d.]+) oob (\d+)% nps (\d+) top (\d+)% runaway (\d+)% takeover (\d+)%', part)
            if f:
                d = ''
                if ref and 'sd' in ref:
                    vals = dict(instruments=float(f[2]), out_of_band=int(f[3]) / 100, notes_per_s=float(f[4]),
                                top_share=int(f[5]) / 100, runaway=int(f[6]) / 100, takeover=int(f[7]) / 100)
                    d = f'{distance(vals, ref):.2f}'
                print(f'{m.group(1):>7} {f[1]:>5} {f[2]:>6} {f[3]:>4}% {f[4]:>5} {f[5]:>4}% {f[6]:>7}% {f[7]:>8}% '
                      f'{d:>6}  {m.group(2)} s')


if __name__ == '__main__':
    main()
