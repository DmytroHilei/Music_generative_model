"""The last N evaluations and sample evaluations of a training log as two small tables (used by cloud/status.sh).
    python3 cloud/status_tables.py logs/long_450m.log [N]
Only the standard library: runs with the system python3."""

import re
import sys


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
    print(f'{"step":>7} {"mode":>5} {"instr":>6} {"oob":>5} {"nps":>5} {"top":>5} {"runaway":>8} {"takeover":>9}  time')
    for line in samples:
        m = re.match(r'samples @(\d+) \((\d+) s\): (.*)', line)
        if not m:  # e.g. "samples @38000: FAILED (...), training continues"
            print('  ' + line[:110])
            continue
        for part in m.group(3).split(' | '):
            f = re.match(r'(\w+): instr ([\d.]+) oob (\d+)% nps (\d+) top (\d+)% runaway (\d+)% takeover (\d+)%', part)
            if f:
                print(f'{m.group(1):>7} {f[1]:>5} {f[2]:>6} {f[3]:>4}% {f[4]:>5} {f[5]:>4}% {f[6]:>7}% {f[7]:>8}%  '
                      f'{m.group(2)} s')


if __name__ == '__main__':
    main()
