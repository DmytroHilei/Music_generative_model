"""cloud/run/plan_queue.sh in dry mode on a fake queue (CPU, ~10 s): python tests/tools/test_plan_queue.py (or pytest).
Covers pop order, need= deferral and promotion, an mv edit while a line runs, a gate inserting lines, the OPT guard,
a timeout killing the whole process group, the retry limit, the catch-all and the idle stop with AUTO_STOP=0."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'cloud/run/plan_queue.sh'

GATES = '''import os, sys
from pathlib import Path
a = sys.argv[1:]
with open('gates_calls.txt', 'a') as f:
    f.write(' '.join(a) + '\\n')
if a[0] == 'guard':
    sys.exit(1 if os.environ.get('GUARD_NO') else 0)
if a[0] == 'd1':
    q = Path('logs/plan_queue.txt'); t = q.with_suffix('.gtmp')
    t.write_text('min=0.01 id=g :: echo ran-g >> out.txt\\n' + q.read_text()); t.replace(q)
'''

QUEUE = '''# comment line
min=0.01 tier=ESS id=a :: echo ran-a >> out.txt
min=0.01 need=late.fl* id=b :: echo ran-b >> out.txt
min=0.01 id=c :: touch late.flag; echo ran-c >> out.txt; printf 'min=0.01 id=e :: echo ran-e >> out.txt\\n' | cat - logs/plan_queue.txt > q.new && mv q.new logs/plan_queue.txt
min=0.02 id=hang :: bash -c 'sleep 999.25 & wait'; echo after-hang >> out.txt

gate=d1
min=0.01 tier=OPT id=o1 :: echo ran-o1 >> out.txt
min=0.01 id=x need=never_there :: echo ran-x >> out.txt
bad line without fields
'''


def sleepers():
    out = subprocess.run(['ps', '-eo', 'args='], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.startswith('sleep 999.25')]


def run_queue(tmp, env_extra):
    (tmp / 'cloud/run').mkdir(parents=True)
    (tmp / 'tools/sweeps').mkdir(parents=True)
    (tmp / 'logs').mkdir()
    shutil.copy(SCRIPT, tmp / 'cloud/run/plan_queue.sh')
    (tmp / 'tools/sweeps/plan_gates.py').write_text(GATES)
    (tmp / 'logs/plan_queue.txt').write_text(QUEUE)
    (tmp / 'logs/plan_deferred.txt').write_text('min=0.01 need=nope id=z try=2 :: echo ran-z >> out.txt\n')
    env = dict(os.environ, PLAN_DRY='1', AUTO_STOP='0', PLAN_IDLE_MIN='0', PLAN_KILL='5s', PLAN_PY=sys.executable,
               **env_extra)
    env.pop('PLAN_COPY', None)
    env.pop('PLAN_ROOT', None)
    return subprocess.run(['bash', 'cloud/run/plan_queue.sh'], cwd=tmp, env=env, capture_output=True, text=True,
                          timeout=120)


def test_dry_queue():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = run_queue(tmp, {'GUARD_NO': '1'})
        log = (tmp / 'logs/plan_queue.log').read_text()
        assert p.returncode == 0, p.stdout + p.stderr
        assert (tmp / 'logs/plan_queue.run.sh').exists()  # ran from the private copy
        # a, then b deferred; c creates its file and mv-edits e onto the queue; b is promoted ahead of e
        assert (tmp / 'out.txt').read_text().split() == ['ran-a', 'ran-c', 'ran-b', 'ran-e', 'ran-g']
        assert 'END hang exit 124 (timeout)' in log and not sleepers()
        assert 'DEFER b: need late.fl*' in log and 'READY 1 deferred' in log
        assert 'SKIP o1: budget guard' in log
        assert 'BAD line' in log
        assert 'DROP z after 2 retries' in log
        calls = (tmp / 'gates_calls.txt').read_text().split('\n')
        assert calls[0] == 'd1' and 'guard --min 0.01' in calls and 'catchall' in calls
        deferred = (tmp / 'logs/plan_deferred.txt').read_text()
        assert 'id=x' in deferred and 'try=2' in deferred  # counted retries: after the first empty queue and after the catch-all
        assert 'hang' in (tmp / 'logs/plan_failed.txt').read_text()
        assert 'DRY upload' in p.stdout and 'AUTO_STOP=0: leaving the machine running' in log
        assert (tmp / 'logs/plan_queue.txt').read_text().strip() == '# comment line'


def test_guard_passes_opt_line():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = run_queue(tmp, {})
        assert p.returncode == 0, p.stdout + p.stderr
        assert 'ran-o1' in (tmp / 'out.txt').read_text().split()


def test_seed_format():
    import re
    ids = []
    for ln in (ROOT / 'cloud/run/plan_queue_seed.txt').read_text().splitlines():
        if not ln.strip() or ln.startswith('#'):
            continue
        if ln.startswith('gate='):
            assert re.fullmatch(r'gate=\w+( need=\S+)?( :: .+)?', ln), ln
            continue
        m = re.fullmatch(r'min=([\d.]+)( need=\S+)?( disk=\d+)? tier=(ESS|REC|OPT) id=(\S+) :: (.+)', ln)
        assert m, ln
        ids.append(m.group(5))
    assert len(ids) == len(set(ids))


if __name__ == '__main__':
    test_dry_queue()
    test_guard_passes_opt_line()
    test_seed_format()
    print('ok')
