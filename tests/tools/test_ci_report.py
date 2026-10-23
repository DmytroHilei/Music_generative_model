"""tests/ci_report.py checks (no GPU): python tests/tools/test_ci_report.py  (or tests/run.py)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_report  # noqa: E402
from helpers import run  # noqa: E402

TB = '''Traceback (most recent call last):
  File "x.py", line 1, in <module>
    f()
KeyError: 'n_kv_head'
'''


def make_run(ci, name, results, plan=None, finished=True, running=None):
    d = ci / name
    d.mkdir()
    plan = plan or list(results)
    summary = dict(commit='abc1234', started=name, gpu='ok', minutes=0.5, finished=finished, running=running,
                   plan=[dict(name=n, gpu=False, needs=[], doc=f'doc of {n}') for n in plan], results=results)
    (d / 'summary.json').write_text(json.dumps(summary))
    return d, summary


def test_failed_test_detail():
    with tempfile.TemporaryDirectory() as tmp:
        ci = Path(tmp)
        d, summary = make_run(ci, '20261023_100000', {
            'lint': dict(status='PASS', seconds=0.1, note='10 files'),
            'unit-cpu': dict(status='FAIL', seconds=2.0, note='1/2 passed; failed: a.py::t'),
            'train-gpu': dict(status='SKIP', seconds=0.0, note='skipped (--cpu-only)')})
        (d / 'unit-cpu.log').write_text(f'$ python tests/run.py\nok    b.py::u\nFAIL  a.py::t\n{TB}1/2 passed\n[exit 1]\n')
        ci_report.write_run(d, ci_dir=ci)
        txt = (d / 'report.txt').read_text()
        assert 'FAIL     unit-cpu' in txt and 'command: python tests/run.py' in txt and "KeyError: 'n_kv_head'" in txt
        page = (d / 'report.html').read_text()
        assert '<details class="FAIL-row" open>' in page and '<details class="PASS-row">' in page
        assert 'href="unit-cpu.log"' in page and 'http-equiv="refresh"' not in page, 'a finished run stops reloading'
        assert 'id="failed-only"' in page and 'KeyError' in page


def test_running_run_and_index():
    with tempfile.TemporaryDirectory() as tmp:
        ci = Path(tmp)
        old, _ = make_run(ci, '20261023_100000', {'lint': dict(status='FAIL', seconds=0.1, note='syntax')})
        new, summary = make_run(ci, '20261023_110000', {'lint': dict(status='PASS', seconds=0.1, note='')},
                                plan=['lint', 'unit-cpu', 'docs'], finished=False, running='unit-cpu')
        ci_report.write_run(old, ci_dir=ci)
        ci_report.write_run(new, ci_dir=ci)
        page = (new / 'report.html').read_text()
        assert 'http-equiv="refresh"' in page and 'running unit-cpu (2 left)' in page
        assert '>RUNNING<' in page and '>PENDING<' in page
        assert 'RUNNING  unit-cpu' in (new / 'report.txt').read_text()
        index = (ci / 'index.html').read_text()
        assert index.index('20261023_110000') < index.index('20261023_100000'), 'newest first'
        assert 'class="cell FAIL"' in index and 'class="cell RUNNING"' in index and 'http-equiv="refresh"' in index
        ci_report.link_latest(new, ci_dir=ci)
        ci_report.link_latest(new, ci_dir=ci)  # replacing an existing link
        assert (ci / 'latest').resolve() == new.resolve()
        assert [d.name for d in ci_report.runs(ci)] == ['20261023_110000', '20261023_100000'], 'latest is no run'


def test_old_summary_without_plan():
    """Runs from before the report module: results only; they count as finished, their results are the plan."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / '20261001_000000'
        d.mkdir()
        (d / 'summary.json').write_text(json.dumps(dict(commit='x', started=d.name, gpu='ok', minutes=1.0,
                                                        results={'lint': dict(status='PASS', seconds=0, note='')})))
        s = ci_report.load(d)
        assert s['finished'] and [t['name'] for t in s['plan']] == ['lint']
        ci_report.write_run(d, ci_dir=Path(tmp))
        assert 'PASS     lint' in (d / 'report.txt').read_text()


if __name__ == '__main__':
    run(globals())
