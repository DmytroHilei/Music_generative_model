"""
Progress of sampling jobs (generate.py, sample_sweep.py) for the dashboard: one JSON file per process in logs/jobs/.

    job = JobStatus('generate', total=512, detail='tiersen prompt')
    job.update(done, detail=...)   # cheap to call every step, written at most once a second
    job.finish('saved samples/x.mid')

status.py lists these files: running while the pid is alive, 'done' after finish(), 'died' if the process is gone
without finishing. An installed package (no repo checkout around it, e.g. pip install) writes none.
"""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / 'logs' / 'jobs'


class JobStatus:
    def __init__(self, name, total, detail=''):
        # a repo checkout (train.py next to the package); in site-packages the job file would land there
        self.path = JOBS_DIR / f'{name}_{os.getpid()}.json' if (ROOT / 'train.py').is_file() else None
        if self.path is not None:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.state = dict(name=name, pid=os.getpid(), total=int(total), done=0, detail=detail, result='',
                          started=time.time(), updated=time.time(), finished=False)
        self._last_write = 0.0
        self._write()

    def update(self, done, detail=None, force=False):
        """force: write now (rare phase changes that must not be swallowed by the once-a-second limit)."""
        self.state['done'] = int(done)
        if detail is not None:
            self.state['detail'] = detail
        if force or time.time() - self._last_write >= 1.0:
            self._write()

    def finish(self, result=''):
        self.state.update(done=self.state['total'], result=result, finished=True)
        self._write()

    def _write(self):
        self.state['updated'] = self._last_write = time.time()
        if self.path is None:
            return
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)  # the dashboard never reads a half-written file


def read_jobs():
    jobs = []
    for p in JOBS_DIR.glob('*.json'):
        try:
            jobs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return sorted(jobs, key=lambda j: j['started'])
