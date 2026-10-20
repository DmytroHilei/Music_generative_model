"""
Generation evaluations queued for a free GPU. Every generation result with drums before the drum-duration fix
(2026-10-20, musicar/generation/sampling.py) is wrong (busy openings collapsed into stacked duplicates), so these are
re-run into new out dirs next to the old ones:
  text_eval        from BOS, 28 EN + UK prompts (T2 v3, v3_long)                                     ~10 min
  text_follow      conflict (contradicting prompts after real openings) + drift (T2 v3, v3_long)    ~25 min
  section_follow   hold / switch with the section-worded prompts (T2 v3_long, ctrl, A0, A256,
                   A256_slots8), cfg 2 as in the sweep                                               ~30 min
  section_follow   the same with the old file-caption prompts for T2 v3_long and A256 (separates
                   the effect of the prompt wording from the fix)                                    ~12 min
Jobs run one after another; a job whose <out>/summary.md exists is skipped (re-run: delete it or --redo). The queue
refuses to start while train.py runs (--force overrides) and stops after the current job when a training process
appears. Logs logs/gen_queue/<job>.log, progress on the dashboard (JobStatus 'gen_queue').

    .venv/bin/python tools/sweeps/gen_queue.py --list
    .venv/bin/python tools/sweeps/gen_queue.py                      # every job not done yet
    .venv/bin/python tools/sweeps/gen_queue.py --only text_eval,text_follow
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from musicar.jobstatus import JobStatus  # noqa: E402

LOGS = ROOT / 'logs/gen_queue'
T2 = 'checkpoints/text_t2_v3,checkpoints/text_t2_v3_long'
SECTION_ARMS = ','.join(['checkpoints/text_t2_v3_long'] + [f'checkpoints/sec_{a}' for a in
                                                           ('ctrl', 'A0', 'A256', 'A256_slots8')])
# name -> (script, arguments, out dir, minutes)
JOBS = {
    'text_eval': ('eval/compare/text_eval.py', ['--checkpoints', T2], 'results/text_eval/t2_v3_drumfix', 10),
    'text_follow': ('eval/compare/text_follow.py', ['--checkpoints', T2, '--mp3'],
                    'results/text_follow/t2_v3_drumfix', 25),
    'section_follow': ('eval/compare/section_follow.py', ['--checkpoints', SECTION_ARMS, '--cfg', '2', '--mp3',
                                                          '--prompts', 'section'],
                       'results/section_follow/drumfix_section', 30),
    'section_follow_file': ('eval/compare/section_follow.py',
                            ['--checkpoints', 'checkpoints/text_t2_v3_long,checkpoints/sec_A256', '--cfg', '2',
                             '--prompts', 'file'],
                            'results/section_follow/drumfix_file', 12),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--only', default='', help='comma-separated job names (default: all)')
    p.add_argument('--list', action='store_true', help='list the jobs and their state')
    p.add_argument('--redo', action='store_true', help='run jobs whose summary.md exists too')
    p.add_argument('--force', action='store_true', help='start although train.py is running')
    return p.parse_args()


def training_running():
    """pids of train.py processes (a chunk of tools/sweeps/section_long.py, a sweep arm, ...)."""
    out = subprocess.run(['pgrep', '-f', r'python.* train\.py'], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def done(out):
    return (ROOT / out / 'summary.md').exists()


def main():
    args = parse_args()
    names = [n for n in args.only.split(',') if n] or list(JOBS)
    unknown = [n for n in names if n not in JOBS]
    if unknown:
        sys.exit(f'unknown job(s) {unknown}; jobs: {", ".join(JOBS)}')
    if args.list:
        for n in names:
            script, extra, out, minutes = JOBS[n]
            print(f'{n:20s} {"done" if done(out) else "to do":6s} ~{minutes:3d} min  -> {out}')
        return
    todo = [n for n in names if args.redo or not done(JOBS[n][2])]
    if not todo:
        print('nothing to do (every summary.md exists; --redo to run again)')
        return
    if training_running() and not args.force:
        sys.exit(f'train.py is running (pid {training_running()}): the GPU is busy (--force to start anyway)')
    LOGS.mkdir(parents=True, exist_ok=True)
    total = sum(JOBS[n][3] for n in todo)
    print(f'{len(todo)} job(s), ~{total} min: {", ".join(todo)}', flush=True)
    job = JobStatus('gen_queue', total, ', '.join(todo))
    spent = 0
    for n in todo:
        script, extra, out, minutes = JOBS[n]
        cmd = ['.venv/bin/python', script, *extra, '--out', out]
        print(f'{time.strftime("%H:%M")} {n}: ' + ' '.join(cmd[1:]), flush=True)
        job.update(spent, f'{n} (~{minutes} min)', force=True)
        t0 = time.time()
        with open(LOGS / f'{n}.log', 'w') as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        spent += minutes
        print(f'{time.strftime("%H:%M")} {n}: exit {code} after {(time.time() - t0) / 60:.0f} min'
              + (f' -> {out}/summary.md' if code == 0 else f' (see {LOGS / n}.log)'), flush=True)
        if training_running() and n != todo[-1]:
            print('train.py started meanwhile: stopping after this job (run the queue again later)')
            break
    job.finish(f'{spent} min of jobs done, logs in {LOGS}')


if __name__ == '__main__':
    main()
