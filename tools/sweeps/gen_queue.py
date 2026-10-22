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
Switch tests of v2_A0 (text_control.md "Next: hypotheses and tests", A1 / A7 / A8; section-worded prompts, the
baseline is results/section_follow/v2_A0_section: cfg 2, switch at note 1024 = the first cache refill):
  a1_contrast      the sections after the switch guided against the previous prompt, cfg 2 and 3      ~8 min
  a1_contrast_1sec the same for the first section after the switch only (then against none)         ~8 min
  a7_switch768     switch at note 768, before the first refill (1024)                                ~4 min
  a7_switch1280    switch at note 1280, on a refill (slide 256)                                       ~4 min
  a7_switch1280_slide192  switch at note 1280 between refills (slide 192: 1024, 1216, 1408, ...)        ~4 min
  a8_file_caption  hold + switch with the file caption 'a piece of music' in both guidance halves     ~7 min
Guidance on the switch (after tools/sweeps/section_forget.py), v2_A0 and forget_all:
  cfg_boost2_*     cfg 2, +2 at each section start, back to 2 over 32 notes                          ~4 min
  cfg_boost4_*     cfg 2, +4 at each section start, back to 2 over 64 notes                          ~4 min
  cfg_double_v2_A0 file caption 'a piece of music', file scale 1, section scale 3 (3-pass)            ~6 min
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
V2 = ['--checkpoints', 'checkpoints/v2_A0', '--cfg', '2', '--prompts', 'section', '--mp3']
JOBS.update({
    'a1_contrast': ('eval/compare/section_follow.py', V2 + ['--parts', 'switch', '--contrast', 'prev', '--cfg', '2,3'],
                    'results/section_follow/v2_A0_a1_contrast', 8),
    'a1_contrast_1sec': ('eval/compare/section_follow.py',
                         V2 + ['--parts', 'switch', '--contrast', 'prev', '--contrast-sections', '1', '--cfg', '2,3'],
                         'results/section_follow/v2_A0_a1_contrast_1sec', 8),
    'a7_switch768': ('eval/compare/section_follow.py', V2 + ['--parts', 'switch', '--switch', '768'],
                     'results/section_follow/v2_A0_a7_switch768', 4),
    'a7_switch1280': ('eval/compare/section_follow.py', V2 + ['--parts', 'switch', '--switch', '1280'],
                      'results/section_follow/v2_A0_a7_switch1280', 4),
    'a7_switch1280_slide192': ('eval/compare/section_follow.py',
                               V2 + ['--parts', 'switch', '--switch', '1280', '--slide', '192'],
                               'results/section_follow/v2_A0_a7_switch1280_slide192', 4),
    'a8_file_caption': ('eval/compare/section_follow.py',
                        V2 + ['--parts', 'hold,switch', '--file-caption', 'a piece of music'],
                        'results/section_follow/v2_A0_a8_file_caption', 7),
})

# guidance schedule / separate scales on the switch (roadmap P0 items 5-6), after tools/sweeps/section_forget.py:
# v2_A0 (baseline results/section_follow/v2_A0_section) and the forget_all arm (results/section_follow/forget_all)
SW = ['--cfg', '2', '--prompts', 'section', '--parts', 'switch', '--mp3']
for ck, tag in (('checkpoints/v2_A0', 'v2_A0'), ('checkpoints/forget_all', 'forget_all')):
    for boost, notes in ((2, 32), (4, 64)):
        JOBS[f'cfg_boost{boost}_{tag}'] = ('eval/compare/section_follow.py',
                                           ['--checkpoints', ck, *SW, '--cfg-boost', str(boost), '--cfg-boost-notes',
                                            str(notes)], f'results/section_follow/{tag}_cfg_boost{boost}', 4)
JOBS['cfg_double_v2_A0'] = ('eval/compare/section_follow.py',
                            ['--checkpoints', 'checkpoints/v2_A0', '--cfg', '1', '--prompts', 'section', '--parts',
                             'switch', '--mp3', '--cfg-section', '3', '--file-caption', 'a piece of music'],
                            'results/section_follow/v2_A0_cfg_double', 6)


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
