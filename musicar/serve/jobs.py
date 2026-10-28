"""
The engine's jobs: requests wait in a FIFO queue, one worker thread runs them one at a time on the model (loaded at the
first job, then kept), and every job keeps the list of its events, so a client reads the stream from the start at any
time (late, or after a dropped connection).

Events (name, data), in order: queued {position}, started {seed, settings}, notes {candidate, notes} (every 16 notes
of each candidate, notes = [onset s, program, pitch, duration s, velocity] lists, program 128 = drums), then done
{best, candidates, ...} or error {message}.
"""

import queue
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field

from musicar.generation.piece import GenerationError, Generator

# the options reported in 'started' (the rest are generate.py defaults or don't apply to the engine)
SHOWN = ('text', 'length_seconds', 'seed', 'best_of', 'max_new_tokens', 'temperature', 'top_k', 'top_p', 'min_p',
         'cfg_scale', 'cfg_notes', 'cfg_end', 'cfg_linear', 'instrument_temperature', 'dry_multiplier', 'dry_allowed',
         'dry_base', 'dry_key', 'eos_ramp', 'length_stop', 'min_notes')


def wire(note):
    """A note as sent: times rounded to the millisecond (the sums of 20 ms steps carry float noise)."""
    onset, program, pitch, duration, velocity = note
    return [round(onset, 3), program, pitch, round(duration, 3), velocity]


@dataclass
class Job:
    id: str
    args: object  # the generate.py options (argparse.Namespace)
    events: list = field(default_factory=list)  # (name, data); only appended to
    state: str = 'queued'  # queued -> running -> done | error, set after the job's last event
    pieces: list = field(default_factory=list)  # the finished candidates, best first

    def emit(self, name, data):
        self.events.append((name, data))

    @property
    def finished(self):
        return self.state in ('done', 'error')


class Engine:
    """The model, the queue and the one worker that runs the jobs."""

    def __init__(self, checkpoint, device='auto', name='laptop'):
        self.checkpoint, self.device, self.name = checkpoint, device, name
        self.generator = None  # loaded by the first job
        self.jobs = {}  # id -> Job, all of this process (kept for now)
        self.queue = queue.Queue()
        self.running = None
        self.started = time.time()
        threading.Thread(target=self._work, name='musicar-engine', daemon=True).start()

    def submit(self, args):
        job = Job(secrets.token_urlsafe(9), args)
        self.jobs[job.id] = job
        job.emit('queued', {'position': self.queue.qsize() + (self.running is not None)})
        self.queue.put(job)
        return job

    def health(self):
        return dict(engine=self.name, model=str(self.checkpoint), loaded=self.generator is not None,
                    busy=self.running is not None, queue_len=self.queue.qsize(),
                    uptime_s=round(time.time() - self.started))

    def _work(self):
        while True:
            job = self.queue.get()
            self.running, job.state = job, 'running'
            try:
                self._run(job)
                state = 'done'
            except GenerationError as e:
                job.emit('error', {'message': str(e)})
                state = 'error'
            except Exception as e:  # noqa: BLE001 - the worker must survive any job
                traceback.print_exc()
                job.emit('error', {'message': f'engine error: {type(e).__name__}: {e}'})
                state = 'error'
            job.state, self.running = state, None

    def _run(self, job):
        if self.generator is None:
            self.generator = Generator(self.checkpoint, self.device)
        args = job.args
        job.emit('started', {'seed': args.seed, 'settings': {k: getattr(args, k) for k in SHOWN}})
        t0 = time.time()
        job.pieces = self.generator.run(
            args, on_notes=lambda c, notes: job.emit('notes', {'candidate': c, 'notes': [wire(n) for n in notes]}))
        candidates = []
        for p in job.pieces:
            notes = p.notes()
            candidates.append(dict(candidate=p.index, n_notes=len(notes),
                                   duration_s=round(max((n[0] + n[3] for n in notes), default=0.0), 3),
                                   eos=p.eos, stopped=p.stopped,
                                   score=None if p.score is None else round(p.score['score'], 3)))
        best = candidates[0]
        job.emit('done', dict(best=best['candidate'], n_notes=best['n_notes'], duration_s=best['duration_s'],
                              seed=args.seed, elapsed_s=round(time.time() - t0, 2), candidates=candidates))
