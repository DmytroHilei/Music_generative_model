"""
The engine's jobs: requests wait in a FIFO queue (at most max_queue), one worker thread runs them one at a time on the
model (loaded at the first job, freed after idle_unload_min without jobs), and every job keeps the list of its events,
so a client reads the stream from the start at any time (late, or after a dropped connection) until the job is
forgotten keep_jobs_min after it finished.

Events (name, data), in order: queued {position}, started {seed, settings}, notes {candidate, notes} (every 16 notes
of each candidate, notes = [onset s, program, pitch, duration s, velocity] lists, program 128 = drums), then one of
done {best, candidates, ...}, error {message}, cancelled {notes_streamed}.
"""

import collections
import gc
import os
import secrets
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

from musicar.data.midi_io import notes_to_midi
from musicar.generation.piece import GenerationError, Generator, caption_index, text_encoder
from musicar.model.checkpoint import resolve_checkpoint

# the options reported in 'started' (the rest are generate.py defaults or don't apply to the engine)
SHOWN = ('text', 'length_seconds', 'seed', 'best_of', 'max_new_tokens', 'temperature', 'top_k', 'top_p', 'min_p',
         'cfg_scale', 'cfg_notes', 'cfg_end', 'cfg_linear', 'instrument_temperature', 'dry_multiplier', 'dry_allowed',
         'dry_base', 'dry_key', 'eos_ramp', 'length_stop', 'eos_floor', 'min_notes', 'retrieve', 'block_size')
FINISHED = ('done', 'error', 'cancelled')
TICK_S = 1.0  # how often an idle worker checks the unload and cleanup timers
OTHER_MB = 256  # GPU memory of other processes that counts as "busy with other work"


def wire(note):
    """A note as sent: times rounded to the millisecond (the sums of 20 ms steps carry float noise)."""
    onset, program, pitch, duration, velocity = note
    return [round(onset, 3), program, pitch, round(duration, 3), velocity]


def gpu_status():
    """{free_mb, total_mb, own_mb, other_mb} of the first GPU from nvidia-smi (own_mb = this process, other_mb = the
    other processes), or None without nvidia-smi."""
    def query(what):
        out = subprocess.run(['nvidia-smi', f'--query-{what}', '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, timeout=5, check=True).stdout
        return [[x.strip() for x in line.split(',')] for line in out.splitlines() if line.strip()]
    try:
        free, total = (int(x) for x in query('gpu=memory.free,memory.total')[0])
        apps = [(int(pid), int(mb)) for pid, mb in query('compute-apps=pid,used_memory')]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    own = sum(mb for pid, mb in apps if pid == os.getpid())
    return dict(free_mb=free, total_mb=total, own_mb=own, other_mb=sum(mb for _, mb in apps) - own)


class QueueFull(Exception):
    def __init__(self, retry_after, message='the queue is full'):
        super().__init__(f'{message}, try again in ~{retry_after} s')
        self.retry_after = retry_after


class Cancelled(Exception):
    """Raised in the note callback of a running job that was cancelled."""


@dataclass
class Job:
    id: str
    args: object  # the generate.py options (argparse.Namespace)
    owner: str = None  # the token's name; only its owner sees the job
    events: list = field(default_factory=list)  # (name, data); only appended to
    state: str = 'queued'  # queued -> running -> done | error | cancelled, set after the job's last event
    pieces: list = field(default_factory=list)  # the finished candidates, best first
    cancel: bool = False  # asked to stop (a running job stops at its next notes)
    finished_at: float = None

    def emit(self, name, data):
        self.events.append((name, data))

    def finish(self, state, name, data):
        self.emit(name, data)
        self.finished_at, self.state = time.time(), state

    @property
    def finished(self):
        return self.state in FINISHED

    def notes(self, candidate):
        """The streamed notes of a candidate so far."""
        return [n for name, d in self.events if name == 'notes' and d['candidate'] == candidate for n in d['notes']]


class Engine:
    """The model, the queue and the one worker that runs the jobs."""

    def __init__(self, config):
        self.config = config
        path = Path(resolve_checkpoint(config.checkpoint))
        self.model_hash = f'{path.stat().st_size:x}-{int(path.stat().st_mtime):x}'  # cheap: size and date
        self.on_gpu = config.device == 'cuda' or (config.device == 'auto' and torch.cuda.is_available())
        self.gpu_status = gpu_status  # replaceable (tests)
        self.generator = None  # loaded by the first job, freed when idle
        self.jobs = {}  # id -> Job, until forgotten
        self.waiting = collections.deque()
        self.cond = threading.Condition()
        self.running = None
        self.started = self.last_active = time.time()
        self.notes_per_s, self.job_s = None, 10.0  # of the last job (job_s: the Retry-After estimate)
        threading.Thread(target=self._work, name='musicar-engine', daemon=True).start()

    def submit(self, args, owner=None):
        with self.cond:
            ahead = len(self.waiting) + (self.running is not None)
            if len(self.waiting) >= self.config.max_queue:
                raise QueueFull(max(1, round(ahead * self.job_s)))
            mine = sum(1 for j in self.jobs.values() if j.owner == owner and not j.finished)
            if owner is not None and mine >= self.config.per_token_jobs:
                raise QueueFull(max(1, round(ahead * self.job_s)), f'{mine} of your jobs are still queued or '
                                                                   f'running (at most {self.config.per_token_jobs})')
            job = Job(secrets.token_urlsafe(9), args, owner)
            self.jobs[job.id] = job
            job.emit('queued', {'position': ahead})
            self.waiting.append(job)
            self.cond.notify()
        return job

    def cancel(self, job):
        """A queued job ends now; a running one at its next notes (within 16 notes). False if already finished."""
        with self.cond:
            if job.finished:
                return False
            job.cancel = True
            if job in self.waiting:
                self.waiting.remove(job)
                job.finish('cancelled', 'cancelled', {'notes_streamed': 0})
        return True

    def midi(self, job, candidate=None):
        """(PrettyMIDI, candidate) of a job: a finished candidate's exact MIDI (default the best), else the notes
        streamed so far (default candidate 0). KeyError for a candidate the job doesn't have."""
        if job.pieces:
            piece = job.pieces[0] if candidate is None else {p.index: p for p in job.pieces}[candidate]
            return piece.midi(job.args.max_polyphony, job.args.max_delta), piece.index
        candidate = candidate or 0
        if not 0 <= candidate < job.args.num_samples * job.args.best_of:
            raise KeyError(candidate)
        return notes_to_midi(job.notes(candidate)), candidate

    def health(self):
        gpu = self.gpu_status() if self.on_gpu else None
        return dict(engine=self.config.name, model=str(self.config.checkpoint), model_hash=self.model_hash,
                    loaded=self.generator is not None, busy=self.running is not None, queue_len=len(self.waiting),
                    notes_per_s=self.notes_per_s, gpu_free_mb=gpu and gpu['free_mb'], engine_mb=gpu and gpu['own_mb'],
                    busy_other=bool(gpu and gpu['other_mb'] >= OTHER_MB), uptime_s=round(time.time() - self.started))

    def unload(self):
        """Free the model and the caption encoder (the next job loads them again)."""
        self.generator = None
        caption_index.cache_clear()  # holds the same LaBSE
        text_encoder.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Engine: model unloaded (idle)', flush=True)

    def _work(self):
        while True:
            with self.cond:
                if not self.waiting:
                    self.cond.wait(TICK_S)
                job = self.waiting.popleft() if self.waiting else None
                if job is not None:
                    self.running, job.state = job, 'running'
            if job is None:
                self._housekeeping()
                continue
            try:
                self._run(job)
            except Cancelled:
                job.finish('cancelled', 'cancelled', {'notes_streamed': sum(len(d['notes']) for name, d in job.events
                                                                            if name == 'notes')})
            except GenerationError as e:
                job.finish('error', 'error', {'message': str(e)})
            except Exception as e:  # noqa: BLE001 - the worker must survive any job
                traceback.print_exc()
                job.finish('error', 'error', {'message': f'engine error: {type(e).__name__}: {e}'})
            if job.state != 'done' and torch.cuda.is_available():
                gc.collect()  # a failed job's tensors (e.g. after running out of memory) go back to the GPU
                torch.cuda.empty_cache()
            with self.cond:
                self.running, self.last_active = None, time.time()

    def _housekeeping(self):
        now = time.time()
        if self.generator is not None and self.config.idle_unload_min and \
                now - self.last_active > self.config.idle_unload_min * 60:
            self.unload()
        for job in list(self.jobs.values()):
            if job.finished and now - job.finished_at > self.config.keep_jobs_min * 60:
                self.jobs.pop(job.id, None)

    def _load(self):
        if self.on_gpu:
            gpu = self.gpu_status()
            if gpu and gpu['free_mb'] < self.config.min_free_mb:
                raise GenerationError(f"the GPU is busy with other work ({gpu['other_mb'] / 1024:.1f} GB used by "
                                      f"other processes, {gpu['free_mb']} MB free, the model needs "
                                      f"{self.config.min_free_mb} MB): try again later")
        self.generator = Generator(self.config.checkpoint, self.config.device)

    def _run(self, job):
        if self.generator is None:
            self._load()
        args = job.args
        job.emit('started', {'seed': args.seed, 'settings': {k: getattr(args, k) for k in SHOWN}})
        streamed = [0]

        def on_notes(c, notes):
            if job.cancel:
                raise Cancelled
            streamed[0] += len(notes)
            job.emit('notes', {'candidate': c, 'notes': [wire(n) for n in notes]})
        t0 = time.time()
        if job.cancel:
            raise Cancelled
        pieces = self.generator.run(args, on_notes=on_notes)
        elapsed = time.time() - t0
        self.job_s, self.notes_per_s = elapsed, round(streamed[0] / max(elapsed, 1e-6))
        candidates = []
        for p in pieces:
            notes = p.notes()
            candidates.append(dict(candidate=p.index, n_notes=len(notes),
                                   duration_s=round(max((n[0] + n[3] for n in notes), default=0.0), 3),
                                   eos=p.eos, stopped=p.stopped,
                                   score=None if p.score is None else round(p.score['score'], 3),
                                   opening=p.label if args.retrieve else None))
        job.pieces = pieces
        best = candidates[0]
        job.finish('done', 'done', dict(best=best['candidate'], n_notes=best['n_notes'],
                                        duration_s=best['duration_s'], seed=args.seed, elapsed_s=round(elapsed, 2),
                                        candidates=candidates))
