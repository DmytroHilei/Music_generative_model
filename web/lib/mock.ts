// A stand-in engine for building the page without the GPU: it answers like /v1 and streams one of the docs/listen
// pieces (web/public/listen) at the laptop's speed, picked by the words of the prompt.

import type { Engine, EngineEvent, GenerateRequest } from './engine';
import type { Note } from './notes';

const NOTES_PER_S = 110; // one candidate on the laptop GPU (engine step 1 measurement)
const BATCH = 16; // the engine sends notes every 16

const PICKS: [string, RegExp][] = [
  ['jazz_band_1', /jazz|джаз|swing/i],
  ['fast_rock_1', /rock|рок|energetic|енергійн/i],
  ['uk_fast_drums_1', /барабан|drum|весел|happy|fast|швидк/i],
  ['guitar_nodrums_1', /guitar|гітар|folk|acoustic/i],
  ['uk_slow_piano_1', /сумн|sad|фортепіано/i],
  ['slow_piano_2', /piano|calm|quiet|slow|minor|повільн|тих/i],
];
const FALLBACK = ['jazz_band_2', 'fast_rock_2', 'uk_slow_piano_2', 'guitar_nodrums_2'];

type MockJob = { req: GenerateRequest; file: string; seed: number };
const jobs = new Map<string, MockJob>();
const started = Date.now();

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const id = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(id);
      reject(new DOMException('aborted', 'AbortError'));
    });
  });

function pick(prompt: string, seed: number): string {
  for (const [file, re] of PICKS) if (re.test(prompt)) return file;
  return FALLBACK[seed % FALLBACK.length];
}

export function mockEngine(): Engine {
  return {
    async health() {
      return { engine: 'mock', model: 'docs/listen replay', loaded: true, busy: false, queue_len: 0, uptime_s: Math.round((Date.now() - started) / 1000) };
    },
    async generate(req) {
      const seed = req.seed ?? Math.floor(Math.random() * 2 ** 31);
      const id = Math.random().toString(36).slice(2, 14);
      jobs.set(id, { req, seed, file: pick(req.prompt, seed) });
      return { job_id: id, position: 0 };
    },
    async events(jobId, onEvent: (e: EngineEvent) => void, signal) {
      const job = jobs.get(jobId);
      if (!job) throw new Error('no such job (the mock forgets its jobs on reload)');
      onEvent({ type: 'queued', position: 0 });
      const res = await fetch(`/listen/${job.file}.json`, { signal });
      const all: Note[] = await res.json();
      const notes = all.filter((n) => n[0] < job.req.length_s);
      await wait(400, signal);
      onEvent({ type: 'started', seed: job.seed, settings: { text: job.req.prompt, length_seconds: job.req.length_s, mock: job.file, ...job.req.overrides } });
      const t0 = performance.now();
      for (let i = 0; i < notes.length; i += BATCH) {
        await wait((BATCH / NOTES_PER_S) * 1000, signal);
        onEvent({ type: 'notes', candidate: 0, notes: notes.slice(i, i + BATCH) });
      }
      const duration = notes.reduce((m, n) => Math.max(m, n[0] + n[3]), 0);
      onEvent({
        type: 'done', best: 0, n_notes: notes.length, duration_s: Math.round(duration * 1000) / 1000, seed: job.seed,
        elapsed_s: Math.round((performance.now() - t0) / 10) / 100,
        candidates: [{ candidate: 0, n_notes: notes.length, duration_s: duration }],
      });
    },
  };
}
