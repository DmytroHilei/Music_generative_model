// Every call to the generation engine goes through this file (the /v1 contract, context/claude/app_2026-10-27.md §2).
// The engine is the laptop (python -m musicar.serve), later the Pi; 'mock' replays docs/listen pieces (mock.ts).

import type { Note } from './notes';
import { mockEngine } from './mock';

export type Health = {
  engine: string;
  model: string;
  loaded: boolean;
  busy: boolean;
  queue_len: number;
  uptime_s: number;
};

export type GenerateRequest = {
  prompt: string;
  length_s: number;
  seed: number | null;
  best_of: number;
  overrides: Record<string, number | string | boolean>;
};

export type Candidate = {
  candidate: number;
  n_notes: number;
  duration_s: number;
  eos?: boolean;
  stopped?: boolean;
  score?: number | null;
};

export type EngineEvent =
  | { type: 'queued'; position: number }
  | { type: 'started'; seed: number; settings: Record<string, unknown> }
  | { type: 'notes'; candidate: number; notes: Note[] }
  | { type: 'done'; best: number; n_notes: number; duration_s: number; seed: number; elapsed_s: number; candidates: Candidate[] }
  | { type: 'error'; message: string };

export interface Engine {
  health(): Promise<Health>;
  generate(req: GenerateRequest): Promise<{ job_id: string; position: number }>;
  // resolves when the job's stream ends (done / error) or the signal aborts it
  events(jobId: string, onEvent: (e: EngineEvent) => void, signal?: AbortSignal): Promise<void>;
}

export class EngineError extends Error {}

async function check(res: Response): Promise<Response> {
  if (res.ok) return res;
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
  } catch {
    // not JSON: keep the status line
  }
  throw new EngineError(detail);
}

// server-sent events read with fetch, so a bearer token header can be added later (EventSource can't send headers)
async function readSSE(res: Response, onEvent: (name: string, data: unknown) => void) {
  if (!res.body) throw new EngineError('the engine sent no event stream');
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += value;
    let cut: number;
    while ((cut = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      let name = 'message';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) name = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (data) onEvent(name, JSON.parse(data));
    }
  }
}

export function httpEngine(base: string): Engine {
  const url = (path: string) => `${base.replace(/\/$/, '')}/v1${path}`;
  return {
    async health() {
      return (await check(await fetch(url('/health'), { cache: 'no-store' }))).json();
    },
    async generate(req) {
      const res = await fetch(url('/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      });
      return (await check(res)).json();
    },
    async events(jobId, onEvent, signal) {
      const res = await check(await fetch(url(`/jobs/${jobId}/events`), { signal, cache: 'no-store' }));
      await readSSE(res, (name, data) => onEvent({ type: name, ...(data as object) } as EngineEvent));
    },
  };
}

// '' = this site's /v1 (the dev server forwards it to the laptop engine, next.config.ts), 'mock', or an engine URL
export const DEFAULT_ENGINE = process.env.NEXT_PUBLIC_ENGINE ?? 'mock';

export function engineFor(setting: string): Engine {
  return setting === 'mock' ? mockEngine() : httpEngine(setting);
}
