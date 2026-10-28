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
  opening?: boolean; // start from a caption-matched real opening (left out of the result)
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
  | { type: 'error'; message: string }
  | { type: 'cancelled'; notes_streamed: number };

export interface Engine {
  health(): Promise<Health>;
  generate(req: GenerateRequest): Promise<{ job_id: string; position: number }>;
  // resolves when the job's stream ends (done / error) or the signal aborts it
  events(jobId: string, onEvent: (e: EngineEvent) => void, signal?: AbortSignal): Promise<void>;
}

export class EngineError extends Error {}

// the engine answers 401 to calls without a valid invite token (musicar/serve/tokens.py)
const NO_INVITE = 'Writing pieces needs access: sign in with GitHub in Settings (once your access request is approved), '
  + 'or open the invite link you were sent.';
const BAD_INVITE = 'Your access code is not valid (any more): sign in with GitHub again in Settings, or ask for a new '
  + 'invite link.';

async function check(res: Response, token = ''): Promise<Response> {
  if (res.ok) return res;
  if (res.status === 401) throw new EngineError(token ? BAD_INVITE : NO_INVITE);
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
  } catch {
    // not JSON: keep the status line
  }
  throw new EngineError(detail);
}

// server-sent events read with fetch, so the bearer token header can be sent (EventSource can't send headers)
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

// token: the invite token, sent on every call but /health (which stays open so the page can show online / offline)
export function httpEngine(base: string, token = ''): Engine {
  const url = (path: string) => `${base.replace(/\/$/, '')}/v1${path}`;
  const auth: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  return {
    async health() {
      return (await check(await fetch(url('/health'), { cache: 'no-store' }))).json();
    },
    async generate(req) {
      const res = await fetch(url('/generate'), {
        method: 'POST',
        headers: { ...auth, 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      });
      return (await check(res, token)).json();
    },
    async events(jobId, onEvent, signal) {
      const res = await check(await fetch(url(`/jobs/${jobId}/events`), { headers: auth, signal, cache: 'no-store' }),
        token);
      await readSSE(res, (name, data) => onEvent({ type: name, ...(data as object) } as EngineEvent));
    },
  };
}

// '' = this site's /v1 (the dev server forwards it to the laptop engine, next.config.ts), 'mock', or an engine URL
export const DEFAULT_ENGINE = process.env.NEXT_PUBLIC_ENGINE ?? 'mock';

export function engineFor(setting: string, token = ''): Engine {
  return setting === 'mock' ? mockEngine() : httpEngine(setting, token);
}
