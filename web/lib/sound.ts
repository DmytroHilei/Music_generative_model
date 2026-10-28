'use client';

// The SoundFont: FluidR3 GM (SF3, public/soundfont, put there by scripts/assets.mjs) played by spessasynth_lib in an
// AudioWorklet. Loaded once, on the first play or Compose, and shared by every player on the page.

import { useSyncExternalStore } from 'react';
import type { WorkletSynthesizer } from 'spessasynth_lib';

export const SOUNDFONT_URL = '/soundfont/FluidR3Mono_GM.sf3';
export const PROCESSOR_URL = '/spessasynth_processor.min.js';

export type SoundStatus = { state: 'idle' | 'loading' | 'ready' | 'failed'; progress: number };

let status: SoundStatus = { state: 'idle', progress: 0 };
const listeners = new Set<() => void>();
const set = (s: SoundStatus) => {
  status = s;
  listeners.forEach((f) => f());
};

export function useSoundStatus(): SoundStatus {
  return useSyncExternalStore(
    (f) => { listeners.add(f); return () => listeners.delete(f); },
    () => status,
    () => status,
  );
}

let bank: Promise<ArrayBuffer> | null = null;

// the soundfont's bytes, fetched once with progress
export function loadBank(): Promise<ArrayBuffer> {
  if (bank) return bank;
  bank = (async () => {
    set({ state: 'loading', progress: 0 });
    const res = await fetch(SOUNDFONT_URL);
    if (!res.ok || !res.body) throw new Error(`soundfont: ${res.status}`);
    const total = Number(res.headers.get('Content-Length')) || 23712790;
    const reader = res.body.getReader();
    const parts: Uint8Array[] = [];
    let got = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parts.push(value);
      got += value.length;
      set({ state: 'loading', progress: Math.min(0.99, got / total) });
    }
    const out = new Uint8Array(got);
    let at = 0;
    for (const p of parts) { out.set(p, at); at += p.length; }
    return out.buffer;
  })();
  bank.catch(() => { bank = null; set({ state: 'failed', progress: 0 }); });
  return bank;
}

export type Live = { ctx: AudioContext; synth: WorkletSynthesizer; out: GainNode };
let live: Promise<Live> | null = null;
let quietUntil = 0; // context time until which notes queued before a stop may still arrive (see silence)
let silenceGen = 0;

// the page's one synthesizer, ready to play
export function liveSynth(): Promise<Live> {
  if (live) return live;
  live = (async () => {
    const ctx = new AudioContext();
    const [{ WorkletSynthesizer }, buf] = await Promise.all([import('spessasynth_lib'), loadBank()]);
    await ctx.audioWorklet.addModule(PROCESSOR_URL);
    const synth = new WorkletSynthesizer(ctx);
    const out = ctx.createGain();
    synth.connect(out);
    out.connect(ctx.destination);
    await synth.soundBankManager.addSoundBank(buf.slice(0), 'main');
    await synth.isReady;
    set({ state: 'ready', progress: 1 });
    return { ctx, synth, out };
  })();
  live.catch(() => { live = null; set({ state: 'failed', progress: 0 }); });
  return live;
}

// stop everything now: notes the scheduler already queued (up to `lookahead` ahead) can't be taken back, so the
// output is muted until they have passed and been stopped too
export function silence(l: Live, lookahead: number) {
  const gen = ++silenceGen;
  const now = l.ctx.currentTime;
  l.out.gain.cancelScheduledValues(now);
  l.out.gain.setTargetAtTime(0, now, 0.01);
  l.synth.stopAll(true);
  quietUntil = now + lookahead + 0.08;
  window.setTimeout(() => {
    if (gen !== silenceGen) return;
    l.synth.stopAll(true);
    l.out.gain.cancelScheduledValues(l.ctx.currentTime);
    l.out.gain.setValueAtTime(1, l.ctx.currentTime);
  }, (lookahead + 0.1) * 1000);
}

// the earliest context time a new playback may start at
export function startTime(l: Live): number {
  return Math.max(l.ctx.currentTime + 0.05, quietUntil + 0.05);
}

// a new playback starting at `at`: notes left from the last one are stopped once they have all arrived (stopAll
// ends sounding voices but keeps queued events, so the new playback's notes are untouched), then the sound returns
export function unmuteAt(l: Live, at: number) {
  silenceGen++; // replaces silence()'s pending unmute
  const now = l.ctx.currentTime;
  const wait = quietUntil - now;
  if (wait > 0) window.setTimeout(() => l.synth.stopAll(true), wait * 1000);
  l.out.gain.cancelScheduledValues(now);
  if (wait > 0) l.out.gain.setValueAtTime(0, now);
  l.out.gain.setValueAtTime(1, Math.max(now, at - 0.01));
}
