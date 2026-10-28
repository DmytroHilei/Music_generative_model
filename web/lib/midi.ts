// A Standard MIDI File (format 1) from the note list, written in the browser: one track per program, drums on
// channel 10, 120 bpm with 480 ticks per beat (so 1 s = 960 ticks).

import type { Note } from './notes';

const TPQ = 480;
const TICKS_PER_S = TPQ * 2;

function vlq(n: number): number[] {
  const out = [n & 0x7f];
  while ((n >>= 7)) out.unshift((n & 0x7f) | 0x80);
  return out;
}

const u32 = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const chunk = (id: string, body: number[]) => [...[...id].map((c) => c.charCodeAt(0)), ...u32(body.length), ...body];

function track(events: [number, number[]][]): number[] {
  events.sort((a, b) => a[0] - b[0] || a[1][0] - b[1][0]); // note-offs (0x8n) before note-ons at the same tick
  const body: number[] = [];
  let last = 0;
  for (const [tick, ev] of events) {
    body.push(...vlq(tick - last), ...ev);
    last = tick;
  }
  body.push(0, 0xff, 0x2f, 0);
  return chunk('MTrk', body);
}

export function toMidi(notes: Note[]): Uint8Array {
  const tempo = track([[0, [0xff, 0x51, 3, 0x07, 0xa1, 0x20]]]); // 500,000 us per beat
  const programs = [...new Set(notes.map((n) => n[1]))].sort((a, b) => a - b);
  const tracks = programs.map((program, i) => {
    const ch = program === 128 ? 9 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15][i % 15];
    const events: [number, number[]][] = program === 128 ? [] : [[0, [0xc0 | ch, program]]];
    for (const [onset, p, pitch, dur, vel] of notes) {
      if (p !== program) continue;
      const on = Math.round(onset * TICKS_PER_S), off = Math.max(on + 1, Math.round((onset + dur) * TICKS_PER_S));
      events.push([on, [0x90 | ch, pitch, Math.max(1, vel)]], [off, [0x80 | ch, pitch, 0]]);
    }
    return track(events);
  });
  const header = chunk('MThd', [0, 1, 0, tracks.length + 1, TPQ >> 8, TPQ & 255]);
  return new Uint8Array([...header, ...tempo, ...tracks.flat()]);
}

export function midiBlob(notes: Note[]): Blob {
  return new Blob([toMidi(notes) as BlobPart], { type: 'audio/midi' });
}
