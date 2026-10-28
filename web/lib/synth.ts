// Plays a note list as it grows: notes (and their note-offs) are scheduled a little ahead of the playhead, so
// playback can start while the engine is still writing. Sound: the FluidR3 SoundFont (sound.ts); if it can't load,
// a small built-in synth. One player sounds at a time.

import { familyOf, type Note } from './notes';
import { liveSynth, silence, startTime, unmuteAt, type Live } from './sound';

const LOOKAHEAD_S = 0.25;
const TICK_MS = 40;
const CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]; // channel 10 (index 9) is the drum kit

interface Output {
  now(): number;
  program(ch: number, program: number, at: number): void;
  on(ch: number, note: Note, at: number): void;
  off(ch: number, pitch: number, at: number): void;
  stop(): void;
}

class SoundFontOut implements Output {
  constructor(private l: Live) {}
  now() { return this.l.ctx.currentTime; }
  program(ch: number, program: number, at: number) { this.l.synth.programChange(ch, program, { time: at }); }
  on(ch: number, n: Note, at: number) { this.l.synth.noteOn(ch, n[2], Math.max(1, n[4]), { time: at }); }
  off(ch: number, pitch: number, at: number) { this.l.synth.noteOff(ch, pitch, { time: at }); }
  stop() { silence(this.l, LOOKAHEAD_S); }
}

// the fallback: triangle waves and filtered noise, a whole voice per note (so off() has nothing to do)
class SimpleOut implements Output {
  private ctx = new AudioContext();
  private master = this.ctx.createGain();
  private noise: AudioBuffer;
  constructor() {
    this.master.gain.value = 0.22;
    this.master.connect(this.ctx.destination);
    this.noise = this.ctx.createBuffer(1, Math.floor(this.ctx.sampleRate * 0.35), this.ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  now() { return this.ctx.currentTime; }
  program() {}
  off() {}
  stop() { void this.ctx.close(); }
  on(_ch: number, [, program, pitch, dur, vel]: Note, at: number) {
    const ctx = this.ctx, g = ctx.createGain(), v = vel / 127;
    g.connect(this.master);
    const fam = familyOf(program);
    if (fam === 'drums') {
      if (pitch === 35 || pitch === 36) {
        const o = ctx.createOscillator();
        o.frequency.setValueAtTime(130, at);
        o.frequency.exponentialRampToValueAtTime(45, at + 0.14);
        g.gain.setValueAtTime(v * 1.2, at);
        g.gain.exponentialRampToValueAtTime(0.001, at + 0.22);
        o.connect(g);
        o.start(at);
        o.stop(at + 0.24);
        return;
      }
      const s = ctx.createBufferSource(), hp = ctx.createBiquadFilter();
      const hat = pitch === 42 || pitch === 44 || pitch === 46, cymbal = pitch === 49 || pitch === 51 || pitch === 57;
      s.buffer = this.noise;
      hp.type = 'highpass';
      hp.frequency.value = hat ? 7500 : cymbal ? 5000 : 1600;
      const len = cymbal ? 0.32 : hat ? 0.05 : 0.09;
      g.gain.setValueAtTime(v * (hat ? 0.35 : 0.5), at);
      g.gain.exponentialRampToValueAtTime(0.001, at + len);
      s.connect(hp);
      hp.connect(g);
      s.start(at);
      s.stop(at + len + 0.02);
      return;
    }
    const o = ctx.createOscillator(), len = Math.max(dur, 0.12);
    o.type = 'triangle';
    o.frequency.value = 440 * Math.pow(2, (pitch + (fam === 'bass' ? 12 : 0) - 69) / 12);
    g.gain.setValueAtTime(0, at);
    g.gain.linearRampToValueAtTime(v * (fam === 'bass' ? 0.8 : 0.4), at + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, at + len + 0.3);
    o.connect(g);
    o.start(at);
    o.stop(at + len + 0.35);
  }
}

let current: Player | null = null;

export class Player {
  private out: Output | null = null;
  private timer: number | null = null;
  private next = 0; // index of the next note to schedule
  private offs: [number, number, number][] = []; // [end s, channel, pitch] waiting to be scheduled
  private channels = new Map<number, number>(); // program -> channel
  private startAt = 0; // output time of piece time 0
  private from = 0;
  private run = 0; // bumps on every play/stop, so a play still loading the sound can tell it was cancelled
  playing = false;

  // onEnd: the piece played out; onStop: stopped by another player starting
  constructor(private notes: () => Note[], private onEnd: () => void, private end: () => number, private onStop?: () => void) {}

  time(): number {
    return this.playing && this.out ? Math.max(this.from, this.out.now() - this.startAt) : this.from;
  }

  async play(from: number): Promise<void> {
    this.halt();
    if (current && current !== this) current.stopFromOutside();
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- the page's one sounding player, not an alias
    current = this;
    const run = ++this.run;
    this.playing = true;
    this.from = from;
    let out: Output;
    try {
      const live = await liveSynth();
      if (live.ctx.state !== 'running') await live.ctx.resume();
      out = new SoundFontOut(live);
      if (run !== this.run) return;
      this.startAt = startTime(live) - from;
      unmuteAt(live, this.startAt + from);
    } catch {
      if (run !== this.run) return;
      out = new SimpleOut();
      this.startAt = out.now() + 0.05 - from;
    }
    this.out = out;
    this.channels.clear();
    this.offs = [];
    const notes = this.notes();
    this.next = notes.findIndex((n) => n[0] >= from);
    if (this.next < 0) this.next = notes.length;
    this.schedule();
    this.timer = window.setInterval(() => this.schedule(), TICK_MS);
  }

  stop(at?: number) {
    this.run++;
    if (this.playing) this.from = at ?? this.time();
    this.halt();
    if (current === this) current = null;
  }

  private stopFromOutside() {
    this.stop();
    this.onStop?.();
  }

  private halt() {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.out?.stop();
    this.out = null;
    this.playing = false;
  }

  private channel(program: number, at: number): number {
    if (program === 128) return 9;
    let ch = this.channels.get(program);
    if (ch === undefined) {
      ch = CHANNELS[this.channels.size % CHANNELS.length];
      this.channels.set(program, ch);
      this.out!.program(ch, program, at);
    }
    return ch;
  }

  private schedule() {
    const out = this.out;
    if (!out) return;
    const notes = this.notes();
    const now = out.now() - this.startAt; // piece time
    const horizon = now + LOOKAHEAD_S;
    while (this.next < notes.length && notes[this.next][0] < horizon) {
      const n = notes[this.next++];
      if (n[0] < now - 0.05) continue; // fell behind: skip rather than pile up
      const at = this.startAt + Math.max(n[0], now);
      const ch = this.channel(n[1], at - 0.002);
      out.on(ch, n, at);
      this.offs.push([n[0] + Math.max(n[3], 0.03), ch, n[2]]);
    }
    if (this.offs.length) {
      const keep: [number, number, number][] = [];
      for (const o of this.offs) {
        if (o[0] < horizon) out.off(o[1], o[2], this.startAt + Math.max(o[0], now));
        else keep.push(o);
      }
      this.offs = keep;
    }
    if (now > this.end()) {
      this.stop(0);
      this.onEnd();
    }
  }
}
