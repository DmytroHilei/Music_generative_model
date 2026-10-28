// A small WebAudio synth that plays the note list as it grows: notes are scheduled a little ahead of the playhead,
// so playback can start while the engine is still writing. A placeholder until the SoundFont player (spessasynth_lib
// + FluidR3 SF3, app_2026-10-27.md §4).

import { familyOf, type Note } from './notes';

const LOOKAHEAD_S = 0.6;
const TICK_MS = 100;

export class Player {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private noise: AudioBuffer | null = null;
  private timer: number | null = null;
  private next = 0; // index of the next note to schedule
  private startAt = 0; // context time of piece time 0
  private from = 0;

  constructor(private notes: () => Note[], private onEnd: () => void, private end: () => number) {}

  get playing() {
    return this.ctx !== null;
  }

  time(): number {
    return this.ctx ? this.ctx.currentTime - this.startAt : this.from;
  }

  play(from: number) {
    this.stop();
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.22;
    this.master.connect(ctx.destination);
    this.noise = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.35), ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    this.from = from;
    this.startAt = ctx.currentTime + 0.05 - from;
    const notes = this.notes();
    this.next = notes.findIndex((n) => n[0] >= from);
    if (this.next < 0) this.next = notes.length;
    this.schedule();
    this.timer = window.setInterval(() => this.schedule(), TICK_MS);
  }

  stop(at?: number) {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    if (this.ctx) {
      this.from = at ?? this.time();
      void this.ctx.close();
    }
    this.ctx = null;
  }

  private schedule() {
    const ctx = this.ctx;
    if (!ctx) return;
    const notes = this.notes();
    const now = ctx.currentTime - this.startAt;
    while (this.next < notes.length && notes[this.next][0] < now + LOOKAHEAD_S) {
      const n = notes[this.next++];
      if (n[0] >= now - 0.05) this.voice(n, this.startAt + Math.max(n[0], now));
    }
    if (now > this.end()) {
      this.stop(0);
      this.onEnd();
    }
  }

  private voice([, program, pitch, dur, vel]: Note, at: number) {
    const ctx = this.ctx!, g = ctx.createGain(), v = vel / 127;
    g.connect(this.master!);
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
