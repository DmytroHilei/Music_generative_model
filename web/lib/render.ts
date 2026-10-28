'use client';

// MP3 in the browser: the piece's MIDI rendered with the same SoundFont in an OfflineAudioContext (faster than
// real time, no speakers), then LAME (wasm-media-encoders) in short chunks so the page stays responsive.
// The same sound as playback (soundfont and compressor), then the peak normalised to PEAK before encoding.

import { toMidi } from './midi';
import { endOf, type Note } from './notes';
import { loadBank, master, PROCESSOR_URL } from './sound';

const RATE = 44100;
const TAIL_S = 2.5; // room for the last notes' release and reverb
const CHUNK = RATE * 2;
const PEAK = 0.89; // -1 dBFS

export type RenderStage = 'sound' | 'render' | 'encode';

export async function renderMp3(notes: Note[], onProgress: (stage: RenderStage, p: number) => void): Promise<Blob> {
  onProgress('sound', 0);
  const [bank, { WorkletSynthesizer }, { BasicMIDI }] = await Promise.all([
    loadBank(), import('spessasynth_lib'), import('spessasynth_core'),
  ]);

  onProgress('render', 0);
  const ctx = new OfflineAudioContext({ numberOfChannels: 2, length: Math.ceil((endOf(notes) + TAIL_S) * RATE), sampleRate: RATE });
  await ctx.audioWorklet.addModule(PROCESSOR_URL);
  const synth = new WorkletSynthesizer(ctx);
  const comp = master(ctx);
  synth.connect(comp);
  comp.connect(ctx.destination);
  const midi = toMidi(notes);
  await synth.startOfflineRender({
    midiSequence: BasicMIDI.fromArrayBuffer(midi.buffer.slice(midi.byteOffset, midi.byteOffset + midi.byteLength) as ArrayBuffer),
    loopCount: 0,
    soundBankList: [{ bankOffset: 0, soundBankBuffer: bank.slice(0) }],
  });
  const audio = await ctx.startRendering();
  synth.destroy();

  onProgress('encode', 0);
  const { createMp3Encoder } = await import('wasm-media-encoders');
  const enc = await createMp3Encoder();
  enc.configure({ sampleRate: RATE, channels: 2, vbrQuality: 2 });
  const left = audio.getChannelData(0), right = audio.getChannelData(1);
  let peak = 0;
  for (const ch of [left, right]) for (let i = 0; i < ch.length; i++) peak = Math.max(peak, Math.abs(ch[i]));
  if (peak > 0) {
    const g = PEAK / peak;
    for (const ch of [left, right]) for (let i = 0; i < ch.length; i++) ch[i] *= g;
  }
  const parts: Uint8Array[] = [];
  for (let i = 0; i < left.length; i += CHUNK) {
    parts.push(enc.encode([left.subarray(i, i + CHUNK), right.subarray(i, i + CHUNK)]).slice()); // the encoder reuses its output
    onProgress('encode', i / left.length);
    await new Promise((r) => setTimeout(r, 0));
  }
  parts.push(enc.finalize().slice());
  return new Blob(parts as BlobPart[], { type: 'audio/mpeg' });
}

export function fileName(prompt: string, ext: string): string {
  return `${prompt.replace(/[^\p{L}\p{N}]+/gu, '_').replace(/^_|_$/g, '').slice(0, 60) || 'musicar'}.${ext}`;
}

export function save(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
