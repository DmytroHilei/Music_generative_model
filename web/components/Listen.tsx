'use client';

// The home page's sample cards (docs/listen) and the faint piano-roll bars behind the headline.

import { useEffect, useState } from 'react';
import Roll from './Roll';
import { Pause, Play } from './Icons';
import { familyOf, type Note } from '@/lib/notes';
import { Player } from '@/lib/synth';

type Sample = { name: string; prompt: string; instruments: string; n_notes: number; duration_s: number; notes?: Note[] };

const THUMB_S = 12;

// one card per prompt (the first take), with its first seconds as the thumbnail
export function useSamples(): Sample[] {
  const [samples, setSamples] = useState<Sample[]>([]);
  useEffect(() => {
    let live = true;
    void (async () => {
      const index: Sample[] = await (await fetch('/listen/index.json')).json();
      const firsts = index.filter((s, i) => index.findIndex((o) => o.prompt === s.prompt) === i);
      const loaded = await Promise.all(firsts.map(async (s) => ({ ...s, notes: await (await fetch(`/listen/${s.name}.json`)).json() as Note[] })));
      if (live) setSamples(loaded);
    })().catch(() => {});
    return () => { live = false; };
  }, []);
  return samples;
}

// skip a sparse opening (jazz_band_1 holds one bass note for 7 s): start a second before the 9th note
const startOf = (notes: Note[]) => Math.max(0, (notes[Math.min(8, notes.length - 1)]?.[0] ?? 0) - 1);
const PREVIEW_S = 30;

function SampleCard({ s, onUse }: { s: Sample; onUse: (p: string) => void }) {
  const [playing, setPlaying] = useState(false);
  const [player] = useState(() => new Player(() => s.notes ?? [], () => setPlaying(false), () => startOf(s.notes ?? []) + PREVIEW_S));
  useEffect(() => () => player.stop(), [player]);
  const start = startOf(s.notes ?? []);
  const thumb = (s.notes ?? []).filter((n) => n[0] >= start && n[0] < start + THUMB_S).map((n) => [n[0] - start, n[1], n[2], n[3], n[4]] as Note);
  return (
    <div className="card">
      <Roll notes={thumb} duration={THUMB_S} thumb />
      <div className="prompt">{s.prompt}</div>
      <div className="foot">
        <span>{s.instruments}</span>
        <div className="card-actions">
          <button type="button" className="act ctl icon" aria-label={playing ? 'Stop' : `Play ${s.prompt}`}
            onClick={() => { if (player.playing) { player.stop(0); setPlaying(false); } else { player.play(start); setPlaying(true); } }}>
            {playing ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button type="button" className="act" onClick={() => onUse(s.prompt)}>Use prompt</button>
        </div>
      </div>
    </div>
  );
}

export function ListenFirst({ samples, onUse }: { samples: Sample[]; onUse: (p: string) => void }) {
  return (
    <section className="listen" aria-labelledby="listen-title">
      <div className="listen-head">
        <h2 id="listen-title">Listen first</h2>
        <span>Written by musicar from the prompt alone. Use one as a starting point.</span>
      </div>
      <div className="cards">
        {samples.map((s) => <SampleCard key={s.name} s={s} onUse={onUse} />)}
      </div>
    </section>
  );
}

// the full band of jazz_band_1 from 6.5 s, stretched across the hero
export function HeroBars({ samples }: { samples: Sample[] }) {
  const notes = samples.find((s) => s.name === 'jazz_band_1')?.notes ?? [];
  const bars = notes.filter((n) => n[1] !== 128 && n[0] >= 6.5 && n[0] < 22);
  return (
    <div className="hero-bars" aria-hidden="true">
      {bars.map(([on, program, pitch, dur], i) => (
        <div key={i} style={{
          left: `${2 + ((on - 6.5) / 15.5) * 96}%`, top: 30 + (68 - pitch) * 6,
          width: `${Math.max((dur / 15.5) * 96, 0.6)}%`, background: `var(--${familyOf(program)})`,
        }} />
      ))}
    </div>
  );
}
