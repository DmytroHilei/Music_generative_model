'use client';

// The piano roll on a canvas: pitched notes above, one lane per drum sound below, coloured by instrument family.
// The same drawing serves the piece card (grid, playhead) and the small sample thumbnails.

import { useEffect, useRef } from 'react';
import { familyOf, type Note } from '@/lib/notes';
import { useApp } from '@/lib/app';

type Props = {
  notes: Note[];
  duration: number; // seconds across the width
  t?: number; // playhead; notes before it are lit while playing
  lit?: boolean;
  thumb?: boolean;
};

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export default function Roll({ notes, duration, t = 0, lit = false, thumb = false }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { theme } = useApp();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) return;
      if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
      if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
      const g = canvas.getContext('2d')!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      g.fillStyle = css('--roll');
      g.fillRect(0, 0, w, h);
      const colour = { piano: css('--piano'), guitar: css('--guitar'), bass: css('--bass'), drums: css('--drums'), other: css('--other') };

      const drumPitches = [...new Set(notes.filter((n) => n[1] === 128).map((n) => n[2]))].sort((a, b) => b - a).slice(0, thumb ? 1 : 10);
      const lane = thumb ? 5 : 9;
      const drumH = drumPitches.length ? drumPitches.length * lane + (thumb ? 4 : 10) : 0;
      const pad = thumb ? 5 : 6;
      const top = h - drumH - pad * 2;
      let lo = 127, hi = 0;
      for (const n of notes) if (n[1] !== 128) { lo = Math.min(lo, n[2]); hi = Math.max(hi, n[2]); }
      if (lo > hi) { lo = 48; hi = 72; }
      const span = Math.max(hi - lo + 4, thumb ? 12 : 24);
      const mid = (lo + hi) / 2;
      const pmax = mid + span / 2;
      const semi = top / span;
      const x = (s: number) => (s / duration) * w;

      if (!thumb) {
        for (let s = 0; s <= duration; s++) {
          g.fillStyle = css(s % 4 === 0 ? '--grid-bar' : '--grid');
          g.fillRect(Math.round(x(s)), 0, 1, h);
        }
        if (drumH) {
          g.fillStyle = css('--line');
          g.fillRect(0, pad + top + 4, w, 1);
        }
      }

      const noteH = thumb ? 3 : 5;
      for (const [onset, program, pitch, dur, vel] of notes) {
        if (onset > duration) continue;
        const fam = familyOf(program);
        g.globalAlpha = thumb ? 0.95 : lit ? (onset <= t ? 1 : 0.35) : 0.5 + (0.5 * vel) / 127;
        g.fillStyle = colour[fam];
        if (fam === 'drums') {
          const li = drumPitches.indexOf(pitch);
          if (li < 0) continue;
          g.fillRect(x(onset), h - pad - drumH + (thumb ? 4 : 10) + li * lane, thumb ? 2 : 3, thumb ? 3 : 6);
        } else {
          const y = pad + (pmax - pitch) * semi - noteH / 2;
          g.beginPath();
          g.roundRect(x(onset), y, Math.max(x(dur), 3), noteH, 2);
          g.fill();
        }
      }
      g.globalAlpha = 1;

      if (!thumb && lit) {
        g.fillStyle = css('--playhead');
        g.globalAlpha = 0.9;
        g.fillRect(x(t) - 1, 0, 2, h);
        g.globalAlpha = 1;
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [notes, duration, t, lit, thumb, theme]);

  return <canvas ref={ref} aria-hidden="true" />;
}
