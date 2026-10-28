// A note as the engine sends it: [onset_s, program, pitch, duration_s, velocity], program 128 = drums.
export type Note = [number, number, number, number, number];

import type { Key } from './i18n';

export type Family = 'piano' | 'guitar' | 'bass' | 'drums' | 'other';

export const FAMILY_LABEL: Record<Family, Key> = {
  piano: 'piano',
  guitar: 'guitar',
  bass: 'bass',
  drums: 'drums',
  other: 'other',
};

export function familyOf(program: number): Family {
  if (program === 128) return 'drums';
  if (program < 8) return 'piano';
  if (program >= 24 && program < 32) return 'guitar';
  if (program >= 32 && program < 40) return 'bass';
  return 'other';
}

export function families(notes: Note[]): Family[] {
  const seen = new Set<Family>();
  for (const n of notes) seen.add(familyOf(n[1]));
  return (['piano', 'guitar', 'bass', 'drums', 'other'] as Family[]).filter((f) => seen.has(f));
}

export function endOf(notes: Note[]): number {
  let end = 0;
  for (const n of notes) end = Math.max(end, n[0] + n[3]);
  return end;
}

export function fmtTime(s: number): string {
  const t = Math.max(0, Math.floor(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}
