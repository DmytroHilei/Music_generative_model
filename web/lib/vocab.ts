// The caption vocabulary the model was trained on (data/captions/templates_en.json, uk_labels.json), used by the
// home page's random prompt, the ingredient chips and the composer's checks.

const R = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)];

const RANDOM = {
  mood: ['calm', 'energetic', 'sad', 'tense', 'happy', 'quiet', 'romantic', 'warm', 'magnificent', 'lazy'],
  genre: ['ambient', 'classical', 'country', 'electronic', 'folk', 'jazz', 'latin', 'pop', 'reggae', 'rock', 'soundtrack'],
  tempo: ['slow', 'fast', 'mid-tempo'],
  inst: ['solo piano', 'piano and strings', 'acoustic guitar', 'electric guitar, bass and drums', 'piano, double bass and drums',
    'strings and flute', 'organ', 'saxophone, bass and drums', 'synth pads and bass', 'harp and flute'],
  key: ['a minor key', 'a major key', 'D minor', 'E-flat major', 'A minor', 'G major'],
  uk: ['Повільна сумна мелодія для фортепіано', 'Швидка весела пісня з барабанами та електрогітарою',
    'Спокійна джазова композиція з фортепіано та контрабасом', 'Тиха класична п’єса для струнних у мінорі',
    'Енергійний рок з барабанами, бас-гітарою та гітарою', 'Романтична мелодія для гітари в мажорі'],
};

// shown one after another in the empty home composer
export const EXAMPLES = [
  'A calm ambient piece with piano and strings in a minor key',
  'Швидка весела пісня з барабанами та електрогітарою',
  'A fast, energetic rock song with drums, electric guitar and bass',
  'Спокійна джазова композиція з фортепіано та контрабасом',
  'A slow, romantic folk tune for acoustic guitar',
  'A tense soundtrack piece with low strings and drums',
];

export const SUGGESTIONS = [
  'Повільна сумна мелодія для фортепіано',
  'A calm acoustic guitar piece without drums',
  'A fast rock song with drums, guitar and bass',
];

const an = (w: string) => (/^[aeiou]/i.test(w) ? 'An ' : 'A ') + w;

export function randomPrompt(): string {
  if (Math.random() < 0.3) return R(RANDOM.uk);
  const m = R(RANDOM.mood), g = R(RANDOM.genre), t = R(RANDOM.tempo), i = R(RANDOM.inst), k = R(RANDOM.key);
  return R([
    () => `${an(m)} ${g} piece with ${i}`,
    () => `${an(t)} ${g} piece in ${k}`,
    () => `${an(t)}, ${m} ${g} piece in ${k} with ${i}`,
    () => `${an(m)} piece for ${i}, ${t}, in ${k}`,
  ])();
}

export type Field = 'mood' | 'tempo' | 'genre' | 'inst' | 'key';
export const INGREDIENTS: { field: Field; label: string; words: string[]; multi?: boolean }[] = [
  { field: 'mood', label: 'Mood', words: ['calm', 'energetic', 'sad', 'happy', 'tense', 'romantic'] },
  { field: 'tempo', label: 'Tempo', words: ['slow', 'fast'] },
  { field: 'genre', label: 'Genre', words: ['jazz', 'rock', 'classical', 'ambient', 'folk', 'electronic', 'pop'] },
  { field: 'inst', label: 'Instruments', words: ['piano', 'guitar', 'strings', 'bass', 'drums', 'flute', 'sax'], multi: true },
  { field: 'key', label: 'Key', words: ['minor', 'major'] },
];

export type Selection = { mood: string | null; tempo: string | null; genre: string | null; key: string | null; inst: string[] };

const has = (low: string, w: string) => new RegExp('(^|[^a-z])' + w).test(low);

export function parse(text: string): Selection {
  const low = text.toLowerCase();
  const found = (f: Field) => INGREDIENTS.find((r) => r.field === f)!.words.filter((w) => has(low, w));
  return { mood: found('mood')[0] ?? null, tempo: found('tempo')[0] ?? null, genre: found('genre')[0] ?? null, key: found('key')[0] ?? null, inst: found('inst') };
}

const joinAnd = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

export function build(sel: Selection): string {
  const adj = [sel.tempo, sel.mood].filter(Boolean).join(', ');
  const head = [adj, sel.genre].filter(Boolean).join(' ');
  if (!head && !sel.key && !sel.inst.length) return '';
  let s = head ? `${an(head)} piece` : 'A piece';
  if (sel.key) s += ` in a ${sel.key} key`;
  if (sel.inst.length) s += ` with ${joinAnd(sel.inst)}`;
  return s;
}

export function toggle(text: string, field: Field, word: string): string {
  const sel = parse(text);
  if (field === 'inst') sel.inst = sel.inst.includes(word) ? sel.inst.filter((x) => x !== word) : [...sel.inst, word];
  else sel[field] = sel[field] === word ? null : word;
  return build(sel);
}

// what the composer checks for (English and Ukrainian word stems)
const CHECKS: Record<string, string[]> = {
  mood: ['lively', 'calm', 'sad', 'happy', 'quiet', 'energetic', 'gentle', 'tense', 'romantic', 'warm', 'весел', 'сумн', 'спокійн', 'тих', 'енергійн', 'романтичн'],
  tempo: ['slow', 'fast', 'mid-tempo', 'moderate', 'повільн', 'швидк'],
  instruments: ['piano', 'bass', 'drum', 'guitar', 'sax', 'string', 'violin', 'flute', 'organ', 'harp', 'synth', 'фортепіано', 'бас', 'барабан', 'гітар', 'скрипк', 'флейт', 'струнн', 'контрабас'],
  key: ['minor', 'major', 'мінор', 'мажор'],
};
const HINTS: Record<string, string> = {
  tempo: 'Add a tempo word: “slow” or “fast”.',
  mood: 'Add a mood: “calm”, “lively”, “sad”.',
  instruments: 'Name an instrument: piano, guitar, drums…',
  key: 'Optional: “in a minor key”.',
};

export function check(text: string): { fields: { name: string; ok: boolean }[]; hint: string } {
  const low = text.toLowerCase();
  const fields = Object.keys(CHECKS).map((name) => ({ name, ok: CHECKS[name].some((w) => low.includes(w)) }));
  const missing = fields.find((f) => !f.ok);
  return { fields, hint: missing ? HINTS[missing.name] : 'Reads like a caption the model knows.' };
}

export function titleOf(prompt: string): string {
  return prompt.replace(/[.,!?]+$/, '').split(/\s+/).slice(0, 4).join(' ');
}
