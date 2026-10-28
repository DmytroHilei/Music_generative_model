// The caption vocabulary the model was trained on (data/captions/templates_en.json, uk_labels.json), used by the
// home page's random prompt, the ingredient chips and the composer's checks. The chips and the random prompt write in
// the page's language.

import type { Key, Lang } from './i18n';

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
export const EXAMPLES: Record<Lang, string[]> = {
  en: [
    'A calm ambient piece with piano and strings in a minor key',
    'Швидка весела пісня з барабанами та електрогітарою',
    'A fast, energetic rock song with drums, electric guitar and bass',
    'Спокійна джазова композиція з фортепіано та контрабасом',
    'A slow, romantic folk tune for acoustic guitar',
    'A tense soundtrack piece with low strings and drums',
  ],
  uk: [
    'Спокійна композиція в стилі ембієнт з фортепіано та струнними в мінорі',
    'Швидка весела пісня з барабанами та електрогітарою',
    'Тиха класична п’єса для струнних у мінорі',
    'A calm ambient piece with piano and strings in a minor key',
    'Спокійна джазова композиція з фортепіано та контрабасом',
    'Напружена музика до фільму з низькими струнними та барабанами',
  ],
};

export const SUGGESTIONS: Record<Lang, string[]> = {
  en: [
    'Повільна сумна мелодія для фортепіано',
    'A calm acoustic guitar piece without drums',
    'A fast rock song with drums, guitar and bass',
  ],
  uk: [
    'Повільна сумна мелодія для фортепіано',
    'Спокійна мелодія для акустичної гітари без барабанів',
    'Швидка рок-пісня з барабанами, гітарою та басом',
  ],
};

const an = (w: string) => (/^[aeiou]/i.test(w) ? 'An ' : 'A ') + w;

export function randomPrompt(lang: Lang): string {
  if (lang === 'uk') return Math.random() < 0.4 ? R(RANDOM.uk) : buildUk(randomSelection());
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
// id: the English chip and caption word; uk: the Ukrainian chip; form: how a Ukrainian caption says it (feminine, to
// agree with «композиція»; instruments as they follow «з»); stem: what parse finds in Ukrainian text
type Word = { id: string; uk: string; form: string; stem: string };
const w = (id: string, uk: string, form: string, stem: string): Word => ({ id, uk, form, stem });

export const INGREDIENTS: { field: Field; label: Key; words: Word[]; multi?: boolean }[] = [
  { field: 'mood', label: 'Mood', words: [
    w('calm', 'спокійна', 'спокійна', 'спокійн'), w('energetic', 'енергійна', 'енергійна', 'енергійн'),
    w('sad', 'сумна', 'сумна', 'сумн'), w('happy', 'весела', 'весела', 'весел'),
    w('tense', 'напружена', 'напружена', 'напружен'), w('romantic', 'романтична', 'романтична', 'романтичн'),
  ] },
  { field: 'tempo', label: 'Tempo', words: [w('slow', 'повільна', 'повільна', 'повільн'), w('fast', 'швидка', 'швидка', 'швидк')] },
  { field: 'genre', label: 'Genre', words: [
    w('jazz', 'джаз', 'джазова композиція', 'джаз'), w('rock', 'рок', 'рок-композиція', 'рок'),
    w('classical', 'класика', 'класична композиція', 'класи[чк]'), w('ambient', 'ембієнт', 'композиція в стилі ембієнт', 'ембієнт'),
    w('folk', 'фолк', 'народна композиція', 'фолк|народн'), w('electronic', 'електроніка', 'електронна композиція', 'електрон'),
    w('pop', 'поп', 'поп-композиція', 'поп'),
  ] },
  { field: 'inst', label: 'Instruments', multi: true, words: [
    w('piano', 'фортепіано', 'фортепіано', 'фортепіано|піаніно'), w('guitar', 'гітара', 'гітарою', 'гітар'),
    w('strings', 'струнні', 'струнними', 'струнн'), w('bass', 'бас', 'басом', 'бас'),
    w('drums', 'барабани', 'барабанами', 'барабан'), w('flute', 'флейта', 'флейтою', 'флейт'),
    w('sax', 'саксофон', 'саксофоном', 'саксофон'),
  ] },
  { field: 'key', label: 'Key', words: [w('minor', 'мінор', 'в мінорі', 'мінор'), w('major', 'мажор', 'в мажорі', 'мажор')] },
];

export type Selection = { mood: string | null; tempo: string | null; genre: string | null; key: string | null; inst: string[] };

const has = (low: string, x: Word) =>
  new RegExp('(^|[^a-z])' + x.id).test(low) || new RegExp(`(^|[^а-щьюяґєії’'])(${x.stem})`).test(low);
const row = (f: Field) => INGREDIENTS.find((r) => r.field === f)!;

// the chips a prompt already has, in either language (a chip clicked after switching language keeps the others)
export function parse(text: string): Selection {
  const low = text.toLowerCase();
  const found = (f: Field) => row(f).words.filter((x) => has(low, x)).map((x) => x.id);
  return { mood: found('mood')[0] ?? null, tempo: found('tempo')[0] ?? null, genre: found('genre')[0] ?? null, key: found('key')[0] ?? null, inst: found('inst') };
}

const joinAnd = (xs: string[], and = 'and') => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} ${and} ${xs[xs.length - 1]}`);

function randomSelection(): Selection {
  const pick = (f: Field) => R(row(f).words).id;
  return {
    mood: pick('mood'), tempo: Math.random() < 0.6 ? pick('tempo') : null, genre: pick('genre'),
    key: Math.random() < 0.4 ? pick('key') : null, inst: [...new Set([pick('inst'), pick('inst')])],
  };
}

function buildUk(sel: Selection): string {
  const form = (f: Field, id: string) => row(f).words.find((x) => x.id === id)!.form;
  const adj = [sel.tempo && form('tempo', sel.tempo), sel.mood && form('mood', sel.mood)].filter(Boolean).join(', ');
  if (!adj && !sel.genre && !sel.key && !sel.inst.length) return '';
  let s = [adj, sel.genre ? form('genre', sel.genre) : 'композиція'].filter(Boolean).join(' ');
  if (sel.key) s += ` ${form('key', sel.key)}`;
  if (sel.inst.length) s += ` з ${joinAnd(sel.inst.map((i) => form('inst', i)), 'та')}`;
  return s[0].toUpperCase() + s.slice(1);
}

export function build(sel: Selection, lang: Lang): string {
  if (lang === 'uk') return buildUk(sel);
  const adj = [sel.tempo, sel.mood].filter(Boolean).join(', ');
  const head = [adj, sel.genre].filter(Boolean).join(' ');
  if (!head && !sel.key && !sel.inst.length) return '';
  let s = head ? `${an(head)} piece` : 'A piece';
  if (sel.key) s += ` in a ${sel.key} key`;
  if (sel.inst.length) s += ` with ${joinAnd(sel.inst)}`;
  return s;
}

export function toggle(text: string, field: Field, word: string, lang: Lang): string {
  const sel = parse(text);
  if (field === 'inst') sel.inst = sel.inst.includes(word) ? sel.inst.filter((x) => x !== word) : [...sel.inst, word];
  else sel[field] = sel[field] === word ? null : word;
  return build(sel, lang);
}

// what the composer checks for (English and Ukrainian word stems)
const CHECKS: Record<string, string[]> = {
  mood: ['lively', 'calm', 'sad', 'happy', 'quiet', 'energetic', 'gentle', 'tense', 'romantic', 'warm', 'весел', 'сумн', 'спокійн', 'тих', 'енергійн', 'романтичн', 'напружен', 'жвав', 'захоплив', 'тепл', 'лагідн'],
  tempo: ['slow', 'fast', 'mid-tempo', 'moderate', 'повільн', 'швидк', 'помірн'],
  instruments: ['piano', 'bass', 'drum', 'guitar', 'sax', 'string', 'violin', 'flute', 'organ', 'harp', 'synth', 'фортепіано', 'піаніно', 'бас', 'барабан', 'гітар', 'скрипк', 'флейт', 'струнн', 'контрабас', 'саксофон', 'орган', 'арф', 'синт'],
  key: ['minor', 'major', 'мінор', 'мажор'],
};
const HINTS: Record<string, Key> = {
  tempo: 'Add a tempo word: “slow” or “fast”.',
  mood: 'Add a mood: “calm”, “lively”, “sad”.',
  instruments: 'Name an instrument: piano, guitar, drums…',
  key: 'Optional: “in a minor key”.',
};

// field names and the hint are keys of lib/i18n.ts, shown in the page's language
export function check(text: string): { fields: { name: Key; ok: boolean }[]; hint: Key } {
  const low = text.toLowerCase();
  const fields = (Object.keys(CHECKS) as Key[]).map((name) => ({ name, ok: CHECKS[name].some((x) => low.includes(x)) }));
  const missing = fields.find((f) => !f.ok);
  return { fields, hint: missing ? HINTS[missing.name] : 'Reads like a caption the model knows.' };
}

export function titleOf(prompt: string): string {
  return prompt.replace(/[.,!?]+$/, '').split(/\s+/).slice(0, 4).join(' ');
}
