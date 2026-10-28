// The page's two languages. The English text is the key; UK holds its Ukrainian. The choice lives in a cookie, so the
// server renders the right language on the first paint (app/layout.tsx); with no cookie, a browser asking for
// Ukrainian gets it.

export type Lang = 'en' | 'uk';
export const LANG_COOKIE = 'musicar_lang';

const UK = {
  // home
  'text to MIDI · English or Ukrainian': 'текст у MIDI · українською чи англійською',
  'What should it play?': 'Що зіграти?',
  'Describe a piece in one sentence. musicar writes it note by note, and you hear it while it is still writing.':
    'Опишіть п’єсу одним реченням. musicar пише її нота за нотою, і ви чуєте її, поки вона ще пишеться.',
  '{model} does not read captions yet: it writes freely and ignores the prompt. Pick musicar-L for text control.':
    '{model} ще не читає описів: пише вільно й не зважає на запит. Для керування текстом оберіть musicar-L.',
  'Mood': 'Настрій',
  'Tempo': 'Темп',
  'Genre': 'Жанр',
  'Instruments': 'Інструменти',
  'Key': 'Лад',
  'Listen first': 'Спершу послухайте',
  'Written by musicar from the prompt alone. Use one as a starting point.':
    'Написано musicar лише за запитом. Візьміть будь-який за основу.',
  'Use prompt': 'Взяти запит',
  'Play {prompt}': 'Відтворити {prompt}',
  'Stop': 'Стоп',

  // chat
  'Not found': 'Не знайдено',
  'New piece': 'Нова п’єса',
  'This chat is not in this browser': 'Цього чату немає в цьому браузері',
  'Chats are kept in the browser that made them.': 'Чати зберігаються в браузері, де їх створили.',
  'Back home': 'На головну',
  'A new piece': 'Нова п’єса',
  'Each message is a caption. After the first piece, Variation and Edit prompt build on it.':
    'Кожне повідомлення — це опис. Після першої п’єси «Варіація» та «Змінити запит» відштовхуються від неї.',
  'musicar writes MIDI from a caption: mood, tempo, instruments and key work best. English or Ukrainian.':
    'musicar пише MIDI за описом: найкраще працюють настрій, темп, інструменти й лад. Українською чи англійською.',
  '{model} ignores the prompt for now; Compose still writes a new piece.':
    '{model} поки не зважає на запит; «Створити» все одно напише нову п’єсу.',

  // composer
  'Describe the music': 'Опишіть музику',
  'Describe the music…': 'Опишіть музику…',
  'Describe the music… e.g. Повільна сумна мелодія для фортепіано':
    'Опишіть музику… напр. Повільна сумна мелодія для фортепіано',
  'Try': 'Спробуйте',
  'mood': 'настрій',
  'tempo': 'темп',
  'instruments': 'інструменти',
  'key': 'лад',
  'Add a tempo word: “slow” or “fast”.': 'Додайте темп: «повільна» чи «швидка».',
  'Add a mood: “calm”, “lively”, “sad”.': 'Додайте настрій: «спокійна», «жвава», «сумна».',
  'Name an instrument: piano, guitar, drums…': 'Назвіть інструмент: фортепіано, гітара, барабани…',
  'Optional: “in a minor key”.': 'Необов’язково: «в мінорі».',
  'Reads like a caption the model knows.': 'Схоже на опис, який модель знає.',
  'Random prompt': 'Випадковий запит',
  'Length {len}, click to change': 'Тривалість {len}, натисніть, щоб змінити',
  '30 s': '30 с',
  '1 min': '1 хв',
  '2 min': '2 хв',
  '4 min': '4 хв',
  '{n} s': '{n} с',
  'Preset': 'Режим',
  'faithful': 'точно',
  'balanced': 'збалансовано',
  'wild': 'вільно',
  'Real opening': 'Справжній початок',
  'Start from the first notes of a real song that fits the caption (same genre and mood); they are not in the result. Variation picks another song.':
    'Почати з перших нот справжньої пісні, що пасує до опису (той самий жанр і настрій); у результат вони не входять. «Варіація» бере іншу пісню.',
  'use this one': 'взяти цей',
  'Compose': 'Створити',

  // piece
  'Generated piece': 'Згенерована п’єса',
  'waiting · {n} ahead': 'у черзі · попереду {n}',
  'starting the engine…': 'запуск рушія…',
  'writing · {n} notes': 'пише · нот: {n}',
  'done': 'готово',
  'done in {s} s': 'готово за {s} с',
  'stopped': 'зупинено',
  'seed {n}': 'зерно {n}',
  'real opening': 'справжній початок',
  '{n} notes': 'нот: {n}',
  'Play': 'Відтворити',
  'Pause': 'Пауза',
  'Seek': 'Перемотати',
  'instruments {p}%': 'інструменти {p}%',
  'The SoundFont did not load; playing with the simple synth.': 'SoundFont не завантажився; грає простий синтезатор.',
  'Variation': 'Варіація',
  'Edit prompt': 'Змінити запит',
  'Loading sounds': 'Завантаження звуків',
  'Rendering': 'Рендеринг',
  'Encoding': 'Кодування',
  'MP3 failed: {msg}': 'MP3 не вдалося: {msg}',
  'piano': 'фортепіано',
  'guitar': 'гітара',
  'bass': 'бас',
  'drums': 'барабани',
  'other': 'інше',
  'guitars': 'гітари',
  'double bass': 'контрабас',

  // sidebar, top bar
  'Chats': 'Чати',
  'musicar home': 'musicar, головна',
  'Home': 'Головна',
  'Your pieces will appear here.': 'Тут з’являться ваші п’єси.',
  'Today': 'Сьогодні',
  'Yesterday': 'Вчора',
  'Earlier': 'Раніше',
  'Delete {title}': 'Видалити {title}',
  'Delete “{title}”?': 'Видалити «{title}»?',
  'Light mode': 'Світла тема',
  'Dark mode': 'Темна тема',
  'Settings': 'Налаштування',
  'checking the engine…': 'перевірка рушія…',
  'laptop engine online': 'рушій на ноутбуці в мережі',
  'writing a piece…': 'пише п’єсу…',
  'engine offline': 'рушій не в мережі',
  'mock engine': 'тестовий рушій',
  'Model': 'Модель',
  'Multi-instrument, reads English and Ukrainian captions': 'Багатоінструментальна, читає описи українською й англійською',
  'Multi-instrument, faster; does not read the prompt yet': 'Багатоінструментальна, швидша; поки не читає запиту',
  'Solo piano; does not read the prompt yet': 'Сольне фортепіано; поки не читає запиту',
  '{desc}. Not on this engine yet.': '{desc}. На цьому рушії її ще немає.',
  'The engine keeps one model in memory; switching will take a few seconds.':
    'Рушій тримає в пам’яті одну модель; перемикання займе кілька секунд.',
  'Dismiss': 'Закрити',

  // settings
  'Engine': 'Рушій',
  'Where pieces are written. Now: {s}': 'Де пишуться п’єси. Зараз: {s}',
  'checking…': 'перевірка…',
  'Laptop engine': 'Рушій на ноутбуці',
  'python -m musicar.serve on 127.0.0.1:8000, reached through this site’s /v1':
    'python -m musicar.serve на 127.0.0.1:8000, через /v1 цього сайту',
  'Engine URL': 'URL рушія',
  'an engine on another address (Tailscale later; needs CORS on the engine)':
    'рушій за іншою адресою (згодом Tailscale; рушію потрібен CORS)',
  'Mock engine': 'Тестовий рушій',
  'replays the docs/listen pieces at the laptop’s speed; no GPU needed':
    'відтворює п’єси з docs/listen зі швидкістю ноутбука; GPU не потрібен',
  'Access': 'Доступ',
  'signed in as {who}': 'ви ввійшли як {who}',
  'with an invite code': 'за кодом запрошення',
  'the engine writes pieces only for approved people': 'рушій пише п’єси лише для схвалених людей',
  'Sign in with GitHub': 'Увійти через GitHub',
  'Request access': 'Попросити доступ',
  'Invite code': 'Код запрошення',
  'set by an invite link or a GitHub sign-in; paste one by hand if you have it':
    'його задає посилання-запрошення або вхід через GitHub; якщо маєте код, вставте вручну',
  'Cancel': 'Скасувати',
  'Save': 'Зберегти',

  // engine state, sign-in, errors (lib/app.tsx, lib/engine.ts)
  'mock engine: replays docs/listen pieces': 'тестовий рушій: відтворює п’єси з docs/listen',
  ' (loads on the first piece)': ' (завантажиться з першою п’єсою)',
  ' · {n} waiting': ' · у черзі {n}',
  ' · not signed in': ' · ви не ввійшли',
  'no answer from {engine}': 'немає відповіді від {engine}',
  'no answer from the laptop engine': 'немає відповіді від рушія на ноутбуці',
  'Signed in as @{as}: you can write pieces now.': 'Ви ввійшли як @{as}: тепер можна писати п’єси.',
  'Your GitHub account': 'Ваш обліковий запис GitHub',
  '{who}: the access request is waiting for approval. Sign in again once it is approved.':
    '{who}: запит на доступ чекає схвалення. Увійдіть знову, коли його схвалять.',
  '{who} has no access request yet.': '{who}: запиту на доступ ще немає.',
  'Access for {who} was revoked.': 'Доступ для {who} відкликано.',
  'GitHub sign-in is not set up on this site yet.': 'Вхід через GitHub на цьому сайті ще не налаштовано.',
  'the sign-in took too long or was opened twice. Try again.': 'вхід тривав задовго або його відкрили двічі. Спробуйте ще раз.',
  'the engine or GitHub did not answer. Try again in a minute.': 'рушій або GitHub не відповіли. Спробуйте за хвилину.',
  'unknown error': 'невідома помилка',
  'Sign-in failed: {why}': 'Не вдалося ввійти: {why}',
  'Lost the engine: {msg}': 'Зв’язок із рушієм втрачено: {msg}',
  'The engine is offline. Check Settings, or try the mock engine.':
    'Рушій не в мережі. Перевірте налаштування або спробуйте тестовий рушій.',
  'Interrupted before the engine took it.': 'Перервано, перш ніж рушій її взяв.',
  'Cancelled.': 'Скасовано.',
  'Lost the engine: the stream kept breaking.': 'Зв’язок із рушієм втрачено: потік постійно обривався.',
  'The engine no longer has this piece (it restarted, or the piece is more than an hour old). Use Variation to write it again.':
    'Рушій більше не має цієї п’єси (він перезапустився або п’єсі понад годину). Натисніть «Варіація», щоб написати її знову.',
  'Writing pieces needs access: sign in with GitHub in Settings (once your access request is approved), or open the invite link you were sent.':
    'Щоб писати п’єси, потрібен доступ: увійдіть через GitHub у налаштуваннях (коли запит на доступ схвалять) або відкрийте надіслане вам посилання-запрошення.',
  'Your access code is not valid (any more): sign in with GitHub again in Settings, or ask for a new invite link.':
    'Ваш код доступу недійсний (вже): увійдіть через GitHub ще раз у налаштуваннях або попросіть нове посилання-запрошення.',
} as const;

export type Key = keyof typeof UK;
export type T = (key: Key, vars?: Record<string, string | number>) => string;

export function translator(lang: Lang): T {
  return (key, vars) => {
    let s: string = lang === 'uk' ? UK[key] : key;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
    return s;
  };
}

// a message that may or may not be one of the keys (an engine error), in the page's language when it is
export function translateAny(lang: Lang, s: string): string {
  return lang === 'uk' && s in UK ? UK[s as Key] : s;
}
