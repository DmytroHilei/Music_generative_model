'use client';

// The page's state: settings, chats (IndexedDB) and the runner that turns a prompt into a streaming piece. It sits in
// the root layout, so a piece keeps streaming while the person moves between home and chats.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { DEFAULT_ENGINE, EngineError, engineFor, type Engine } from './engine';
import { LANG_COOKIE, translateAny, translator, type Key, type Lang, type T } from './i18n';
import { loadBank } from './sound';
import { loadChats, newId, removeChat, saveChat, type Chat, type Piece } from './store';
import { titleOf } from './vocab';

// labels, descriptions: keys of lib/i18n.ts, shown with useApp().t
export const LENGTHS: { label: Key; s: number }[] = [
  { label: '30 s', s: 30 },
  { label: '1 min', s: 60 },
  { label: '2 min', s: 120 },
  { label: '4 min', s: 240 },
];

// faithful / wild are first guesses around the gen_search winner (config/generate/f1.toml = balanced)
export const PRESETS: { id: string; label: Key; overrides: Record<string, number> }[] = [
  { id: 'faithful', label: 'faithful', overrides: { cfg_scale: 3.0, temperature: 1.0 } },
  { id: 'balanced', label: 'balanced', overrides: {} },
  { id: 'wild', label: 'wild', overrides: { cfg_scale: 1.5, temperature: 1.2 } },
];

export const MODELS: { id: string; size: string; desc: Key; captions: boolean; available: boolean }[] = [
  { id: 'musicar-L', size: '477M', desc: 'Multi-instrument, reads English and Ukrainian captions', captions: true, available: true },
  { id: 'musicar-S', size: '108M', desc: 'Multi-instrument, faster; does not read the prompt yet', captions: false, available: false },
  { id: 'musicar-S-piano', size: '107M', desc: 'Solo piano; does not read the prompt yet', captions: false, available: false },
];

export type Theme = 'dark' | 'light';
// token: the engine token (an invite link https://…/?invite=mus_… sets it once, GitHub sign-in sets it, Settings can
// paste it); account: '@login' after a GitHub sign-in, '' otherwise
// opening: start pieces from a caption-matched real opening (the composer's switch)
export type Settings = {
  engine: string; model: string; length: number; preset: string; token: string; account: string; opening: boolean;
};
export type EngineStatus = 'checking' | 'online' | 'busy' | 'offline' | 'mock';
export type Notice = { kind: 'ok' | 'info' | 'error'; text: string; link?: { href: string; label: string } };

export const REQUEST_ACCESS_URL =
  'https://github.com/DmytroHilei/Music_generative_model/issues/new?template=access-request.yml';
const SETTINGS_KEY = 'musicar.settings';
const THEME_KEY = 'musicar.theme';
const RECONNECTS = 3; // a broken event stream is opened again this many times (after 1, 2, 4 s)
const SIGNIN_COOKIE = 'musicar_signin'; // set by app/api/auth/github/callback (lib/github.ts), read once here
const DEFAULTS: Settings = {
  engine: DEFAULT_ENGINE, model: 'musicar-L', length: 60, preset: 'balanced', token: '', account: '', opening: false,
};

// an invite link's token, taken out of the address bar (so it isn't shared on by copying the URL), else null
function takeInvite(): string | null {
  const url = new URL(window.location.href);
  const token = url.searchParams.get('invite')?.trim();
  if (!token) return null;
  url.searchParams.delete('invite');
  window.history.replaceState(window.history.state, '', url.pathname + url.search + url.hash);
  return token;
}

// the GitHub sign-in's result (?signin=…&as=…&reason=…, and the token cookie on success), taken out of the address
// bar and the cookie jar, else null
function takeSignin(): { status: string; as: string; reason: string; token: string } | null {
  const url = new URL(window.location.href);
  const status = url.searchParams.get('signin');
  if (!status) return null;
  const as = url.searchParams.get('as') ?? '';
  const reason = url.searchParams.get('reason') ?? '';
  for (const k of ['signin', 'as', 'reason']) url.searchParams.delete(k);
  window.history.replaceState(window.history.state, '', url.pathname + url.search + url.hash);
  const m = document.cookie.match(new RegExp(`(?:^|;\\s*)${SIGNIN_COOKIE}=([^;]+)`));
  if (m) document.cookie = `${SIGNIN_COOKIE}=; Max-Age=0; path=/`;
  return { status, as, reason, token: m ? decodeURIComponent(m[1]) : '' };
}

function signinNotice({ status, as, reason, token }: { status: string; as: string; reason: string; token: string }, t: T): Notice {
  const who = as ? `@${as}` : t('Your GitHub account');
  if (status === 'ok' && token) return { kind: 'ok', text: t('Signed in as @{as}: you can write pieces now.', { as }) };
  if (status === 'pending') return { kind: 'info', text: t('{who}: the access request is waiting for approval. Sign in again once it is approved.', { who }) };
  if (status === 'none') {
    return { kind: 'info', text: t('{who} has no access request yet.', { who }), link: { href: REQUEST_ACCESS_URL, label: t('Request access') } };
  }
  if (status === 'revoked') return { kind: 'error', text: t('Access for {who} was revoked.', { who }) };
  const why = reason === 'off' ? t('GitHub sign-in is not set up on this site yet.')
    : reason === 'state' ? t('the sign-in took too long or was opened twice. Try again.')
      : reason === 'offline' ? t('the engine or GitHub did not answer. Try again in a minute.') : `${reason || t('unknown error')}.`;
  return { kind: 'error', text: reason === 'off' ? why : t('Sign-in failed: {why}', { why }) };
}

type AppState = {
  ready: boolean;
  chats: Chat[];
  settings: Settings;
  setSettings: (s: Partial<Settings>) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
  lang: Lang;
  setLang: (l: Lang) => void;
  t: T;
  status: EngineStatus;
  statusDetail: string;
  notice: Notice | null;
  dismissNotice: () => void;
  // like: a piece whose length, preset and opening a Variation keeps (default: the current settings)
  compose: (chatId: string | null, prompt: string, like?: Piece) => string;
  deleteChat: (id: string) => void;
};

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp outside AppProvider');
  return v;
}

function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}

// lang: the language the server rendered (app/layout.tsx)
export function AppProvider({ children, lang: firstLang }: { children: React.ReactNode; lang: Lang }) {
  const [ready, setReady] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const chatsRef = useRef<Chat[]>([]);
  const [settings, setSettingsState] = useState<Settings>(DEFAULTS);
  const settingsRef = useRef<Settings>(DEFAULTS);
  const [theme, setThemeState] = useState<Theme>('dark');
  const [lang, setLangState] = useState<Lang>(firstLang);
  const t = useMemo(() => translator(lang), [lang]);
  // messages kept with a piece are written in the language of the moment
  const langRef = useRef(lang);
  langRef.current = lang;
  const tr = useCallback<T>((key, vars) => translator(langRef.current)(key, vars), []);
  const [status, setStatus] = useState<EngineStatus>('checking');
  const [statusDetail, setStatusDetail] = useState('');
  const [notice, setNotice] = useState<Notice | null>(null);
  const dismissNotice = useCallback(() => setNotice(null), []);

  const commit = useCallback((next: Chat[]) => {
    chatsRef.current = next;
    setChats(next);
  }, []);

  const patch = useCallback((chatId: string, pieceId: string, change: (p: Piece) => Partial<Piece>, persist = false) => {
    const next = chatsRef.current.map((c) =>
      c.id !== chatId ? c : { ...c, pieces: c.pieces.map((p) => (p.id === pieceId ? { ...p, ...change(p) } : p)) });
    commit(next);
    if (persist) {
      const chat = next.find((c) => c.id === chatId);
      if (chat) void saveChat(chat);
    }
  }, [commit]);

  // a job's event stream; a stream that breaks (a network error, or the end before done / error / cancelled) is opened
  // again up to RECONNECTS times: the engine replays the job's events from the start in the same order, so the ones
  // already handled are skipped and the piece continues where it was
  const follow = useCallback(async (engine: Engine, chatId: string, pieceId: string, jobId: string) => {
    let handled = 0;
    let finished = false;
    for (let attempt = 0; ; attempt++) {
      let i = 0;
      try {
        await engine.events(jobId, (e) => {
          if (i++ < handled) return; // replayed: already handled
          handled++;
          if (e.type === 'queued') patch(chatId, pieceId, () => ({ status: 'queued', position: e.position }));
          else if (e.type === 'started') patch(chatId, pieceId, () => ({ status: 'streaming', seed: e.seed, notes: [] }));
          else if (e.type === 'notes' && e.candidate === 0) patch(chatId, pieceId, (p) => ({ notes: p.notes.concat(e.notes) }));
          else if (e.type === 'done') {
            finished = true;
            patch(chatId, pieceId, () => ({ status: 'done', elapsedS: e.elapsed_s, seed: e.seed }), true);
          } else if (e.type === 'error' || e.type === 'cancelled') {
            finished = true;
            patch(chatId, pieceId, () => ({ status: 'error', error: e.type === 'error' ? e.message : tr('Cancelled.') }), true);
          }
        });
        if (finished) return;
      } catch (err) {
        const msg = (err as Error).message;
        if ((err as Error).name === 'AbortError') return;
        if (/no such job/.test(msg)) {
          patch(chatId, pieceId, () => ({
            status: 'error',
            error: tr('The engine no longer has this piece (it restarted, or the piece is more than an hour old). Use Variation to write it again.'),
          }), true);
          return;
        }
        const access = err instanceof EngineError && /invite|access code/.test(msg); // 401: retrying can't help
        if (access || attempt >= RECONNECTS) {
          patch(chatId, pieceId, () => ({ status: 'error', error: tr('Lost the engine: {msg}', { msg: translateAny(langRef.current, msg) }) }), true);
          return;
        }
      }
      if (attempt >= RECONNECTS) {
        patch(chatId, pieceId, () => ({ status: 'error', error: tr('Lost the engine: the stream kept breaking.') }), true);
        return;
      }
      await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt)); // 1, 2, 4 s
    }
  }, [patch, tr]);

  const run = useCallback(async (chatId: string, piece: Piece) => {
    const engine = engineFor(settingsRef.current.engine, settingsRef.current.token);
    try {
      const preset = PRESETS.find((p) => p.id === piece.preset) ?? PRESETS[1];
      const { job_id, position } = await engine.generate({
        prompt: piece.prompt, length_s: piece.lengthS, seed: null, best_of: 1, opening: !!piece.opening,
        overrides: preset.overrides,
      });
      patch(chatId, piece.id, () => ({ jobId: job_id, position }), true);
      await follow(engine, chatId, piece.id, job_id);
    } catch (err) {
      const msg = (err as Error).message;
      patch(chatId, piece.id, () => ({
        status: 'error',
        error: /fetch|network|Failed/i.test(msg) ? tr('The engine is offline. Check Settings, or try the mock engine.')
          : translateAny(langRef.current, msg),
      }), true);
    }
  }, [follow, patch, tr]);

  const compose = useCallback((chatId: string | null, prompt: string, like?: Piece) => {
    const now = Date.now();
    const s = settingsRef.current;
    const piece: Piece = {
      id: newId(), prompt, lengthS: like?.lengthS ?? s.length, preset: like?.preset ?? s.preset, model: s.model,
      opening: like ? !!like.opening : s.opening, notes: [], status: 'queued', createdAt: now,
    };
    const old = chatId ? chatsRef.current.find((c) => c.id === chatId) : undefined;
    const chat: Chat = old
      ? { ...old, updatedAt: now, pieces: [...old.pieces, piece] }
      : { id: newId(), title: titleOf(prompt), createdAt: now, updatedAt: now, pieces: [piece] };
    commit([chat, ...chatsRef.current.filter((c) => c.id !== chat.id)]);
    void saveChat(chat);
    void run(chat.id, piece);
    loadBank().catch(() => {}); // the SoundFont downloads while the engine writes
    return chat.id;
  }, [commit, run]);

  const deleteChat = useCallback((id: string) => {
    commit(chatsRef.current.filter((c) => c.id !== id));
    void removeChat(id);
  }, [commit]);

  const setSettings = useCallback((change: Partial<Settings>) => {
    const next = { ...settingsRef.current, ...change };
    settingsRef.current = next;
    setSettingsState(next);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
    } catch {
      // storage blocked: settings last for this visit
    }
  }, []);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch {
      // see setSettings
    }
  }, []);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    document.documentElement.lang = l;
    document.cookie = `${LANG_COOKIE}=${l}; Max-Age=31536000; path=/; SameSite=Lax`;
  }, []);

  // first load: settings (and an invite link's or a GitHub sign-in's token), theme, chats; pieces left mid-stream
  // reattach to their job (the engine keeps its events)
  useEffect(() => {
    let s = readJSON(SETTINGS_KEY, DEFAULTS);
    const invite = takeInvite();
    const signin = takeSignin();
    if (invite) s = { ...s, token: invite, account: '' };
    if (signin?.status === 'ok' && signin.token) s = { ...s, token: signin.token, account: `@${signin.as}` };
    if (signin) setNotice(signinNotice(signin, tr));
    if (invite || signin?.token) {
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
      } catch {
        // storage blocked: the token lasts for this visit
      }
    }
    settingsRef.current = s;
    setSettingsState(s);
    const t = document.documentElement.dataset.theme;
    setThemeState(t === 'light' || t === 'dark' ? t : matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    void loadChats().then((loaded) => {
      commit(loaded);
      setReady(true);
      const engine = engineFor(s.engine, s.token);
      for (const c of loaded) for (const p of c.pieces) {
        if (p.status === 'done' || p.status === 'error') continue;
        if (p.jobId) void follow(engine, c.id, p.id, p.jobId);
        else patch(c.id, p.id, () => ({ status: 'error', error: tr('Interrupted before the engine took it.') }), true);
      }
    });
  }, [commit, follow, patch, tr]);

  useEffect(() => {
    let live = true;
    const ask = async () => {
      if (settings.engine === 'mock') {
        setStatus('mock');
        setStatusDetail(t('mock engine: replays docs/listen pieces'));
        return;
      }
      try {
        const h = await engineFor(settings.engine).health();
        if (!live) return;
        setStatus(h.busy ? 'busy' : 'online');
        setStatusDetail(`${h.engine} · ${h.model}${h.loaded ? '' : t(' (loads on the first piece)')}${h.queue_len ? t(' · {n} waiting', { n: h.queue_len }) : ''}${settings.token ? '' : t(' · not signed in')}`);
      } catch {
        if (!live) return;
        setStatus('offline');
        setStatusDetail(settings.engine ? t('no answer from {engine}', { engine: settings.engine }) : t('no answer from the laptop engine'));
      }
    };
    setStatus('checking');
    void ask();
    const id = window.setInterval(ask, 15000);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [settings.engine, settings.token, t]);

  const value = useMemo(() => ({
    ready, chats, settings, setSettings, theme, setTheme, lang, setLang, t, status, statusDetail, notice, dismissNotice,
    compose, deleteChat,
  }), [ready, chats, settings, setSettings, theme, setTheme, lang, setLang, t, status, statusDetail, notice,
    dismissNotice, compose, deleteChat]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
