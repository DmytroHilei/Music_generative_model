'use client';

// The prompt box. Home: large, types example captions while empty (Tab takes the one shown), dice, Compose.
// Chat: smaller, suggestions while empty, then checks of the caption fields the model knows. On a phone the chat box
// folds to one line (prompt + send) while the person reads or swipes, and opens again when they tap into it.

import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Clock, Dice, Note, Up } from './Icons';
import { LENGTHS, MODELS, PRESETS, useApp } from '@/lib/app';
import { check, EXAMPLES, randomPrompt, SUGGESTIONS } from '@/lib/vocab';

type Props = {
  variant: 'home' | 'chat';
  text: string;
  setText: (s: string) => void;
  onSubmit: (prompt: string) => void;
  typeRef?: React.MutableRefObject<((s: string) => void) | null>;
  startOpen?: boolean; // chat on a phone: open at first (an empty chat) or folded (a chat with pieces)
};

const PHONE = '(max-width: 800px)';

function usePhone(): boolean {
  return useSyncExternalStore(
    (f) => { const m = matchMedia(PHONE); m.addEventListener('change', f); return () => m.removeEventListener('change', f); },
    () => matchMedia(PHONE).matches,
    () => false,
  );
}

export default function Composer({ variant, text, setText, onSubmit, typeRef, startOpen = true }: Props) {
  const { settings, setSettings, lang, t } = useApp();
  const examples = EXAMPLES[lang];
  const home = variant === 'home';
  const [ph, setPh] = useState('');
  const [turn, setTurn] = useState(0);
  const typing = useRef<number | null>(null);
  const shown = useRef(0);
  const area = useRef<HTMLTextAreaElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const empty = text.trim() === '';
  const model = MODELS.find((m) => m.id === settings.model) ?? MODELS[0];
  const phone = usePhone();
  const foldable = !home && phone;
  const [open, setOpen] = useState(startOpen);
  const collapsed = foldable && !open;

  // the chat's pieces arrive after the first render (IndexedDB), so follow startOpen when it changes
  useEffect(() => { setOpen(startOpen); }, [startOpen]);

  // a swipe or a tap anywhere else folds the box (and closes the keyboard); the same textarea stays mounted
  // throughout, since iOS drops the keyboard when the focused element is replaced
  useEffect(() => {
    if (!foldable) return;
    const fold = (e: Event) => {
      if (box.current?.contains(e.target as Node)) return;
      setOpen(false);
      area.current?.blur();
    };
    document.addEventListener('touchmove', fold, { passive: true });
    document.addEventListener('pointerdown', fold);
    return () => {
      document.removeEventListener('touchmove', fold);
      document.removeEventListener('pointerdown', fold);
    };
  }, [foldable]);

  // type a prompt into the box a few letters at a time (dice, sample cards, Tab)
  const typeInto = (s: string) => {
    if (typing.current) window.clearInterval(typing.current);
    let i = 0;
    typing.current = window.setInterval(() => {
      i += 2;
      setText(s.slice(0, i));
      if (i >= s.length && typing.current) {
        window.clearInterval(typing.current);
        typing.current = null;
        area.current?.focus();
      }
    }, 16);
  };
  if (typeRef) typeRef.current = typeInto;

  useEffect(() => () => { if (typing.current) window.clearInterval(typing.current); }, []);

  // the empty home box types the examples one after another
  useEffect(() => {
    if (!home || text) return;
    let pos = 0, dir = 1, hold = 8;
    const id = window.setInterval(() => {
      if (hold > 0) { hold--; return; }
      const full = examples[shown.current];
      if (dir > 0) {
        pos++;
        if (pos >= full.length) { dir = -1; hold = 48; }
      } else {
        pos -= 3;
        if (pos <= 0) { pos = 0; dir = 1; hold = 8; shown.current = (shown.current + 1) % examples.length; }
      }
      setPh(examples[shown.current].slice(0, pos) + '▍');
    }, 45);
    return () => window.clearInterval(id);
  }, [home, text, examples]);

  const submit = () => {
    if (empty) return;
    if (typing.current) window.clearInterval(typing.current);
    onSubmit(text.trim());
    if (foldable) {
      setOpen(false); // make room for the piece that is starting
      area.current?.blur();
    }
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (home && e.key === 'Tab' && !e.shiftKey && !text) {
      e.preventDefault();
      typeInto(examples[shown.current]);
    } else if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const dice = () => {
    setTurn((v) => v + 180 + 90 * Math.floor(Math.random() * 3));
    typeInto(randomPrompt(lang));
  };

  const len = LENGTHS.findIndex((l) => l.s === settings.length);
  const lenLabel = len >= 0 ? t(LENGTHS[len].label) : t('{n} s', { n: settings.length });
  const checks = !empty && !home ? check(text) : null;

  return (
    <div ref={box} className={`composer${home ? ' big' : ''}${collapsed ? ' collapsed' : ''}`}>
      <label htmlFor={`prompt-${variant}`} className="sr-only">{t('Describe the music')}</label>
      <textarea
        id={`prompt-${variant}`} ref={area} rows={home ? 3 : collapsed ? 1 : 2} value={text} onKeyDown={onKey}
        onFocus={() => setOpen(true)}
        onChange={(e) => { if (typing.current) window.clearInterval(typing.current); setText(e.target.value); }}
        placeholder={home ? ph || '▍' : collapsed ? t('Describe the music…') : t('Describe the music… e.g. Повільна сумна мелодія для фортепіано')}
      />
      {!home && empty && (
        <div className="hints">
          <span style={{ padding: '0 4px' }}>{t('Try')}</span>
          {SUGGESTIONS[lang].map((s) => <button key={s} type="button" className="try" onClick={() => typeInto(s)}>{s}</button>)}
        </div>
      )}
      {checks && model.captions && (
        <div className="hints">
          {checks.fields.map((f) => <span key={f.name} className={`check${f.ok ? ' ok' : ''}`}>{f.ok ? '✓' : '–'} {t(f.name)}</span>)}
          <span style={{ paddingLeft: 4 }}>{t(checks.hint)}</span>
        </div>
      )}
      <div className="controls">
        {home ? (
          <button type="button" className="ctl dice" onClick={dice}><Dice turn={turn} />{t('Random prompt')}</button>
        ) : (
          <button type="button" className="ctl icon dice" onClick={dice} aria-label={t('Random prompt')}><Dice size={18} turn={turn} /></button>
        )}
        <button type="button" className="ctl" onClick={() => setSettings({ length: LENGTHS[(len + 1) % LENGTHS.length].s })}
          aria-label={t('Length {len}, click to change', { len: lenLabel })}>
          <Clock />{lenLabel}
        </button>
        <div className="seg" role="group" aria-label={t('Preset')}>
          {PRESETS.map((p) => (
            <button key={p.id} type="button" aria-pressed={settings.preset === p.id} onClick={() => setSettings({ preset: p.id })}>{t(p.label)}</button>
          ))}
        </div>
        {model.captions && (
          <button type="button" className="ctl" aria-pressed={settings.opening}
            onClick={() => setSettings({ opening: !settings.opening })}
            title={t('Start from the first notes of a real song that fits the caption (same genre and mood); they are not in the result. Variation picks another song.')}>
            {t('Real opening')}
          </button>
        )}
        <div className="grow" />
        {home && empty && <span className="tabhint"><kbd>Tab</kbd>{t('use this one')}</span>}
        {home ? (
          <button type="button" className="compose-btn" onClick={submit} disabled={empty}><Note />{t('Compose')}</button>
        ) : (
          <button type="button" className="send-btn" onClick={submit} disabled={empty} aria-label={t('Compose')}><Up /></button>
        )}
      </div>
    </div>
  );
}
