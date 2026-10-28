'use client';

// The prompt box. Home: large, types example captions while empty (Tab takes the one shown), dice, Compose.
// Chat: smaller, suggestions while empty, then checks of the caption fields the model knows.

import { useEffect, useRef, useState } from 'react';
import { Clock, Dice, Note, Up } from './Icons';
import { LENGTHS, MODELS, PRESETS, useApp } from '@/lib/app';
import { check, EXAMPLES, randomPrompt, SUGGESTIONS } from '@/lib/vocab';

type Props = {
  variant: 'home' | 'chat';
  text: string;
  setText: (s: string) => void;
  onSubmit: (prompt: string) => void;
  typeRef?: React.MutableRefObject<((s: string) => void) | null>;
};

export default function Composer({ variant, text, setText, onSubmit, typeRef }: Props) {
  const { settings, setSettings } = useApp();
  const home = variant === 'home';
  const [ph, setPh] = useState('');
  const [turn, setTurn] = useState(0);
  const typing = useRef<number | null>(null);
  const shown = useRef(0);
  const area = useRef<HTMLTextAreaElement>(null);
  const empty = text.trim() === '';
  const model = MODELS.find((m) => m.id === settings.model) ?? MODELS[0];

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
      const full = EXAMPLES[shown.current];
      if (dir > 0) {
        pos++;
        if (pos >= full.length) { dir = -1; hold = 48; }
      } else {
        pos -= 3;
        if (pos <= 0) { pos = 0; dir = 1; hold = 8; shown.current = (shown.current + 1) % EXAMPLES.length; }
      }
      setPh(EXAMPLES[shown.current].slice(0, pos) + '▍');
    }, 45);
    return () => window.clearInterval(id);
  }, [home, text]);

  const submit = () => {
    if (empty) return;
    if (typing.current) window.clearInterval(typing.current);
    onSubmit(text.trim());
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (home && e.key === 'Tab' && !e.shiftKey && !text) {
      e.preventDefault();
      typeInto(EXAMPLES[shown.current]);
    } else if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const dice = () => {
    setTurn((v) => v + 180 + 90 * Math.floor(Math.random() * 3));
    typeInto(randomPrompt());
  };

  const len = LENGTHS.findIndex((l) => l.s === settings.length);
  const checks = !empty && !home ? check(text) : null;

  return (
    <div className={`composer${home ? ' big' : ''}`}>
      <label htmlFor={`prompt-${variant}`} className="sr-only">Describe the music</label>
      <textarea
        id={`prompt-${variant}`} ref={area} rows={home ? 3 : 2} value={text} onKeyDown={onKey}
        onChange={(e) => { if (typing.current) window.clearInterval(typing.current); setText(e.target.value); }}
        placeholder={home ? ph || '▍' : 'Describe the music… e.g. Повільна сумна мелодія для фортепіано'}
      />
      {!home && empty && (
        <div className="hints">
          <span style={{ padding: '0 4px' }}>Try</span>
          {SUGGESTIONS.map((s) => <button key={s} type="button" className="try" onClick={() => typeInto(s)}>{s}</button>)}
        </div>
      )}
      {checks && model.captions && (
        <div className="hints">
          {checks.fields.map((f) => <span key={f.name} className={`check${f.ok ? ' ok' : ''}`}>{f.ok ? '✓' : '–'} {f.name}</span>)}
          <span style={{ paddingLeft: 4 }}>{checks.hint}</span>
        </div>
      )}
      <div className="controls">
        {home ? (
          <button type="button" className="ctl dice" onClick={dice}><Dice turn={turn} />Random prompt</button>
        ) : (
          <button type="button" className="ctl icon dice" onClick={dice} aria-label="Random prompt"><Dice size={18} turn={turn} /></button>
        )}
        <button type="button" className="ctl" onClick={() => setSettings({ length: LENGTHS[(len + 1) % LENGTHS.length].s })}
          aria-label={`Length ${LENGTHS[len]?.label ?? ''}, click to change`}>
          <Clock />{LENGTHS[len]?.label ?? `${settings.length} s`}
        </button>
        <div className="seg" role="group" aria-label="Preset">
          {PRESETS.map((p) => (
            <button key={p.id} type="button" aria-pressed={settings.preset === p.id} onClick={() => setSettings({ preset: p.id })}>{p.label}</button>
          ))}
        </div>
        <div className="grow" />
        {home && empty && <span className="tabhint"><kbd>Tab</kbd>use this one</span>}
        {home ? (
          <button type="button" className="compose-btn" onClick={submit} disabled={empty}><Note />Compose</button>
        ) : (
          <button type="button" className="send-btn" onClick={submit} disabled={empty} aria-label="Compose"><Up /></button>
        )}
      </div>
    </div>
  );
}
