'use client';

// Home: the big prompt box, ingredient chips and the sample cards. Compose opens a new chat with the piece writing.

import { useRouter } from 'next/navigation';
import { useRef, useState } from 'react';
import Composer from '@/components/Composer';
import { HeroBars, ListenFirst, useSamples } from '@/components/Listen';
import TopBar from '@/components/TopBar';
import { useMenu } from '@/components/Shell';
import { MODELS, useApp } from '@/lib/app';
import { INGREDIENTS, parse, toggle } from '@/lib/vocab';

export default function Home() {
  const { compose, settings, lang, t } = useApp();
  const router = useRouter();
  const openMenu = useMenu();
  const [text, setText] = useState('');
  const typeRef = useRef<((s: string) => void) | null>(null);
  const samples = useSamples();
  const sel = parse(text);
  const model = MODELS.find((m) => m.id === settings.model) ?? MODELS[0];

  const submit = (prompt: string) => {
    const id = compose(null, prompt);
    router.push(`/chat/${id}`);
  };

  return (
    <>
      <TopBar onMenu={openMenu} />
      <div className="hero">
        <HeroBars samples={samples} />
        <div className="hero-text">
          <span className="eyebrow mono">{t('text to MIDI · English or Ukrainian')}</span>
          <h1>{t('What should it play?')}</h1>
          <p>{t('Describe a piece in one sentence. musicar writes it note by note, and you hear it while it is still writing.')}</p>
        </div>
      </div>
      <div className="wrap" style={{ paddingTop: 28 }}>
        <div className="col">
          <Composer variant="home" text={text} setText={setText} onSubmit={submit} typeRef={typeRef} />
          {!model.captions && (
            <div className="note-box">{t('{model} does not read captions yet: it writes freely and ignores the prompt. Pick musicar-L for text control.', { model: model.id })}</div>
          )}
          <div className="ingredients">
            {INGREDIENTS.map((row) => (
              <div key={row.field} className="ing-row">
                <span>{t(row.label)}</span>
                {row.words.map((w) => {
                  const on = row.field === 'inst' ? sel.inst.includes(w.id) : sel[row.field] === w.id;
                  return (
                    <button key={w.id} type="button" className={`chip${on ? ' on' : ''}`} aria-pressed={on}
                      onClick={() => setText(toggle(text, row.field, w.id, lang))}>{lang === 'uk' ? w.uk : w.id}</button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
        <div className="col wide">
          <ListenFirst samples={samples} onUse={(p) => { typeRef.current?.(p); window.scrollTo({ top: 0, behavior: 'smooth' }); }} />
        </div>
      </div>
    </>
  );
}
