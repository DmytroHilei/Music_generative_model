'use client';

// The strip above home and chats: (title), the model picker and the engine's state.

import { useEffect, useRef, useState } from 'react';
import { Chevron, Menu } from './Icons';
import { MODELS, useApp } from '@/lib/app';

const LABEL = { checking: 'checking the engine…', online: 'laptop engine online', busy: 'writing a piece…', offline: 'engine offline', mock: 'mock engine' };

export default function TopBar({ title, line = false, onMenu }: { title?: string; line?: boolean; onMenu: () => void }) {
  const { settings, setSettings, status, statusDetail } = useApp();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const model = MODELS.find((m) => m.id === settings.model) ?? MODELS[0];

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', close);
    box.current?.addEventListener('keydown', esc);
    const el = box.current;
    return () => { document.removeEventListener('mousedown', close); el?.removeEventListener('keydown', esc); };
  }, [open]);

  return (
    <div className={`topbar${line ? ' line' : ''}`}>
      <button type="button" className="menu-btn" onClick={onMenu} aria-label="Chats"><Menu /></button>
      {title && <h1>{title}</h1>}
      <div className="topbar-right">
        <div className="picker" ref={box}>
          <button type="button" className="picker-btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)}>
            {model.id}<span className="mono">{model.size}</span><Chevron />
          </button>
          {open && (
            <div className="picker-menu" role="listbox" aria-label="Model">
              {MODELS.map((m) => (
                <button key={m.id} type="button" role="option" aria-selected={m.id === model.id} disabled={!m.available}
                  className={`picker-opt${m.id === model.id ? ' on' : ''}`}
                  onClick={() => { setSettings({ model: m.id }); setOpen(false); }}>
                  <span style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <span className="name">{m.id} <span className="mono muted" style={{ fontSize: 11, fontWeight: 400 }}>{m.size}</span></span>
                    <span className="desc">{m.available ? m.desc : `${m.desc}. Not on this engine yet.`}</span>
                  </span>
                  <span className="check" style={{ opacity: m.id === model.id ? 1 : 0 }}>✓</span>
                </button>
              ))}
              <div className="picker-note">The engine keeps one model in memory; switching will take a few seconds.</div>
            </div>
          )}
        </div>
        <span className="pill" title={statusDetail}><span className={`dot ${status}`} />{LABEL[status]}</span>
      </div>
    </div>
  );
}
