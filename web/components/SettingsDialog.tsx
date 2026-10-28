'use client';

// Where the page finds the engine: the mock, the laptop through this site's /v1 (next dev forwards it), or a URL; and
// access: Sign in with GitHub (app/api/auth/github/*) or the code of an invite link.

import { useEffect, useRef, useState } from 'react';
import { REQUEST_ACCESS_URL, useApp } from '@/lib/app';

export default function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { settings, setSettings, statusDetail, t } = useApp();
  const ref = useRef<HTMLDialogElement>(null);
  const kind = settings.engine === 'mock' ? 'mock' : settings.engine === '' ? 'local' : 'url';
  const [choice, setChoice] = useState(kind);
  const [url, setUrl] = useState(kind === 'url' ? settings.engine : '');
  const [code, setCode] = useState(settings.token);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      setChoice(kind);
      setUrl(kind === 'url' ? settings.engine : '');
      setCode(settings.token);
      d.showModal();
    }
    if (!open && d.open) d.close();
  }, [open, kind, settings.engine, settings.token]);

  const save = () => {
    const token = code.trim();
    setSettings({
      engine: choice === 'mock' ? 'mock' : choice === 'local' ? '' : url.trim(), token,
      account: token === settings.token ? settings.account : '', // a pasted code is no GitHub sign-in
    });
    onClose();
  };

  return (
    <dialog ref={ref} className="settings" onClose={onClose} aria-labelledby="settings-title">
      <h2 id="settings-title">{t('Engine')}</h2>
      <p>{t('Where pieces are written. Now: {s}', { s: statusDetail || t('checking…') })}</p>
      <label className="radio">
        <input type="radio" name="engine" checked={choice === 'local'} onChange={() => setChoice('local')} />
        <span>{t('Laptop engine')}<small>{t('python -m musicar.serve on 127.0.0.1:8000, reached through this site’s /v1')}</small></span>
      </label>
      <label className="radio">
        <input type="radio" name="engine" checked={choice === 'url'} onChange={() => setChoice('url')} />
        <span style={{ flex: 1 }}>{t('Engine URL')}<small>{t('an engine on another address (Tailscale later; needs CORS on the engine)')}</small>
          <input className="field mono" value={url} placeholder="https://laptop.tailnet.ts.net" aria-label={t('Engine URL')}
            onChange={(e) => { setUrl(e.target.value); setChoice('url'); }} />
        </span>
      </label>
      <label className="radio">
        <input type="radio" name="engine" checked={choice === 'mock'} onChange={() => setChoice('mock')} />
        <span>{t('Mock engine')}<small>{t('replays the docs/listen pieces at the laptop’s speed; no GPU needed')}</small></span>
      </label>
      <div className="signin">
        <span>{t('Access')}<small>{settings.account ? t('signed in as {who}', { who: settings.account })
          : settings.token ? t('with an invite code') : t('the engine writes pieces only for approved people')}</small></span>
        <a className="act primary" href="/api/auth/github/login">{t('Sign in with GitHub')}</a>
        <a className="act quiet" href={REQUEST_ACCESS_URL} target="_blank" rel="noreferrer">{t('Request access')}</a>
      </div>
      <label className="radio">
        <span style={{ flex: 1 }}>{t('Invite code')}<small>{t('set by an invite link or a GitHub sign-in; paste one by hand if you have it')}</small>
          <input className="field mono" type="password" value={code} placeholder="mus_…" aria-label={t('Invite code')}
            autoComplete="off" spellCheck={false} onChange={(e) => setCode(e.target.value)} />
        </span>
      </label>
      <div className="dialog-foot">
        <button type="button" className="act" onClick={onClose}>{t('Cancel')}</button>
        <button type="button" className="act primary" onClick={save} disabled={choice === 'url' && !url.trim()}>{t('Save')}</button>
      </div>
    </dialog>
  );
}
