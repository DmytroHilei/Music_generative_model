'use client';

// Where the page finds the engine: the mock, the laptop through this site's /v1 (next dev forwards it), or a URL.

import { useEffect, useRef, useState } from 'react';
import { useApp } from '@/lib/app';

export default function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { settings, setSettings, statusDetail } = useApp();
  const ref = useRef<HTMLDialogElement>(null);
  const kind = settings.engine === 'mock' ? 'mock' : settings.engine === '' ? 'local' : 'url';
  const [choice, setChoice] = useState(kind);
  const [url, setUrl] = useState(kind === 'url' ? settings.engine : '');

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      setChoice(kind);
      setUrl(kind === 'url' ? settings.engine : '');
      d.showModal();
    }
    if (!open && d.open) d.close();
  }, [open, kind, settings.engine]);

  const save = () => {
    setSettings({ engine: choice === 'mock' ? 'mock' : choice === 'local' ? '' : url.trim() });
    onClose();
  };

  return (
    <dialog ref={ref} className="settings" onClose={onClose} aria-labelledby="settings-title">
      <h2 id="settings-title">Engine</h2>
      <p>Where pieces are written. Now: {statusDetail || 'checking…'}</p>
      <label className="radio">
        <input type="radio" name="engine" checked={choice === 'local'} onChange={() => setChoice('local')} />
        <span>Laptop engine<small>python -m musicar.serve on 127.0.0.1:8000, reached through this site’s /v1</small></span>
      </label>
      <label className="radio">
        <input type="radio" name="engine" checked={choice === 'url'} onChange={() => setChoice('url')} />
        <span style={{ flex: 1 }}>Engine URL<small>an engine on another address (Tailscale later; needs CORS on the engine)</small>
          <input className="field mono" value={url} placeholder="https://laptop.tailnet.ts.net" aria-label="Engine URL"
            onChange={(e) => { setUrl(e.target.value); setChoice('url'); }} />
        </span>
      </label>
      <label className="radio">
        <input type="radio" name="engine" checked={choice === 'mock'} onChange={() => setChoice('mock')} />
        <span>Mock engine<small>replays the docs/listen pieces at the laptop’s speed; no GPU needed</small></span>
      </label>
      <div className="dialog-foot">
        <button type="button" className="act" onClick={onClose}>Cancel</button>
        <button type="button" className="act primary" onClick={save} disabled={choice === 'url' && !url.trim()}>Save</button>
      </div>
    </dialog>
  );
}
