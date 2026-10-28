'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Gear, HomeIcon, Logo, Moon, Plus, Sun, Trash } from './Icons';
import { useApp } from '@/lib/app';
import type { Chat } from '@/lib/store';

const DAY = 86400000;

function groups(chats: Chat[]): [string, Chat[]][] {
  const today = new Date().setHours(0, 0, 0, 0);
  const out: [string, Chat[]][] = [['Today', []], ['Yesterday', []], ['Earlier', []]];
  for (const c of chats) out[c.updatedAt >= today ? 0 : c.updatedAt >= today - DAY ? 1 : 2][1].push(c);
  return out.filter(([, cs]) => cs.length);
}

export default function Sidebar({ open, onClose, onSettings }: { open: boolean; onClose: () => void; onSettings: () => void }) {
  const { chats, ready, theme, setTheme, deleteChat } = useApp();
  const path = usePathname();

  return (
    <>
      <div className={`scrim${open ? ' open' : ''}`} onClick={onClose} />
      <nav className={`side${open ? ' open' : ''}`} aria-label="Chats" onClick={(e) => { if ((e.target as HTMLElement).closest('a')) onClose(); }}>
        <Link href="/" className="brand" aria-label="musicar home"><Logo /><span>musicar</span></Link>
        <Link href="/chat/new" className="side-btn new"><Plus />New piece</Link>
        <Link href="/" className={`side-btn${path === '/' ? ' on' : ''}`}><HomeIcon />Home</Link>
        {ready && chats.length === 0 && <div className="side-empty" style={{ marginTop: 18 }}>Your pieces will appear here.</div>}
        {groups(chats).map(([label, cs]) => (
          <div key={label}>
            <div className="side-group">{label}</div>
            {cs.map((c) => (
              <div key={c.id} className={`chat-item${path === `/chat/${c.id}` ? ' on' : ''}`}>
                <Link href={`/chat/${c.id}`}>{c.title}</Link>
                <button type="button" className="del" aria-label={`Delete ${c.title}`}
                  onClick={() => { if (confirm(`Delete “${c.title}”?`)) deleteChat(c.id); }}><Trash /></button>
              </div>
            ))}
          </div>
        ))}
        <div className="side-fill" />
        <button type="button" className="side-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
          {theme === 'dark' ? <Sun /> : <Moon />}{theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
        <button type="button" className="side-btn" onClick={onSettings}><Gear />Settings</button>
      </nav>
    </>
  );
}
