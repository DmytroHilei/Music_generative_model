'use client';

import { createContext, useContext, useState } from 'react';
import Sidebar from './Sidebar';
import NoticeBar from './NoticeBar';
import SettingsDialog from './SettingsDialog';
import { AppProvider } from '@/lib/app';
import type { Lang } from '@/lib/i18n';

const MenuCtx = createContext<() => void>(() => {});
export const useMenu = () => useContext(MenuCtx);

export default function Shell({ children, lang }: { children: React.ReactNode; lang: Lang }) {
  const [menu, setMenu] = useState(false);
  const [settings, setSettings] = useState(false);
  return (
    <AppProvider lang={lang}>
      <MenuCtx.Provider value={() => setMenu(true)}>
        <div className="shell">
          <Sidebar open={menu} onClose={() => setMenu(false)} onSettings={() => { setMenu(false); setSettings(true); }} />
          <main className="main"><NoticeBar />{children}</main>
        </div>
        <SettingsDialog open={settings} onClose={() => setSettings(false)} />
      </MenuCtx.Provider>
    </AppProvider>
  );
}
