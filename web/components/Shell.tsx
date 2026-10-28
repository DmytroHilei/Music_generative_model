'use client';

import { createContext, useContext, useState } from 'react';
import Sidebar from './Sidebar';
import SettingsDialog from './SettingsDialog';
import { AppProvider } from '@/lib/app';

const MenuCtx = createContext<() => void>(() => {});
export const useMenu = () => useContext(MenuCtx);

export default function Shell({ children }: { children: React.ReactNode }) {
  const [menu, setMenu] = useState(false);
  const [settings, setSettings] = useState(false);
  return (
    <AppProvider>
      <MenuCtx.Provider value={() => setMenu(true)}>
        <div className="shell">
          <Sidebar open={menu} onClose={() => setMenu(false)} onSettings={() => { setMenu(false); setSettings(true); }} />
          <main className="main">{children}</main>
        </div>
        <SettingsDialog open={settings} onClose={() => setSettings(false)} />
      </MenuCtx.Provider>
    </AppProvider>
  );
}
