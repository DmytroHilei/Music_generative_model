import type { Metadata, Viewport } from 'next';
import { Inter_Tight, JetBrains_Mono } from 'next/font/google';
import { cookies, headers } from 'next/headers';
import Shell from '@/components/Shell';
import { LANG_COOKIE, type Lang } from '@/lib/i18n';
import './globals.css';

const ui = Inter_Tight({ subsets: ['latin', 'cyrillic'], variable: '--font-ui' });
const mono = JetBrains_Mono({ subsets: ['latin', 'cyrillic'], variable: '--font-mono' });

export const metadata: Metadata = {
  title: 'musicar',
  description: 'Describe a piece in one sentence; musicar writes it as MIDI, note by note.',
};

// viewport-fit=cover: the page reaches under the iPhone's home bar, and env(safe-area-inset-*) says by how much
export const viewport: Viewport = { width: 'device-width', initialScale: 1, viewportFit: 'cover' };

// the saved theme is set before the first paint, so a light-mode visitor never sees a dark flash
const THEME = `try{var t=localStorage.getItem('musicar.theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}`;

// the language: the sidebar switch's cookie, else Ukrainian for a browser that asks for it first
async function pageLang(): Promise<Lang> {
  const saved = (await cookies()).get(LANG_COOKIE)?.value;
  if (saved === 'en' || saved === 'uk') return saved;
  return /^uk\b/i.test((await headers()).get('accept-language') ?? '') ? 'uk' : 'en';
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const lang = await pageLang();
  return (
    <html lang={lang} className={`${ui.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME }} />
      </head>
      <body>
        <Shell lang={lang}>{children}</Shell>
      </body>
    </html>
  );
}
