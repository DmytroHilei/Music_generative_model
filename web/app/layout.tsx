import type { Metadata, Viewport } from 'next';
import { Inter_Tight, JetBrains_Mono } from 'next/font/google';
import Shell from '@/components/Shell';
import './globals.css';

const ui = Inter_Tight({ subsets: ['latin', 'cyrillic'], variable: '--font-ui' });
const mono = JetBrains_Mono({ subsets: ['latin', 'cyrillic'], variable: '--font-mono' });

export const metadata: Metadata = {
  title: 'musicar',
  description: 'Describe a piece in one sentence; musicar writes it as MIDI, note by note.',
};

export const viewport: Viewport = { width: 'device-width', initialScale: 1 };

// the saved theme is set before the first paint, so a light-mode visitor never sees a dark flash
const THEME = `try{var t=localStorage.getItem('musicar.theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${ui.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME }} />
      </head>
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
