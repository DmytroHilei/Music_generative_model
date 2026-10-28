'use client';

// A chat: each message is a caption and the piece it produced. /chat/new is an empty chat until the first prompt.

import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import Composer from '@/components/Composer';
import { Logo } from '@/components/Icons';
import PieceCard from '@/components/PieceCard';
import TopBar from '@/components/TopBar';
import { useMenu } from '@/components/Shell';
import { MODELS, useApp } from '@/lib/app';

export default function ChatPage() {
  const { id } = useParams<{ id: string }>();
  const { chats, ready, compose, settings } = useApp();
  const router = useRouter();
  const openMenu = useMenu();
  const [text, setText] = useState('');
  const end = useRef<HTMLDivElement>(null);
  const chat = id === 'new' ? undefined : chats.find((c) => c.id === id);
  const count = chat?.pieces.length ?? 0;
  const model = MODELS.find((m) => m.id === settings.model) ?? MODELS[0];

  useEffect(() => {
    if (count) end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [count]);

  const submit = (prompt: string) => {
    setText('');
    const chatId = compose(chat?.id ?? null, prompt);
    if (!chat) router.replace(`/chat/${chatId}`);
  };

  const missing = ready && id !== 'new' && !chat;

  return (
    <>
      <TopBar title={chat?.title ?? (missing ? 'Not found' : 'New piece')} line onMenu={openMenu} />
      <div className="thread">
        <div className="col chat">
          {missing && (
            <div className="empty-chat">
              <strong>This chat is not in this browser</strong>
              <div>Chats are kept in the browser that made them. <Link href="/" style={{ color: 'var(--acc-text)' }}>Back home</Link></div>
            </div>
          )}
          {!chat && !missing && (
            <div className="empty-chat">
              <Logo size={40} />
              <strong>A new piece</strong>
              <div>Each message is a caption. After the first piece, Variation and Edit prompt build on it.</div>
            </div>
          )}
          {chat?.pieces.map((p) => (
            <div key={p.id} className="turn">
              <div className="bubble">{p.prompt}</div>
              <PieceCard piece={p} onVariation={() => compose(chat.id, p.prompt)} onEdit={() => setText(p.prompt)} />
            </div>
          ))}
          <div ref={end} />
        </div>
      </div>
      {!missing && (
        <div className="composer-dock">
          <div className="col chat">
            <Composer variant="chat" text={text} setText={setText} onSubmit={submit} startOpen={!chat} />
            <p className="foot-note">
              {model.captions
                ? 'musicar writes MIDI from a caption: mood, tempo, instruments and key work best. English or Ukrainian.'
                : `${model.id} ignores the prompt for now; Compose still writes a new piece.`}
            </p>
          </div>
        </div>
      )}
    </>
  );
}
