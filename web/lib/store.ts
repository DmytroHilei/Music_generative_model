// Chats live in this browser's IndexedDB (accounts and server-side chats come later, app_2026-10-27.md §7 step 6).

import type { Note } from './notes';

export type PieceStatus = 'queued' | 'streaming' | 'done' | 'error';

export type Piece = {
  id: string;
  prompt: string;
  lengthS: number;
  preset: string;
  model: string;
  jobId?: string;
  seed?: number;
  position?: number;
  notes: Note[];
  status: PieceStatus;
  error?: string;
  elapsedS?: number;
  createdAt: number;
};

export type Chat = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  pieces: Piece[];
};

const DB = 'musicar';
const STORE = 'chats';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function run<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const req = fn(db.transaction(STORE, mode).objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function loadChats(): Promise<Chat[]> {
  try {
    const chats = await run<Chat[]>('readonly', (s) => s.getAll() as IDBRequest<Chat[]>);
    return chats.sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return []; // private window or blocked storage: the page still works, chats just don't persist
  }
}

export async function saveChat(chat: Chat): Promise<void> {
  try {
    await run('readwrite', (s) => s.put(chat));
  } catch {
    // see loadChats
  }
}

export async function removeChat(id: string): Promise<void> {
  try {
    await run('readwrite', (s) => s.delete(id));
  } catch {
    // see loadChats
  }
}

export const newId = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
