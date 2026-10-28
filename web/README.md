# musicar web

The chat page for musicar (design: `context/claude/app_2026-10-27.md` §3-4; look: the design canvas of 2026-10-28).
Next.js 15, App Router, plain CSS; chats in the browser's IndexedDB. Every engine call is in `lib/engine.ts`.

## Run

```bash
cd web
npm install
npm run dev            # http://localhost:3000, mock engine by default
```

Settings (sidebar) picks the engine:

- **Mock**: replays the `docs/listen` pieces (`public/listen/*.json`) at the laptop's speed. No GPU.
- **Laptop engine**: `python -m musicar.serve` on 127.0.0.1:8000. `next dev` forwards this site's `/v1` there
  (`next.config.ts`), so no CORS is needed. `ENGINE_URL=http://127.0.0.1:8011 npm run dev` forwards elsewhere.
- **Engine URL**: an engine on another address (Tailscale later); the engine must then send CORS headers.

`NEXT_PUBLIC_ENGINE=` (empty) makes the laptop engine the default instead of the mock.

## Layout

| path | what |
|---|---|
| `app/page.tsx` | home: prompt box (typing examples, Tab, dice), ingredient chips, sample cards |
| `app/chat/[id]/page.tsx` | a chat: prompts and their pieces; `/chat/new` is an empty chat |
| `lib/app.tsx` | settings, chats and the runner (generate → event stream → piece), kept across pages |
| `lib/engine.ts` | the `/v1` client (fetch + streamed server-sent events) and the engine choice |
| `lib/mock.ts` | the mock engine |
| `lib/synth.ts` | placeholder WebAudio synth; the SoundFont player replaces it |
| `lib/midi.ts` | MIDI file download, written in the browser |
| `lib/vocab.ts` | caption vocabulary: random prompts, chips, composer checks |
| `components/Roll.tsx` | the canvas piano roll (piece card and thumbnails) |
| `scripts/listen_to_json.py` | `docs/listen` MIDI → `public/listen` (run from the repository root with `.venv/bin/python`) |
