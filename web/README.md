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

## Sound

Pieces play with the FluidR3 GM SoundFont (MuseScore 3's compressed `FluidR3Mono_GM.sf3`, 23.7 MB, MIT), the same
instruments as the FluidSynth MP3s, through spessasynth_lib in an AudioWorklet. `scripts/assets.mjs` runs before
`dev` and `build`: it copies spessasynth's worklet processor into `public/` and downloads the soundfont into
`public/soundfont/` (checked against its SHA-256); both are gitignored. The browser fetches the soundfont once, on
the first play or Compose. If it can't load, playback falls back to a small built-in synth.

**MP3**: rendered in the browser with the same soundfont in an OfflineAudioContext, then encoded with LAME
(wasm-media-encoders, VBR quality 2). A 1-minute piece takes about 2 s. Same instruments as the FluidSynth MP3s, not
identical audio (reverb and interpolation differ).

Versions are pinned (`spessasynth_lib` 4.3.14 with `spessasynth_core` 4.3.20 through `overrides`, as the lib asks for
`latest`).

## Deploy (Vercel)

1. vercel.com → Add New → Project → import the GitHub repository.
2. **Root Directory: `web`**. Vercel detects Next.js; the default build command runs `prebuild`, which fetches the
   soundfont.
3. Production deploys come from `main`; another branch deploys as a preview (or set it under Settings → Git →
   Production Branch).
4. Optional, Settings → Git → Ignored Build Step: `git diff --quiet HEAD^ HEAD -- .` skips builds for commits that
   don't touch `web/`.

On Vercel the page works with the mock engine. The laptop engine needs to be reachable (Tailscale Funnel) and to
send CORS headers for the Vercel origin; then set `NEXT_PUBLIC_ENGINE=https://<laptop>.ts.net` in the project's
environment variables. The `/v1` forwarding in `next.config.ts` is for `next dev` only.

## Layout

| path | what |
|---|---|
| `app/page.tsx` | home: prompt box (typing examples, Tab, dice), ingredient chips, sample cards |
| `app/chat/[id]/page.tsx` | a chat: prompts and their pieces; `/chat/new` is an empty chat |
| `lib/app.tsx` | settings, chats and the runner (generate → event stream → piece), kept across pages |
| `lib/engine.ts` | the `/v1` client (fetch + streamed server-sent events) and the engine choice |
| `lib/mock.ts` | the mock engine |
| `lib/sound.ts` | the SoundFont and the page's one synthesizer (spessasynth_lib) |
| `lib/synth.ts` | the player: schedules notes ahead of the playhead, also while they stream; fallback synth |
| `lib/render.ts` | MP3: offline render + LAME |
| `lib/midi.ts` | MIDI file, written in the browser |
| `scripts/assets.mjs` | worklet processor + soundfont into `public/` (runs before dev and build) |
| `lib/vocab.ts` | caption vocabulary: random prompts, chips, composer checks |
| `components/Roll.tsx` | the canvas piano roll (piece card and thumbnails) |
| `scripts/listen_to_json.py` | `docs/listen` MIDI → `public/listen` (run from the repository root with `.venv/bin/python`) |
