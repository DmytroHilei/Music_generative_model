// Puts the two runtime files the sound needs into public/ (both gitignored), before `next dev` and `next build`:
// - spessasynth's AudioWorklet processor, copied from node_modules (it must be served as its own file)
// - the FluidR3 GM soundfont as SF3 (MuseScore 3's compressed FluidR3Mono_GM, MIT), the same instruments as the
//   FluidSynth MP3s in docs/listen; downloaded once and checked against its SHA-256

import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const SF = {
  url: 'https://github.com/musescore/MuseScore/raw/v3.6.2/share/sound/FluidR3Mono_GM.sf3',
  path: join(web, 'public/soundfont/FluidR3Mono_GM.sf3'),
  sha256: '2aacd036d7058d40a371846ef2f5dc5f130d648ab3837fe2626591ba49a71254',
};

const sha = (buf) => createHash('sha256').update(buf).digest('hex');

copyFileSync(join(web, 'node_modules/spessasynth_lib/dist/spessasynth_processor.min.js'), join(web, 'public/spessasynth_processor.min.js'));

if (existsSync(SF.path) && sha(readFileSync(SF.path)) === SF.sha256) {
  console.log('assets: soundfont present');
} else {
  console.log(`assets: downloading ${SF.url}`);
  const res = await fetch(SF.url);
  if (!res.ok) throw new Error(`soundfont download failed: ${res.status} ${res.statusText}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (sha(buf) !== SF.sha256) throw new Error('soundfont checksum mismatch: not writing it');
  mkdirSync(dirname(SF.path), { recursive: true });
  writeFileSync(SF.path, buf);
  console.log(`assets: soundfont saved (${(buf.length / 1e6).toFixed(1)} MB)`);
}
