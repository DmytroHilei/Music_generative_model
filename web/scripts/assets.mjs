// Puts the two runtime files the sound needs into public/ (both gitignored), before `next dev` and `next build`:
// - spessasynth's AudioWorklet processor, copied from node_modules (it must be served as its own file)
// - the MuseScore_General GM soundfont v0.2 as SF3 (S. Christian Collins, MIT; FluidR3Mono with the weakest
//   instruments replaced), 39.9 MB; downloaded once and checked against its SHA-256

import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const SF = {
  url: 'https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3',
  path: join(web, 'public/soundfont/MuseScore_General.sf3'),
  sha256: '5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3',
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
