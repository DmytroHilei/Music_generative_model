'use client';

// One generated piece: the roll filling in as notes arrive, the player, its settings and what to do next.

import { useEffect, useMemo, useRef, useState } from 'react';
import Roll from './Roll';
import { Arrow, Download, Pause, Pencil, Play, Redo } from './Icons';
import { LENGTHS, PRESETS } from '@/lib/app';
import { midiBlob } from '@/lib/midi';
import { endOf, families, fmtTime, FAMILY_LABEL } from '@/lib/notes';
import { fileName, renderMp3, save, type RenderStage } from '@/lib/render';
import { useSoundStatus } from '@/lib/sound';
import { Player } from '@/lib/synth';
import type { Piece } from '@/lib/store';

type Props = { piece: Piece; onVariation: () => void; onEdit: () => void };

const MP3_LABEL: Record<RenderStage, string> = { sound: 'Loading sounds', render: 'Rendering', encode: 'Encoding' };

export default function PieceCard({ piece, onVariation, onEdit }: Props) {
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [mp3, setMp3] = useState<{ stage: RenderStage; p: number } | null>(null);
  const [mp3Error, setMp3Error] = useState('');
  const sound = useSoundStatus();
  const notesRef = useRef(piece.notes);
  notesRef.current = piece.notes;
  const streaming = piece.status === 'queued' || piece.status === 'streaming';
  const end = useMemo(() => endOf(piece.notes), [piece.notes]);
  // while writing, the roll spans the asked length, so notes grow left to right; after, the piece's real length
  const duration = Math.max(streaming ? piece.lengthS : end, 1);
  const endRef = useRef(end);
  endRef.current = streaming ? Infinity : end;

  const player = useMemo(() => new Player(
    () => notesRef.current, () => { setPlaying(false); setT(0); }, () => endRef.current, () => setPlaying(false),
  ), []);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      setT(player.time());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, player]);

  useEffect(() => () => player.stop(), [player]);

  const toggle = () => {
    if (player.playing) {
      player.stop();
      setPlaying(false);
    } else {
      void player.play(!streaming && t >= end - 0.05 ? 0 : t);
      setPlaying(true);
    }
  };

  const seek = (e: React.MouseEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const to = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * duration;
    setT(to);
    if (player.playing) void player.play(to);
    else player.stop(to);
  };

  const downloadMp3 = async () => {
    setMp3Error('');
    setMp3({ stage: 'sound', p: 0 });
    try {
      const blob = await renderMp3(piece.notes, (stage, p) => setMp3({ stage, p }));
      save(blob, fileName(piece.prompt, 'mp3'));
    } catch (err) {
      setMp3Error(`MP3 failed: ${(err as Error).message}`);
    } finally {
      setMp3(null);
    }
  };

  const loadingSound = playing && sound.state === 'loading';

  const fams = families(piece.notes);
  const preset = PRESETS.find((p) => p.id === piece.preset);
  const length = LENGTHS.find((l) => l.s === piece.lengthS)?.label ?? `${piece.lengthS} s`;
  const status =
    piece.status === 'queued' ? (piece.position ? `waiting · ${piece.position} ahead` : 'starting the engine…')
      : piece.status === 'streaming' ? `writing · ${piece.notes.length} notes`
        : piece.status === 'done' ? `done${piece.elapsedS != null ? ` in ${piece.elapsedS.toFixed(1)} s` : ''}`
          : 'stopped';
  const settings = [piece.model, piece.seed != null && `seed ${piece.seed}`, preset?.label,
    piece.opening && 'real opening', length, `${piece.notes.length} notes`]
    .filter(Boolean).join(' · ');

  return (
    <div className="turn">
      <div className="who">
        <span><b>musicar</b></span>
        <span>{status}</span>
      </div>
      <section className="piece" aria-label="Generated piece" aria-busy={streaming}>
        <div className="stream-bar" style={{ width: `${Math.min(100, (end / piece.lengthS) * 100)}%`, opacity: streaming ? 1 : 0 }} />
        <div className="piece-head">
          <div className="legend">
            {fams.map((f) => <span key={f}><i style={{ background: `var(--${f})` }} />{FAMILY_LABEL[f]}</span>)}
          </div>
          <span className="mono muted" style={{ fontSize: 12 }}>{fmtTime(end)}</span>
        </div>
        <div className="roll">
          <Roll notes={piece.notes} duration={duration} t={t} lit={playing || t > 0} />
          {piece.notes.length === 0 && piece.status !== 'error' && <div className="roll-wait">{status}</div>}
        </div>
        <div className="transport">
          <button type="button" className="play" onClick={toggle} disabled={piece.notes.length === 0} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? <Pause /> : <Play />}
          </button>
          <span className="time mono">
            {loadingSound ? `instruments ${Math.round(sound.progress * 100)}%` : `${fmtTime(t)} / ${fmtTime(end)}`}
          </span>
          <button type="button" className="track" onClick={seek} aria-label="Seek">
            <div style={{ width: `${end ? Math.min(100, (t / end) * 100) : 0}%` }} />
          </button>
        </div>
        {piece.error && <div className="piece-error" role="alert">{piece.error}</div>}
        {mp3Error && <div className="piece-error" role="alert">{mp3Error}</div>}
        {sound.state === 'failed' && playing && <div className="piece-error">The SoundFont did not load; playing with the simple synth.</div>}
        <div className="settings-line mono">{settings}</div>
        <div className="actions">
          <button type="button" className="act" onClick={onVariation} disabled={streaming}><Redo />Variation</button>
          <button type="button" className="act primary" disabled title="Needs section captions (F1); coming after the first version">
            Continue with the next section<Arrow />
          </button>
          <button type="button" className="act" onClick={onEdit}><Pencil />Edit prompt</button>
          <div className="grow" />
          <button type="button" className="act quiet" disabled={piece.notes.length === 0}
            onClick={() => save(midiBlob(piece.notes), fileName(piece.prompt, 'mid'))}><Download />MIDI</button>
          <button type="button" className="act quiet" disabled={streaming || piece.notes.length === 0 || mp3 !== null}
            onClick={() => void downloadMp3()} aria-live="polite">
            <Download />
            {mp3 ? `${MP3_LABEL[mp3.stage]}${mp3.stage === 'render' ? '…' : ` ${Math.round(mp3.p * 100)}%`}` : 'MP3'}
          </button>
        </div>
      </section>
    </div>
  );
}
