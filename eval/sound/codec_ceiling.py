"""
Codec ceiling (sound ladder, level 3 gate; context/claude/future_directions_2026-10-30.md section 3): real song excerpts
and their Demucs stems go through a neural audio codec and back (encode -> tokens -> decode). A renderer that predicts
codec tokens from notes cannot sound better than this round trip, so the check shows which codec, at how many tokens per
second, keeps enough of our audio. Everything stays local: the songs are copyrighted.

    .venv-audio/bin/python eval/sound/codec_ceiling.py prep     # excerpts + Demucs stems -> ref/
    .venv/bin/python eval/sound/codec_ceiling.py codec          # every ref x setting -> dec/<setting>/
    .venv-audio/bin/python eval/sound/codec_ceiling.py score    # metrics -> scores.jsonl
    .venv-audio/bin/python eval/sound/codec_ceiling.py report   # report.md + listen.html

prep  CLIPS: 20 s of each song (mono downmix = part 'mix') plus its htdemucs_6s stems above --stem-gate-db vs the mix,
  and two FluidSynth renders of our own MIDI (FluidR3_GM, what render_mp3 uses). Mono 44.1 kHz FLAC, 24-bit.
codec  SETTINGS: EnCodec 24 kHz at 1.5-24 kbps, EnCodec 32 kHz (MusicGen's), DAC 44.1 / 24 kHz at several codebook
  counts, SNAC 32 / 44 kHz, plus MP3 320 / 128 kbps as familiar anchors. The ref is resampled (soxr) to the codec's
  rate and set to LEVEL_DB RMS (peak capped at 0.99: codecs are trained on normalized audio, quiet stems at -30 to
  -48 dBFS lost up to 5 dB SI-SDR on DAC), the decoded audio is divided by the same gain, trimmed / padded to the input
  length and written at the codec's rate (float WAV, unclipped: EnCodec overshoots +-1). tokens/s = code elements per
  second of real signal (SNAC's internal padding not counted); kbps = tokens/s x log2(codebook size). MP3: LAME's
  1,105-sample delay through a pipe is cut.
score  per (setting, clip, part), on ref and decoded both resampled to 24 kHz and aligned by cross-correlation (+-50 ms):
  multi-resolution log-mel L1 distance on level-matched audio (FFT 512 / 1024 / 2048, 80 mels to 12 kHz, magnitude,
  clamp 1e-5, log10, mean of the 3 scales: DAC-style, but not DAC's 7-scale full-band sum, so not comparable with
  published values), SI-SDR, level change in dB; 'air' at 44.1 kHz: the decoded high-band share of total energy vs the
  ref's, in dB, for 12-16 kHz and above 16 kHz, only where the ref has a share of at least -40 dB there and the codec's
  rate can carry the band. Round-trip notes: basic-pitch on ref and decoded (tonal parts and the mix), note onsets
  matched with mir_eval (50 ms, 50 cents; also with offsets at ratio 0.2); drums: ADTOF hits per class at 50 ms;
  decoded times shifted by the measured lag. The ref's transcription is the target, so F1 says how much of what a
  transcriber hears survives, not accuracy vs the true notes. The MP3 anchors give the metric's noise floor.
report  report.md and listen.html (ref next to every setting, per clip and part). F1 is pooled over notes. Headline
  stems = at least MAIN_STEM_DB vs the mix and, for the renders, an instrument family the MIDI has (Demucs leakage
  out); the other stems are in the per-part tables only.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import time
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'codec_ceiling'
SR_REF = 44100
LEVEL_DB = -20.0
MAIN_STEM_DB = -20.0
AIR_BANDS = ((12000, 16000), (16000, 22050))
SECONDS = 20.0
SOUNDFONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
STEMS = ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other')
TONAL = ('mix', 'vocals', 'bass', 'guitar', 'piano', 'other')

# (name, source glob relative to the repo, excerpt start in s); a song's first sorted match is used
CLIPS = [
    ('okean_911', 'data/audio/Океан Ельзи/*911*.mp3', 60),
    ('kanoe_vesnianka', 'data/audio/Один в каное/*Веснянка*.mp3', 60),
    ('tartak', 'data/audio/Тартак/*.mp3', 60),
    ('hardkiss', 'data/audio/The Hardkiss/*.mp3', 60),
    ('boombox', 'data/audio/Бумбокс/*.mp3', 60),
    ('skryabin_100', 'data/audio/Скрябін/*100 годин*.mp3', 60),
    ('render_sad_piano', 'samples/engine_step2/sad_piano.mid', 0),
    ('render_uk_guitar', 'samples/engine_step2/uk_fast_guitar_wild_0.mid', 0),
]

# (name, kind, repo or encoder, options)
SETTINGS = [
    ('mp3_320', 'mp3', None, {'kbps': 320}),
    ('mp3_128', 'mp3', None, {'kbps': 128}),
    ('encodec24_1.5k', 'encodec', 'facebook/encodec_24khz', {'bandwidth': 1.5}),
    ('encodec24_3k', 'encodec', 'facebook/encodec_24khz', {'bandwidth': 3.0}),
    ('encodec24_6k', 'encodec', 'facebook/encodec_24khz', {'bandwidth': 6.0}),
    ('encodec24_12k', 'encodec', 'facebook/encodec_24khz', {'bandwidth': 12.0}),
    ('encodec24_24k', 'encodec', 'facebook/encodec_24khz', {'bandwidth': 24.0}),
    ('encodec32_4q', 'encodec', 'facebook/encodec_32khz', {'bandwidth': 2.2}),
    ('dac44_4q', 'dac', 'descript/dac_44khz', {'n_q': 4}),
    ('dac44_6q', 'dac', 'descript/dac_44khz', {'n_q': 6}),
    ('dac44_9q', 'dac', 'descript/dac_44khz', {'n_q': 9}),
    ('dac24_8q', 'dac', 'descript/dac_24khz', {'n_q': 8}),
    ('dac24_32q', 'dac', 'descript/dac_24khz', {'n_q': 32}),
    ('snac32', 'snac', 'hubertsiuzdak/snac_32khz', {}),
    ('snac44', 'snac', 'hubertsiuzdak/snac_44khz', {}),
]


# ---- audio I/O through ffmpeg (the training .venv has no soundfile) ----

def read_audio(path, sr):
    """Any audio file -> mono float32 at sr (soxr)."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-af', f'aresample={sr}:resampler=soxr',
                          '-f', 'f32le', '-'], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_wav(path, audio, sr):
    """Mono float32 -> float32 WAV, unclipped (EnCodec overshoots +-1 on loud masters); returns the peak."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.part.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-', '-c:a',
                    'pcm_f32le', str(tmp)], input=audio.astype(np.float32).tobytes(), check=True)
    os.replace(tmp, path)
    return float(np.abs(audio).max()) if audio.size else 0.0


# ---- prep ----

def render_midi(mid, wav):
    """FluidSynth render, bounded by the MIDI's length + 5 s (render_mp3's guard against runaway renders)."""
    import pretty_midi
    seconds = pretty_midi.PrettyMIDI(str(mid)).get_end_time() + 5.0
    proc = subprocess.Popen(['fluidsynth', '-ni', '-r', str(SR_REF), '-F', str(wav), SOUNDFONT, str(mid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=max(60, seconds))
    except subprocess.TimeoutExpired:
        proc.kill()
        raise


def prep(args):
    import soundfile as sf
    import torch
    from demucs.api import Separator
    separator = Separator(model='htdemucs_6s', device='cuda' if torch.cuda.is_available() else 'cpu', shifts=1,
                          split=True, overlap=0.25, progress=False)
    tmp = OUT / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, pattern, start in CLIPS:
        matches = sorted(glob.glob(str(ROOT / pattern)))
        if not matches:
            print(f'{name}: no file for {pattern}, skipped')
            continue
        src = Path(matches[0])
        if src.suffix == '.mid':
            full = tmp / f'{name}_full.wav'
            render_midi(src, full)
            src_audio = full
        else:
            src_audio = src
        excerpt = tmp / f'{name}.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-t', str(SECONDS), '-i', str(src_audio),
                        '-ac', '2', '-af', f'aresample={SR_REF}:resampler=soxr', str(excerpt)], check=True)
        stereo, sr = sf.read(str(excerpt), dtype='float32', always_2d=True)
        assert sr == SR_REF
        parts = {'mix': stereo.mean(axis=1)}
        _, res = separator.separate_tensor(torch.from_numpy(stereo.T.copy()), sr)
        stems = {s: res[s].mean(dim=0).cpu().numpy() for s in STEMS}
        mix_power = np.mean(sum(stems.values()) ** 2) + 1e-12
        levels = {s: float(10 * np.log10(np.mean(a ** 2) / mix_power + 1e-12)) for s, a in stems.items()}
        parts.update({s: a for s, a in stems.items() if levels[s] >= args.stem_gate_db})
        (OUT / 'ref' / name).mkdir(parents=True, exist_ok=True)
        for part, audio in parts.items():
            sf.write(str(OUT / 'ref' / name / f'{part}.flac'), audio, SR_REF, subtype='PCM_24')
        manifest[name] = {'source': str(src.relative_to(ROOT)), 'start': start, 'seconds': len(stereo) / SR_REF,
                          'levels_db': levels, 'parts': list(parts)}
        print(f'{name}: {len(stereo) / SR_REF:.1f} s, parts {", ".join(parts)}')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    shutil.rmtree(tmp)


# ---- codec ----

MP3_DELAY = 1105  # LAME encoder (576) + decoder (529) delay at 44.1 kHz when ffmpeg decodes from a pipe (no Xing tag)


@cache
def load_codec(kind, repo):
    import torch
    if kind == 'encodec':
        from transformers import EncodecModel
        model = EncodecModel.from_pretrained(repo)
        sr, size = model.config.sampling_rate, model.config.codebook_size
    elif kind == 'dac':
        from transformers import DacModel
        model = DacModel.from_pretrained(repo)
        sr, size = model.config.sampling_rate, model.config.codebook_size
    else:
        from snac import SNAC
        model = SNAC.from_pretrained(repo)
        sr, size = model.sampling_rate, model.codebook_size
    return model.eval().to('cuda' if torch.cuda.is_available() else 'cpu'), sr, size


def round_trip(kind, repo, opts, audio, sr):
    """audio (mono float32 at the codec's rate) -> (decoded audio, code elements, codebook size, layout)."""
    import torch
    seconds = len(audio) / sr
    if kind == 'mp3':
        enc = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-',
                              '-c:a', 'libmp3lame', '-b:a', f'{opts["kbps"]}k', '-f', 'mp3', '-'],
                             input=audio.tobytes(), check=True, capture_output=True).stdout
        dec = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'mp3', '-i', '-', '-ac', '1', '-ar', str(sr),
                              '-f', 'f32le', '-'], input=enc, check=True, capture_output=True).stdout
        return np.frombuffer(dec, dtype=np.float32)[MP3_DELAY:], len(enc) * 8, 2, f'{opts["kbps"]} kbps'
    model, _, size = load_codec(kind, repo)
    device = next(model.parameters()).device
    x = torch.from_numpy(audio).to(device)[None, None]
    torch.manual_seed(0)  # SNAC's decoder adds noise
    with torch.inference_mode():
        if kind == 'encodec':
            mask = torch.ones_like(x, dtype=torch.long)[:, 0]
            enc = model.encode(x, mask, bandwidth=opts['bandwidth'])
            codes = enc.audio_codes  # (chunks = 1, batch, codebooks, frames)
            y = model.decode(codes, enc.audio_scales, mask, last_frame_pad_length=enc.last_frame_pad_length)[0]
            n_codes = codes.numel()
            layout = f'{codes.shape[-1] / seconds:.0f} Hz x {codes.shape[-2]}'
        elif kind == 'dac':
            # descript's preprocess pads to a whole hop; HF's encode drops the last partial hop
            x = torch.nn.functional.pad(x, (0, -x.shape[-1] % model.config.hop_length))
            enc = model.encode(x, n_quantizers=opts['n_q'])
            codes = enc.audio_codes  # (batch, codebooks, frames)
            assert codes.shape[1] == opts['n_q'], codes.shape
            y = model.decode(audio_codes=codes).audio_values
            n_codes = codes.numel() * len(audio) / x.shape[-1]
            layout = f'{sr / model.config.hop_length:.0f} Hz x {codes.shape[1]}'
        else:
            codes = model.encode(x)
            y = model.decode(codes)
            # encode pads to hop x lcm(strides, attention window); count the codes of the real signal only
            n_codes = len(audio) / model.hop_length * sum(1 / s for s in model.vq_strides)
            layout = ' + '.join(f'{sr / model.hop_length / s:.0f}' for s in model.vq_strides) + ' Hz'
    y = y.float().reshape(-1).cpu().numpy()
    return y, n_codes, size, layout


def fit_length(y, n):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def level_gain(audio):
    """Gain that puts audio at LEVEL_DB RMS, capped so the peak stays at 0.99."""
    rms_db = 10 * np.log10(np.mean(audio ** 2) + 1e-12)
    return min(10 ** ((LEVEL_DB - rms_db) / 20), 0.99 / max(float(np.abs(audio).max()), 1e-9))


def codec(args):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    info = {}
    for name, kind, repo, opts in SETTINGS:
        if args.only and name not in args.only:
            continue
        sr = SR_REF if kind == 'mp3' else load_codec(kind, repo)[1]
        stats = {'sr': sr, 'codes': 0, 'seconds': 0.0, 'time': 0.0, 'max_peak': 0.0}
        for clip, meta in manifest.items():
            for part in meta['parts']:
                out = OUT / 'dec' / name / clip / f'{part}.wav'
                audio = read_audio(OUT / 'ref' / clip / f'{part}.flac', sr)
                gain = level_gain(audio)
                t0 = time.time()
                y, n_codes, size, layout = round_trip(kind, repo, opts, (audio * gain).astype(np.float32), sr)
                stats['time'] += time.time() - t0
                stats['codes'] += n_codes
                stats['seconds'] += len(audio) / sr
                stats['codebook'], stats['layout'] = size, layout
                stats['max_peak'] = max(stats['max_peak'], write_wav(out, fit_length(y / gain, len(audio)), sr))
        stats['tokens_per_s'] = stats['codes'] / stats['seconds'] if kind != 'mp3' else None
        stats['kbps'] = (stats['codes'] / stats['seconds'] / 1000 if kind == 'mp3'
                         else stats['tokens_per_s'] * math.log2(stats['codebook']) / 1000)
        stats['realtime_x'] = stats['seconds'] / max(stats['time'], 1e-9)
        info[name] = stats
        print(f'{name}: {stats["seconds"]:.0f} s of audio, {layout}, '
              f'{stats["tokens_per_s"] or 0:.0f} tokens/s, {stats["kbps"]:.2f} kbps, {stats["realtime_x"]:.0f}x realtime, '
              f'peak {stats["max_peak"]:.2f}')
    path = OUT / 'codec.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **info}, indent=1))
    scores = OUT / 'scores.jsonl'
    if scores.exists():  # rows of re-encoded settings are stale
        keep = [line for line in scores.read_text().splitlines() if json.loads(line)['setting'] not in info]
        scores.write_text(''.join(line + '\n' for line in keep))


# ---- score ----

def align(ref, dec, max_lag):
    """Shift dec so it lines up with ref (FFT cross-correlation over +-max_lag samples); returns (dec, lag), lag < 0
    when dec is late."""
    n = len(ref) + len(dec)
    corr = np.fft.irfft(np.fft.rfft(ref, n) * np.conj(np.fft.rfft(dec, n)), n)
    lags = np.r_[0:max_lag + 1, -max_lag:0]
    lag = int(lags[np.argmax(corr[lags])])
    shifted = np.roll(dec, lag)
    if lag > 0:
        shifted[:lag] = 0
    elif lag < 0:
        shifted[lag:] = 0
    return shifted, lag


def mel_distance(ref, dec, sr):
    import librosa
    total = 0.0
    for n_fft in (512, 1024, 2048):
        kw = dict(sr=sr, n_fft=n_fft, hop_length=n_fft // 4, n_mels=80, fmax=12000, power=1.0)
        a = np.log10(np.maximum(librosa.feature.melspectrogram(y=ref, **kw), 1e-5))
        b = np.log10(np.maximum(librosa.feature.melspectrogram(y=dec, **kw), 1e-5))
        total += float(np.mean(np.abs(a - b)))
    return total / 3


def si_sdr(ref, dec):
    alpha = np.dot(dec, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum((dec - target) ** 2) + 1e-12)))


def air_db(ref, dec, sr, codec_sr):
    """Per AIR_BANDS: dec's share of its energy in the band vs ref's share, in dB; NaN where the ref has less than
    -40 dB there or the codec's Nyquist is at or below the band's start."""
    freqs = np.fft.rfftfreq(len(ref), 1 / sr)
    spec_ref, spec_dec = np.abs(np.fft.rfft(ref)) ** 2, np.abs(np.fft.rfft(dec)) ** 2
    out = []
    for lo, hi in AIR_BANDS:
        band = (freqs > lo) & (freqs <= hi)
        share_ref = spec_ref[band].sum() / (spec_ref.sum() + 1e-12)
        share_dec = spec_dec[band].sum() / (spec_dec.sum() + 1e-12)
        ok = share_ref >= 1e-4 and codec_sr / 2 > lo
        out.append(float(10 * np.log10((share_dec + 1e-12) / share_ref)) if ok else float('nan'))
    return out


def rms_db(x):
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12))


@cache
def notes_of(path, drums):
    """Transcription of an audio file: (intervals, pitches) arrays; cached per path within one run."""
    import sys
    sys.path.insert(0, str(ROOT / 'data_prep' / 'songs'))
    from audio_to_piano import transcribe, transcribe_drums
    notes = transcribe_drums(path, 'cuda') if drums else transcribe(path, 127.70)
    iv = np.array([[n.start, n.end] for n in notes]).reshape(-1, 2)
    return iv, np.array([n.pitch for n in notes], dtype=float)


def note_f1(ref_path, dec_path, part, shift):
    """Matches of decoded vs ref transcription; decoded times + shift (s). Returns hits / n_ref / n_est and F1."""
    import mir_eval
    drums = part == 'drums'
    ri, rp = notes_of(str(ref_path), drums)
    ei, ep = notes_of(str(dec_path), drums)
    ei = ei + shift
    keep = ei[:, 0] >= 0
    ei, ep = ei[keep], ep[keep]
    row = {'n_ref': len(rp), 'n_est': len(ep)}
    if drums:
        row['hits'] = sum(len(mir_eval.util.match_events(ri[rp == p, 0], ei[ep == p, 0], 0.05))
                          for p in np.unique(np.r_[rp, ep]))
    elif len(rp) and len(ep):
        hz = lambda p: 440.0 * 2 ** ((p - 69) / 12)
        match = lambda off: len(mir_eval.transcription.match_notes(ri, hz(rp), ei, hz(ep), onset_tolerance=0.05,
                                                                   pitch_tolerance=50.0, offset_ratio=off))
        row['hits'], row['hits_off'] = match(None), match(0.2)
    else:
        row['hits'] = row['hits_off'] = 0
    row['f1'] = 2 * row['hits'] / max(row['n_ref'] + row['n_est'], 1)
    return row


def score(args):
    import librosa
    manifest = json.loads((OUT / 'manifest.json').read_text())
    path = OUT / 'scores.jsonl'
    done = set()
    if path.exists():
        done = {(r['setting'], r['clip'], r['part']) for r in map(json.loads, path.read_text().splitlines())}
    with path.open('a') as f:
        for name, *_ in SETTINGS:
            if args.only and name not in args.only:
                continue
            for clip, meta in manifest.items():
                for part in meta['parts']:
                    if (name, clip, part) in done:
                        continue
                    ref_path, dec_path = OUT / 'ref' / clip / f'{part}.flac', OUT / 'dec' / name / clip / f'{part}.wav'
                    if not dec_path.exists():
                        continue
                    ref24 = librosa.load(str(ref_path), sr=24000, res_type='soxr_hq')[0]
                    dec24 = librosa.load(str(dec_path), sr=24000, res_type='soxr_hq')[0]
                    dec24, lag = align(ref24, fit_length(dec24, len(ref24)), 1200)
                    ref44 = librosa.load(str(ref_path), sr=SR_REF, res_type='soxr_hq')[0]
                    dec44 = librosa.load(str(dec_path), sr=SR_REF, res_type='soxr_hq')[0]
                    dec44 = np.roll(fit_length(dec44, len(ref44)), round(lag * SR_REF / 24000))
                    match = np.sqrt((np.mean(ref24 ** 2) + 1e-12) / (np.mean(dec24 ** 2) + 1e-12))
                    air_lo, air_hi = air_db(ref44, dec44, SR_REF, librosa.get_samplerate(str(dec_path)))
                    row = {'setting': name, 'clip': clip, 'part': part, 'lag_ms': lag / 24,
                           'mel': mel_distance(ref24, match * dec24, 24000), 'si_sdr': si_sdr(ref24, dec24),
                           'level_db': rms_db(dec24) - rms_db(ref24), 'air_12_16': air_lo, 'air_16': air_hi}
                    if part in TONAL or part == 'drums':
                        row.update(note_f1(ref_path, dec_path, part, lag / 24000))
                    f.write(json.dumps(row) + '\n')
                    f.flush()
            print(f'{name}: scored')


# ---- report ----

def midi_stems(source):
    """Demucs stems a render can really contain, from its MIDI's programs (GM families)."""
    import pretty_midi
    stems = set()
    for inst in pretty_midi.PrettyMIDI(str(ROOT / source)).instruments:
        p = inst.program
        stems.add('drums' if inst.is_drum else 'piano' if p < 8 else 'guitar' if 24 <= p < 32
                  else 'bass' if 32 <= p < 40 else 'vocals' if 52 <= p < 55 else 'other')
    return stems


def report(args):
    import html
    rows = [json.loads(line) for line in (OUT / 'scores.jsonl').read_text().splitlines()]
    info = json.loads((OUT / 'codec.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    main = {}
    for clip, m in manifest.items():
        real = midi_stems(m['source']) if m['source'].endswith('.mid') else set(STEMS)
        main[clip] = {p for p in m['parts'] if p != 'mix' and p in real and m['levels_db'][p] >= MAIN_STEM_DB}
    nan = float('nan')
    mean = lambda xs: float(np.mean(xs)) if xs else nan
    median = lambda xs: (float(np.median(xs)), len(xs)) if xs else (nan, 0)
    pooled = lambda rs: 2 * sum(r['hits'] for r in rs) / max(sum(r['n_ref'] + r['n_est'] for r in rs), 1) if rs else nan
    fmt = lambda v, f='{:.2f}': '—' if v is None or math.isnan(v) else f.format(v)
    names = [s[0] for s in SETTINGS if s[0] in info]
    lines = ['# Codec ceiling', '',
             f'{len(manifest)} clips x {SECONDS:.0f} s; each part set to {LEVEL_DB:.0f} dBFS RMS before the codec and '
             'restored after. mel = multi-resolution log-mel L1 at 24 kHz on level-matched audio (lower is better), '
             'SI-SDR in dB, level = decoded minus ref RMS. F1 = round-trip notes pooled over notes (basic-pitch; drums '
             f'ADTOF hits). Headline stems = at least {MAIN_STEM_DB:.0f} dB vs the mix and, for renders, an instrument '
             'the MIDI has. air = high-band share vs the ref in dB, median over the clips with content there (n).', '',
             '| setting | layout | tokens/s | kbps | mel mix | mel stems | SI-SDR mix | level dB | F1 mix | F1 tonal stems '
             '| F1 drums | air 12-16k (n) | air >16k (n) | x realtime |',
             '|---' * 14 + '|']
    for n in names:
        r = [x for x in rows if x['setting'] == n]
        mix = [x for x in r if x['part'] == 'mix']
        stems = [x for x in r if x['part'] in main[x['clip']]]
        a_lo = median([x['air_12_16'] for x in mix if not math.isnan(x['air_12_16'])])
        a_hi = median([x['air_16'] for x in mix if not math.isnan(x['air_16'])])
        tps = info[n]['tokens_per_s']
        lines.append(
            f'| {n} | {info[n].get("layout", "")} | {fmt(tps, "{:.0f}")} | {info[n]["kbps"]:.1f} | '
            f'{fmt(mean([x["mel"] for x in mix]), "{:.3f}")} | {fmt(mean([x["mel"] for x in stems]), "{:.3f}")} | '
            f'{fmt(mean([x["si_sdr"] for x in mix]), "{:.1f}")} | {fmt(mean([x["level_db"] for x in r]), "{:+.2f}")} | '
            f'{fmt(pooled(mix))} | {fmt(pooled([x for x in stems if x["part"] in TONAL]))} | '
            f'{fmt(pooled([x for x in stems if x["part"] == "drums"]))} | '
            f'{fmt(a_lo[0], "{:+.1f}")} ({a_lo[1]}) | {fmt(a_hi[0], "{:+.1f}")} ({a_hi[1]}) | '
            f'{info[n]["realtime_x"]:.0f} |')
    lags = [abs(x['lag_ms']) for x in rows]
    lines += ['', f'Alignment: largest |lag| {max(lags):.2f} ms over {len(rows)} rows (0 = no hidden codec delay).']
    parts = [p for p in ('mix',) + STEMS if any(x['part'] == p for x in rows)]
    for title, take, cell in (('mel per part, headline parts', lambda x: x['part'] == 'mix' or x['part'] in main[x['clip']],
                               lambda rs: fmt(mean([x['mel'] for x in rs]), '{:.3f}')),
                              ('F1 per part, headline parts', lambda x: x['part'] == 'mix' or x['part'] in main[x['clip']],
                               lambda rs: fmt(pooled([x for x in rs if 'hits' in x]))),
                              ('F1 per part, quiet / leakage stems (not in the headline)',
                               lambda x: x['part'] != 'mix' and x['part'] not in main[x['clip']],
                               lambda rs: fmt(pooled([x for x in rs if 'hits' in x])))):
        lines += ['', f'## {title}', '', '| setting | ' + ' | '.join(parts) + ' |', '|---' * (len(parts) + 1) + '|']
        for n in names:
            lines.append(f'| {n} | ' + ' | '.join(
                cell([x for x in rows if x['setting'] == n and x['part'] == p and take(x)]) for p in parts) + ' |')
    lines += ['', '## Clips', ''] + [
        f'- {c}: {m["source"]} from {m["start"]} s; headline stems {", ".join(sorted(main[c])) or "none"}; '
        f'other stems {", ".join(p for p in m["parts"] if p != "mix" and p not in main[c]) or "none"}'
        for c, m in manifest.items()]
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')

    cells = lambda clip, part: ''.join(
        f'<td><audio controls preload="none" src="{html.escape(p)}"></audio></td>'
        for p in [f'ref/{clip}/{part}.flac'] + [f'dec/{n}/{clip}/{part}.wav' for n in names])
    body = ''.join(
        f'<h2>{html.escape(clip)}</h2><p>{html.escape(m["source"])} from {m["start"]} s</p><div class="w"><table><tr><th></th>'
        + ''.join(f'<th>{html.escape(h)}</th>' for h in ['ref'] + names) + '</tr>'
        + ''.join(f'<tr><th>{part}{"" if part == "mix" or part in main[clip] else " (quiet)"}</th>{cells(clip, part)}</tr>'
                  for part in m['parts']) + '</table></div>'
        for clip, m in manifest.items())
    (OUT / 'listen.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Codec ceiling</title><style>body{font:14px system-ui;margin:16px}'
        'th{text-align:left;font-weight:600;white-space:nowrap}td audio{width:180px}.w{overflow-x:auto}</style>'
        '<h1>Codec ceiling</h1><p>Each row: the ref, then the same audio through each codec. Numbers in report.md.</p>'
        + body)
    print(OUT / 'report.md')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('prep', 'codec', 'score', 'report'))
    p.add_argument('--only', nargs='*', help='settings to run (codec / score); default all')
    p.add_argument('--stem-gate-db', type=float, default=-30, help='leave out stems quieter than this vs the mix')
    args = p.parse_args()
    {'prep': prep, 'codec': codec, 'score': score, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
