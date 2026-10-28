"""Model tokens -> MIDI file, and MIDI -> MP3 rendering (fluidsynth + ffmpeg)."""

import shutil
import subprocess
import time
from pathlib import Path

import pretty_midi

from musicar.data.loader import DRUM_PROGRAM


def note_events(pitches, velocities, durations, delta_times, velocity_bins=32, time_resolution=0.02,
                max_delta=None, programs=None, start=0.0):
    """(onset s, program, pitch, duration s, MIDI velocity) of every note, as tokens_to_midi writes them; onsets count
    from `start` (a stream of notes continues from its last onset and gets the same floats as the whole file)."""
    programs = programs if programs is not None else [0] * len(pitches)
    current_time = start
    for p, v, d, dt, prog in zip(pitches, velocities, durations, delta_times, programs):
        delta = dt * time_resolution
        if max_delta is not None:
            delta = min(delta, max_delta)  # cap runaway gaps
        if delta > 0:
            current_time += delta
        duration = 0.1 if prog == DRUM_PROGRAM else max(0.08, d * time_resolution)
        yield current_time, int(prog), int(p), duration, min(127, int(v * 128 / velocity_bins) + 2)


def tokens_to_midi(pitches, velocities, durations, delta_times,
                   velocity_bins=32, time_resolution=0.02,
                   max_polyphony=None, max_delta=None, programs=None):
    """programs (optional, one per note, 128 = drums): one MIDI track per instrument, drums on channel 10 with a
    fixed short length (the model's drum durations are untrained: their loss is masked)."""
    notes = []
    current_time = 0.0
    pending = []  # notes at the current time slot, flushed when time advances

    def flush(pending):
        pending.sort(key=lambda n: n[4], reverse=True)
        notes.extend(pending[:max_polyphony] if max_polyphony else pending)

    for note in note_events(pitches, velocities, durations, delta_times, velocity_bins, time_resolution, max_delta,
                            programs):
        if note[0] != current_time:
            flush(pending)
            pending = []
            current_time = note[0]
        pending.append(note)

    flush(pending)
    return notes_to_midi(notes)


def notes_to_midi(notes):
    """[onset s, program, pitch, duration s, velocity] notes (note_events) -> PrettyMIDI, one track per instrument in
    program order, drums (program 128) on channel 10."""
    tracks = {}
    for onset, prog, pitch, duration, velocity in notes:
        if prog not in tracks:
            drum = prog == DRUM_PROGRAM
            name = 'Drums' if drum else pretty_midi.program_to_instrument_name(prog)
            tracks[prog] = pretty_midi.Instrument(program=0 if drum else prog, is_drum=drum, name=name)
        tracks[prog].notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=onset, end=onset + duration))
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(tracks[k] for k in sorted(tracks))
    return midi


def render_mp3(midi_path: Path, soundfont: str) -> None:
    if not shutil.which("fluidsynth"):
        print("fluidsynth not found — skipping MP3 render (sudo apt install fluidsynth)")
        return
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found — skipping MP3 render (sudo apt install ffmpeg)")
        return
    if not Path(soundfont).exists():
        print(f"Soundfont not found: {soundfont}")
        print("Install with: sudo apt install fluid-soundfont-gm")
        return

    raw_path = midi_path.with_suffix(".raw")
    mp3_path = midi_path.with_suffix(".mp3")
    # bounded render: fluidsynth's fast render once kept going on a 16 s multi-instrument MIDI until it had written
    # 14.5 GB (2026-10-03). Raw 16-bit stereo PCM has no header to break, so the watchdog can stop it at the
    # MIDI's length + 5 s of release tail and ffmpeg encodes exactly that much.
    seconds = pretty_midi.PrettyMIDI(str(midi_path)).get_end_time() + 5.0
    rate = 44100
    limit = int(seconds * rate * 4)  # 2 channels x 2 bytes

    print("Rendering MP3 with fluidsynth...")
    proc = subprocess.Popen(
        ["fluidsynth", "-ni", "-r", str(rate), "-O", "s16", "-T", "raw", "-F", str(raw_path), soundfont,
         str(midi_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if raw_path.exists() and raw_path.stat().st_size >= limit:
            proc.kill()
            break
        time.sleep(0.05)
    proc.wait()
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(rate), "-ac", "2", "-i", str(raw_path), "-t", f"{seconds:.2f}",
         "-q:a", "2", str(mp3_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    raw_path.unlink()
    print(f"Saved MP3  → {mp3_path}")
