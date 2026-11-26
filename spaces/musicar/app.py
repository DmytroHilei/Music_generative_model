"""musicar demo Space on ZeroGPU: text to MIDI with musicar-L-caption (English or Ukrainian), and MIDI from the start
or continuing an uploaded file with musicar-L, musicar-S and musicar-S-piano.

Models: https://huggingface.co/Dimitri-206265 - code: https://github.com/DmytroHilei/Music_generative_model

On ZeroGPU every model is placed on `cuda` at startup (CUDA is emulated outside @spaces.GPU functions) and each
request's generation runs in run_model, for which ZeroGPU lends a GPU: a forked worker process that receives the
arguments and returns its results pickled, so run_model takes plain values and returns the MIDI file's bytes. Errors
it raises end the worker, so it returns generate()'s option errors instead. Audio and the piano roll are made on the
CPU. Run locally: pip install gradio==6.31.0 -r requirements.txt, then python app.py (a CUDA GPU; on the CPU ask for
fewer notes; audio needs fluidsynth and fluid-soundfont-gm).
"""

import io
import math
import os
import random
import shutil
import subprocess
import tempfile
import time

try:
    import spaces  # ZeroGPU; imported before torch, whose CUDA calls it patches
except ImportError:  # a local run without it: run_model is a plain function
    spaces = None

import gradio as gr
import numpy as np
import pretty_midi
import torch
from matplotlib.figure import Figure

import musicar
from musicar.generation import piece

ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
DEVICE = "cuda" if ZERO_GPU or torch.cuda.is_available() else "cpu"
try:  # ZeroGPU multiplies a request's duration by its GPU's factor (1.5 on the RTX Pro 6000) for the quota and limit
    from spaces.zero.config import get_config
    DURATION_FACTOR = float(get_config()["duration_factor"]) if ZERO_GPU else 1.0
except Exception:
    DURATION_FACTOR = 1.0
CUDA_GRAPH = os.environ.get("MUSICAR_CUDA_GRAPH", "1") != "0"  # decode by replaying a captured CUDA graph
SOUNDFONT = os.environ.get("SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2")  # packages.txt: fluid-soundfont-gm
RATE = 32000  # audio sample rate
COLAB = ("https://colab.research.google.com/github/DmytroHilei/Music_generative_model/blob/transformer-v2/"
         "notebooks/musicar_colab.ipynb")

# cap = most notes per request; notes / temperature = the controls' start values; rate = notes per second on the
# Space's GPU (bf16), about 75% of the speed measured there at 2,048 notes (caption 96 at 1,173 notes, L 156, S 326,
# S-piano 717), for the GPU time a request asks ZeroGPU for (gpu_seconds)
MODELS = {
    "musicar-L-caption": dict(cap=2048, notes=2048, temperature=0.9, piano=False, text=True, rate=70),
    "musicar-L": dict(cap=2048, notes=1024, temperature=0.9, piano=False, text=False, rate=120),
    "musicar-S": dict(cap=2048, notes=1024, temperature=0.85, piano=False, text=False, rate=250),
    "musicar-S-piano": dict(cap=2048, notes=1024, temperature=0.9, piano=True, text=False, rate=500),
}
NOTES_PER_SECOND = 80  # the caption model's note budget per second of music asked for (the densest example wrote 82)
START_S, MAX_GPU_S = 8, 60  # a request's GPU start-up (worker, weights, CUDA graph) and the most it asks for
# the caption model follows pace, loudness, register, sad / happy and piano / strings / guitar / drums (model card)
EXAMPLES = ["a calm, slow and quiet piano piece",
            "a fast, loud piece with electric guitar and drums",
            "a sad, slow piece for strings",
            "a happy, fast, high melody without drums",
            "сумна повільна балада для фортепіано",
            "швидка гучна музика з гітарою та барабанами"]
INSTRUMENTS = [("the model chooses", "sample"), ("piano only", "piano"), ("the uploaded MIDI's", "prompt"),
               ("this list", "list")]
OUT = os.path.join(tempfile.gettempdir(), "musicar_space")  # one folder per request, removed after 6 h
KEEP_S = 6 * 3600
SEED_MAX = 2**31 - 1


def builtin_start(path):
    """A plain 4-bar band start for musicar-S, which has no start-of-piece token and gets very dense when started from
    nothing: I-V-vi-IV in C at 110 bpm, piano half-note chords, eighth-note bass roots, kick / snare / hi-hat."""
    beat = 60 / 110
    pm = pretty_midi.PrettyMIDI(initial_tempo=110)
    piano, bass = pretty_midi.Instrument(program=0), pretty_midi.Instrument(program=33)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for bar, (root, chord) in enumerate([(36, (60, 64, 67)), (43, (59, 62, 67)), (45, (57, 60, 64)),
                                         (41, (57, 60, 65))]):
        t = bar * 4 * beat
        for half in (0, 2):
            piano.notes += [pretty_midi.Note(72, p, t + half * beat, t + (half + 1.9) * beat) for p in chord]
        bass.notes += [pretty_midi.Note(84, root, t + k * beat / 2, t + (k + 0.9) * beat / 2) for k in range(8)]
        for k in range(8):
            onset = t + k * beat / 2
            drums.notes.append(pretty_midi.Note(60, 42, onset, onset + 0.1))
            if k % 2 == 0:
                drums.notes.append(pretty_midi.Note(96, 36 if k % 4 == 0 else 38, onset, onset + 0.1))
    pm.instruments += [piano, bass, drums]
    pm.write(path)
    return path, sum(len(i.notes) for i in pm.instruments)


START, START_NOTES = builtin_start(os.path.join(tempfile.gettempdir(), "musicar_space_start.mid"))

if DEVICE == "cpu":
    torch.set_num_threads(int(os.environ.get("MUSICAR_THREADS", os.cpu_count() or 2)))
# all models on DEVICE at startup (on ZeroGPU emulated until a request), bf16 on ZeroGPU's Blackwell GPUs
LOADED = {name: musicar.from_pretrained(name, device=DEVICE, dtype="bf16" if ZERO_GPU else "auto") for name in MODELS}
piece.text_encoder(DEVICE)  # LaBSE for the captions, loaded once here (cached per device; the GPU worker inherits it)


def sweep():
    """Remove the request folders older than KEEP_S."""
    if not os.path.isdir(OUT):
        return
    for d in os.listdir(OUT):
        path = os.path.join(OUT, d)
        if time.time() - os.path.getmtime(path) > KEEP_S:
            shutil.rmtree(path, ignore_errors=True)


def instruments_arg(name, mode, programs, upload):
    """generate()'s instruments= for the chosen mode; upload = the uploaded MIDI (pretty_midi) or None."""
    if MODELS[name]["piano"] or mode == "piano":
        return "piano"
    if mode == "prompt":  # every instrument of the file, not only those of its first notes
        if upload is None:
            return "sample"  # nothing uploaded: the model chooses
        return sorted({128 if i.is_drum else i.program for i in upload.instruments if i.notes})
    if mode == "list":
        try:
            values = [int(v) for v in programs.replace(",", " ").split()]
        except ValueError:
            values = []
        if not values or not all(0 <= v <= 128 for v in values):
            raise gr.Error("Instrument list: General MIDI programs 0-127 and 128 for drums, e.g. 0, 33, 25, 128")
        return values
    return "sample"


def on_gpu(duration):
    """spaces.GPU(duration=duration): a GPU from ZeroGPU for each call (a plain function elsewhere)."""
    return spaces.GPU(duration=duration) if spaces is not None else (lambda fn: fn)


def gpu_seconds(name, n_notes, *rest):
    """The GPU time a request asks ZeroGPU for: start-up plus its notes at the model's rate, with a margin, divided by
    ZeroGPU's factor, which it multiplies back. ZeroGPU refuses a request when the visitor's daily quota left is below
    that amount, stops it there, and charges the time actually used."""
    return math.ceil(min(MAX_GPU_S, START_S + 1.25 * n_notes / MODELS[name]["rate"]) / DURATION_FACTOR)


@on_gpu(gpu_seconds)
def run_model(name, n_notes, text, temperature, seed, instruments, prompt, prompt_notes, keep_prompt, length):
    """One generate() call on the GPU: (the MIDI file's bytes or None, generate()'s error or None, seconds)."""
    progress = gr.Progress()

    def report(done, total, detail):
        if done % 32 == 0:
            try:
                progress(done / max(total, 1), desc=f"{done:,} / {total:,} notes")
            except Exception:  # the progress bar is cosmetic
                pass

    options = dict(n_notes=n_notes, temperature=temperature, seed=seed, instruments=instruments, prompt=prompt,
                   prompt_notes=prompt_notes, keep_prompt=keep_prompt, no_cuda_graph=not CUDA_GRAPH, progress=report)
    if MODELS[name]["text"]:  # the f1 preset: guidance 2.5, DRY on pitch-class chords, the length input
        # + top-p 0.9 (and temperature 0.9 by default), the model card's caption evaluation settings: at the preset's
        # T 1.1 without top-p a piece spread over 5-19 instruments, often odd ones (6 seeds x 3 captions, 2026-11-26).
        # The length counts from the prompt's start when the prompt is kept, else from its end
        options.update(text=text, preset="f1", top_p=0.9, length_seconds=length, length_after_prompt=not keep_prompt)
        if instruments == "piano":  # the explicit choice over a caption's own mask ("solo violin", "without drums")
            options.update(loose_instruments=True)
    t0 = time.time()
    try:
        midi = musicar.generate(LOADED[name], **options)
    except ValueError as e:  # GenerationError: an option this model or this MIDI can't follow
        return None, str(e), 0.0
    took = time.time() - t0
    buf = io.BytesIO()
    midi.write(buf)
    return buf.getvalue(), None, took


def render(midi_path, seconds):
    """The MIDI as (rate, int16 stereo samples) through fluidsynth, or None without fluidsynth or the soundfont. The
    render stops at the end of the music + 3 s: fluidsynth's fast render has been seen to run far past a file's end."""
    if not shutil.which("fluidsynth") or not os.path.exists(SOUNDFONT):
        return None
    floats = int((seconds + 3) * RATE) * 2  # stereo
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "audio.raw")
        proc = subprocess.Popen(["fluidsynth", "-niq", "-r", str(RATE), "-T", "raw", "-O", "float", "-F", raw,
                                 SOUNDFONT, midi_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while proc.poll() is None:
            if (os.path.exists(raw) and os.path.getsize(raw) >= 4 * floats) or time.time() > deadline:
                proc.kill()
            time.sleep(0.05)
        audio = np.fromfile(raw, dtype="<f4")[:floats] if os.path.exists(raw) else np.zeros(0, "<f4")
    if audio.size < 2:
        return None
    audio = audio[:audio.size // 2 * 2].reshape(-1, 2)
    peak = float(np.abs(audio).max(initial=0.0))
    if peak < 1e-4:  # no notes: fluidsynth's noise floor, which normalising would turn into a loud thump
        return None
    audio = audio * (0.89 / peak)  # peaks at -1 dBFS
    return RATE, (audio * 32767).astype(np.int16)


# piano roll colours: the three busiest instruments, every other instrument in grey, drums in a lane below
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES, OTHER = ("#2a78d6", "#eb6834", "#1baf7a"), "#b4b2aa"


def piano_roll(midi):
    pitched = sorted((i for i in midi.instruments if i.notes and not i.is_drum), key=lambda i: -len(i.notes))
    hits = [n.start for i in midi.instruments if i.is_drum for n in i.notes]
    fig = Figure(figsize=(9, 4.2 if hits else 3.6), dpi=110, facecolor=SURFACE, layout="constrained")
    if hits:
        ax, lane = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[6, 1]))
    else:
        ax, lane = fig.subplots(), None
    groups = [(SERIES[k], pretty_midi.program_to_instrument_name(i.program), i.notes)
              for k, i in enumerate(pitched[:3])]
    rest = [n for i in pitched[3:] for n in i.notes]
    if rest:
        groups.append((OTHER, f"{len(pitched) - 3} more instruments", rest))
    for color, label, notes in reversed(groups):  # the busiest instrument is drawn last, on top
        ax.barh([n.pitch for n in notes], [n.end - n.start for n in notes], left=[n.start for n in notes],
                height=0.8, color=color, edgecolor=SURFACE, linewidth=0.3, label=f"{label} ({len(notes):,})")
    for a in (ax, lane) if lane else (ax,):
        a.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(AXIS)
        a.tick_params(colors=MUTED, labelsize=8, length=0)
    if groups:
        lo = min(n.pitch for _, _, notes in groups for n in notes) - 2
        hi = max(n.pitch for _, _, notes in groups for n in notes) + 2
        ticks = [p for p in range(0, 128, 12) if lo <= p <= hi] or [lo + 2, hi - 2]
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks, [pretty_midi.note_number_to_name(p) for p in ticks])
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) == 1:  # one instrument: a label, no legend
            ax.set_title(labels[0], loc="left", color=INK, fontsize=8)
        else:
            ax.legend(handles[::-1], labels[::-1], loc="lower left", bbox_to_anchor=(0, 1), ncol=4, frameon=False,
                      fontsize=8, labelcolor=INK, handlelength=1, columnspacing=1.2, borderaxespad=0.2)
    else:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "no pitched notes", transform=ax.transAxes, ha="center", color=MUTED)
    if lane:
        lane.vlines(hits, 0.15, 0.85, color=INK, linewidth=0.6)
        lane.set_ylim(0, 1)
        lane.set_yticks([0.5], [f"drums ({len(hits):,})"])
    (lane or ax).set_xlabel("seconds", color=MUTED, fontsize=8)
    ax.set_xlim(0, max(midi.get_end_time(), 1.0))
    return fig


def generate(name, text, length, n_notes, temperature, seed, mode, programs, prompt, prompt_notes, keep_prompt,
             progress=gr.Progress()):
    spec = MODELS[name]
    if spec["text"]:
        text = (text or "").strip()
        if not text:
            raise gr.Error(f"Describe the music first, e.g. '{EXAMPLES[0]}'")
        length = float(length)
        n_notes = min(spec["cap"], int(length * NOTES_PER_SECOND) + 64)
    else:
        text, length = None, None
        n_notes = int(min(max(n_notes or 16, 16), spec["cap"]))
    if seed and not 0 <= seed <= SEED_MAX:
        raise gr.Error(f"Seed: 0 (random) to {SEED_MAX:,}")
    seed = int(seed) if seed else random.randrange(1, SEED_MAX)  # 0 = random, shown in the status
    upload = None
    if prompt:
        try:
            upload = pretty_midi.PrettyMIDI(prompt)
        except Exception as e:
            raise gr.Error(f"Can't read the uploaded file as MIDI ({e})")
        if not any(i.notes for i in upload.instruments):
            raise gr.Error("The uploaded MIDI file has no notes")
    instruments = instruments_arg(name, mode, programs or "", upload)
    keep = bool(prompt and keep_prompt)
    prompt_notes = int(prompt_notes)
    if spec["text"] and keep:  # the length counts from the prompt's first note: leave room for new notes
        onsets = sorted(note.start for i in upload.instruments for note in i.notes)
        heard = onsets[min(prompt_notes, len(onsets)) - 1] - onsets[0]
        if heard > 0.8 * length:
            raise gr.Error(f"The first {min(prompt_notes, len(onsets))} notes of the file already last {heard:.0f} s "
                           f"of the {length:.0f} s asked for: raise the Length, lower the Prompt notes, or untick "
                           "'Start the result with the prompt'")
    started = name == "musicar-S" and not prompt  # no start token: continue the built-in start, left out
    if started:
        prompt, prompt_notes = START, START_NOTES
    progress(0, desc="Waiting for a GPU")
    t0 = time.time()
    data, error, took = run_model(name, n_notes, text, float(temperature), seed, instruments, prompt or None,
                                  prompt_notes, keep, length)
    if error:
        raise gr.Error(error)
    start = time.time() - t0 - took  # GPU queue, worker start and weight transfer
    sweep()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(tempfile.mkdtemp(prefix="req_", dir=OUT), f"{name}_seed{seed}.mid")
    with open(path, "wb") as f:
        f.write(data)
    midi = pretty_midi.PrettyMIDI(path)
    notes = sum(len(i.notes) for i in midi.instruments)
    seconds = midi.get_end_time()
    names = sorted({"Drums" if i.is_drum else pretty_midi.program_to_instrument_name(i.program)
                    for i in midi.instruments if i.notes})
    status = (f"**{notes:,} notes, {seconds:.1f} s of music**" + (", the prompt included" if keep else "")
              + f". Generated in {took:.1f} s" + ("" if keep else f" ({notes / max(took, 1e-6):.0f} notes/s)")
              + (f" after {start:.0f} s getting a GPU" if start >= 1 else "") + f"; seed {seed}.  \n"
              + f"{len(names)} instrument{'s' if len(names) != 1 else ''}: {', '.join(names)}"
              + ("  \nmusicar-S has no start token, so it continued a built-in 4-bar start (left out of the result)"
                 if started else ""))
    return (render(path, seconds) if notes else None), path, piano_roll(midi), status


def model_changed(name):
    spec = MODELS[name]
    return (gr.Column(visible=spec["text"]), gr.Slider(visible=not spec["text"], maximum=spec["cap"],
                                                       value=spec["notes"]),
            gr.Slider(value=spec["temperature"]),
            gr.Radio(value="piano" if spec["piano"] else "sample", interactive=not spec["piano"]))


with gr.Blocks(title="musicar", delete_cache=(3600, KEEP_S)) as demo:  # Gradio's copies: hourly, older than 6 h
    gr.Markdown(
        "# musicar\n"
        "MIDI from the [musicar](https://github.com/DmytroHilei/Music_generative_model) models, written one note at "
        "a time. [musicar-L-caption](https://huggingface.co/Dimitri-206265/musicar-L-caption) writes from a "
        "description in English or Ukrainian (477M). [musicar-L](https://huggingface.co/Dimitri-206265/musicar-L) (464M), "
        "[musicar-S](https://huggingface.co/Dimitri-206265/musicar-S) (108M, 128 General MIDI instruments and drums) "
        "and [musicar-S-piano](https://huggingface.co/Dimitri-206265/musicar-S-piano) (107M, solo piano) start on "
        "their own or continue an uploaded MIDI file, which usually sounds best. From the start musicar-L mostly "
        "stays with one instrument, and musicar-S, which has no start token, continues a built-in 4-bar start.  \n"
        "Each request borrows a GPU for a few seconds (ZeroGPU). Hugging Face gives every visitor a daily GPU "
        "allowance: about 2 minutes without an account, 5 when logged in. For longer sessions use the "
        f"[Colab notebook]({COLAB}). The models are CC BY-NC-SA 4.0 (non-commercial use).")
    with gr.Row():
        with gr.Column():
            name = gr.Dropdown(list(MODELS), value="musicar-L-caption", label="Model")
            with gr.Column(visible=True) as caption_box:
                text = gr.Textbox(EXAMPLES[0], label="Description", lines=2,
                                  info="English or Ukrainian. It follows pace, loudness, register, sad / happy and "
                                       "piano, strings, guitar, drums; genre and key only weakly")
                gr.Examples([[e] for e in EXAMPLES], inputs=text, label="Examples")
                length = gr.Slider(10, 60, value=30, step=5, label="Length (seconds)",
                                   info="The model's length input: the music ends near it (at most 2,048 notes). A "
                                        "kept prompt counts toward it")
            n_notes = gr.Slider(16, MODELS["musicar-L"]["cap"], value=1024, step=16, label="Notes to generate",
                                visible=False,
                                info="One note per step. Band music has about 20-40 notes per second, piano about 10")
            temperature = gr.Slider(0.5, 1.3, value=MODELS["musicar-L-caption"]["temperature"], step=0.05,
                                    label="Temperature", info="Lower = safer and more repetitive, higher = more varied")
            seed = gr.Number(0, precision=0, minimum=0, maximum=SEED_MAX, label="Seed",
                             info="0 = random. Each result shows its seed: enter it here to get the same piece again")
            mode = gr.Radio(INSTRUMENTS, value="sample", label="Instruments")
            programs = gr.Textbox("0, 33, 25, 128", label="Instrument list",
                                  info="General MIDI programs (0 piano, 25 guitar, 33 bass, 48 strings; 128 = drums)")
            prompt = gr.File(label="MIDI to continue (optional)", file_types=[".mid", ".midi"], type="filepath")
            prompt_notes = gr.Slider(16, 512, value=64, step=16, label="Prompt notes",
                                     info="How many notes from the start of the uploaded file the model hears")
            keep_prompt = gr.Checkbox(True, label="Start the result with the prompt")
            button = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio = gr.Audio(label="Audio (FluidR3 GM soundfont)", type="numpy")
            midi_file = gr.File(label="MIDI file")
            roll = gr.Plot(label="Piano roll")
            status = gr.Markdown()
    name.change(model_changed, name, [caption_box, n_notes, temperature, mode])
    button.click(generate, [name, text, length, n_notes, temperature, seed, mode, programs, prompt, prompt_notes,
                            keep_prompt], [audio, midi_file, roll, status])

# ZeroGPU runs each request in its own worker; locally two requests would share the models and the CUDA graph capture
demo.queue(default_concurrency_limit=2 if ZERO_GPU else 1, max_size=20)

if __name__ == "__main__":
    demo.launch()
