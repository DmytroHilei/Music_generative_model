"""generate.py and its library (musicar/generation/piece.py): prompt rows, caption masks, presets (CPU). Run: python
tests/generation/test_generate_prompt.py  (or pytest tests/)."""

import sys
import tempfile
import types
from pathlib import Path

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run, tiny  # noqa: E402
import torch  # noqa: E402
import generate  # noqa: E402
from musicar.generation import piece  # noqa: E402
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM  # noqa: E402


def prompt_file(path):
    pm = pretty_midi.PrettyMIDI()
    for program, drum, pitch in ((0, False, 60), (33, False, 36), (0, True, 42)):
        inst = pretty_midi.Instrument(program=program, is_drum=drum)
        inst.notes = [pretty_midi.Note(velocity=90, pitch=pitch, start=k * 0.25, end=k * 0.25 + 0.2) for k in range(8)]
        pm.instruments.append(inst)
    pm.write(str(path))


def test_midi_prompt_keeps_instruments():
    # a multi-instrument model reads a --prompt MIDI with every track and its instrument, drums included (as its
    # token stores); a piano model merges the pitched tracks into piano and drops the drums
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'prompt.mid'
        prompt_file(path)
        cfg = types.SimpleNamespace(n_programs=129, velocity_size=32, duration_size=512, delta_time_size=512,
                                    pitch_size=BOS_PITCH + 2)
        args = types.SimpleNamespace(prompt=str(path), prompt_start=0, prompt_notes=24)
        streams, labels, programs = piece.prompt_rows(cfg, args, None, 0, 2, 'cpu')
        assert streams[0].shape == (2, 25) and streams[0][0, 0] == BOS_PITCH and programs[0, 0] == 0
        assert programs[0, 1:4].tolist() == [0, 33, DRUM_PROGRAM]  # (onset, program, pitch) order
        assert streams[0][0, 1:4].tolist() == [60, 36, 42] and 'Drums' in labels[0]
        cfg.n_programs = 0
        streams, _, programs = piece.prompt_rows(cfg, args, None, 0, 2, 'cpu')
        assert streams[0].shape == (2, 17) and DRUM_PROGRAM not in programs.tolist()[0]


def test_caption_instrument_mask():
    """generate.py --text with instrument limits on a multi-instrument checkpoint without text input: 'solo piano'
    samples only piano, 'only piano and drums' only those, 'without drums' never drums; --loose-instruments keeps
    the plain sampler."""
    from musicar.data.loader import tokenize_tracks
    with tempfile.TemporaryDirectory() as d:
        m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2)
        args = {k: v for k, v in vars(m.config).items() if k != 'dropout'}
        torch.save(dict(model=m.state_dict(), model_args=args, iter_num=0), Path(d) / 'ckpt.pt')

        def programs(*flags):
            out = Path(d) / 'g.mid'
            argv = sys.argv
            sys.argv = ['generate.py', '--checkpoint', str(Path(d) / 'ckpt.pt'), '--output', str(out), '--num-samples',
                        '2', '--max-new-tokens', '60', '--no-mp3', '--device', 'cpu', '--seed', '1',
                        '--temperature', '1.0', *flags]
            try:
                generate.main()
            finally:
                sys.argv = argv
            return {p for f in Path(d).glob('g_*.mid') for p in tokenize_tracks(str(f))[1].tolist()}

        assert programs('--text', 'solo piano') <= {0, 1}
        assert programs('--text', 'only piano and drums') <= {0, 1, DRUM_PROGRAM}
        assert DRUM_PROGRAM not in programs('--sample-instruments', '--text', 'a song without drums')
        loose = programs('--sample-instruments', '--text', 'solo piano', '--loose-instruments', '--best-of', '2')
        assert len(loose) > 2, 'an untrained head samples many instruments when nothing limits it'


def test_config_presets():
    # --config: TOML presets (dashes or underscores), later presets over earlier ones, the command line over both;
    # every shipped preset in config/generate/ parses
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / 'a.toml', Path(d) / 'b.toml'
        a.write_text('checkpoint = "ck"\ndry-multiplier = 3\ndry_key = "pitch_class"\nbest-of = 4\n')
        b.write_text('best-of = 2\n')
        args = generate.parse_args(['--config', str(a), '--config', str(b), '--dry-multiplier', '1'])
        assert (args.checkpoint, args.dry_multiplier, args.dry_key, args.best_of) == ('ck', 1.0, 'pitch_class', 2)
        for text in ('dry_multipler = 3\n', 'dry-key = "chords"\n'):
            a.write_text(text)
            try:
                generate.parse_args(['--config', str(a), '--checkpoint', 'ck'])
                raise AssertionError(f'accepted {text!r}')
            except SystemExit:
                pass
    for path in sorted(generate.PRESETS.glob('*.toml')):
        assert generate.parse_args(['--config', path.stem]).checkpoint


if __name__ == '__main__':
    run(globals())
