"""musicar/hub.py and musicar/cli.py (CPU, tiny random models, no network): Hub names, device / dtype choice,
from_pretrained + generate on models saved as the Hub repos hold them (model.safetensors + config.json), the
musicar-generate command line. Run: python tests/generation/test_hub.py (or pytest tests/)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import ROOT, run, tiny  # noqa: E402
import pretty_midi  # noqa: E402
import torch  # noqa: E402
import musicar  # noqa: E402
from musicar import hub  # noqa: E402
from musicar.data.loader import BOS_PITCH, DRUM_PROGRAM  # noqa: E402
from musicar.generation.piece import GenerationError, pick_device, pick_dtype  # noqa: E402
from generation.test_generate_prompt import prompt_file  # noqa: E402


def save_hub_format(model, d, iter_num=7):
    """model.safetensors (bf16) + config.json (model_args with dropout, iter_num, dtype), as on the Hub."""
    from safetensors.torch import save_file
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
              str(d / 'model.safetensors'))
    (d / 'config.json').write_text(json.dumps(dict(model_args=vars(model.config), dtype='bfloat16', iter_num=iter_num)))
    return d


def programs(midi):
    return {DRUM_PROGRAM if i.is_drum else i.program for i in midi.instruments if i.notes}


def expect_error(fn, error=GenerationError):
    try:
        fn()
    except error:
        return
    raise AssertionError('no error')


def test_names():
    # short names -> the project's repos, user/repo ids as they are; a local path wins; a missing path that is no Hub
    # name fails before any download
    assert hub.repo_id('musicar-S') == 'Dimitri-206265/musicar-S'
    assert hub.repo_id('musicar-L-caption') == 'Dimitri-206265/musicar-L-caption'
    assert hub.repo_id('someone/my-model.v2') == 'someone/my-model.v2'
    assert hub.repo_id('musicar-XL') is None and hub.repo_id('/a/b/c') is None
    with tempfile.TemporaryDirectory() as d:
        assert hub.resolve(d) == Path(d)
        for missing in ('musicar-XL', str(Path(d) / 'no' / 'model_bf16.pt')):
            expect_error(lambda: hub.resolve(missing), FileNotFoundError)
    # names meant as local paths never reach the Hub (no network here): FileNotFoundError at once
    for name in ('./missing.pt', '../m.pt', '~/nope', 'runs/best.pt', 'a/-b', 'x--y/z', 'tests/no_such_run'):
        assert hub.looks_local(name) or hub.repo_id(name) is None, name
        expect_error(lambda: hub.resolve(name), FileNotFoundError)


def test_device_dtype():
    assert pick_device('auto') == ('cuda' if torch.cuda.is_available() else 'cpu') and pick_device('cpu') == 'cpu'
    assert pick_dtype('cpu') == torch.float32 and pick_dtype('mps') == torch.float32
    assert pick_dtype('cuda') == torch.bfloat16 and pick_dtype('cuda:1') == torch.bfloat16
    assert pick_dtype('cpu', 'bf16') == torch.bfloat16 and pick_dtype('cuda', 'fp32') == torch.float32
    assert pick_dtype('cuda', torch.float16) == torch.float16
    expect_error(lambda: pick_dtype('cpu', 'int8'), ValueError)
    # a GPU without native bf16 (compute capability < 8, e.g. Colab's T4 = 7.5) gets float32
    real = torch.cuda.is_available, torch.cuda.get_device_capability
    try:
        torch.cuda.is_available, torch.cuda.get_device_capability = (lambda: True), (lambda d=None: (7, 5))
        assert pick_dtype('cuda') == torch.float32 and pick_dtype('cuda', 'bf16') == torch.bfloat16
        torch.cuda.get_device_capability = lambda d=None: (8, 0)
        assert pick_dtype('cuda:0') == torch.bfloat16
    finally:
        torch.cuda.is_available, torch.cuda.get_device_capability = real


def test_import_is_light():
    # import musicar loads no torch; the API names resolve on first use
    out = subprocess.run([sys.executable, '-c', "import sys, musicar; print('torch' in sys.modules); "
                          "print(musicar.from_pretrained.__module__, musicar.generate.__module__, musicar.Generator)"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split('\n')
    assert out[0] == 'False' and out[1].startswith('musicar.hub musicar.hub'), out


def test_from_pretrained_and_generate():
    """A multi-instrument model with BOS/EOS in the Hub format: loaded in eval mode with its (bf16) weights, then
    generate(): the same notes for the same seed, num_samples, the instrument choices, presets and options, text only
    for text models."""
    with tempfile.TemporaryDirectory() as d:
        m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2, pos_emb='rope')
        model = musicar.from_pretrained(str(save_hub_format(m, Path(d) / 'repo')), device='cpu')
        assert model.dtype == torch.float32 and model.iter == 7 and not model.model.training
        assert model.config.dropout == 0.0 and model.checkpoint.name == 'model.safetensors'
        w = m.state_dict()
        assert all(torch.equal(v, w[k].to(torch.bfloat16).float()) for k, v in model.model.state_dict().items())

        midi = musicar.generate(model, n_notes=40, seed=1)
        assert isinstance(midi, pretty_midi.PrettyMIDI) and sum(len(i.notes) for i in midi.instruments) > 0
        again = musicar.generate(model, n_notes=40, seed=1)
        assert [(i.program, i.is_drum, [(n.pitch, n.start, n.velocity) for n in i.notes]) for i in midi.instruments] \
            == [(i.program, i.is_drum, [(n.pitch, n.start, n.velocity) for n in i.notes]) for i in again.instruments]
        many = musicar.generate(model, n_notes=40, seed=1, num_samples=2, best_of=2, temperature=1.0)
        assert isinstance(many, list) and len(many) == 2
        assert len(programs(many[0]) | programs(many[1])) > 2, 'an untrained head samples many instruments'
        assert programs(musicar.generate(model, n_notes=40, seed=1, instruments=[0, 33], temperature=1.0)) <= {0, 33}
        assert programs(musicar.generate(model, n_notes=40, seed=1, instruments='piano')) == {0}
        prompt = Path(d) / 'prompt.mid'
        prompt_file(prompt)  # piano, bass, drums
        assert programs(musicar.generate(model, n_notes=40, prompt=prompt, prompt_notes=24, instruments='prompt',
                                         temperature=1.0)) <= {0, 33, DRUM_PROGRAM}

        # presets, then the named arguments, then options (dashes or underscores)
        preset = Path(d) / 'p.toml'
        preset.write_text('temperature = 0.5\ntop-k = 3\nbest-of = 2\n')
        seen, run_ = [], model.run
        model.run = lambda args, **kw: seen.append(args) or run_(args, **kw)
        musicar.generate(model, n_notes=20, preset=str(preset), top_k=5, **{'dry-multiplier': 1.0})
        a = seen[-1]
        assert (a.temperature, a.top_k, a.best_of, a.max_new_tokens, a.dry_multiplier) == (0.5, 5, 2, 20, 1.0)
        # None = the preset's value, else generate()'s own default: 1000 notes, a random seed, prompt notes by the
        # command line's rule (64, 128 with retrieve); a preset's seed is kept
        model.run = lambda args, **kw: seen.append(args) or (_ for _ in ()).throw(StopIteration)
        for kw in (dict(), dict(), dict(preset=str(preset))):
            expect_error(lambda: musicar.generate(model, **kw), StopIteration)
        a, b, c = seen[-3:]
        assert a.max_new_tokens == b.max_new_tokens == 1000 and a.seed != b.seed and a.prompt_notes is None
        long_preset = Path(d) / 'long.toml'
        long_preset.write_text('max-new-tokens = 77\nseed = 5\n')
        expect_error(lambda: musicar.generate(model, preset=str(long_preset)), StopIteration)
        assert (seen[-1].max_new_tokens, seen[-1].seed) == (77, 5)
        expect_error(lambda: musicar.generate(model, n_notes=9, seed=3, preset=str(long_preset)), StopIteration)
        assert (seen[-1].max_new_tokens, seen[-1].seed) == (9, 3)
        model.run = run_
        # a pathlib.Path works as a preset; options generate() has no use for are refused with a hint
        assert isinstance(musicar.generate(model, n_notes=8, seed=1, preset=preset), pretty_midi.PrettyMIDI)
        for key in ('device', 'output', 'no_mp3', 'checkpoint', 'config', 'threads'):
            expect_error(lambda: musicar.generate(model, n_notes=8, **{key: 'x'}))
        # instruments='prompt' on a model with BOS: the BOS row's instrument 0 is not one of the prompt's
        no_piano = Path(d) / 'bass_drums.mid'
        pm = pretty_midi.PrettyMIDI()
        for program, drum, pitch in ((33, False, 40), (0, True, 36)):
            inst = pretty_midi.Instrument(program=program, is_drum=drum)
            inst.notes = [pretty_midi.Note(90, pitch + k % 5, 0.25 * k, 0.25 * k + 0.2) for k in range(32)]
            pm.instruments.append(inst)
        pm.write(str(no_piano))
        got = programs(musicar.generate(model, n_notes=60, prompt=no_piano, prompt_notes=24, instruments='prompt',
                                        temperature=1.0, seed=2))
        assert got <= {33, DRUM_PROGRAM} and got, got
        expect_error(lambda: musicar.generate(model, n_notes=20, dry_multipler=1.0))
        expect_error(lambda: musicar.generate(model, n_notes=20, text='a sad piano ballad'))
        expect_error(lambda: musicar.generate(model, n_notes=20, instruments='prompt'))
        expect_error(lambda: musicar.generate(model, n_notes=20, instruments=[0, 200]))


def test_piano_model():
    # a piano model without BOS (as musicar-S-piano): 'sample' = piano; instrument lists are refused
    with tempfile.TemporaryDirectory() as d:
        model = musicar.from_pretrained(str(save_hub_format(tiny(), Path(d))), device='cpu')
        assert programs(musicar.generate(model, n_notes=30, seed=0, temperature=1.0)) == {0}
        expect_error(lambda: musicar.generate(model, n_notes=30, instruments=[0, 33]))


def test_cli():
    """musicar-generate (musicar.cli.main) with --model on Hub-format directories: the files hold the notes generate()
    returns for the same options and seed, also when generate() stops early because every row has ended (rows of the
    BOS/EOS model forced to end by --length-stop after 0.3 s)."""
    import contextlib
    import io
    from musicar import cli
    with tempfile.TemporaryDirectory() as d:
        for name, kw, extra in (('plain', dict(n_programs=129), {}),
                                ('eos', dict(n_programs=129, pitch_size=BOS_PITCH + 2),
                                 dict(length_seconds=0.3, length_stop=1.0))):
            repo = save_hub_format(tiny(**kw), Path(d) / name)
            cli.main(['--model', str(repo), '--output', str(repo / 'g.mid'), '--num-samples', '2',
                      '--max-new-tokens', '30', '--no-mp3', '--device', 'cpu',
                      *[x for k, v in extra.items() for x in (f"--{k.replace('_', '-')}", str(v))]])
            files = sorted(repo.glob('g_*.mid'))
            assert len(files) == 2
            if name == 'plain':  # no EOS: every row has its 30 notes
                assert all(len(pretty_midi.PrettyMIDI(str(f)).instruments[0].notes) == 30 for f in files)
            model = musicar.from_pretrained(str(repo), device='cpu')
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                midis = musicar.generate(model, n_notes=30, num_samples=2, instruments='piano', seed=42, **extra)
            assert (name == 'eos') == ('every row ended' in log.getvalue()), log.getvalue()
            for k, midi in enumerate(midis):
                midi.write(str(repo / f'api_{k}.mid'))
                assert (repo / f'api_{k}.mid').read_bytes() == files[k].read_bytes(), (name, k)


if __name__ == '__main__':
    run(globals())
