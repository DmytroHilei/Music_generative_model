"""musicar/generation/piece.py (CPU, tiny model): settings, the note stream. Run: python tests/generation/test_piece.py
(or pytest tests/)."""

import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run, tiny  # noqa: E402
import torch  # noqa: E402
from musicar.data.loader import BOS_PITCH  # noqa: E402
from musicar.generation.piece import GenerationError, Generator, settings  # noqa: E402
from generation.test_generate_prompt import prompt_file  # noqa: E402


def test_settings():
    # parser defaults, then presets, then keyword options (dashes or underscores); unknown names and bad choices raise
    args = settings(['caption'], best_of=2, **{'dry-key': 'notes'})
    assert (args.retrieve, args.best_of, args.dry_key, args.temperature) == ('gigamidi_replay_train', 2, 'notes', 0.85)
    for bad in (dict(dry_multipler=3), dict(dry_key='chords')):
        try:
            settings(**bad)
            raise AssertionError(f'accepted {bad}')
        except GenerationError:
            pass


def test_stream_matches_pieces():
    """on_notes hands over every candidate's notes as its MIDI has them (prompt kept or not, cut at EOS), and
    streaming changes no note."""
    with tempfile.TemporaryDirectory() as d:
        m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2)
        args = {k: v for k, v in vars(m.config).items() if k != 'dropout'}
        torch.save(dict(model=m.state_dict(), model_args=args, iter_num=0), Path(d) / 'ckpt.pt')
        prompt_file(Path(d) / 'prompt.mid')
        gen = Generator(str(Path(d) / 'ckpt.pt'), device='cpu')
        for extra in (dict(), dict(prompt=str(Path(d) / 'prompt.mid'), prompt_notes=20, keep_prompt=True)):
            args = settings(sample_instruments=True, max_new_tokens=70, num_samples=2, best_of=2, batch_size=3,
                            temperature=1.0, seed=3, **extra)
            got = defaultdict(list)
            pieces = gen.run(args, on_notes=lambda c, notes: got[c].extend(notes))
            assert [p.tokens for p in gen.run(args)] == [p.tokens for p in pieces]
            assert sorted(p.index for p in pieces) == [0, 1, 2, 3]
            for p in pieces:
                assert got[p.index] == p.notes() and p.notes(), (extra, p.index)
            if extra:
                assert all(p.tokens[0][:20] == pieces[0].tokens[0][:20] for p in pieces), 'kept prompt first'


if __name__ == '__main__':
    run(globals())
