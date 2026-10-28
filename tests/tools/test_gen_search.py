"""gen_search's pick, grid and cost logic (CPU, no model): python tests/tools/test_gen_search.py  (or pytest tests/)."""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval/compare'))
sys.path.insert(0, str(ROOT / 'tools/sweeps'))

import gen_search as gs  # noqa: E402
import text_eval as te  # noqa: E402


def _b(name, J, gated=False):
    return dict(setting=name, spec='', rows=16, F=J, F_late=None, D=0.0, runaway=0.0, copies=0.0, Q=0.0, J=J,
                gated=gated)


def test_pick_needs_more_than_the_noise():
    board = [_b('cfg2', 0.60), _b('base', 0.50), _b('base_s2', 0.45), _b('best8', 0.90, gated=True)]
    name, noise, _ = gs.pick(board)
    assert (name, round(noise, 6)) == ('cfg2', 0.05)
    board[0]['J'] = 0.54  # within the noise floor
    assert gs.pick(board)[0] == 'base'
    assert gs.pick([_b('base', 0.5), _b('x', 0.9, gated=True)])[0] == 'base'
    assert gs.pick([_b('x', 0.9), _b('base', None)])[0] == 'x'


def test_stage_a_specs_parse():
    for name, spec in gs.STAGE_A:
        spec = spec.format(store='some_store')
        s = te.parse_setting(f'{name}:{spec}' if spec else name, temperature=0.9, top_p=0.9, seed=1234)
        assert s.name == name
    names = [n for n, _ in gs.STAGE_A]
    assert len(set(names)) == len(names)


def test_sample_settings():
    rng = random.Random(1)
    pool = gs.sample_settings('cfg=3,cfg_notes=64,dry=3,dry_key=pitch_class', 20, rng)
    assert len(pool) == 20 and len({s for _, s in pool}) == 20
    for name, spec in pool:
        s = te.parse_setting(f'{name}:{spec}' if spec else name, temperature=0.9, top_p=0.9, seed=1234)
        if s.cfg == 1.0:
            assert s.cfg_notes == 0 and s.neg == 'none', spec
        if s.dry:
            assert s.dry_key == 'pitch_class'
        if s.min_p:
            assert s.top_p != s.top_p, 'min_p replaces top_p'  # nan
    assert sum('cfg_notes=64' in s for _, s in pool) >= 5, 'the pool centres on the pick'
    base_pool = gs.sample_settings('', 5, random.Random(2), best_of_ok=True, retrieve='st')
    assert all('best_of' not in s and 'retrieve' not in s for _, s in base_pool)


def test_cost_and_toml():
    units = gs.cost_units([('base', ''), ('cfg2', 'cfg=2'), ('b4', 'best_of=4')], 16, 512)
    assert units == 16 + 32 + 64
    assert gs.cost_units([('w', 'cfg=1,cfg_notes=250,cfg_end=3,cfg_linear=1')], 8, 1024) == 32
    toml = gs.toml_snippet('checkpoints/f1_cap', 'cfg=3,cfg_notes=64,dry=3,dry_key=pitch_class,top_p=none,min_p=0.035')
    text = '\n'.join(toml)
    assert 'cfg-scale = 3' in text and 'cfg-notes = 64' in text and 'dry-key = "pitch_class"' in text
    assert 'min-p = 0.035' in text and 'top-p' not in text


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
