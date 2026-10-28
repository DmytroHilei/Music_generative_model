"""text_eval settings and scores (CPU, no model): python tests/eval/test_text_eval.py  (or pytest tests/)."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval/compare'))

import text_eval as te  # noqa: E402


def test_parse_setting():
    s = te.parse_setting('base', temperature=0.9, top_p=0.9, seed=1234)
    assert (s.name, s.temperature, s.top_p, s.cfg, s.best_of, s.guided) == ('base', 0.9, 0.9, 1.0, 1, False)
    assert s.short() == 'defaults'
    s = te.parse_setting('open64:cfg=3,cfg_notes=64,dry=3,dry_key=pitch_class,top_p=none,neg=pair,best_of=4',
                         temperature=0.9, top_p=0.9, seed=1234)
    assert s.guided and s.cfg == 3 and s.cfg_notes == 64 and math.isnan(s.top_p) and s.dry_key == 'pitch_class'
    assert s.neg == 'pair' and s.best_of == 4 and 'cfg_notes=64' in s.short()
    warm = te.parse_setting('w:cfg=1,cfg_notes=250,cfg_end=3,cfg_linear=1', temperature=0.9, top_p=0.9, seed=1)
    assert warm.guided and warm.cfg_linear
    for bad in ('x:bogus=1', 'x:neg=up', 'x:best_of=0'):
        try:
            te.parse_setting(bad, temperature=0.9, top_p=0.9, seed=1)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_pair_text():
    prompts = dict(te.PROMPTS)
    assert te.pair_text('fast+', prompts) == prompts['fast-']
    assert te.pair_text('uk_loud-', prompts) == prompts['uk_loud+']
    assert te.pair_text('piano+', prompts) is None and te.pair_text('sad', prompts) is None


def _rows(setting, nps_fast=30.0, nps_slow=15.0, drums_none=0.6, drums_nodrums=0.1, runaway=0.0, copies=0.1):
    """Synthetic rows of one setting: 3 prompts x 2 rows with the F metrics and the quality columns."""
    base = {k: np.nan for k in te.METRICS}
    base.update(n_notes=512, instruments=6.0, top_share=0.4, takeover=0.0, se_runaway=0.0, npo=4.0)
    rows = []
    for prompt, nps, has_drums in (('none', 20.0, drums_none), ('fast+', nps_fast, drums_none),
                                   ('fast-', nps_slow, drums_none), ('nodrums+', 20.0, drums_nodrums)):
        for r in range(2):
            rows.append(dict(base, checkpoint='ck', setting=setting, spec='', cfg=1.0, prompt=prompt, text='x',
                             row=r, nps=nps, has_drums=has_drums, runaway=runaway, copies=copies))
    return rows


def test_scores():
    rows = _rows('a') + _rows('b', nps_fast=20.0, nps_slow=20.0, drums_nodrums=0.6) \
        + _rows('c', nps_fast=90.0, runaway=0.5)
    board = {b['setting']: b for b in te.scoreboard(rows)}
    # a: fast pair +15 notes/s = 1.0 x REF; nodrums 0.1 - 0.6 = -0.5 -> sign flipped = 1.0 x REF -> F = 1
    assert abs(board['a']['F'] - 1.0) < 1e-9, board['a']
    assert abs(board['b']['F'] - 0.0) < 1e-9, 'no contrast = 0'
    # c: fast +75 clipped at 1.5, nodrums 1.0 -> F 1.25; runaway 0.5 gates it and raises Q
    assert abs(board['c']['F'] - 1.25) < 1e-9 and board['c']['gated'] and not board['a']['gated']
    assert board['c']['Q'] > board['a']['Q'] and board['a']['J'] > board['b']['J']
    order = [b['setting'] for b in te.scoreboard(rows)]
    assert order[-1] == 'c' and order[0] == 'a', order
    summary = te.summarize(rows)
    assert '## Scoreboard' in summary and '| a |' in summary and 'OUT' in summary
    late = [dict(r, late_nps=r['nps'], late_has_drums=r['has_drums']) for r in _rows('d')]
    b = te.scoreboard(late)[0]
    assert abs(b['F_late'] - 1.0) < 1e-9


def test_quality_reference():
    """D uses the real-song reference without the out-of-band term and is 0 at the reference itself."""
    ref = te.sample_eval._load_reference()
    assert ref, 'musicar/eval/sample_reference.json'
    rows = _rows('a')
    for r in rows:
        r.update(takeover=ref['takeover'], top_share=ref['top_share'], nps=ref['notes_per_s'],
                 instruments=ref['instruments'], se_runaway=ref['runaway'])
    q = te.quality(rows)
    assert abs(q['D']) < 1e-9 and not q['gated'] and abs(q['Q']) < 1e-9
    for r in rows:
        r['copies'] = 0.5
    q = te.quality(rows)
    assert q['gated'] and abs(q['Q'] - te.PENALTY * 0.25) < 1e-9


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
