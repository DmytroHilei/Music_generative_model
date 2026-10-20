"""section_follow prompt sets and summaries (CPU, no model): python tests/eval/test_section_prompts.py  (or pytest tests/)."""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval/compare'))
sys.path.insert(0, str(ROOT / 'tools/sweeps'))

from section_follow import HOLD, PROMPT_SETS, SWITCH, read_rows, summarize  # noqa: E402
from text_follow import target  # noqa: E402

NAMES = sorted(set(HOLD) | {p for a, b, _ in SWITCH.values() for p in (a, b)})


def test_every_prompt_has_texts_and_a_metric():
    for name in NAMES:
        for set_name, texts in PROMPT_SETS.items():
            assert texts.get(name), (set_name, name)
        assert target(name)[0] is not None, name
    for pair, (before, after, metric) in SWITCH.items():
        assert target(after) == (metric, 1), pair  # the 'after' prompt raises the switch metric
        for texts in PROMPT_SETS.values():
            assert texts[before] != texts[after], pair


def rows_for(prompts=None):
    """A few synthetic generation rows: hold none / fast+, switch 'fast' both ways, 2 segments."""
    rows = []
    for part, pool, prompt in [('hold', '-', 'none'), ('hold', '-', 'fast+'), ('switch', 'fast', 'AB'),
                               ('switch', 'fast', 'BA')]:
        for seg in range(2):
            nps = 10.0 + (prompt == 'fast+') * 5 + (prompt == 'AB') * 4 * seg - (prompt == 'BA') * 4 * seg
            r = dict(checkpoint='sec_X', iter=1.0, cfg=2.0, part=part, pool=pool, prompt=prompt, row=0.0,
                     segment=float(seg), nps=nps, ops=nps / 2, runaway=0.0, notes_per_onset=2.0)
            if prompts:
                r['prompts'] = prompts
            rows.append(r)
    return rows


def test_summarize_old_and_new_rows():
    args = argparse.Namespace(seg=256, switch=257)  # segment 1 starts at the switch
    with tempfile.TemporaryDirectory() as d:
        for prompts in (None, 'section'):  # rows.csv from before --prompts has no 'prompts' column
            path = Path(d) / f'rows_{prompts}.csv'
            rows = rows_for(prompts)
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            back = read_rows(path)
            assert all(r['prompts'] == (prompts or 'file') for r in back)
            text = summarize(back, args)
            assert f'Prompt wording: {prompts or "file"}' in text
            assert '| fast+ | nps |' in text and '| fast | nps |' in text


def test_pick_weights_register_down():
    import section_sweep
    assert section_sweep.SWITCH_WEIGHT.get('high', 1.0) < 1.0
    assert set(section_sweep.SWITCH_WEIGHT) <= set(SWITCH)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
