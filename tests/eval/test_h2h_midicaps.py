"""h2h_midicaps scores and selection (CPU, no model, no store): python tests/run.py h2h_midicaps  (or pytest tests/)."""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval/compare'))

import h2h_midicaps as m  # noqa: E402


def test_tempo_bins_match_midicaps_words():
    assert m.tempo_bin(40) == 0 and m.tempo_bin(41) == 1  # Grave <= 40, Largo 41-60
    assert m.tempo_bin(60) == 1 and m.tempo_bin(61) == 2  # Adagio 61-70
    assert m.tempo_bin(140) == 5 and m.tempo_bin(141) == 6  # Allegro 111-140, Vivace 141-160
    assert m.tempo_bin(210) == 7 and m.tempo_bin(211) == 8  # Presto 161-210, Prestissimo above


def test_key_index_and_relative():
    assert m.key_index('C major') == 0 and m.key_index('A minor') == 21
    assert m.key_index('Bb major') == m.key_index('A# major') == 10
    assert m.relative(m.key_index('C major')) == m.key_index('A minor')
    assert m.relative(m.key_index('A minor')) == m.key_index('C major')
    assert all(m.relative(m.relative(k)) == k for k in range(24))


def test_instrument_scores():
    assert m.instrument_scores([0, 40, 128], [0, 41]) == (1 / 3, 2 / 3, 1 / 4)  # 41 is 40's family, drums are missing
    assert m.instrument_scores([128], [128]) == (1, 1, 1)
    assert all(np.isnan(m.instrument_scores([], [1])))
    assert m.instrument_scores([0], []) == (0, 0, 0)


def test_first_seconds():
    tok = np.zeros((5, 4), np.int64)
    tok[:, 3] = 1000  # 20 s apart: onsets 20, 40, 60, ...
    assert len(m.first_seconds(tok, np.zeros(5, np.int64))[0]) == 2


def test_eligible_md5s():
    rows = dict.fromkeys(['train', 'outside', 'replay', 'exact', 'loose', 'clean'])
    hashes = dict(train=(0, 1, 1), replay=(2, 2, 2), exact=(1, 7, 3), loose=(1, 4, 8), clean=(2, 5, 5))
    keep, counts = m.eligible_md5s(rows, hashes, {7}, {8}, {'replay'})
    assert keep == ['clean']
    assert counts == dict(test_rows=6, outside_gigamidi=1, train_split=1, replay=1, same_notes=2, eligible=1)


def piece(bpm, seconds=30):
    """A pulse of C major notes at bpm, 4 pitches, one piano."""
    step = int(60 / bpm / m.DT)
    n = int(seconds / (step * m.DT))
    pitch = np.array([60, 64, 67, 72] * (n // 4 + 1))[:n]
    tok = np.stack([pitch, np.full(n, 80), np.full(n, step), np.full(n, step)], 1).astype(np.int64)
    return tok, np.zeros(n, np.int64)


def test_file_row_on_a_pulse():
    target = dict(tempo=120, key=m.key_index('C major'), programs=[0], caption='a piano piece')
    r = m.file_row(*piece(120), target)
    assert r['tb'] == 1 and r['tbt'] == 1 and r['ck'] == 1 and r['ckd'] == 1
    assert r['recall'] == 1 and r['jaccard'] == 1
    far = m.file_row(*piece(120), dict(target, tempo=45, key=m.key_index('F# major'), programs=[40]))
    assert far['tb'] == 0 and far['tbt'] == 0 and far['ck'] == 0 and far['recall'] == 0


def test_relative_key_counts_only_in_ckd():
    target = dict(tempo=120, key=m.key_index('A minor'), programs=[0], caption='a piano piece')
    r = m.file_row(*piece(120), target)
    assert r['ck'] == 0 and r['ckd'] == 1


def test_program_names_and_asked_programs():
    rows = [dict(instrument_numbers_sorted=[0, 24], instrument_summary=['Piano', 'Acoustic Guitar']),
            dict(instrument_numbers_sorted=[4, 25, 22], instrument_summary=['Piano', 'Acoustic Guitar', 'Harmonica']),
            dict(instrument_numbers_sorted=[0, 0, 1], instrument_summary=['Piano'])]  # lists differ: not used
    names = m.program_names(rows)
    assert names == {0: 'Piano', 24: 'Acoustic Guitar', 4: 'Piano', 25: 'Acoustic Guitar', 22: 'Harmonica'}
    assert m.asked_programs([4, 0, 25, 25, 22], ['Piano', 'Acoustic Guitar', 'Harmonica'], names) == [4, 25, 22]
    assert m.asked_programs([], [], names) == []


def test_missing_file_scores_zero_and_keeps_the_unjudged():
    with tempfile.TemporaryDirectory() as d:
        table = m.system_rows(Path(d), [dict(id='m000'), dict(id='m001')],
                              [dict(tempo=120, key=0, programs=[0], caption='x'),
                               dict(tempo=120, key=0, programs=[], caption='x')])
    n = len(m.SCORES)
    assert (table[0, :n] == 0).all() and np.isnan(table[:, n:]).all()
    assert table[1, m.SCORES.index('tb')] == 0
    assert np.isnan(table[1, m.SCORES.index('recall')]) and np.isnan(table[1, m.SCORES.index('jaccard')])


def test_prompt_without_instruments_drops_out_of_their_mean():
    tok, prog = piece(120)
    target = dict(tempo=120, key=0, programs=[], caption='a piano piece')
    assert np.isnan(m.file_row(tok, prog, target)['recall'])
    result = m.bootstrap(dict(a=np.array([1.0, np.nan]), b=np.array([0.0, np.nan])), 10, 0)
    assert result['rows']['a']['mean'] == 1 and result['rows']['b']['mean'] == 0
