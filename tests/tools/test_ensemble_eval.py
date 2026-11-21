"""ensemble_eval: K=1 equals estimate_val_loss, the mixture bounds, the summary keys plan_gates reads (CPU, tiny model):
python tests/tools/test_ensemble_eval.py  (or pytest tests/)."""

import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/sweeps'), str(ROOT / 'tests')]

import ensemble_eval as E  # noqa: E402
from helpers import make_store, notes  # noqa: E402
from musicar.eval.val_loss import estimate_val_loss  # noqa: E402
from musicar.model.gpt import GPT, MusicConfig  # noqa: E402

ARGS = dict(n_layer=1, n_head=2, n_embd=32, block_size=2048, pitch_size=130, n_programs=129, pos_emb='rope',
            norm='rmsnorm', mlp='swiglu', cond_inst=True, n_density=16)


def _setup():
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    files = [notes(n, seed=i) for i, n in enumerate((2200, 4300, 2500, 90))]
    make_store(d / 'v_validation', files, [rng.integers(0, 129, len(f)) for f in files])
    dirs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        m = GPT(MusicConfig(**ARGS))
        (d / f'm{seed}').mkdir()
        torch.save(dict(model_args=ARGS, model=m.state_dict()), d / f'm{seed}' / 'model_bf16.pt')
        dirs.append(str(d / f'm{seed}'))
    return d, dirs


def test_k1_equals_estimate_val_loss():
    d, dirs = _setup()
    models = [E.load_model(x)[0] for x in dirs]
    src = f'store:{d / "v"}'
    res, parts, _ = E.run_set(models, E.loader(src, 'validation', 2, 4), 0.0, ['prob', 'logit', 'joint'],
                              nullcontext(), 'cpu')
    s = E.summarize(res, 2)
    for i, m in enumerate(models):
        batches = (((st, ex), Y) for st, ex, Y in (E.prepare(X, Y, m.config, 0.0, 'cpu')
                                                   for X, Y in E.loader(src, 'validation', 2, 4)))
        ref = estimate_val_loss(m, batches, nullcontext())
        assert abs(s['prob']['members'][i] - ref['ce/total']) < 1e-5, (s['prob']['members'][i], ref['ce/total'])
        assert abs(s['logit']['members'][i] - ref['ce/total']) < 1e-5
        assert abs(parts[i]['pitch'] - ref['ce/pitch']) < 1e-5
    # Jensen: the probability mixture is never worse than the members' mean; L_inf from the two points
    assert s['prob']['L2'] <= s['prob']['L1'] + 1e-6 and s['joint']['L2'] <= s['joint']['L1'] + 1e-6
    assert abs(s['prob']['L_inf'] - (2 * s['prob']['L2'] - s['prob']['L1'])) < 1e-6


def test_identical_members_change_nothing():
    d, dirs = _setup()
    models = [E.load_model(dirs[0])[0] for _ in range(2)]
    res, _, lps = E.run_set(models, E.loader(f'store:{d / "v"}', 'validation', 2, 4), 1.0, ['prob', 'logit'],
                            nullcontext(), 'cpu', save_lp=True)
    s = E.summarize(res, 2)
    assert abs(s['prob']['L2'] - s['prob']['L1']) < 1e-5 and abs(s['logit']['L2'] - s['logit']['L1']) < 1e-5
    assert lps[0].dtype == np.float16 and lps[0].shape[0] == len(models[0].head_names)


def test_summary_keys_for_plan_gates():
    d, dirs = _setup()

    class A:  # the CLI arguments evaluate_set reads
        val, windows, micro, train_frac, nocond, save_lp = f'store:{d / "v"}', 4, 2, 1.0, True, False
        train_store, train_windows, modes = f'store:{d / "v"}', 0, ['prob']
    arm = E.evaluate_set(dirs, A, 'cpu', nullcontext(), d / 'ens')
    for k in ('L1', 'L2', 'mean_member', 'g4', 'heldout', 'L_inf'):
        assert isinstance(arm[k], float), k
    assert abs(arm['g4'] - (arm['L1'] - arm['L2'])) < 1e-9 and arm['heldout'] == arm['mean_member'] == arm['L1']
    assert set(arm['eval']['heldout']) == {'cond', 'nocond'}
    assert E.member_dir('control_s3') == ROOT / 'checkpoints/members/control_s3'
    assert E.arm_name(['checkpoints/members/fresh_s3']) == 'fresh'


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
