"""tools/sweeps/plan_gates.py on fake runs.json files, one case per gate branch (CPU, < 5 s):
python tests/tools/test_plan_gates.py  (or pytest tests/)."""

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/sweeps'))

import plan_gates as pg  # noqa: E402

SEED_Q = ['min=17 tier=ESS id=A-S4 :: $A x', 'min=17.5 tier=REC id=B-T1aug :: $A y', 'min=3 tier=REC id=E-E3 :: z',
          'min=17 tier=OPT id=D-O4 :: $A --arms embwd0', 'min=12.1 tier=OPT id=A-U3 :: $A u3',
          'min=17 tier=OPT id=D-O3 :: $A --arms wstar_lr5 --arm-json logs/reg_arms.json']


class Env:
    """A temporary plan root with runs.json, the queue and members' logs."""

    def __init__(self, queue=SEED_Q):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        pg.ROOT = self.root
        (self.root / 'logs/members').mkdir(parents=True)
        (self.root / 'results/aug_sweep').mkdir(parents=True)
        (self.root / 'logs/plan_queue.txt').write_text(''.join(x + '\n' for x in queue))
        self.runs = {}

    def run(self, arm, ce, seeds=(1337, 2), size='42M', frac=0.25, notes=160e6, prog=1.0, **kw):
        for i, s in enumerate(seeds):
            c = ce[i] if isinstance(ce, (list, tuple)) else ce
            self.runs[f'{arm}_{s}_{size}_{frac}_{notes}'] = dict(arm=arm, seed=s, size=size, frac=frac, notes=notes,
                                                                  ce=c, prog=prog, **kw)
        (self.root / 'results/aug_sweep/runs.json').write_text(json.dumps(self.runs))
        return self

    def member(self, name, ce):
        (self.root / f'logs/members/{name}.log').write_text(
            f'Model parameters : 42,000,000\nstep 4882: train loss 2.0 val loss {ce} | gigamidi CE {ce:.4f} '
            f'(pit 1.0 vel 1.0 dur 1.0 del 1.0) | prog CE 1.000\n')

    def gate(self, *argv):
        return pg.main(list(argv))

    def queue(self):
        return (self.root / 'logs/plan_queue.txt').read_text().splitlines()

    def ids(self):
        return [pg.line_id(x) or x for x in self.queue()]

    def arms(self):
        return json.loads((self.root / 'logs/reg_arms.json').read_text())

    def decisions(self):
        return (self.root / 'logs/plan_decisions.md').read_text()

    def state(self):
        return json.loads((self.root / 'logs/plan_gates_state.json').read_text())


def base(env, **deltas):
    """control 2.60 / fresh 2.45 at 42M plus the D1 core arms (control + delta, same on both seeds)."""
    env.run('control', [2.600, 2.600]).run('fresh', 2.45, frac=1.0)
    for arm in pg.D1_CORE:
        env.run(arm, 2.600 + deltas.get(arm, 0.0))
    for arm, dl in deltas.items():
        if arm not in pg.D1_CORE:
            env.run(arm, 2.600 + dl)
    return env


def test_rstar_reference_values():
    U, R = 41.48e6, 160e6 / 41.48e6 - 1
    for ps, want in ((0.20, 7.2), (0.30, 4.6), (0.40, 3.35), (0.50, 2.6)):  # §8.4 reference, epoch basis
        got = pg.rstar(ps, 1.0, U, Uf=1e12, basis='epoch')
        assert abs(got - want) / want < 0.03, (ps, got)
    assert abs(pg.rstar(0.133, 0.416) - 4.6) < 0.1  # Poisson basis at 42M (frontier brief 4.61)
    assert pg.rstar(-0.01, 0.4) == math.inf and R > 0
    r = pg.rstar(0.15, 0.4)
    assert abs(pg.penalty_pred(r, 0.4, U) - 0.15) < 1e-6


def test_se_pair_and_threshold():
    env = Env()
    env.run('control', [2.60, 2.60]).run('dyn', [2.59, 2.60]).run('family', [2.59, 2.59])
    d = pg.Data()
    assert abs(d.se_pair() - math.sqrt((0.005 ** 2 + 0) / 2)) < 1e-9
    assert d.T() == 0.012 and d.T('108M') == 0.017
    env.run('octave', [2.55, 2.61])
    assert abs(pg.Data().T() - 2 * pg.Data().se_pair()) < 1e-12 and pg.Data().T() > 0.012


def test_d1_neither_passes():
    env = base(Env(), wd3x=-0.005, wd10x=0.002)
    env.gate('d1')
    st, arms = env.state(), env.arms()
    assert st['reg_name'] == '' and 'REG' not in arms
    assert arms['BEST'] == ['--hypothesis=stack:parts=aug_dyn/shift=4;aug_family/rows=0.5'] == arms['dynfam']
    ids = env.ids()
    assert ids[:5] == ['D-E6s', 'D-E6', 'gate=d1 :: --resolve', 'D-E9', 'gate=confirm']
    assert 'D-E5' not in ids and 'D-R4' not in ids and 'D-O3' not in ids and ids[-1] == 'A-U3'
    env.gate('d1')  # a second call adds nothing
    assert env.ids().count('D-E6') == 1


def test_d1_one_passes_and_third_seed():
    env = base(Env(), wd10x=-0.030, drop10=-0.008)
    env.gate('d1')
    st, arms = env.state(), env.arms()
    assert st['reg_name'] == 'wd10x' and arms['REG'] == ['--weight_decay=1.0']
    assert arms['BEST'] == ['--weight_decay=1.0', '--hypothesis=stack:parts=aug_dyn/shift=4;aug_family/rows=0.5']
    assert arms['BEST_wd2x'][0] == '--weight_decay=2' and arms['wstar_lr5'][-1] == '--learning_rate=5e-4'
    q = env.queue()
    assert pg.line_id(q[0]) == 'D-R4' and '--seeds 3 --arms drop10 ' in q[0]
    assert 'D-O3' in env.ids()


def test_d1_both_pass_stacks_hypotheses_and_prog_guard():
    env = base(Env(), wd30x_trunk=-0.04, drop10_late=-0.03, wd10x=-0.05)
    env.run('wd10x', 2.55, prog=1.02)  # better CE but prog CE +0.02: not eligible
    env.gate('d1')
    st, arms = env.state(), env.arms()
    assert st['wstar'] == 'wd30x_trunk' and st['dstar'] == 'drop10_late' and st['reg_combo']
    assert arms['REG'] == ['--hypothesis=stack:parts=wd_split/muon=3.0;drop_late/p=0.1/start=1266']
    assert arms['REG_wd2x'] == ['--hypothesis=stack:parts=wd_split/muon=6;drop_late/p=0.1/start=1266']
    assert arms['BEST'] == ['--hypothesis=stack:parts=wd_split/muon=3.0;drop_late/p=0.1/start=1266;aug_dyn/shift=4;'
                            'aug_family/rows=0.5']  # one flat stack
    assert env.ids()[:3] == ['D-E5', 'gate=d1 :: --resolve', 'D-E6s']


def test_d1_waits_for_missing_core_arms():
    env = Env(['min=1 tier=REC id=R1 :: a', 'min=17 tier=ESS id=D-E3 :: $A --arms wd30x'] + SEED_Q[3:])
    (env.root / 'logs/plan_deferred.txt').write_text('min=17 need=x tier=ESS id=D-E4 :: $A --arms drop10\n')
    env.run('control', 2.6).run('fresh', 2.45, frac=1.0).run('wd3x', 2.58)
    env.gate('d1')  # D-E3 queued: after it
    assert env.ids()[2] == 'gate=d1 need=tests/tools/test_aug_sweep.py' and not env.state().get('d1_done')
    (env.root / 'logs/plan_queue.txt').write_text('')
    env.gate('d1')  # D-E4 deferred: after it in the deferred list, with its needs
    assert (env.root / 'logs/plan_deferred.txt').read_text().splitlines()[1] == 'gate=d1 need=x'
    env = Env([])
    env.run('control', 2.6).run('fresh', 2.45, frac=1.0).run('wd3x', 2.58)
    env.gate('d1')  # nothing queued: one rerun of the missing arms, then D1
    assert env.ids() == ['D-E1to4r', 'gate=d1 need=tests/tools/test_aug_sweep.py'] and '--arms wd10x,wd30x,drop10 ' in env.queue()[0]
    (env.root / 'logs/plan_queue.txt').write_text('')
    env.gate('d1')
    assert env.state()['d1_done'] and 'after a rerun' in env.decisions()


def test_resolve_reg_then_best():
    env = base(Env(), wd10x=-0.030, drop10=-0.020)
    env.gate('d1')
    env.run('REG', 2.575)  # the pair does not beat wd10x (2.570)
    env.gate('d1', '--resolve')
    st = env.state()
    assert st['reg_name'] == 'wd10x' and env.arms()['REG'] == ['--weight_decay=1.0']
    env.run('BEST', 2.571)  # BEST does not beat REG (wd10x 2.570): later lines run REG
    env.gate('d1', '--resolve')
    q = '\n'.join(env.queue())
    assert '--arms fresh_REG ' in q and '--seeds 3,4 --arms REG ' in q and 'BEST' not in q.replace('BEST := ', '')


def test_confirm_pass_and_fail():
    for best3, renamed in ((2.57, False), (2.595, True)):
        env = base(Env(), wd10x=-0.030)
        env.gate('d1')
        env.member('control_s3', 2.60)
        env.member('control_s4', 2.60)
        env.run('BEST', best3, seeds=(3, 4))
        env.gate('confirm')
        q = {pg.line_id(x): x for x in env.queue()}
        assert ('--arms dynfam ' in q['D-E10']) == renamed and ('--arms fresh_dynfam ' in q['D-E11']) == renamed
        assert '--arms BEST ' in q['D-E8']


def test_rgates_branches():
    env = base(Env(), wd10x=-0.030, wd30x=-0.045, drop10=-0.02, embwd0=0.02, b16k=-0.02, b16k_lr7=-0.025)
    env.run('control', 2.70, size='6M').run('fresh', 2.64, size='6M', frac=1.0)
    env.run('fresh_wd03', 2.44, frac=1.0).run('fresh_wd1', 2.45, frac=1.0)
    env.gate('d1')
    env.run('REG', 2.54)  # the pair beats its single arms
    env.gate('d1', '--resolve')
    env.gate('rgates')
    ids, text = env.ids(), env.decisions()
    assert 'D-R5' in ids and 'D-R9b' in ids and 'D-R12' in ids  # REG = pair, P_6M 0.06, both b16k arms done
    assert 'D-O1' in ids and 'D-O2' in ids and 'D-O5' in ids and 'D-O7' in ids
    assert 'fresh_b16k_lr7' in {x.split(' :: ')[1].split('--arms ')[1].split()[0] for x in env.queue()
                                if pg.line_id(x) == 'D-R12'}
    assert 'D-L5 fires' in text and 'D-L5' in ids  # P*(wd10x) = 2.570 - 2.45 = 0.12: |0.12 - 0.133| >= 0.012
    # OPT rows by §9 rank: D-O2 before D-O1 before D-O4 (seeded) before D-O5
    order = [i for i in ids if i in ('D-O2', 'D-O1', 'D-O4', 'D-O5', 'A-U3')]
    assert order == ['D-O2', 'D-O1', 'D-O4', 'D-O5', 'A-U3']
    env.gate('rgates')  # nothing twice
    assert env.ids().count('D-O1') == 1


def test_rgates_r5_skip_and_r7_r8():
    env = base(Env(), wd3x=-0.03)
    env.gate('d1')
    env.run('control', 2.52, size='108M').run('fresh', 2.36, size='108M', frac=1.0)
    env.run('BEST', 2.55).run('fresh_BEST', 2.44, frac=1.0)
    env.run('BEST', 2.43, size='108M').run('fresh_BEST', 2.36, size='108M', frac=1.0)
    env.gate('rgates')
    text, ids = env.decisions(), env.ids()
    assert 'D-R5 skipped: REG = wd3x' in text
    assert 'D-R7 fires' in text and 'D-R7' in ids and 'U7' in ids  # P'_108 0.07 -> high R*'
    assert 'D-R8' in text and 'D-R8' not in ids  # dP_108/P_108 = 0.09/0.16 vs half of 0.04/0.15


def test_frontier_branches():
    env = Env()
    env.run('control', 2.53, seeds=(1337,), size='217M').run('control', 2.54, size='108M')
    env.run('control', 2.70, size='6M').run('fresh', 2.66, size='6M', frac=1.0, minutes=3.7, iters=4883)
    env.run('lr2e3', 2.69, size='6M').run('fresh_lr2e3', 2.67, size='6M', frac=1.0)
    env.gate('frontier')
    ids, text = env.ids(), env.decisions()
    assert ids.index('A-S5') == ids.index('B-T1aug') - 1  # REC before its anchor
    assert 'A-L2' in ids  # P_lr2 0.02 vs P_lr1 0.04
    assert 'A-U5' in ids and ids.index('A-U5') < ids.index('D-O4')  # P_6M 0.04 -> R* > 8, OPT rank 1
    assert 'G0: 6M 36.9 ms/step (above 35' in text
    env.run('sub168_census', 2.63, size='6M', frac=0.088, unique_notes=168e6)
    env.run('fresh_census', 2.62, size='6M', frac=1.0, unique_notes=1.897e9)
    env.gate('frontier')
    assert env.state()['dprime_basis'] == 'poisson' and "D' basis" in env.decisions()


def test_discover_bands():
    for disc_ctrl, want in ((2.90, 'one corpus'), (3.10, 'B-T3'), (2.927, 'B-T1s3')):
        env = Env()
        env.run('control', 2.60).run('fresh', 2.45, frac=1.0).run('fresh', 2.57, frac=1.0, notes=120e6)
        env.run('disc_control', disc_ctrl, frac=0.247).run('disc_fresh', 2.75, frac=1.0)
        env.run('disc_fresh120', 2.87, frac=1.0, notes=120e6)
        env.gate('discover')
        if want == 'one corpus':
            assert want in env.decisions() and 'B-T3' not in env.ids() and 'B-T1s3' not in env.ids()
        else:
            assert want in env.ids(), (disc_ctrl, env.decisions())


def test_items34_ensemble_retention_and_g4a():
    env = Env()
    env.run('control', 2.60).run('fresh', 2.45, frac=1.0)
    (env.root / 'results/aug_sweep/ensemble_eval.json').write_text(json.dumps(
        dict(mean_member=2.60, L4=2.57, heldout=2.60, train_windows=2.40)))
    env.gate('items34')
    ids = env.ids()
    assert ids[:3] == ['E-E4', 'E-E5', 'gate=items34'] and 'E-E9' in ids and 'E-E6' in ids
    env.run('distill4', 2.58)  # E5 -0.02, retention 0.67
    env.gate('items34')
    ids = env.ids()
    assert all(i in ids for i in ('E-E7', 'E-E8', 'E-E10', 'E-E11'))
    sbp = env.root / 'logs/sbp'
    sbp.mkdir(parents=True)
    for name, ce, shuf in (('base', 2.50, 2.51), ('it800', 2.40, 2.45), ('it1600', 2.39, 2.43)):
        (sbp / f'{name}.log').write_text(f'step 0: train loss 0 val loss {ce} | val CE {ce} (pit 1.0) | shuf CE {shuf}\n')
    env.gate('items34')
    ids = env.ids()
    assert all(i in ids for i in ('E-S5', 'E-S7', 'E-S8')) and ids[ids.index('E-S8') + 1] == 'gate=items34'
    assert env.arms()['sbp25'] == {'synth': [['data/cache/sbp_synth'], 0.25]}


def test_guard_cap_and_deadline():
    env = Env(['min=600 tier=ESS id=X :: a'])
    (env.root / 'logs/T0').write_text('1000000\n')
    (env.root / 'logs/DEADLINE').write_text(str(1000000 + 46 * 3600))
    t = 1000000 + 3600 * 10  # spend 2.01 + 5.02 = 7.03; + 1.15 x 610 min x 0.502/60 = 12.9
    assert pg.gate_guard(10, now=t) == 0
    assert pg.gate_guard(10, now=1000000 + 3600 * 36) == 1  # 2.01 + 18.07 + 5.87 = 25.95 > 25.50
    (env.root / 'logs/plan_queue.txt').write_text('')
    assert pg.gate_guard(60, now=1000000 + 3600 * 42) == 0  # ends 44.2 h + 10 min < 46 h
    assert pg.gate_guard(60, now=1000000 + 3600 * 44) == 1


def test_catchall_requeues_aug_and_failed_lines():
    env = Env([])
    (env.root / 'logs/plan_done.txt').write_text(
        'min=17 tier=ESS id=D-E1 :: $A --arms wd3x # exit 0 2026-11-21 06:00:00\n'
        'min=3 tier=ESS id=D-E4s :: $A --smoke --arms drop10 # exit 0 2026-11-21 06:10:00\n'
        'min=10 tier=ESS id=F-U1e :: upkeep keep # exit 0 2026-11-21 06:20:00\n'
        'min=4 tier=REC id=E-S3 :: plain cmd # exit 1 2026-11-21 06:30:00\n'
        'gate=d1 # exit 0 2026-11-21 06:31:00\n')
    env.gate('catchall')
    q = env.queue()
    assert [pg.line_id(x) for x in q] == ['D-E1', 'E-S3'] and all(' try=9 :: ' in x for x in q)
    (env.root / 'logs/plan_queue.txt').write_text('')
    with open(env.root / 'logs/plan_done.txt', 'a') as f:  # the reruns ran (and failed again)
        f.writelines(x + ' # exit 1 2026-11-21 07:00:00\n' for x in q)
    env.gate('catchall')  # a rerun is not requeued again
    assert env.queue() == []

if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
