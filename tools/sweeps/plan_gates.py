"""
Decision gates of the 2026-11-20 scaling plan (context/claude/scaling_plan_inputs_2026-11-20/01_MASTER_PLAN_assembled.md
§8, K3b), called by cloud/run/plan_queue.sh from `gate=<name>` queue lines:

    plan_gates.py d1 [--resolve]  # §8.5 D1: w*, d*, REG, BEST -> logs/reg_arms.json, inserts D-R4*, D-E5*..D-E11;
                                  # --resolve: REG vs its better single arm after D-E5, BEST vs REG after D-E6
    plan_gates.py confirm         # D-E9: BEST - control <= -T on seeds 3/4, else D-E10/E11 run dyn + family
    plan_gates.py rgates          # §8.5 gate table: D-R5, D-R7 (+U7), D-R8, D-R9b, D-R12, D-O1, D-O2, D-O5..O7, D-L5
    plan_gates.py frontier        # §8.2 G0 (note), G2 (A-S5), G3 (A-L2), G4 (A-U5), B1 (D' basis), A-L3 LR gate
    plan_gates.py discover        # §8.4 G1 (B-T3 / B-T1s3), G2 (B-T2), T1-aug (note)
    plan_gates.py items34         # §8.6 G3a (E-E4/E5, E-E9), G3b (E-E6), retention (E-E7/E8/E10/E11), G4a (E-S5/S7/S8),
                                  # E-S9 / E-S10 / E-S11 from the SBP arms
    plan_gates.py guard --min M   # §9 budget guard for one OPTIONAL line: exit 0 = start it, 1 = skip
    plan_gates.py catchall        # every gate once more, then reruns: aug_sweep lines (finished runs are skipped,
                                  # failed ones retried) and failed plain lines from logs/plan_done.txt

Inputs: results/aug_sweep/runs.json (aug_sweep.py records: arm, seed, size, frac, notes, unique_notes, ce, prog; failed
runs carry failed=True), the members' train.py logs logs/members/<name>.log (control_s3..s6 / fresh_s3..s6 = control /
fresh at seeds 3-6, q108_control_s3 / q108_fresh_s3 = 108M seed 3), aug_sweep logs (panel extras), logs/sbp/*.log
(pair evals) and the ensemble summary (results/aug_sweep/ensemble*.json). Insertions go into logs/plan_queue.txt under
the runner's lock (temp file + rename): D-chain lines at the top (where the gate line was), gated RECOMMENDED rows
before their anchor line, OPTIONAL rows at their §9 priority. Every decision is appended to logs/plan_decisions.md;
logs/plan_gates_state.json remembers what was inserted and decided, so a gate called again only adds what is new.
Thresholds (§8.0): SE_pair = RMS over the finished 42M arms of |d_1337 - d_2|/2, T = max(0.012, 2 SE_pair) at 42M and
0.017 at 108M. One rstar() for every R* read: same-machine P, Poisson D' (epoch once B1 says so), slope s from
B-Gf120 (42M) / A-R1 (6M) when present. PLAN_ROOT moves every path (tests).
"""
import argparse
import fcntl
import glob
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_ladder import parse  # noqa: E402

ROOT = Path(os.environ.get('PLAN_ROOT') or Path(__file__).resolve().parents[2])
RUNS, QUEUE, QLOCK = 'results/aug_sweep/runs.json', 'logs/plan_queue.txt', 'logs/plan_queue.txt.lock'
DEFERRED, DONE, DECISIONS = 'logs/plan_deferred.txt', 'logs/plan_done.txt', 'logs/plan_decisions.md'
STATE, REG_ARMS = 'logs/plan_gates_state.json', 'logs/reg_arms.json'
NOTES, SEEDS = 160e6, (1337, 2)
U_Q, U_STORE = 41.48e6, 167.9e6  # replay quarter / whole replay store, unique notes
LOG2_160_120 = math.log2(160 / 120)  # 0.415
SLOPE = {'3M': 0.40, '6M': 0.404, '20M': 0.41, '42M': 0.416, '108M': 0.468, '217M': 0.50}  # 3M/20M/217M: guesses
T108 = 0.017
RATE, SUNK, CAP = 0.502, 2.01, 25.50  # $/h, $ before T0 (idle + storage), plannable cap
T0_DEFAULT, DEADLINE_DEFAULT = 1795223160, 1795390560  # 01:06 UTC Sat 21 Nov, 23:36 UTC Sun 22 Nov
K1 = 'tests/tools/test_aug_sweep.py'
NEED_RJ = f'{K1},{REG_ARMS}'
DISC_NEED = f'{K1},data/cache/cdisc_t1*'
AJ = '--arm-json logs/reg_arms.json'
FLAGS = {  # aug_sweep ARMS of the D1 candidates and the augmentations
    'wd3x': ['--weight_decay=0.3'], 'wd10x': ['--weight_decay=1.0'], 'wd30x': ['--weight_decay=3.0'],
    'wd60x': ['--weight_decay=6.0'], 'wd30x_trunk': ['--hypothesis=wd_split:muon=3.0'],
    'drop10': ['--dropout=0.1'], 'drop20': ['--dropout=0.2'], 'drop10_late': ['--hypothesis=drop_late:p=0.1:start=1266'],
    'dyn': ['--hypothesis=aug_dyn:shift=4'], 'family': ['--hypothesis=aug_family:rows=0.5'],
}
WD_ARMS, DROP_ARMS = ('wd3x', 'wd10x', 'wd30x', 'wd30x_trunk', 'wd60x'), ('drop10', 'drop10_late', 'drop20')
D1_CORE = ('wd3x', 'wd10x', 'wd30x', 'drop10')  # D1 waits for these (2 seeds each)
D1_LINES = ('D-E1', 'D-E2', 'D-E3', 'D-E4', 'D-E1to4r')  # their queue lines
OPT_ORDER = ['A-U5', 'A-L2', 'D-O2', 'D-O1', 'D-O4', 'D-O5', 'C-O3', 'A-U3', 'A-U4', 'B-T1s3', 'C-O2', 'C-O1', 'E-E10',
             'B-T2', 'B-T5', 'D-O3', 'D-L5', 'D-O8', 'D-O6', 'D-O7', 'F-Q7', 'F-Q8', 'F-U1o', 'E-S4', 'E-S5', 'E-S7',
             'E-S8', 'E-S9', 'E-S11', 'E-E7', 'E-E8', 'E-E6', 'E-E9', 'E-S10', 'E-E11', 'A-L3', 'C-O4', 'E-LIT6']
ANCHOR = {'D-R5': 'D-R6', 'D-R7': 'D-R9a', 'U7': 'D-R9a', 'D-R8': 'D-R9a', 'D-R9b': 'D-R10s', 'D-R12': 'A-R4',
          'A-S5': 'B-T1aug', 'B-T3': 'E-E3'}  # gated RECOMMENDED rows go before this line (top if it has run)
MEMBERS108 = ','.join(['checkpoints/keep/q108_control_s3'] + [f'checkpoints/members108/control_s{s}' for s in (4, 5, 6)])


def path(rel):
    return ROOT / rel


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def utc():
    return time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())


def read_json(rel, default):
    try:
        return json.loads(path(rel).read_text())
    except (OSError, ValueError):
        return default


def write_json(rel, obj):
    p = path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f'{p.name}.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def decide(gate, text):
    p = path(DECISIONS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a') as f:
        f.write(f'- {utc()} **{gate}** {text}\n')
    print(f'{gate}: {text}', flush=True)


# ---------------------------------------------------------------------------------------------------------- data
def fixed_frac(arm):
    """Arms with their own train_file_frac (recorded frac is theirs, not the CLI's)."""
    return (arm == 'fresh' or arm.startswith(('fresh_', 'disc_', 'aria_', 'giga_')) or 'census' in arm
            or arm == 'full')


def member_final(name):
    """Last evaluation of logs/members/<name>.log if it reached the end of its 4,883 iterations."""
    log = path(f'logs/members/{name}.log')
    if not log.exists():
        return None
    evals, _ = parse(log)
    return evals[-1] if evals and evals[-1]['step'] >= 4882 else None


def last_extras(log):
    """{name: CE} of the ' | <name> CE x' extras on the last eval line of a train.py log."""
    try:
        text = Path(log).read_text(errors='replace')
    except OSError:
        return {}
    lines = [ln for ln in text.splitlines() if re.match(r'\s*step \d+: ', ln)]
    return {k: float(v) for k, v in re.findall(r'\| (\w+) CE ([\d.]+)', lines[-1])} if lines else {}


class Data:
    def __init__(self):
        self.runs = read_json(RUNS, {})
        self.state = read_json(STATE, {})

    def recs(self, arm, size='42M', frac=0.25, notes=NOTES):
        """{seed: record} of the finished runs of an arm (members' logs fill control / fresh seeds 3-6)."""
        out = {}
        for r in self.runs.values():
            if r.get('failed') or r.get('arm') != arm or r.get('size', '42M') != size or r.get('ce') is None:
                continue
            if not fixed_frac(arm) and abs(r.get('frac', 0.25) - frac) > 1e-6:
                continue
            if abs(r.get('notes', NOTES) - notes) > 0.02 * notes:
                continue
            out[r['seed']] = r
        if arm in ('control', 'fresh') and abs(notes - NOTES) < 0.02 * NOTES and (arm == 'fresh' or frac == 0.25):
            names = {'42M': {s: f'{arm}_s{s}' for s in (3, 4, 5, 6)}, '108M': {3: f'q108_{arm}_s3'}}.get(size, {})
            for s, name in names.items():
                if s not in out and (e := member_final(name)):
                    out[s] = dict(e, arm=arm, seed=s, size=size, member=name)
        return out

    def m(self, arm, size='42M', frac=0.25, notes=NOTES, seeds=SEEDS, need=2):
        """Mean CE over the given seeds, None unless at least `need` of them finished."""
        r = self.recs(arm, size, frac, notes)
        got = [r[s]['ce'] for s in seeds if s in r]
        return mean(got) if len(got) >= need else None

    def se(self, arm, size='42M', frac=0.25, notes=NOTES):
        r = self.recs(arm, size, frac, notes)
        return abs(r[1337]['ce'] - r[2]['ce']) / 2 if 1337 in r and 2 in r else None

    def unique(self, arm, size='42M', frac=0.25, notes=NOTES, default=None):
        return next((r['unique_notes'] for r in self.recs(arm, size, frac, notes).values() if r.get('unique_notes')),
                    default)

    def effect(self, arm, ref='control', size='42M', frac=0.25, seeds=SEEDS, notes=NOTES):
        """Paired arm - ref over the seeds both finished: dict(delta, dprog, n) or None."""
        a, r = self.recs(arm, size, frac, notes), self.recs(ref, size, frac, notes)
        common = [s for s in seeds if s in a and s in r]
        if not common:
            return None
        dprog = mean([a[s]['prog'] - r[s]['prog'] for s in common
                      if a[s].get('prog') is not None and r[s].get('prog') is not None])
        return dict(delta=mean([a[s]['ce'] - r[s]['ce'] for s in common]), dprog=dprog, n=len(common))

    def se_pair(self):
        """RMS over the finished 42M quarter arms (both seeds, paired with control) of |d_1337 - d_2| / 2."""
        ctrl = self.recs('control')
        if 1337 not in ctrl or 2 not in ctrl:
            return None
        halves = []
        for arm in {r.get('arm') for r in self.runs.values()}:
            if not arm or arm == 'control' or fixed_frac(arm):
                continue
            a = self.recs(arm)
            if 1337 in a and 2 in a:
                halves.append(((a[1337]['ce'] - ctrl[1337]['ce']) - (a[2]['ce'] - ctrl[2]['ce'])) / 2)
        return math.sqrt(sum(h * h for h in halves) / len(halves)) if halves else None

    def T(self, size='42M'):
        if size == '108M':
            return T108
        se = self.se_pair()
        return max(0.012, 2 * se) if se is not None else 0.012

    def slope(self, size):
        """Fresh CE per doubling of notes seen: B-Gf120 (42M, 120M notes), A-R1 (6M, 80M notes), else the default."""
        f = self.m('fresh', size, need=1)
        if f is not None:
            for notes in (120e6, 80e6):
                g = self.m('fresh', size, notes=notes, need=1)
                if g is not None and g > f:
                    return (g - f) / math.log2(NOTES / notes)
        return SLOPE.get(size, 0.416)

    def basis(self):
        return self.state.get('dprime_basis', 'poisson')


# ------------------------------------------------------------------------------------------------------- rstar()
def dprime(D, U, R, basis='poisson'):
    """Effective data of D notes seen over U unique notes with repetition half-life R* (Muennighoff). Poisson: the
    loader samples windows with replacement, so a note is seen k ~ Poisson(D/U) times; epoch: every note D/U times."""
    if basis == 'epoch':
        return D if D <= U else U * (1 + R * (1 - math.exp(-(D / U - 1) / R)))
    lam, total = D / U, 0.0
    for k in range(1, int(lam + 12 * math.sqrt(lam) + 30)):
        p = math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
        total += p * (1 + R * (1 - math.exp(-(k - 1) / R)))
    return U * total


def penalty_pred(R, s, U, Uf=U_STORE, D=NOTES, basis='poisson'):
    """Predicted repeat penalty CE(U) - CE(Uf) at D notes seen: s per doubling of effective data."""
    return s * math.log2(dprime(D, Uf, R, basis) / dprime(D, U, R, basis))


def rstar(P, s, U=U_Q, Uf=U_STORE, D=NOTES, basis='poisson'):
    """R* that makes the predicted penalty equal P (bisection in log R*); inf if P is below any R*'s prediction."""
    if P is None or s is None or s <= 0:
        return None
    lo, hi = 0.05, 1e4
    if P <= penalty_pred(hi, s, U, Uf, D, basis):
        return math.inf
    if P >= penalty_pred(lo, s, U, Uf, D, basis):
        return lo
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if penalty_pred(mid, s, U, Uf, D, basis) > P:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def rstar_at(d, size='42M', arm='control', ref='fresh', frac=0.25):
    """Same-machine R* of an arm against its fresh reference (seeds 1337/2), or None."""
    a, f = d.m(arm, size, frac), d.m(ref, size)
    if a is None or f is None:
        return None
    U = d.unique(arm, size, frac, default=U_Q * frac / 0.25)
    return rstar(a - f, d.slope(size), U, d.unique(ref, size, default=U_STORE), basis=d.basis())


# ---------------------------------------------------------------------------------------------------------- queue
def line(minutes, tier, pid, cmd, need=K1, disk=None):
    head = f'min={minutes:g}' + (f' need={need}' if need else '') + (f' disk={disk}' if disk else '')
    return f'{head} tier={tier} id={pid} :: {cmd}'


def line_id(ln):
    m = re.search(r'(?:^| )id=(\S+)', ln.split(' :: ')[0])
    return m.group(1) if m else None


def line_tier(ln):
    m = re.search(r'(?:^| )tier=(\S+)', ln.split(' :: ')[0])
    return m.group(1) if m else None


def line_min(ln):
    m = re.search(r'(?:^| )min=([\d.]+)', ln.split(' :: ')[0])
    return float(m.group(1)) if m else 0.0


def edit_queue(fn):
    """fn(lines) -> lines on logs/plan_queue.txt under the runner's lock; written to a temp file, then renamed."""
    q = path(QUEUE)
    q.parent.mkdir(parents=True, exist_ok=True)
    with open(path(QLOCK), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = q.read_text().splitlines() if q.exists() else []
        new = fn(lines)
        tmp = q.with_name(f'{q.name}.tmp.{os.getpid()}')
        tmp.write_text(''.join(ln + '\n' for ln in new))
        os.replace(tmp, q)


def read_lines(rel):
    p = path(rel)
    return [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith('#')] \
        if p.exists() else []


def pending_lines():
    out = []
    for rel in (QUEUE, DEFERRED):
        p = path(rel)
        if p.exists():
            out += [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    return out


def place(lines, ln):
    """Index for a gated line: OPTIONAL by §9 rank, RECOMMENDED before its anchor, else the top."""
    pid, tier = line_id(ln), line_tier(ln)
    if tier == 'OPT' and pid in OPT_ORDER:
        rank = OPT_ORDER.index(pid)
        for i, x in enumerate(lines):
            if line_tier(x) == 'OPT' and line_id(x) in OPT_ORDER and OPT_ORDER.index(line_id(x)) > rank:
                return i
        return len(lines)
    if pid in ANCHOR:
        for i, x in enumerate(lines):
            if line_id(x) == ANCHOR[pid]:
                return i
    return None  # top


def insert(st, gate, new):
    """Gated lines into the queue, each id once (state 'inserted'); lines for the top keep their order. -> ids."""
    present = {line_id(x) for x in pending_lines()}
    todo = [ln for ln in new if line_id(ln) not in st.setdefault('inserted', []) and line_id(ln) not in present]
    if not todo:
        return []

    def fn(lines):
        lines, top = list(lines), []
        for ln in todo:
            i = place(lines, ln)
            if i is None:
                top.append(ln)
            else:
                lines.insert(i, ln)
        return top + lines

    edit_queue(fn)
    ids = [line_id(ln) for ln in todo]
    st['inserted'] += ids
    decide(gate, 'inserted ' + ', '.join(ids))
    return ids


def rename_arms(gate, mapping, ids=None):
    """Arm names in queued commands (all lines, or only these ids): e.g. BEST -> REG."""
    pat = re.compile(r'(?<![\w])(' + '|'.join(sorted(map(re.escape, mapping), key=len, reverse=True)) + r')(?![\w])')
    changed = []

    def fn(lines):
        out = []
        for ln in lines:
            if ' :: ' in ln and '--arm-json' in ln and (ids is None or line_id(ln) in ids):
                head, cmd = ln.split(' :: ', 1)
                new = pat.sub(lambda m: mapping[m.group(1)], cmd)
                if new != cmd:
                    changed.append(line_id(ln))
                ln = f'{head} :: {new}'
            out.append(ln)
        return out

    edit_queue(fn)
    if changed:
        decide(gate, f'renamed {mapping} in ' + ', '.join(changed))


# ----------------------------------------------------------------------------------------------------------- arms
def stack(*flag_lists):
    """Flags of several arms in one: hypotheses merged into one stack:parts=... spec."""
    flags = [f for fl in flag_lists for f in fl]
    hyps = [f.split('=', 1)[1] for f in flags if f.startswith('--hypothesis=')]
    rest = [f for f in flags if not f.startswith('--hypothesis=')]
    if len(hyps) <= 1:
        return rest + [f'--hypothesis={h}' for h in hyps]
    return rest + ['--hypothesis=stack:parts=' + ';'.join(h.replace(':', '/') for h in hyps)]


def wd2x(flags):
    """BEST_wd2x: the arm's weight decay doubled (Muon's for wd_split), 0.2 if it has none."""
    out, done = [], False
    for f in flags:
        if f.startswith('--weight_decay='):
            f, done = f'--weight_decay={2 * float(f.split("=")[1]):g}', True
        elif 'wd_split:muon=' in f or 'wd_split/muon=' in f:
            f, done = re.sub(r'muon=([\d.]+)', lambda m: f'muon={2 * float(m.group(1)):g}', f), True
        out.append(f)
    return out if done else stack(['--weight_decay=0.2'], out)


def write_arms(st):
    """logs/reg_arms.json from the state: REG, BEST and their fresh / wd2x variants, dynfam (the fallback BEST),
    wstar_lr5 (D-O3), fresh_b16k_lr7 (D-R12) and the items 3-4 arms."""
    aug = stack(FLAGS['dyn'], FLAGS['family'])
    reg = st.get('reg_flags') or []
    best = stack(reg, FLAGS['dyn'], FLAGS['family'])
    arms = read_json(REG_ARMS, {})
    arms.update({'BEST': best, 'fresh_BEST': best, 'BEST_wd2x': wd2x(best),
                 'dynfam': aug, 'fresh_dynfam': aug, 'dynfam_wd2x': wd2x(aug),
                 'fresh_b16k_lr7': {'args': ['--learning_rate=7.07e-4'], 'rows': 8, 'warmup': 200}})
    if reg:
        arms.update({'REG': reg, 'fresh_REG': reg, 'REG_wd2x': wd2x(reg)})
    if st.get('wstar'):
        arms['wstar_lr5'] = FLAGS[st['wstar']] + ['--learning_rate=5e-4']
    write_json(REG_ARMS, arms)


def add_arms(new):
    arms = read_json(REG_ARMS, {})
    arms.update(new)
    write_json(REG_ARMS, arms)


# ------------------------------------------------------------------------------------------------------------- D1
def eligible(e, T):
    return e is not None and e['n'] >= 2 and e['delta'] <= -T and (e['dprog'] is None or e['dprog'] <= 0.01)


def d1_lines(st):
    B = st.get('best', 'BEST')
    out = []
    if st.get('reg_combo'):
        out += [line(19.3, 'ESS', 'D-E5', f'$A --arms REG {AJ}', NEED_RJ), 'gate=d1 :: --resolve']
    out += [line(3, 'ESS', 'D-E6s', f'$A --smoke --arms {B} {AJ}', NEED_RJ),
            line(17.8, 'ESS', 'D-E6', f'$A --arms {B} {AJ}', NEED_RJ), 'gate=d1 :: --resolve',
            line(19.3, 'ESS', 'D-E9', f'$A --seeds 3,4 --arms {B} --force {AJ}', NEED_RJ), 'gate=confirm',
            line(19.3, 'ESS', 'D-E7', f'$A --arms fresh_{B} {AJ}', NEED_RJ),
            line(19.3, 'ESS', 'D-E8', f'$A --frac 0.125 --force --arms {B} {AJ}', NEED_RJ),
            line(38.6, 'ESS', 'D-E10', f'$A --size 108M --arms {B} --keep --force {AJ}', NEED_RJ, disk=3),
            line(38.6, 'ESS', 'D-E11', f'$A --size 108M --arms fresh_{B} --force {AJ}', NEED_RJ)]
    return out


def insert_block(st, gate, lines_):
    """Lines (with gate= lines between them) at the top, in order; ids already inserted are skipped."""
    present = {line_id(x) for x in pending_lines()}
    keep = [ln for ln in lines_ if ln.startswith('gate=') or (line_id(ln) not in st.setdefault('inserted', [])
                                                               and line_id(ln) not in present)]
    ids = [line_id(ln) for ln in keep if not ln.startswith('gate=')]
    if not ids:
        return []
    edit_queue(lambda lines: keep + list(lines))
    st['inserted'] += ids
    decide(gate, 'inserted ' + ', '.join(ln if ln.startswith('gate=') else line_id(ln) for ln in keep))
    return ids


def gate_d1(d, st, resolve=False, wait=True):
    T = d.T()
    if resolve:
        return d1_resolve(d, st, T)
    if st.get('d1_done'):
        return decide('d1', 'already decided; nothing to do')
    missing = [a for a in D1_CORE if d.effect(a) is None or d.effect(a)['n'] < 2]
    if missing and not wait:
        return decide('d1', f'catch-all: {missing} unfinished, D1 not decided')
    if missing:
        queued = [line_id(x) for x in read_lines(QUEUE) if line_id(x) in D1_LINES]
        deferred = [x for x in read_lines(DEFERRED) if line_id(x) in D1_LINES]
        if queued:  # their lines are still queued: D1 goes after the last of them
            edit_queue(lambda lines: after_id(lines, queued[-1], f'gate=d1 need={K1}'))
            return decide('d1', f'waiting for {missing} (2 seeds each): gate=d1 requeued after {queued[-1]}')
        if deferred:  # deferred (the runner waits on this gate, so the file is ours): after it, with its needs
            last = deferred[-1]
            m = re.search(r'(?:^| )need=(\S+)', last.split(' :: ')[0])
            gl = f'gate=d1 need={m.group(1) if m else K1}'
            p = path(DEFERRED)
            p.write_text(''.join(x + '\n' for x in after_id(p.read_text().splitlines(), line_id(last), gl)))
            return decide('d1', f'waiting for {missing}: gate=d1 deferred after {line_id(last)} ({gl})')
        if not st.get('d1_rerun'):  # they ran and failed: one rerun, then D1
            st['d1_rerun'] = True
            edit_queue(lambda lines: [line(round(8.5 * len(missing), 1), 'ESS', 'D-E1to4r',
                                           f'$A --arms {",".join(missing)} --force'), f'gate=d1 need={K1}'] + lines)
            return decide('d1', f'{missing} unfinished and not queued: one rerun, then gate=d1')
        decide('d1', f'still missing {missing} after a rerun: deciding on what finished')
    eff = {a: d.effect(a) for a in WD_ARMS + DROP_ARMS}
    se = d.se_pair()
    desc = ', '.join(f'{a} {e["delta"]:+.4f}' + (f' (prog {e["dprog"]:+.3f})' if e['dprog'] is not None else '')
                     for a, e in eff.items() if e)
    el = {a: e for a, e in eff.items() if eligible(e, T)}
    w = min((a for a in el if a in WD_ARMS), key=lambda a: el[a]['delta'], default=None)
    dd = min((a for a in el if a in DROP_ARMS), key=lambda a: el[a]['delta'], default=None)
    st.update(wstar=w, dstar=dd, T=T, d1_done=True)
    if w and dd:
        st.update(reg_flags=stack(FLAGS[w], FLAGS[dd]), reg_combo=True, reg_name=f'{w}+{dd}')
    elif w or dd:
        st.update(reg_flags=FLAGS[w or dd], reg_combo=False, reg_name=w or dd)
    else:
        st.update(reg_flags=[], reg_combo=False, reg_name='')
    write_arms(st)
    decide('d1', f'SE_pair {se if se is None else round(se, 4)}, T {T:.4f}; {desc}; w* {w}, d* {dd}; '
                 f'REG = {st["reg_name"] or "empty"}; BEST = REG + dyn + family -> {REG_ARMS}')
    # D-R4: third seed for arms between T/2 and T, at most 2
    mid = sorted((a for a, e in eff.items() if e and e['n'] >= 2 and T / 2 <= -e['delta'] < T),
                 key=lambda a: abs(-eff[a]['delta'] - T))[:2]
    block = []
    if mid:
        block.append(line(round(9.65 * len(mid), 1), 'REC', 'D-R4', f'$A --seeds 3 --arms {",".join(mid)} --force',
                          K1))
    if not st.get('wstar'):
        edit_queue(lambda lines: [x for x in lines if line_id(x) != 'D-O3'])
        decide('d1', 'no w*: D-O3 (w* + LR 5e-4) removed')
    insert_block(st, 'd1', block + d1_lines(st))


def after_id(lines, pid, ln):
    out = []
    for x in lines:
        out.append(x)
        if line_id(x) == pid:
            out.append(ln)
    return out


def d1_resolve(d, st, T):
    if not st.get('d1_done'):
        return decide('d1', '--resolve before D1: nothing to do')
    if st.get('reg_combo') and not st.get('reg_resolved'):
        combo = d.m('REG')
        singles = {a: d.m(a) for a in (st['wstar'], st['dstar'])}
        if combo is None or None in singles.values():
            return decide('d1', 'resolve: REG (D-E5) or its single arms unfinished; REG stays the pair')
        best_single = min(singles, key=singles.get)
        st['reg_resolved'] = True
        if combo >= singles[best_single]:
            st.update(reg_flags=FLAGS[best_single], reg_name=best_single, reg_combo=False)
            write_arms(st)
            return decide('d1', f'D-E5 REG {combo:.4f} does not beat {best_single} {singles[best_single]:.4f}: '
                                f'REG := {best_single}; BEST, fresh_BEST, BEST_wd2x rewritten')
        return decide('d1', f'D-E5 REG {combo:.4f} beats its single arms {singles}: REG = {st["reg_name"]}')
    if not st.get('best_checked'):
        best = d.m('BEST')
        if not st.get('reg_flags'):
            ref_arm = 'control'
        else:  # a single REG arm ran under its own name, the pair as REG (D-E5)
            ref_arm = st['reg_name'] if st.get('reg_name') in FLAGS else 'REG'
        ref = d.m(ref_arm)
        if best is None or ref is None:
            return decide('d1', f'resolve: BEST or {ref_arm} unfinished; nothing to do')
        st['best_checked'] = True
        if best >= ref and st.get('reg_flags'):
            st['best'] = 'REG'
            rename_arms('d1', {'fresh_BEST': 'fresh_REG', 'BEST_wd2x': 'REG_wd2x', 'BEST': 'REG'})
            return decide('d1', f'BEST {best:.4f} does not beat REG ({st["reg_name"]}) {ref:.4f}: BEST := REG '
                                '(later lines run REG / fresh_REG / REG_wd2x)')
        if best >= ref:
            return decide('d1', f'BEST (= dyn + family, REG empty) {best:.4f} does not beat control {ref:.4f}; '
                                'kept as BEST (no REG to fall back to)')
        return decide('d1', f'BEST {best:.4f} beats {ref_arm} {ref:.4f}')
    decide('d1', 'resolve: nothing pending')


def gate_confirm(d, st):
    if st.get('confirm_done'):
        return decide('confirm', 'already decided')
    B, T = st.get('best', 'BEST'), d.T()
    e = d.effect(B, 'control', seeds=(3, 4))
    if e is None or e['n'] < 2:
        return decide('confirm', f'{B} seeds 3/4 or members control_s3/s4 unfinished; BEST kept for D-E10/E11')
    st['confirm_done'] = True
    if e['delta'] <= -T:
        return decide('confirm', f'{B} - control on seeds 3/4 = {e["delta"]:+.4f} <= -T ({T:.4f}): confirmed')
    if B == 'BEST' and not st.get('reg_flags'):
        return decide('confirm', f'{B} - control on seeds 3/4 = {e["delta"]:+.4f} > -T, but BEST is already dyn + '
                                 'family')
    st['best108'] = 'dynfam'
    rename_arms('confirm', {f'fresh_{B}': 'fresh_dynfam', f'{B}_wd2x': 'dynfam_wd2x', B: 'dynfam'},
                ids={'D-E10', 'D-E11'})
    decide('confirm', f'{B} - control on seeds 3/4 = {e["delta"]:+.4f} > -T ({T:.4f}): D-E10/E11 run dynfam '
                      '(dyn + family)')


# ---------------------------------------------------------------------------------------------------------- rgates
def once(st, key):
    """True the first time a gate item is decided (then remembered)."""
    done = st.setdefault('decided', {})
    if key in done:
        return False
    done[key] = True
    return True


def se_dp(d, size, B):
    parts = [d.se(a, size) for a in ('control', 'fresh', B, f'fresh_{B}')]
    parts = [p for p in parts if p is not None]
    return max(0.0085, math.sqrt(sum(p * p for p in parts))) if parts else 0.0085


def gate_rgates(d, st):
    T, B, B108 = d.T(), st.get('best', 'BEST'), st.get('best108', st.get('best', 'BEST'))
    new = []
    # D-R5: fresh_REG unless REG is empty or wd3x / wd10x alone
    if st.get('d1_done') and (not st.get('reg_combo') or st.get('reg_resolved')) and once(st, 'D-R5'):
        if st.get('reg_name') in ('', 'wd3x', 'wd10x'):
            decide('rgates', f'D-R5 skipped: REG = {st.get("reg_name") or "empty"}')
        else:
            new.append(line(19.3, 'REC', 'D-R5', f'$A --arms fresh_REG {AJ}', NEED_RJ))
            decide('rgates', f'D-R5 fires: REG = {st["reg_name"]}')
    # D-R7 (+U7): predicted P'_108(f0.125) <= s_108
    b, fb = d.m(B108, '108M'), d.m(f'fresh_{B108}', '108M')
    if b is not None and fb is not None and once(st, 'D-R7'):
        s, U = SLOPE['108M'], d.unique(B108, '108M', default=U_Q)
        r = rstar(b - fb, s, U, basis=d.basis())
        pred = penalty_pred(r, s, U / 2, basis=d.basis()) if r and r != math.inf else 0.0
        fire = pred <= s
        decide('rgates', f"D-R7 {'fires' if fire else 'skipped'}: P'_108 {b - fb:+.4f}, R*' {r:.2f}, predicted "
                         f"P'_108(f0.125) {pred:.3f} vs s_108 {s}")
        if fire:
            new += [line(38.6, 'REC', 'D-R7', f'$A --size 108M --frac 0.125 --force --arms {B108} {AJ}', NEED_RJ),
                    line(17, 'REC', 'U7', '$A --size 108M --frac 0.125 --force --arms control --seeds 1337')]
    # D-R8: dP_108 / P_108 < half of dP_42 / P_42, and dP_42 >= 2 SE
    vals = [d.m('control'), d.m('fresh'), d.m(B), d.m(f'fresh_{B}'),
            d.m('control', '108M'), d.m('fresh', '108M'), b, fb]
    if None not in vals and once(st, 'D-R8'):
        c42, f42, b42, fb42, c108, f108 = vals[:6]
        P42, P108 = c42 - f42, c108 - f108
        dP42, dP108 = P42 - (b42 - fb42), P108 - (b - fb)
        se = se_dp(d, '42M', B)
        fire = P42 > 0 and P108 > 0 and dP108 / P108 < 0.5 * dP42 / P42 and dP42 >= 2 * se
        decide('rgates', f"D-R8 {'fires' if fire else 'skipped'}: dP/P 42M {dP42:+.4f}/{P42:.4f}, 108M "
                         f"{dP108:+.4f}/{P108:.4f}, SE(dP) {se:.4f}")
        if fire:
            new.append(line(38.6, 'REC', 'D-R8', f'$A --size 108M --arms {B108}_wd2x --force {AJ}', NEED_RJ))
    # D-R9b: P_6M >= 0.04
    c6, f6 = d.m('control', '6M'), d.m('fresh', '6M')
    if c6 is not None and f6 is not None and st.get('d1_done') and once(st, 'D-R9b'):
        fire = c6 - f6 >= 0.04
        decide('rgates', f"D-R9b {'fires' if fire else 'skipped'}: P_6M {c6 - f6:+.4f}")
        if fire:
            new.append(line(8, 'REC', 'D-R9b', f'$A --size 6M --arms fresh_{B} {AJ}', NEED_RJ))
    # D-O1: drop10 <= -T
    e = d.effect('drop10')
    if e and e['n'] >= 2 and once(st, 'D-O1'):
        fire = e['delta'] <= -T
        decide('rgates', f"D-O1 {'fires' if fire else 'skipped'}: drop10 {e['delta']:+.4f}, T {T:.4f}")
        if fire:
            new.append(line(19.3, 'OPT', 'D-O1', '$A --arms drop20'))
    # D-O2: wd30x (or trunk) is the best wd arm and beats wd10x by >= SE
    wds = {a: d.effect(a) for a in ('wd3x', 'wd10x', 'wd30x', 'wd30x_trunk')}
    if all(wds[a] and wds[a]['n'] >= 2 for a in ('wd3x', 'wd10x', 'wd30x')) and once(st, 'D-O2'):
        got = {a: e['delta'] for a, e in wds.items() if e and e['n'] >= 2}
        bestwd = min(got, key=got.get)
        se = d.se_pair() or 0.006
        fire = bestwd in ('wd30x', 'wd30x_trunk') and got['wd10x'] - got[bestwd] >= se
        decide('rgates', f"D-O2 {'fires' if fire else 'skipped'}: best wd {bestwd} {got[bestwd]:+.4f}, wd10x "
                         f"{got['wd10x']:+.4f}, SE {se:.4f}")
        if fire:
            new.append(line(17, 'OPT', 'D-O2', '$A --arms wd60x'))
    # D-O5: |embwd0 - control| >= T
    e = d.effect('embwd0')
    if e and e['n'] >= 2 and once(st, 'D-O5'):
        fire = abs(e['delta']) >= T
        decide('rgates', f"D-O5 {'fires' if fire else 'skipped'}: embwd0 {e['delta']:+.4f}")
        if fire:
            new.append(line(17, 'OPT', 'D-O5', '$A --arms fresh_embwd0', f'{K1},hypotheses/wd_split.py'))
    # D-R12 / D-O7 / D-O6: the batch arms
    bk = {a: d.effect(a) for a in ('b16k', 'b16k_lr7')}
    if all(e and e['n'] >= 2 for e in bk.values()):
        better = min(bk, key=lambda a: bk[a]['delta'])
        twin = f'fresh_{better}'
        if once(st, 'D-R12'):
            decide('rgates', f'D-R12 fires: {twin} (b16k {bk["b16k"]["delta"]:+.4f}, b16k_lr7 '
                             f'{bk["b16k_lr7"]["delta"]:+.4f})')
            new.append(line(21.9, 'REC', 'D-R12', f'$A --arms {twin} {AJ}', NEED_RJ))
            st['b16k_better'] = better
        if once(st, 'D-O7'):
            fire = bk[better]['delta'] <= -T
            decide('rgates', f"D-O7 {'fires' if fire else 'skipped'}: best b16k {bk[better]['delta']:+.4f}")
            if fire:
                new.append(line(21.9, 'OPT', 'D-O7', '$A --arms b16k_wd05'))
        xb, fxb, c, f = d.m(better), d.m(twin), d.m('control'), d.m('fresh')
        if None not in (xb, fxb, c, f) and once(st, 'D-O6'):
            dP = (c - f) - (xb - fxb)
            se = max(0.0085, math.sqrt(sum((d.se(a) or 0) ** 2 for a in ('control', 'fresh', better, twin))))
            fire = bk[better]['delta'] <= -T and dP >= se
            decide('rgates', f"D-O6 {'fires' if fire else 'skipped'}: best b16k {bk[better]['delta']:+.4f}, "
                             f"dP_b16k {dP:+.4f}, SE {se:.4f}")
            if fire:
                new.append(line(31.7, 'OPT', 'D-O6', '$A --arms b8k'))
    # D-L5: |P*(lambda) - 0.133| >= 0.012, P*_X = X - min(fresh, fresh_X); lambda rule notes
    f = d.m('fresh')
    pstar = {}
    for x, fx in (('wd3x', 'fresh_wd03'), ('wd10x', 'fresh_wd1')):
        vx, vfx = d.m(x), d.m(fx)
        if None not in (vx, vfx, f):
            pstar[x] = vx - min(f, vfx)
            if f - vfx >= 0.012 and once(st, f'lambda_{fx}'):
                decide('rgates', f'lambda rule: {fx} beats fresh by {f - vfx:.4f} >= 0.012: it becomes the fresh '
                                 'reference (analysis)')
    if len(pstar) == 2 and once(st, 'D-L5'):
        best = min(pstar.values())
        fire = abs(best - 0.133) >= 0.012
        decide('rgates', f"D-L5 {'fires' if fire else 'skipped'}: P*(best lambda) {best:.4f} vs 0.133 "
                         f"({'R_D* >= 7.5 under the tuned recipe' if best <= 0.080 else 'lambda trades E for P' if best > 0.11 else 'middle band'})")
        if fire:
            new.append(line(14.2, 'OPT', 'D-L5', '$A --size 6M --arms wd06,fresh_wd06 --force'))
    insert(st, 'rgates', new)


# -------------------------------------------------------------------------------------------------------- frontier
def gate_frontier(d, st):
    new = []
    # G0: 6M ms/step of A-S1 (note only)
    r6 = d.recs('fresh', '6M') or d.recs('control', '6M')
    if r6 and once(st, 'G0'):
        r = next(iter(r6.values()))
        if r.get('minutes') and r.get('iters'):
            ms = (r['minutes'] - 0.7) * 60000 / r['iters']
            decide('frontier', f'G0: 6M {ms:.1f} ms/step' + (' (above 35: expect +~1.5 h)' if ms > 35 else ''))
    # G2: A-S5 if |control_217 - control_108| < 0.015
    c217, c108 = d.m('control', '217M', need=1), d.m('control', '108M', need=1)
    if c217 is not None and c108 is not None and once(st, 'G2'):
        fire = abs(c217 - c108) < 0.015
        decide('frontier', f"G2 {'fires' if fire else 'skipped'} (A-S5): control_217 {c217:.4f}, control_108 "
                           f"{c108:.4f}" + ('; crossover at N/U ~ 5' if c217 > c108 + 0.012 else ''))
        if fire:
            new.append(line(31.5, 'REC', 'A-S5', '$A --size 217M --arms control --seeds 2 --force'))
    # G3: A-L2 if |P_lr2 - P_lr1| > 0.012 at 6M
    lr2, flr2, c6, f6 = d.m('lr2e3', '6M'), d.m('fresh_lr2e3', '6M'), d.m('control', '6M'), d.m('fresh', '6M')
    if None not in (lr2, flr2, c6, f6) and once(st, 'G3'):
        diff = (lr2 - flr2) - (c6 - f6)
        fire = abs(diff) > 0.012
        decide('frontier', f"G3 {'fires' if fire else 'skipped'} (A-L2): P_lr2 - P_lr1 at 6M {diff:+.4f}")
        if fire:
            new.append(line(10.9, 'OPT', 'A-L2', '$A --size 3M --arms lr2e3,fresh_lr2e3 --force'))
    # A-L3 (LR gate): an LR check moves P or fresh CE by >= 0.02
    moves, complete = [], True
    for size in ('6M', '42M'):
        v = [d.m('lr2e3', size), d.m('fresh_lr2e3', size), d.m('control', size), d.m('fresh', size)]
        if None in v:
            complete = False
            continue
        moves.append((size, (v[0] - v[1]) - (v[2] - v[3]), v[1] - v[3]))
    if moves and (complete or any(abs(a) >= 0.02 or abs(b) >= 0.02 for _, a, b in moves)) and once(st, 'A-L3'):
        fire = any(abs(a) >= 0.02 or abs(b) >= 0.02 for _, a, b in moves)
        decide('frontier', f"A-L3 {'fires' if fire else 'skipped'}: LR 2e-3 moves (size, dP, d fresh) "
                           + ', '.join(f'{s} {a:+.4f} {b:+.4f}' for s, a, b in moves))
        if fire:
            new.append(line(63, 'OPT', 'A-L3', '$A --size 217M --arms lr05,fresh_lr05 --seeds 1337 --force', disk=6))
    # B1: Poisson vs epoch D' from the store-offset test
    fr, sub, cen = d.m('fresh', '6M'), d.m('sub168_census', '6M'), d.m('fresh_census', '6M')
    if None not in (fr, sub, cen) and once(st, 'B1'):
        s6 = d.slope('6M')
        R = rstar_at(d, '6M') or 4.6
        R = 4.6 if R == math.inf else R
        Usub = d.unique('sub168_census', '6M', default=168e6)
        Ucen = d.unique('fresh_census', '6M', default=1.897e9)
        pred = s6 * math.log2(dprime(NOTES, Ucen, R) / dprime(NOTES, Usub, R))
        deficit = sub - cen
        st['dprime_basis'] = 'epoch' if abs(deficit - pred) > 0.015 else 'poisson'
        decide('frontier', f'B1: delta_s (replay fresh - sub168) {fr - sub:+.4f}, deficit (sub168 - census) '
                           f'{deficit:+.4f} vs Poisson {pred:+.4f}: D\' basis {st["dprime_basis"]}')
    # G4: A-U5 if R_D*(6M) > 8
    r6s = rstar_at(d, '6M')
    if r6s is not None and once(st, 'G4'):
        fire = r6s > 8
        decide('frontier', f"G4 {'fires' if fire else 'skipped'} (A-U5): R_D*(6M) {r6s:.2f} "
                           f"(s {d.slope('6M'):.3f}, basis {d.basis()})")
        if fire:
            new.append(line(6.4, 'OPT', 'A-U5', '$A --size 6M --frac 0.0625 --notes 320e6 --arms control --seeds 1337 '
                                                '--force'))
    insert(st, 'frontier', new)


# -------------------------------------------------------------------------------------------------------- discover
def disc_rstar(d, extra=None):
    """R* of the Discover pair (disc_control vs disc_fresh, slope from disc_fresh120); extra = a panel CE name."""
    def ce(arm):
        rs = d.recs(arm, notes=120e6 if arm.endswith('120') else NOTES)
        vals = []
        for s in SEEDS:
            if s not in rs:
                continue
            if extra is None:
                vals.append(rs[s]['ce'])
            else:
                v = rs[s].get('extras', {}).get(extra) or last_extras(path(f'logs/aug_sweep/{arm}_s{s}.log')).get(extra)
                vals.append(v)
        return mean(vals) if len([v for v in vals if v is not None]) >= 2 else None
    c, f, f120 = ce('disc_control'), ce('disc_fresh'), ce('disc_fresh120')
    if None in (c, f, f120) or f120 <= f:
        return None
    U = d.unique('disc_control', default=41.5e6)
    return rstar(c - f, (f120 - f) / LOG2_160_120, U, d.unique('disc_fresh', default=168e6), basis=d.basis())


def gate_discover(d, st):
    new = []
    rd, rg = disc_rstar(d), rstar_at(d, '42M')
    if rd is not None and rg is not None and once(st, 'G1'):
        rho = rd / rg if rg not in (0, math.inf) else math.inf
        if 0.8 <= rho <= 1.25:
            decide('discover', f'G1: rho {rho:.2f} (R*_D {rd:.2f} / R*_G {rg:.2f}) in [0.8, 1.25]: one corpus R_D*')
        elif rho < 0.75 or rho > 1.33:
            decide('discover', f'G1: rho {rho:.2f} (R*_D {rd:.2f} / R*_G {rg:.2f}): per-source R_D*, B-T3 fires')
            new.append(line(61.6, 'REC', 'B-T3', '$A --size 108M --arms disc_control,disc_fresh,disc_fresh120,'
                                                 'giga_fresh120 --seeds 1337', DISC_NEED))
        else:
            decide('discover', f'G1: rho {rho:.2f} (R*_D {rd:.2f} / R*_G {rg:.2f}) in the in-between band: B-T1s3')
            new.append(line(17.5, 'OPT', 'B-T1s3', '$A --seeds 3 --arms disc_control,disc_fresh', DISC_NEED))
    ra = disc_rstar(d, 'disc_aria')
    if ra is not None and rd is not None and once(st, 'G2d'):
        ratio = max(ra, rd) / min(ra, rd) if min(ra, rd) > 0 else math.inf
        fire = ratio > 1.33
        decide('discover', f"G2 {'fires' if fire else 'skipped'} (B-T2): R* Aria part {ra:.2f} vs non-Aria {rd:.2f} "
                           f"(x{ratio:.2f})")
        if fire:
            new.append(line(48.5, 'OPT', 'B-T2', '$A --arms aria_control,aria_fresh,aria_fresh120',
                            f'{K1},data/cache/aria_train*', disk=5))
    dc, dd, df = d.m('disc_control'), d.m('disc_dyn'), d.m('disc_fresh')
    gc, gdyn, gf = d.m('control'), d.m('dyn'), d.m('fresh')
    if None not in (dc, dd, df, gc, gdyn, gf) and dc > df and gc > gf and once(st, 'T1aug'):
        wd, wg = (dc - dd) / (dc - df), (gc - gdyn) / (gc - gf)
        verdict = ('dyn corpus-wide' if wd >= wg / 2 else 'dyn credited only for GigaMIDI\'s 24% share'
                   if wd <= 0.05 else 'between: dyn partly')
        decide('discover', f'T1-aug: dyn won back {wd:.0%} on Discover vs {wg:.0%} on GigaMIDI: {verdict}')
    insert(st, 'discover', new)


# --------------------------------------------------------------------------------------------------------- items34
def ensemble_summary():
    """g4 and the memorization gap from ensemble_eval.py's summary (keys tolerated: g4 | mean_member + L4;
    gap | heldout + train_windows)."""
    files = sorted(glob.glob(str(path('results/aug_sweep/ensemble*.json'))) +
                   glob.glob(str(path('results/ensemble/*.json'))))
    for fn in files:
        try:
            s = json.loads(Path(fn).read_text())
        except (OSError, ValueError):
            continue
        L4 = s.get('L4', s.get('L_4', s.get('ensemble')))
        mm = s.get('mean_member')
        g4 = s.get('g4', mm - L4 if mm is not None and L4 is not None else None)
        gap = s.get('gap', s.get('memorization_gap'))
        if gap is None and s.get('heldout') is not None and s.get('train_windows') is not None:
            gap = s['heldout'] - s['train_windows']
        if g4 is not None:
            return dict(g4=g4, L4=L4, gap=gap, file=fn)
    return None


def sbp_eval(name):
    """(pairs CE, shuffled CE) of logs/sbp/<name>.log."""
    log = path(f'logs/sbp/{name}.log')
    if not log.exists():
        return None
    evals, _ = parse(log)
    shuf = last_extras(log).get('shuf')
    return (evals[-1]['ce'], shuf) if evals and shuf is not None else None


def gate_items34(d, st):
    top, new, follow, ens = [], [], False, ensemble_summary()
    members = ','.join(f'checkpoints/members/control_s{s}' for s in (3, 4, 5, 6))
    fresh_members = ','.join(f'checkpoints/members/fresh_s{s}' for s in (3, 4, 5, 6))
    if ens and once(st, 'G3a'):
        g3a = ens['g4'] >= 0.015
        st['g3a'] = g3a
        decide('items34', f"G3a {'fires' if g3a else 'skipped'}: g4 {ens['g4']:.4f} ({ens['file']})")
        if g3a:
            top += [line(3, 'REC', 'E-E4', '$A --smoke --arms distill4', f'{K1},hypotheses/distill.py'),
                    line(52, 'REC', 'E-E5', '$A --arms distill4 --force', f'{K1},hypotheses/distill.py'),
                    'gate=items34']
            reg = st.get('reg_flags') or []
            half = [f'--weight_decay={float(f.split("=")[1]) / 2:g}' if f.startswith('--weight_decay=') else f
                    for f in reg] or ['--weight_decay=0.05']
            kim = ' '.join(shlex.quote(f) for f in half)
            new.append(line(76.8, 'OPT', 'E-E9', '; '.join(
                f'm kim_s{s} $C42 --train_file_frac=0.25 --seed={s} --max_iters=9766 --lr_decay_iters=9766 {kim} '
                f'--out_dir=checkpoints/members/kim_s{s}' for s in (7, 8, 9, 10)), None))
    if ens and ens.get('gap') is not None and st.get('g3a') and once(st, 'G3b'):
        fire = ens['gap'] >= 0.133
        decide('items34', f"G3b {'fires' if fire else 'skipped'} (E-E6): memorization gap {ens['gap']:.4f}")
        if fire:
            add_arms({'kd_seq25': {'synth': [['data/cache/kd_members'], 0.25]}})
            new.append(line(77.5, 'OPT', 'E-E6', f'$A --arms kd_seq25 --force --keep {AJ}',
                            f'{NEED_RJ},data/cache/kd_members*'))
    # retention after E-E5
    e5, c = d.effect('distill4'), d.m('control')
    if ens and ens.get('L4') is not None and e5 and e5['n'] >= 2 and c is not None and once(st, 'retention'):
        student = c + e5['delta']
        ret = (c - student) / (c - ens['L4']) if c != ens['L4'] else 0.0
        decide('items34', f"retention {ret:.2f}, E5 {e5['delta']:+.4f} (student {student:.4f}, L_4 {ens['L4']:.4f})")
        if e5['delta'] <= -0.012:
            add_arms({'distill1': [f'--hypothesis=distill:members=checkpoints/members/control_s3:alpha=0.0'],
                      'distill4_fresh': ['--train_file_frac=1.0',
                                         f'--hypothesis=distill:members={fresh_members}:alpha=0.0']})
            new += [line(27.2, 'OPT', 'E-E7', f'$A --arms distill1 --force {AJ}', NEED_RJ),
                    line(52, 'OPT', 'E-E8', f'$A --arms distill4_fresh --force {AJ}', NEED_RJ)]
            if ret >= 0.5:
                add_arms({'distill4_108': [f'--hypothesis=distill:members={MEMBERS108}:alpha=0.0']})
                new += [line(54.5, 'OPT', 'E-E10', '; '.join(
                    f'm q108_control_s{s} $C108 --train_file_frac=0.25 --seed={s} '
                    f'--out_dir=checkpoints/members108/control_s{s}' for s in (4, 5, 6)) + '; upkeep members108', None,
                    disk=3),
                    line(101.6, 'OPT', 'E-E11', f'$A --size 108M --arms distill4_108 --force --keep {AJ}', NEED_RJ,
                         disk=2)]
    # G4a: synthesizer pair gain
    base, ft = sbp_eval('base'), [x for x in (sbp_eval('it800'), sbp_eval('it1600')) if x]
    if base and ft and once(st, 'G4a'):
        b = base[1] - base[0]
        gain = max(sh - ce for ce, sh in ft)
        bar = max(0.02, b + 0.01)
        fire = gain >= bar
        decide('items34', f"G4a {'fires' if fire else 'skipped'}: pair gain {gain:.4f} vs bar {bar:.4f} (base {b:.4f})")
        if fire:
            add_arms({'sbp25': {'synth': [['data/cache/sbp_synth'], 0.25]}})
            new += [line(131, 'OPT', 'E-S5', '$SS --boundary --checkpoint checkpoints/sbp_synth --out data/cache/sbp_synth',
                         'checkpoints/sbp_synth', disk=4),
                    line(18.5, 'OPT', 'E-S7', f'$A --smoke --arms sbp25 {AJ} && $A --arms sbp25 --force --keep {AJ}',
                         f'{NEED_RJ},data/cache/sbp_synth*'),
                    line(34, 'OPT', 'E-S8', f'$A --size 108M --arms sbp25 --force --keep {AJ}',
                         f'{NEED_RJ},data/cache/sbp_synth*', disk=2)]
            follow = True
    # E-S9 / E-S11 after sbp25, E-S10 after sbp_clean25
    sbp, ctl, syn = d.m('sbp25'), d.m('control'), d.m('synth25')
    if sbp is not None and ctl is not None and once(st, 'E-S9'):
        fire9 = sbp - ctl <= -0.012 or (syn is not None and sbp - syn <= -0.012)
        decide('items34', f"E-S9 {'fires' if fire9 else 'skipped'}, E-S11 {'fires' if sbp - ctl <= -0.012 else 'skipped'}: "
                          f"sbp25 - control {sbp - ctl:+.4f}" + (f', - synth25 {sbp - syn:+.4f}' if syn else ''))
        if fire9:
            add_arms({'sbp_clean25': {'synth': [['data/cache/sbp_clean'], 0.25]}})
            new.append(line(80.1, 'OPT', 'E-S9',
                            '$TB --init_from=finetune --init_ckpt=checkpoints/members/control_s3/model_bf16.pt $C42 $AW '
                            '--csv_path=store:data/cache/sbp_pairs --out_dir=checkpoints/sbp_synth42 --batch_size=16 '
                            '--gradient_accumulation_steps=1 --max_iters=3192 --lr_decay_iters=3192 --eval_iters=75 '
                            '--save_iters=1596 --always_save_checkpoint=True > logs/sbp/train42.log 2>&1 && $SS --boundary '
                            '--checkpoint checkpoints/sbp_synth42 --out data/cache/sbp_clean && $A --arms sbp_clean25 '
                            f'--force --keep {AJ}', f'{NEED_RJ},data/cache/sbp_pairs*', disk=3))
            follow = True
        if sbp - ctl <= -0.012:
            add_arms({'sbp12': {'synth': [['data/cache/sbp_synth'], 0.125]}})
            new.append(line(17, 'OPT', 'E-S11', f'$A --arms sbp12 --force {AJ}', f'{NEED_RJ},data/cache/sbp_synth*'))
    sc, scl = d.m('sbp_clean25'), d.m('synth_clean25')
    if sc is not None and scl is not None and once(st, 'E-S10'):
        fire = sc - scl <= -0.012
        decide('items34', f"E-S10 {'fires' if fire else 'skipped'}: sbp_clean25 - synth_clean25 {sc - scl:+.4f}")
        if fire:
            add_arms({'rand_clean25': {'synth': [['data/cache/sbp_rand_clean'], 0.25]}})
            new.append(line(80.1, 'OPT', 'E-S10',
                            '$TB --init_from=finetune --init_ckpt=checkpoints/members/control_s3/model_bf16.pt $C42 $AW '
                            '--csv_path=store:data/cache/sbp_rand --out_dir=checkpoints/sbp_rand42 --batch_size=16 '
                            '--gradient_accumulation_steps=1 --max_iters=3192 --lr_decay_iters=3192 --eval_iters=75 '
                            '--save_iters=1596 --always_save_checkpoint=True > logs/sbp/rand42.log 2>&1 && $SS --boundary '
                            '--checkpoint checkpoints/sbp_rand42 --out data/cache/sbp_rand_clean && $A --arms rand_clean25 '
                            f'--force --keep {AJ}', f'{NEED_RJ},data/cache/sbp_rand*', disk=3))
    if top:
        insert_block(st, 'items34', top)
    added = insert(st, 'items34', new)
    if follow and added:  # read the new SBP arms once they ran: gate=items34 after the last line added
        last = added[-1]
        edit_queue(lambda lines: sum(([x, 'gate=items34'] if line_id(x) == last else [x] for x in lines), []))


# ----------------------------------------------------------------------------------------------------------- guard
def epoch_file(rel, default):
    try:
        return float(path(rel).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def gate_guard(minutes, now=None):
    """0 if an OPTIONAL line of `minutes` may start now, else 1 (§9 with the review's spend model)."""
    now = time.time() if now is None else now
    t0, deadline = epoch_file('logs/T0', T0_DEFAULT), epoch_file('logs/DEADLINE', DEADLINE_DEFAULT)
    spend = SUNK + (now - t0) / 3600 * RATE
    rem = sum(line_min(x) for x in pending_lines() if line_tier(x) in ('ESS', 'REC'))
    if rem > 0:
        need = spend + 1.15 * (rem + minutes) * RATE / 60
        ok = need <= CAP
        why = (f'spend ${spend:.2f} + 1.15 x ({rem:.0f} min ESS/REC left + {minutes:g}) = ${need:.2f} vs cap ${CAP}')
    else:
        end = now + 2.2 * minutes * 60 + 600
        need = spend + 2.2 * minutes / 60 * RATE
        ok = end <= deadline and need <= CAP
        why = (f'ends by {time.strftime("%a %H:%M", time.gmtime(end))} UTC at the 2.2x bound (+10 min upload) vs '
               f'deadline {time.strftime("%a %H:%M", time.gmtime(deadline))}; spend at the bound ${need:.2f}')
    decide('guard', f"{'go' if ok else 'skip'} ({minutes:g} min): {why}")
    return 0 if ok else 1


# -------------------------------------------------------------------------------------------------------- catchall
def gate_catchall(d, st):
    for name, fn in (('d1', lambda: gate_d1(d, st, wait=False) if not st.get('d1_done') else None),
                     ('d1 --resolve', lambda: d1_resolve(d, st, d.T()) if st.get('d1_done') else None),
                     ('confirm', lambda: gate_confirm(d, st) if st.get('d1_done') else None),
                     ('rgates', lambda: gate_rgates(d, st)), ('frontier', lambda: gate_frontier(d, st)),
                     ('discover', lambda: gate_discover(d, st)), ('items34', lambda: gate_items34(d, st))):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - one broken gate must not stop the catch-all
            decide('catchall', f'{name} failed: {e!r}')
        write_json(STATE, st)
    p = path(DONE)
    if not p.exists():
        return decide('catchall', f'no {DONE}: nothing to rerun')
    last = {}
    for ln in p.read_text().splitlines():
        if ' # exit ' not in ln or ln.startswith('gate='):
            continue
        body, tail = ln.rsplit(' # exit ', 1)
        code = tail.split()[0]
        pid = line_id(body)
        if pid:
            last[pid] = (body, code)
    present = {line_id(x) for x in pending_lines()}
    rerun = []
    for pid, (body, code) in last.items():
        cmd = body.split(' :: ', 1)[1] if ' :: ' in body else ''
        aug = '$A ' in cmd and '--smoke' not in cmd
        if pid not in present and (aug or code != '0') and ' try=9' not in body:
            rerun.append(re.sub(r'(^| )try=\d+', '', body.split(' :: ')[0]) + ' try=9 :: ' + cmd)
    if rerun:
        edit_queue(lambda lines: list(lines) + rerun)
    decide('catchall', f'requeued {len(rerun)} line(s) (aug_sweep lines skip finished runs and retry failed ones; '
                       'plain lines that failed): ' + ', '.join(line_id(x) for x in rerun))


# ------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('d1').add_argument('--resolve', action='store_true')
    for name in ('confirm', 'rgates', 'frontier', 'discover', 'items34', 'catchall'):
        sub.add_parser(name)
    g = sub.add_parser('guard')
    g.add_argument('--min', type=float, required=True)
    a = p.parse_args(argv)
    if a.cmd == 'guard':
        return gate_guard(a.min)
    d = Data()
    st = d.state
    fn = {'d1': lambda: gate_d1(d, st, a.resolve), 'confirm': lambda: gate_confirm(d, st),
          'rgates': lambda: gate_rgates(d, st), 'frontier': lambda: gate_frontier(d, st),
          'discover': lambda: gate_discover(d, st), 'items34': lambda: gate_items34(d, st),
          'catchall': lambda: gate_catchall(d, st)}[a.cmd]
    try:
        fn()
    finally:
        write_json(STATE, st)
    return 0


if __name__ == '__main__':
    sys.exit(main())
