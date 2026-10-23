"""Reports of tests/ci.py sessions, written from each run's summary.json (layout after pytest-html's
--self-contained-html: one file that opens from disk, no server):

  logs/ci/<run>/report.txt   every test: status, time, note; a failed one also its command and traceback / log tail
  logs/ci/<run>/report.html  the same as a page: failed rows red and expanded, passed collapsed, a failed-only
                             filter, links to the test logs; reloads itself every 5 s while the run is going
  logs/ci/index.html         every run with its counts, and a test x run grid of the last HISTORY runs
  logs/ci/latest             symlink to the newest run

tests/ci.py calls write_run after every test. Rebuild everything from the summary.json files (also for runs from
before this module):

  .venv/bin/python tests/ci_report.py
"""

import html
import json
import os
import sys
import time
from pathlib import Path

CI_DIR = Path(__file__).resolve().parents[1] / 'logs' / 'ci'
HISTORY = 30   # runs in the index grid
TAIL = 40      # log lines shown for a failed test
STATUSES = ('FAIL', 'PASS', 'SKIP')

CSS = '''
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --card: #f5f5f7;
  --pass: #1e7d32; --pass-bg: #e8f5e9; --fail: #c62828; --fail-bg: #fdecea; --skip: #8a6d00; --skip-bg: #fff8e1;
  --run: #0b57d0; --run-bg: #e8f0fe; --code: #f0f0f2; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #161618; --fg: #f2f2f7; --muted: #a1a1a6; --line: #3a3a3c; --card: #1f1f22;
  --pass: #81c784; --pass-bg: #1b2e1c; --fail: #ff8a80; --fail-bg: #3b1c1a; --skip: #ffd54f; --skip-bg: #332b10;
  --run: #8ab4f8; --run-bg: #1c2a40; --code: #26262a; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.meta { color: var(--muted); margin-bottom: 12px; }
.counts { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.chip { padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: 13px; }
.PASS { color: var(--pass); background: var(--pass-bg); }
.FAIL { color: var(--fail); background: var(--fail-bg); }
.SKIP { color: var(--skip); background: var(--skip-bg); }
.RUNNING, .PENDING { color: var(--run); background: var(--run-bg); }
.PENDING { opacity: .6; }
label.filter { display: inline-flex; gap: 6px; align-items: center; margin: 4px 0 12px; cursor: pointer; }
details { border: 1px solid var(--line); border-radius: 8px; margin: 6px 0; background: var(--card); }
details.FAIL-row { border-color: var(--fail); background: var(--fail-bg); }
summary { display: grid; grid-template-columns: 70px minmax(120px, 200px) 70px 1fr; gap: 8px; padding: 6px 10px;
  cursor: pointer; align-items: baseline; }
summary .note { color: var(--muted); overflow-wrap: anywhere; }
details.FAIL-row summary .note { color: var(--fail); }
.body { padding: 0 10px 10px; }
.doc { color: var(--muted); margin: 4px 0 8px; }
pre { background: var(--code); padding: 8px; border-radius: 6px; overflow-x: auto; margin: 6px 0;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
a { color: var(--run); }
body.failed-only details:not(.FAIL-row) { display: none; }
table { border-collapse: collapse; margin: 8px 0 20px; }
.scroll { overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--line); padding: 3px 6px; text-align: left; white-space: nowrap; font-size: 13px; }
th { background: var(--card); }
td.cell { text-align: center; padding: 0; }
td.cell a { display: block; padding: 3px 6px; text-decoration: none; font-weight: 600; color: inherit; }
@media (max-width: 600px) { summary { grid-template-columns: 60px 1fr 56px; } summary .note { grid-column: 1 / -1; } }
'''

FILTER_JS = '''
const box = document.getElementById('failed-only');
try { box.checked = localStorage.getItem('ci-failed-only') === '1'; } catch (e) {}
const apply = () => { document.body.classList.toggle('failed-only', box.checked);
  try { localStorage.setItem('ci-failed-only', box.checked ? '1' : '0'); } catch (e) {} };
box.addEventListener('change', apply); apply();
'''


def write_atomic(path, text):
    """path.tmp, then rename: a page reloading mid-write never reads half a file."""
    tmp = Path(f'{path}.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def load(run_dir):
    """A run's summary.json (None if it has none). Runs from before the report module have no plan / finished keys:
    their results are the plan and they are finished."""
    p = Path(run_dir) / 'summary.json'
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    s.setdefault('finished', True)
    s.setdefault('running', None)
    s.setdefault('plan', [dict(name=n, gpu=False, doc='') for n in s.get('results', {})])
    return s


def status_of(summary, name):
    r = summary['results'].get(name)
    if r:
        return r['status']
    if summary['running'] == name:
        return 'RUNNING'
    return 'PENDING' if not summary['finished'] else 'NOT RUN'


def counts(summary):
    return {k: sum(r['status'] == k for r in summary['results'].values()) for k in STATUSES}


def log_lines(run_dir, name):
    p = Path(run_dir) / f'{name}.log'
    return p.read_text(errors='replace').replace('\r', '\n').splitlines() if p.exists() else []


def log_tracebacks(lines, limit=5):
    """The Python tracebacks printed in a log ('Traceback (most recent call last):' to the exception line), last
    `limit` of them."""
    blocks, cur = [], None
    for ln in lines:
        if ln.startswith('Traceback (most recent call last):'):
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
            if ln and not ln[0].isspace():  # the exception line ends the block
                blocks.append('\n'.join(cur))
                cur = None
    return blocks[-limit:]


def failure_detail(run_dir, name, result):
    """(last command, tracebacks (the test's own, then those in its log), log tail) of a failed test."""
    lines = log_lines(run_dir, name)
    cmd = next((ln[2:] for ln in reversed(lines) if ln.startswith('$ ')), '')
    tbs = ([result['traceback'].rstrip()] if result.get('traceback') else []) + log_tracebacks(lines)
    tail = [ln for ln in lines if ln.strip()][-TAIL:]
    return cmd, '\n\n'.join(tbs) or None, tail


def headline(summary):
    c = counts(summary)
    state = 'running' if not summary['finished'] else f'finished in {summary.get("minutes", 0):g} min'
    return (f'CI run {summary["started"]}  commit {summary.get("commit", "?")}  {state}  '
            f'{c["PASS"]} passed, {c["FAIL"]} failed, {c["SKIP"]} skipped')


def write_text(run_dir, summary):
    out = [headline(summary), f'GPU: {summary.get("gpu", "?")}', '']
    for t in summary['plan']:
        name = t['name']
        r = summary['results'].get(name, {})
        out.append(f'{status_of(summary, name):8s} {name:20s} {r.get("seconds", 0):7.1f} s  {r.get("note", "")}')
        if r.get('status') == 'FAIL':
            cmd, tb, tail = failure_detail(run_dir, name, r)
            out.append(f'    why:     {r.get("note", "")}')
            if cmd:
                out.append(f'    command: {cmd}')
            if tb:
                out += ['    traceback:'] + ['      ' + ln for ln in tb.rstrip().splitlines()]
            out += [f'    log tail ({name}.log):'] + ['      ' + ln for ln in tail]
    write_atomic(Path(run_dir) / 'report.txt', '\n'.join(out) + '\n')


def page(title, body, refresh=False, script=''):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            + ('<meta http-equiv="refresh" content="5">' if refresh else '')
            + f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main>'
            + (f'<script>{script}</script>' if script else '') + '</body></html>\n')


def write_html(run_dir, summary):
    e = html.escape
    c = counts(summary)
    chips = ''.join(f'<span class="chip {k}">{c[k]} {k.lower()}</span>' for k in STATUSES if c[k])
    if not summary['finished']:
        n_left = sum(t['name'] not in summary['results'] for t in summary['plan'])
        chips += f'<span class="chip RUNNING">running {e(summary["running"] or "")} ({n_left} left)</span>'
    rows = []
    for t in summary['plan']:
        name = t['name']
        r = summary['results'].get(name, {})
        st = status_of(summary, name)
        cls = st.replace(' ', '')
        log = f'<a href="{e(name)}.log">{e(name)}.log</a>' if (Path(run_dir) / f'{name}.log').exists() else ''
        body = (f'<div class="doc">{e(t.get("doc", ""))}</div>' if t.get('doc') else '') + \
            (f'<div>log: {log}</div>' if log else '')
        if st == 'FAIL':
            cmd, tb, tail = failure_detail(run_dir, name, r)
            body += f'<div><b>why</b></div><pre>{e(r.get("note", ""))}</pre>'
            if cmd:
                body += f'<div><b>command</b></div><pre>{e(cmd)}</pre>'
            if tb:
                body += f'<div><b>traceback</b></div><pre>{e(tb)}</pre>'
            body += f'<div><b>log tail</b> ({len(tail)} lines)</div><pre>{e(chr(10).join(tail))}</pre>'
        secs = f'{r["seconds"]:.1f} s' if 'seconds' in r else ''
        rows.append(f'<details class="{cls}-row"{" open" if st == "FAIL" else ""}><summary>'
                    f'<span class="chip {cls}">{e(st)}</span><b>{e(name)}</b><span>{secs}</span>'
                    f'<span class="note">{e(r.get("note", ""))}</span></summary><div class="body">{body}</div></details>')
    body = (f'<h1>CI run {e(summary["started"])}</h1>'
            f'<div class="meta">commit {e(summary.get("commit", "?"))} · GPU: {e(str(summary.get("gpu", "?")))} · '
            + ('running' if not summary['finished'] else f'{summary.get("minutes", 0):g} min')
            + f' · <a href="report.txt">report.txt</a> · <a href="../index.html">all runs</a></div>'
            f'<div class="counts">{chips}</div>'
            f'<label class="filter"><input type="checkbox" id="failed-only"> failed only</label>' + ''.join(rows))
    write_atomic(Path(run_dir) / 'report.html', page(f'CI {summary["started"]}', body,
                                                     refresh=not summary['finished'], script=FILTER_JS))


def runs(ci_dir=CI_DIR):
    """Run dirs with a summary.json, newest first."""
    return sorted((d for d in Path(ci_dir).iterdir()
                   if d.is_dir() and not d.is_symlink() and (d / 'summary.json').exists()),
                  key=lambda d: d.name, reverse=True)


def write_index(ci_dir=CI_DIR):
    e = html.escape
    ci_dir = Path(ci_dir)
    all_runs = [(d, load(d)) for d in runs(ci_dir)]
    lines = ['<table><tr><th>run</th><th>commit</th><th>state</th><th>passed</th><th>failed</th><th>skipped</th>'
             '<th>min</th></tr>']
    for d, s in all_runs:
        c = counts(s)
        state = 'running' if not s['finished'] else ('failed' if c['FAIL'] else 'ok')
        cls = 'RUNNING' if not s['finished'] else ('FAIL' if c['FAIL'] else 'PASS')
        lines.append(f'<tr><td><a href="{e(d.name)}/report.html">{e(d.name)}</a></td><td>{e(s.get("commit", "?"))}</td>'
                     f'<td><span class="chip {cls}">{state}</span></td><td>{c["PASS"]}</td><td>{c["FAIL"]}</td>'
                     f'<td>{c["SKIP"]}</td><td>{s.get("minutes", "")}</td></tr>')
    lines.append('</table>')
    recent = all_runs[:HISTORY]
    names = []  # tests in plan order, the newest runs first
    for _, s in recent:
        for t in s['plan']:
            if t['name'] not in names:
                names.append(t['name'])
    grid = ['<div class="scroll"><table><tr><th>test</th>'
            + ''.join(f'<th title="{e(d.name)}">{e(d.name[4:8])}<br>{e(d.name[9:13])}</th>' for d, _ in recent)
            + '</tr>']
    for name in names:
        cells = []
        for d, s in recent:
            st = status_of(s, name) if any(t['name'] == name for t in s['plan']) else ''
            mark = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '–', 'RUNNING': '…', 'PENDING': '·'}.get(st, '')
            note = s['results'].get(name, {}).get('note', '')
            cells.append(f'<td class="cell {st.replace(" ", "")}"><a href="{e(d.name)}/report.html" '
                         f'title="{e(d.name)}: {e(st)} {e(note)}">{mark}</a></td>' if st else '<td></td>')
        grid.append(f'<tr><th>{e(name)}</th>' + ''.join(cells) + '</tr>')
    grid.append('</table></div>')
    running = any(not s['finished'] for _, s in all_runs[:1])
    body = (f'<h1>CI runs</h1><div class="meta">{len(all_runs)} runs in {e(str(ci_dir))} · updated '
            f'{time.strftime("%Y-%m-%d %H:%M:%S")}' + (' · <a href="latest/report.html">latest</a>' if all_runs else '')
            + '</div>' + ''.join(lines) + f'<h2>Last {len(recent)} runs by test</h2>' + ''.join(grid))
    write_atomic(ci_dir / 'index.html', page('CI runs', body, refresh=running))


def link_latest(run_dir, ci_dir=CI_DIR):
    """logs/ci/latest -> run_dir (relative symlink, replaced atomically)."""
    link, tmp = Path(ci_dir) / 'latest', Path(ci_dir) / f'.latest.{os.getpid()}'
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(Path(run_dir).name)
    os.replace(tmp, link)


def write_run(run_dir, summary=None, ci_dir=CI_DIR):
    """report.txt + report.html of a run, then the index (summary: the run's state, else its summary.json)."""
    summary = summary or load(run_dir)
    write_text(run_dir, summary)
    write_html(run_dir, summary)
    write_index(ci_dir)


def main():
    done = runs()
    for d in done:
        write_text(d, load(d))
        write_html(d, load(d))
    write_index()
    if done:
        link_latest(done[0])
    print(f'{len(done)} runs: {CI_DIR}/index.html')


if __name__ == '__main__':
    sys.exit(main())
