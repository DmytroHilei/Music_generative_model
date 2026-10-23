"""
The project to-do list as a page of the web dashboard (tools/status.py --web): /todo?t=<token>, laid out like
Reminders / Microsoft To Do: a sidebar with smart lists (in progress, all open, completed, rejected), the tracks and
the papers; the selected list (open items by priority, completed and rejected ones folded); a details pane that saves
each field on change. On a phone one screen at a time (lists -> list -> details; the view is in the URL hash, so the
back button works). Everything can be edited from the page (add, edit, status, move, delete); the list lives in
context/todo.json, which agents edit too (keep the format). Papers (title, link, one-line idea) are a list of their
own; items name theirs in 'papers' (ids), the details pane links / unlinks them.

    GET  /todo?t=<token>        the page
    GET  /api/todo?t=<token>    the list as JSON
    POST /api/todo?t=<token>    one operation, JSON: {"op": "add" | "update" | "delete" | "move", ...} -> the new list

Operations apply to the file as it is on disk (agent edits in between are kept): add {item}, update {id, fields},
delete {id}, move {id, dir: -1 | 1} (within the item's track and priority); paper_add {paper}, paper_update {id,
fields}, paper_delete {id} (also removed from the items).
"""

import json
import os
import secrets
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TODO = ROOT / 'context/todo.json'
STATUSES = ('todo', 'doing', 'done', 'rejected')
PRIORITIES = ('P0', 'P1', 'P2', 'P3', '')
FIELDS = ('track', 'title', 'priority', 'status', 'cost', 'note', 'evidence')
PAPER_FIELDS = ('title', 'url', 'idea')
_lock = threading.Lock()


def load():
    data = json.loads(TODO.read_text()) if TODO.exists() else {'tracks': [], 'items': []}
    data.setdefault('papers', [])
    return data


def save(data):
    tmp = TODO.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    os.replace(tmp, TODO)


def clean(fields, names=FIELDS):
    """Only known fields, as strings ('papers': a list of ids); status / priority / track / papers checked by
    apply()."""
    out = {k: str(fields[k]).strip() for k in names if k in fields and fields[k] is not None}
    if 'papers' in fields and names is FIELDS:
        out['papers'] = [str(x) for x in (fields['papers'] or []) if str(x)]
    return out


def apply_paper(data, req):
    """paper_add / paper_update / paper_delete."""
    papers = data['papers']
    op = req['op']
    if op == 'paper_add':
        paper = clean(req.get('paper', {}), PAPER_FIELDS)
        paper['id'] = secrets.token_hex(4)
        papers.append(paper)
    else:
        paper = next((x for x in papers if x['id'] == req.get('id')), None)
        if paper is None:
            raise ValueError('no such paper (deleted or changed elsewhere: reload)')
        if op == 'paper_delete':
            papers.remove(paper)
            for it in data['items']:
                if paper['id'] in it.get('papers', []):
                    it['papers'].remove(paper['id'])
            return
        paper.update(clean(req.get('fields', {}), PAPER_FIELDS))
    if not paper.get('title'):
        raise ValueError('title required')
    if paper.get('url') and not paper['url'].startswith(('https://', 'http://')):
        raise ValueError('the link must start with https:// or http://')


def apply(data, req):
    """One edit operation on the list (in place); raises ValueError on a bad request."""
    items, tracks = data['items'], {t['id'] for t in data['tracks']}
    by_id = {it['id']: it for it in items}
    op = req.get('op')
    if str(op).startswith('paper_'):
        return apply_paper(data, req)
    today = time.strftime('%Y-%m-%d')
    if op == 'add':
        it = clean(req.get('item', {}))
        if not it.get('title'):
            raise ValueError('title required')
        it.setdefault('status', 'todo')
        it.setdefault('priority', '')
        it['id'] = secrets.token_hex(4)
        it['order'] = max([x.get('order', 0) for x in items] + [0]) + 1
        it['updated'] = today
        target = it
    elif op in ('update', 'delete', 'move'):
        target = by_id.get(req.get('id'))
        if target is None:
            raise ValueError('no such item (deleted or changed elsewhere: reload)')
        if op == 'delete':
            items.remove(target)
            return
        if op == 'move':
            peers = sorted((x for x in items if x['track'] == target['track']
                            and x.get('priority', '') == target.get('priority', '')
                            and (x['status'] in ('todo', 'doing')) == (target['status'] in ('todo', 'doing'))),
                           key=lambda x: x.get('order', 0))
            i = peers.index(target)
            j = i + (1 if int(req.get('dir', 1)) > 0 else -1)
            if 0 <= j < len(peers):
                target['order'], peers[j]['order'] = peers[j].get('order', 0), target.get('order', 0)
                if target['order'] == peers[j]['order']:  # equal orders (hand-edited file): make them distinct
                    target['order'] += -0.5 if j < i else 0.5
            return
        target.update(clean(req.get('fields', {})))
        target['updated'] = today
    else:
        raise ValueError(f'unknown op {op!r}')
    if target.get('status') not in STATUSES:
        raise ValueError(f'status must be one of {STATUSES}')
    if target.get('priority', '') not in PRIORITIES:
        raise ValueError(f'priority must be one of {PRIORITIES}')
    if target.get('track') not in tracks:
        raise ValueError('unknown track')
    known = {x['id'] for x in data['papers']}
    if any(x not in known for x in target.get('papers', [])):
        raise ValueError('unknown paper')
    if op == 'add':
        items.append(target)


def handle(method, path, body=b''):
    """(status code, content type, body bytes) for /todo and /api/todo, None for other paths (token already
    checked by the caller)."""
    route = path.split('?', 1)[0].rstrip('/')
    if route == '/todo' and method == 'GET':
        return 200, 'text/html; charset=utf-8', PAGE.encode()
    if route != '/api/todo':
        return None
    with _lock:
        data = load()
        if method == 'POST':
            try:
                apply(data, json.loads(body or b'{}'))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return 400, 'application/json', json.dumps({'error': str(e)}).encode()
            save(data)
    return 200, 'application/json', json.dumps(data, ensure_ascii=False).encode()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f2f2f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#1c1c1e" media="(prefers-color-scheme: dark)">
<title>To-do</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#fff;--side:#f2f2f7;--pane:#f2f2f7;--card:#fff;--seg-on:#fff;--fg:#000;--fg2:rgba(60,60,67,.6);
 --fg3:rgba(60,60,67,.3);--sep:rgba(60,60,67,.2);--fill:rgba(118,118,128,.12);--hover:rgba(0,0,0,.035);--ring:#c7c7cc;
 --red:#ff3b30;--blue:#007aff;--grey:#8e8e93;--c0:#007aff;--c1:#ff9500;--c2:#34c759;--c3:#af52de;--c4:#ff2d55;
 --c5:#32ade6;--c6:#ffcc00;--c7:#a2845e;--c8:#5856d6;--c9:#ff3b30;
 --font:-apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",Inter,Roboto,"Helvetica Neue",sans-serif;
 color-scheme:light dark}
@media (prefers-color-scheme:dark){:root{--bg:#000;--side:#1c1c1e;--pane:#1c1c1e;--card:#2c2c2e;--seg-on:#636366;
 --fg:#fff;--fg2:rgba(235,235,245,.6);--fg3:rgba(235,235,245,.3);--sep:rgba(84,84,88,.65);--fill:rgba(118,118,128,.24);
 --hover:rgba(255,255,255,.05);--ring:#48484a;--red:#ff453a;--blue:#0a84ff;--grey:#98989d;--c0:#0a84ff;--c1:#ff9f0a;
 --c2:#30d158;--c3:#bf5af2;--c4:#ff375f;--c5:#64d2ff;--c6:#ffd60a;--c7:#ac8e68;--c8:#5e5ce6;--c9:#ff453a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.35 var(--font);-webkit-font-smoothing:antialiased;
 -webkit-tap-highlight-color:transparent}
a{color:inherit;text-decoration:none}
button,input,select,textarea{font:inherit;color:inherit}
svg{display:block;flex:none}
#shell{display:grid;grid-template-columns:272px minmax(0,1fr);height:100vh}
body.has-sel #shell{grid-template-columns:272px minmax(0,1fr) 384px}
#side{background:var(--side);border-right:.5px solid var(--sep);overflow-y:auto;padding:14px 12px 28px}
#list{overflow-y:auto;padding:30px 44px 72px}
#pane{display:none;background:var(--pane);border-left:.5px solid var(--sep);overflow-y:auto;padding:10px 16px 32px}
body.has-sel #pane{display:block}
.m-only{display:none}

/* sidebar */
.stop{display:flex;justify-content:flex-end;margin:-4px 0 8px}
.runs{display:flex;align-items:center;gap:5px;font-size:13px;color:var(--fg2);padding:4px 8px;border-radius:6px}
.runs:hover{background:var(--fill);color:var(--fg)}
.sw{position:relative;margin-bottom:14px}
.sw svg{position:absolute;left:9px;top:50%;transform:translateY(-50%);color:var(--fg2);font-size:15px;pointer-events:none}
.search{width:100%;border:0;border-radius:9px;background:var(--fill);padding:7px 10px 7px 30px;outline:none;font-size:14px}
.search::placeholder{color:var(--fg2)}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:24px}
.tile{background:var(--card);border-radius:11px;padding:8px 10px 8px 8px;display:grid;grid-template-columns:auto 1fr;row-gap:6px;
 cursor:pointer;box-shadow:0 .5px 1.5px rgba(0,0,0,.07)}
.ti{width:28px;height:28px;border-radius:50%;background:var(--tc);color:#fff;display:grid;place-items:center;font-size:16px}
.tile b{justify-self:end;font-size:22px;font-weight:700;line-height:28px;font-variant-numeric:tabular-nums}
.tl{grid-column:1/-1;color:var(--fg2);font-weight:600;font-size:13px;padding-left:2px}
.tile.on{background:var(--tc);color:#fff}
.tile.on .ti{background:#fff;color:var(--tc)}
.tile.on .tl{color:rgba(255,255,255,.85)}
.scap{font-size:13px;font-weight:600;color:var(--fg2);margin:0 8px 4px}
.nav{margin-bottom:20px}
.nr{display:flex;align-items:center;gap:10px;padding:5px 8px;border-radius:7px;font-size:14px;min-height:34px}
.nr:hover{background:var(--hover)}
.nr.on{background:var(--fill)}
.ni{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;color:#fff;font-size:13px;flex:none}
.nn{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nc{color:var(--fg2);font-size:13px;font-variant-numeric:tabular-nums}
.chev{color:var(--fg3);font-size:16px}

/* list */
#list h1{font-size:30px;font-weight:700;letter-spacing:-.02em;margin:0 0 16px;display:flex;align-items:baseline;gap:12px}
#list h1 .hc{margin-left:auto;font-variant-numeric:tabular-nums}
.add{display:flex;align-items:center;gap:12px;margin:0 -12px 8px;padding:0 13px;height:46px;border-radius:10px;
 background:var(--fill)}
.add svg{color:var(--tc);font-size:20px}
.add input{flex:1;border:0;background:none;outline:none;padding:0;height:100%}
.add input::placeholder{color:var(--tc)}
.add input:focus::placeholder{color:var(--fg3)}
.list{margin:0 -12px}
.row{display:flex;gap:12px;align-items:flex-start;padding:0 12px;border-radius:10px;cursor:default}
.row:hover{background:var(--hover)}
.row.on{background:color-mix(in srgb,var(--tc) 13%,transparent)}
.row > .chk{margin-top:11px}
.body{flex:1;min-width:0;padding:11px 0;border-top:.5px solid var(--sep)}
.row:first-child .body,.row:hover .body,.row:hover + .row .body,.row.on .body,.row.on + .row .body{border-top-color:transparent}
.t{overflow-wrap:anywhere}
.bang{color:var(--tc);font-weight:700;margin-right:4px}
.nt{color:var(--fg2);font-size:13px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:0 6px;color:var(--fg2);font-size:12px;margin-top:3px}
.meta > * + *::before{content:"\2022";margin-right:6px;opacity:.6}
.meta .tk{font-weight:600}
.meta .pp{display:inline-flex;gap:3px;align-items:center}
.row.done .t,.row.rejected .t{color:var(--fg2)}
.row.rejected .t{text-decoration:line-through}
.row.leaving{opacity:.45;transition:opacity .4s .15s}
.pi{width:22px;height:22px;margin-top:11px;color:var(--c7);display:grid;place-items:center;font-size:19px}
h3{font-size:20px;font-weight:700;letter-spacing:-.01em;margin:26px 0 4px}
#list h1 + h3{margin-top:4px}
.fold{margin-top:22px}
.grp{display:inline-flex;align-items:center;gap:6px;border:0;background:var(--fill);border-radius:7px;padding:4px 10px 4px 6px;
 font-size:14px;font-weight:600;cursor:pointer}
.grp svg{transition:transform .15s;font-size:14px}
.grp.open svg{transform:rotate(90deg)}
.gc{color:var(--fg2);font-weight:400}
.fold .list{margin-top:6px}
.empty{color:var(--fg2);padding:40px 0;text-align:center}

/* checkbox */
.chk{flex:none;width:22px;height:22px;border-radius:50%;border:1.5px solid var(--ring);background:none;padding:0;display:grid;
 place-items:center;color:transparent;font-size:14px;cursor:pointer;position:relative;transition:background .2s,border-color .2s}
.chk:hover{color:var(--ring)}
.chk.doing{border-color:var(--tc)}
.chk.doing::after{content:"";position:absolute;inset:3px;border-radius:50%;background:conic-gradient(var(--tc) 0 50%,transparent 0)}
.chk.doing:hover::after{opacity:.35}
.chk.done,.chk.ticking{background:var(--tc);border-color:var(--tc);color:#fff}
.chk.ticking::after{display:none}
.chk.rejected{background:var(--ring);border-color:var(--ring);color:#fff}

/* details pane */
.ptop{display:flex;align-items:center;min-height:36px;margin-bottom:6px}
.ptop .x{margin-left:auto}
.ib{border:0;background:none;width:32px;height:32px;border-radius:8px;display:grid;place-items:center;color:var(--fg2);
 cursor:pointer;font-size:18px}
.ib:hover{background:var(--fill);color:var(--fg)}
.ib.del:hover{color:var(--red)}
.back{align-items:center;gap:1px;color:var(--blue);font-size:17px;margin-left:-8px}
.back svg{font-size:24px}
.card{background:var(--card);border-radius:12px;margin-bottom:14px;overflow:hidden}
.card.head{display:flex;gap:12px;align-items:flex-start;padding:14px}
.card.head .chk,.card.head .ni{margin-top:2px}
textarea{border:0;outline:none;background:none;resize:none;width:100%;padding:12px 14px;display:block;line-height:1.4;
 overflow:hidden;field-sizing:content}
textarea::placeholder,input::placeholder{color:var(--fg3)}
.ttl{padding:0;font-size:19px;font-weight:600;line-height:1.3}
.seg{display:flex;background:var(--fill);border-radius:9px;padding:2px;gap:2px;margin-bottom:14px}
.seg button{flex:1;border:0;background:none;border-radius:7px;padding:6px 4px;font-size:13px;font-weight:500;cursor:pointer;
 white-space:nowrap}
.seg button.on{background:var(--seg-on);box-shadow:0 1px 3px rgba(0,0,0,.12),0 0 0 .5px rgba(0,0,0,.04);font-weight:600}
.seg.small{margin:0 0 0 auto}
.seg.small button{flex:none;padding:3px 8px}
.form .f{display:flex;align-items:center;gap:12px;padding:0 14px;min-height:46px}
.form .f + .f{border-top:.5px solid var(--sep)}
.fl{flex:none}
.form select,.form input{flex:1;min-width:0;border:0;background:none;outline:none;text-align:right;text-align-last:right;
 color:var(--fg2);padding:12px 0;appearance:none;cursor:pointer}
.form input{cursor:text}
.form .open{display:flex;align-items:center;gap:4px;color:var(--blue);font-size:14px}
.cap{font-size:13px;color:var(--fg2);margin:4px 14px 6px}
.plist .pr{display:flex;gap:8px;align-items:flex-start;padding:10px 6px 10px 14px}
.plist .pr + .pr,.ul .ur + .ur{border-top:.5px solid var(--sep)}
.pr .pb{flex:1;min-width:0}
.pr a{color:var(--blue);font-weight:500}
.idea{color:var(--fg2);font-size:13px;margin-top:2px}
.pr.add-p{align-items:center;color:var(--blue);gap:10px;padding:0 14px;font-size:16px}
.pr.add-p select{flex:1;border:0;background:none;outline:none;color:var(--blue);padding:12px 0;appearance:none;cursor:pointer;
 font-size:15px}
.ur{display:flex;align-items:center;gap:10px;padding:11px 14px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--tc);flex:none}
.card.empty{padding:14px;text-align:left;font-size:14px}
.pfoot{display:flex;align-items:center;justify-content:space-between;color:var(--fg2);font-size:12px;margin-top:22px;
 padding-top:8px;border-top:.5px solid var(--sep)}
.pfoot .mv{display:flex}
#err{position:fixed;bottom:16px;left:16px;right:16px;max-width:520px;margin:auto;background:var(--fg);color:var(--bg);
 padding:10px 14px;border-radius:12px;display:none;box-shadow:0 6px 24px rgba(0,0,0,.2);font-size:14px;z-index:9}

/* phone: Lists -> list -> details, one screen at a time */
@media (max-width:899px){
 #shell,body.has-sel #shell{display:block;height:auto}
 #side,#list,body.has-sel #pane{display:none;border:0;overflow:visible;min-height:100vh}
 body.s-home #side,body.s-search #side,body.s-list #list,body.s-search #list,body.s-pane #pane{display:block}
 body.s-home,body.s-pane{background:var(--side)}
 body.s-search #side{min-height:0;padding-bottom:4px}
 body.s-search #side > :not(.sw){display:none}
 #side{padding:12px 16px 40px}
 #list{padding:6px 16px 56px}
 #pane{padding:6px 16px 40px}
 .m-only{display:flex}
 .d-only{display:none}
 .mtop{min-height:40px;align-items:center}
 #list h1{font-size:32px}
 .tile{padding:10px 12px 9px 10px}
 .tile b{font-size:26px}
 .scap{font-size:20px;font-weight:700;color:var(--fg);margin:0 4px 8px}
 .nav{background:var(--card);border-radius:12px;padding:0}
 .nr{border-radius:0;padding:8px 14px;gap:12px;font-size:16px;min-height:46px}
 .nr + .nr{background:linear-gradient(var(--sep),var(--sep)) right top/calc(100% - 56px) .5px no-repeat}
 .ni{width:30px;height:30px;font-size:16px}
 .nr.on{background:none}
 input,textarea,select{font-size:16px}
}
</style></head>
<body>
<div id="shell"><aside id="side"></aside><main id="list"></main><section id="pane"></section></div>
<div id="err"></div>
<script>
const T = new URLSearchParams(location.search).get('t') || '';
const API = '/api/todo?t=' + encodeURIComponent(T);
const PRI = ['P0', 'P1', 'P2', 'P3', ''];
const BANG = {P0: '!!!', P1: '!!', P2: '!'};
const ST = [['todo', 'To do'], ['doing', 'In progress'], ['done', 'Done'], ['rejected', 'Rejected']];
const S = (d, w = 2) => `<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor"
  stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
const I = {
  check: S('<path d="M5.5 12.5l4 4L18.5 7.5"/>', 2.6), x: S('<path d="M7.5 7.5l9 9M16.5 7.5l-9 9"/>', 2.6),
  clock: S('<circle cx="12" cy="12" r="7.5"/><path d="M12 8v4.2l2.6 1.8"/>', 2.2),
  tray: S('<path d="M4 13.5L6.5 6h11l2.5 7.5V18H4z"/><path d="M4 13.5h4.5a3.5 3.5 0 0 0 7 0H20"/>'),
  list: S('<path d="M9.5 7.5h8M9.5 12h8M9.5 16.5h8"/><path d="M6 7.5h.01M6 12h.01M6 16.5h.01"/>', 2.4),
  book: S('<path d="M5.5 5a1.5 1.5 0 0 1 1.5-1.5h10.5v14H7A1.5 1.5 0 0 0 5.5 19z"/><path d="M5.5 19A1.5 1.5 0 0 0 7 20.5h10.5"/>'),
  doc: S('<path d="M7 3.5h6.5l4 4v13H7z"/><path d="M13.5 3.5V7.5h4"/>', 1.8),
  search: S('<circle cx="11" cy="11" r="6"/><path d="M19.5 19.5l-4.2-4.2"/>', 2.2),
  right: S('<path d="M9.5 6l6 6-6 6"/>', 2.2), left: S('<path d="M15 5.5L8.5 12l6.5 6.5"/>', 2.4),
  plus: S('<path d="M12 5.5v13M5.5 12h13"/>', 2.2), up: S('<path d="M12 18.5v-13M6.5 11L12 5.5l5.5 5.5"/>'),
  down: S('<path d="M12 5.5v13M6.5 13l5.5 5.5 5.5-5.5"/>'), close: S('<path d="M7 7l10 10M17 7L7 17"/>'),
  trash: S('<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13M10 11v5M14 11v5"/>', 1.7),
  out: S('<path d="M13.5 5H19v5.5M19 5l-8 8"/><path d="M17 14v5H5V7h5"/>', 1.8),
  pulse: S('<path d="M3 12h4l2.5-6 5 12 2.5-6h4"/>'),
};
const SMART = {doing: ['In progress', 'c0', I.clock], open: ['All open', 'grey', I.tray],
  done: ['Completed', 'c2', I.check], rejected: ['Rejected', 'grey', I.x]};
const wide = matchMedia('(min-width: 900px)');
let data = null, q = '';
const expanded = new Set();

const $ = id => document.getElementById(id);
const svg = s => { const t = document.createElement('template'); t.innerHTML = s.trim(); return t.content.firstChild; };
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) e.append(k);
  return e;
};
const safe = u => /^https?:\/\//.test(u || '') ? u : undefined;
const paper = id => (data.papers || []).find(p => p.id === id);
const live = it => it.status === 'todo' || it.status === 'doing';
const rank = it => PRI.indexOf(it.priority || '') * 1e6 + (it.order || 0);
const recent = (a, b) => (b.updated || '').localeCompare(a.updated || '');
const tIndex = id => data.tracks.findIndex(t => t.id === id);
const tcol = id => `var(--c${Math.max(0, tIndex(id)) % 10})`;
const trackName = id => (data.tracks.find(t => t.id === id) || {name: id}).name;
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const fit = t => { t.style.height = 'auto'; t.style.height = t.scrollHeight + 'px'; };

function err(msg) {
  const b = $('err'); b.textContent = msg; b.style.display = 'block';
  clearTimeout(err.t); err.t = setTimeout(() => b.style.display = 'none', 5000);
}
async function load() {
  try {
    const r = await fetch(API); if (!r.ok) { err('Could not load the list (' + r.status + ')'); return; }
    data = await r.json();
  } catch (e) { err('Could not load the list: ' + e.message); return; }
  render();
}
// pane: false = an autosave from the details pane, which keeps its fields (and the cursor) as they are
async function op(body, {pane = true} = {}) {
  let r, j;
  try {
    r = await fetch(API, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    j = await r.json();
  } catch (e) { err('Not saved: ' + e.message); return false; }
  if (!r.ok) { err(j.error || 'Not saved (' + r.status + ')'); await load(); return false; }
  data = j;
  if (pane) render(); else { const rt = route(); renderSide(rt); renderList(rt); }
  return true;
}

// location.hash = '<view>[/i:<item id> | /p:<paper id>]', view = track:<id> | doing | open | done | rejected | papers
function route() {
  const [v, s] = location.hash.slice(1).split('/');
  const view = v || (wide.matches && data.tracks.length ? 'track:' + data.tracks[0].id : '');
  return {view, sel: s ? {kind: s[0] === 'p' ? 'paper' : 'item', id: s.slice(2)} : null};
}
function go(view, sel) {
  const h = '#' + (view || '') + (sel ? '/' + sel.kind[0] + ':' + sel.id : '');
  if (location.hash === h || (h === '#' && !location.hash)) render(); else location.hash = h;
}
function meta(view) {
  if (view.startsWith('track:')) {
    const id = view.slice(6);
    return tIndex(id) < 0 ? null : {title: trackName(id), color: tcol(id), track: id};
  }
  if (view === 'papers') return {title: 'Papers', color: 'var(--c7)'};
  const s = SMART[view];
  return s ? {title: s[0], color: `var(--${s[1]})`} : null;
}

function render() {
  if (!data) return;
  const a = document.activeElement, k = a && a.dataset && a.dataset.k;
  const r = route();
  renderSide(r); renderList(r); renderPane(r); stage(r);
  document.querySelectorAll('textarea').forEach(fit);
  if (k) { const b = document.querySelector(`[data-k="${k}"]`); if (b && b !== a) b.focus(); }
}

// which screen a phone shows: s-home (lists), s-list, s-search (search box over the results) or s-pane (details)
function stage(r) {
  document.body.className = 's-' + (r.sel ? 'pane' : q ? 'search' : r.view ? 'list' : 'home') + (r.sel ? ' has-sel' : '');
}
function renderSide(r) {
  const side = $('side'), items = data.items; side.textContent = '';
  const counts = {doing: items.filter(i => i.status === 'doing').length, open: items.filter(live).length,
    done: items.filter(i => i.status === 'done').length, rejected: items.filter(i => i.status === 'rejected').length};
  const search = el('input', {class: 'search', type: 'search', placeholder: 'Search', 'data-k': 'q', value: q,
    oninput: e => { q = e.target.value.trim(); const rt = route(); renderList(rt); stage(rt); }});
  const nav = (view, name, color, icon, n) => el('a', {class: 'nr' + (r.view === view && !q ? ' on' : ''), href: '#' + view,
    onclick: e => { e.preventDefault(); q = ''; go(view); }},
    el('span', {class: 'ni', style: 'background:' + color}, svg(icon)), el('span', {class: 'nn'}, name),
    el('span', {class: 'nc'}, n), el('span', {class: 'chev m-only'}, svg(I.right)));
  side.append(
    el('div', {class: 'stop'}, el('a', {class: 'runs', href: '/?t=' + encodeURIComponent(T)}, svg(I.pulse), 'Runs')),
    el('div', {class: 'sw'}, svg(I.search), search),
    el('div', {class: 'tiles'}, ...Object.entries(SMART).map(([k, [name, c, icon]]) =>
      el('a', {class: 'tile' + (r.view === k && !q ? ' on' : ''), href: '#' + k, style: `--tc:var(--${c})`,
        onclick: e => { e.preventDefault(); q = ''; go(k); }},
        el('span', {class: 'ti'}, svg(icon)), el('b', {}, counts[k]), el('span', {class: 'tl'}, name)))),
    el('div', {class: 'scap'}, 'Tracks'),
    el('nav', {class: 'nav'}, ...data.tracks.map(t => nav('track:' + t.id, t.name, tcol(t.id), I.list,
      items.filter(i => i.track === t.id && live(i)).length))),
    el('nav', {class: 'nav'}, nav('papers', 'Papers', 'var(--c7)', I.book, (data.papers || []).length)));
}

function chk(it) {
  return el('button', {class: 'chk ' + it.status, title: live(it) ? 'Mark done' : 'Reopen', onclick: e => {
    e.stopPropagation();
    if (!live(it)) { op({op: 'update', id: it.id, fields: {status: 'todo'}}); return; }
    e.currentTarget.classList.add('ticking');
    const row = e.currentTarget.closest('.row'); if (row) row.classList.add('leaving');
    setTimeout(() => op({op: 'update', id: it.id, fields: {status: 'done'}}), 450);
  }}, svg(it.status === 'rejected' ? I.x : I.check));
}
function itemRow(it, r, showTrack) {
  const on = r.sel && r.sel.kind === 'item' && r.sel.id === it.id;
  const info = [
    showTrack ? el('span', {class: 'tk', style: 'color:' + tcol(it.track)}, trackName(it.track)) : null,
    it.cost ? el('span', {}, it.cost) : null,
    (it.papers || []).length ? el('span', {class: 'pp'}, svg(I.doc), it.papers.length) : null,
    !live(it) && it.updated ? el('span', {}, it.updated) : null].filter(Boolean);
  return el('div', {class: 'row ' + it.status + (on ? ' on' : ''), style: '--tc:' + tcol(it.track),
    onclick: () => go(r.view, {kind: 'item', id: it.id})},
    chk(it),
    el('div', {class: 'body'},
      el('div', {class: 't'}, BANG[it.priority] ? el('span', {class: 'bang'}, BANG[it.priority]) : null, it.title),
      it.note ? el('div', {class: 'nt'}, it.note) : null,
      info.length ? el('div', {class: 'meta'}, ...info) : null));
}
function paperRow(p, r) {
  const on = r.sel && r.sel.kind === 'paper' && r.sel.id === p.id;
  const n = data.items.filter(i => (i.papers || []).includes(p.id)).length;
  let host = ''; try { host = new URL(p.url).hostname.replace(/^www\./, ''); } catch (e) {}
  return el('div', {class: 'row' + (on ? ' on' : ''), style: '--tc:var(--c7)',
    onclick: () => go(r.view, {kind: 'paper', id: p.id})},
    el('span', {class: 'pi'}, svg(I.doc)),
    el('div', {class: 'body'}, el('div', {class: 't'}, p.title), p.idea ? el('div', {class: 'nt'}, p.idea) : null,
      el('div', {class: 'meta'}, el('span', {}, n ? 'Used by ' + plural(n, 'item') : 'Not linked'),
        host ? el('span', {}, host) : null)));
}
const listOf = (rows) => el('div', {class: 'list'}, ...rows);
function byTrack(its, r, sort) {
  const out = [];
  for (const t of data.tracks) {
    const g = its.filter(i => i.track === t.id).sort(sort);
    if (g.length) out.push(el('h3', {style: 'color:' + tcol(t.id)}, t.name), listOf(g.map(i => itemRow(i, r, false))));
  }
  return out;
}
function fold(key, label, rows) {
  if (!rows.length) return null;
  const open = expanded.has(key);
  return el('div', {class: 'fold'},
    el('button', {class: 'grp' + (open ? ' open' : ''),
      onclick: () => { open ? expanded.delete(key) : expanded.add(key); renderList(route()); }},
      svg(I.right), label, el('span', {class: 'gc'}, rows.length)),
    open ? listOf(rows) : null);
}
function addRow(placeholder, color, onAdd) {
  const inp = el('input', {placeholder, 'data-k': 'add', enterkeyhint: 'done'});
  return el('form', {class: 'add', style: '--tc:' + color, onsubmit: e => {
    e.preventDefault(); const v = inp.value.trim(); if (v) onAdd(v);
  }}, svg(I.plus), inp);
}

function renderList(r) {
  const main = $('list'), items = data.items; main.textContent = '';
  const m = q ? {title: 'Results', color: 'var(--fg)'} : meta(r.view);
  main.append(el('div', {class: 'mtop m-only'}, el('a', {class: 'back m-only', href: '#',
    onclick: e => { e.preventDefault(); q = ''; go(''); }}, svg(I.left), 'Lists')));
  if (!m) return;
  const h1 = el('h1', {style: 'color:' + m.color}, m.title);
  main.append(h1);
  let body = [], n = 0;
  if (q) {
    const s = q.toLowerCase(), has = (...f) => f.some(x => (x || '').toLowerCase().includes(s));
    const its = items.filter(i => has(i.title, i.note, i.evidence, i.cost));
    const ps = (data.papers || []).filter(p => has(p.title, p.idea, p.url));
    body = byTrack(its, r, (a, b) => live(b) - live(a) || rank(a) - rank(b));
    if (ps.length) body.push(el('h3', {style: 'color:var(--c7)'}, 'Papers'), listOf(ps.map(p => paperRow(p, r))));
    n = its.length + ps.length;
    if (!n) body.push(el('div', {class: 'empty'}, 'No results for “' + q + '”'));
  } else if (m.track) {
    const its = items.filter(i => i.track === m.track);
    const open = its.filter(live).sort((a, b) => rank(a) - rank(b));
    n = open.length;
    body.push(addRow('Add a task', m.color, v => {
      op({op: 'add', item: {title: v, track: m.track, status: 'todo'}});
    }));
    body.push(open.length ? listOf(open.map(i => itemRow(i, r, false))) : el('div', {class: 'empty'}, 'All done'));
    body.push(fold(r.view + ':done', 'Completed', its.filter(i => i.status === 'done').sort(recent).map(i => itemRow(i, r, false))),
      fold(r.view + ':rejected', 'Rejected', its.filter(i => i.status === 'rejected').sort(recent).map(i => itemRow(i, r, false))));
  } else if (r.view === 'papers') {
    const ps = [...(data.papers || [])].sort((a, b) => a.title.localeCompare(b.title));
    n = ps.length;
    body.push(addRow('Add a paper', m.color, v => {
      const before = new Set((data.papers || []).map(p => p.id));
      op({op: 'paper_add', paper: {title: v}}).then(ok => {
        const p = ok && data.papers.find(x => !before.has(x.id)); if (p) go('papers', {kind: 'paper', id: p.id});
      });
    }), listOf(ps.map(p => paperRow(p, r))));
  } else {
    const pick = {doing: i => i.status === 'doing', open: live, done: i => i.status === 'done',
      rejected: i => i.status === 'rejected'}[r.view];
    const its = items.filter(pick);
    n = its.length;
    body = its.length ? byTrack(its, r, live(its[0]) ? (a, b) => rank(a) - rank(b) : recent)
      : [el('div', {class: 'empty'}, 'Nothing here')];
  }
  h1.append(el('span', {class: 'hc'}, n));
  main.append(...body.filter(Boolean));
}

function iconBtn(icon, title, onclick, cls = '') {
  return el('button', {class: 'ib ' + cls, title, 'aria-label': title, onclick}, svg(icon));
}
function area(k, v, placeholder, save, cls) {
  const t = el('textarea', {class: cls, 'data-k': k, placeholder, rows: 1});
  t.value = v || '';
  t.addEventListener('input', () => fit(t));
  t.addEventListener('change', () => {
    const x = t.value.trim();
    if (k === 'title' && !x) { t.value = v; fit(t); return; }
    if (x !== (v || '')) { v = x; save(x); }
  });
  if (k === 'title') t.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); t.blur(); } });
  return t;
}
function input(k, v, placeholder, save) {
  const i = el('input', {'data-k': k, placeholder, value: v || ''});
  i.addEventListener('change', () => { const x = i.value.trim(); if (x !== (v || '')) { v = x; save(x); } });
  i.addEventListener('keydown', e => { if (e.key === 'Enter') i.blur(); });
  return i;
}
function seg(opts, cur, pick, cls = '') {
  return el('div', {class: 'seg ' + cls}, ...opts.map(([v, label]) =>
    el('button', {type: 'button', class: v === cur ? 'on' : '', onclick: () => v !== cur && pick(v)}, label)));
}
const field = (label, ctrl, tag = 'label') => el(tag, {class: 'f'}, el('span', {class: 'fl'}, label), ctrl);
function paneTop(r) {
  const back = q ? 'Results' : (meta(r.view) || {title: 'Back'}).title;
  return el('div', {class: 'ptop'},
    el('a', {class: 'back m-only', href: '#' + r.view, onclick: e => { e.preventDefault(); go(r.view); }}, svg(I.left), back),
    iconBtn(I.close, 'Close', () => go(r.view), 'x d-only'));
}

function renderPane(r) {
  const pane = $('pane'); pane.textContent = '';
  if (!r.sel) return;
  if (r.sel.kind === 'paper') return paperPane(pane, r);
  const it = data.items.find(i => i.id === r.sel.id);
  if (!it) return go(r.view);
  const set = fields => op({op: 'update', id: it.id, fields});
  const save = fields => op({op: 'update', id: it.id, fields}, {pane: false});
  const ids = it.papers || [];
  const papers = el('div', {class: 'card plist'});
  for (const id of ids) {
    const p = paper(id); if (!p) continue;
    papers.append(el('div', {class: 'pr'},
      el('div', {class: 'pb'}, el('a', {href: safe(p.url), target: '_blank', rel: 'noopener'}, p.title),
        p.idea ? el('div', {class: 'idea'}, p.idea) : null),
      iconBtn(I.close, 'Unlink', () => set({papers: ids.filter(x => x !== id)}))));
  }
  const rest = (data.papers || []).filter(p => !ids.includes(p.id)).sort((a, b) => a.title.localeCompare(b.title));
  papers.append(el('label', {class: 'pr add-p'}, svg(I.plus),
    el('select', {onchange: e => e.target.value && set({papers: [...ids, e.target.value]})},
      el('option', {value: ''}, 'Link a paper'), ...rest.map(p => el('option', {value: p.id}, p.title)))));
  const track = el('select', {onchange: e => set({track: e.target.value})},
    ...data.tracks.map(t => el('option', {value: t.id, selected: t.id === it.track}, t.name)));
  pane.style.setProperty('--tc', tcol(it.track));
  pane.append(paneTop(r),
    el('div', {class: 'card head'}, chk(it), area('title', it.title, 'Title', v => save({title: v}), 'ttl')),
    seg(ST, it.status, s => set({status: s})),
    el('div', {class: 'card form'},
      field('Track', track),
      field('Priority', seg(PRI.map(p => [p, p || 'None']), it.priority || '', p => set({priority: p}), 'small'), 'div'),
      field('Cost', input('cost', it.cost, 'e.g. ~2 h, €5', v => save({cost: v})))),
    el('div', {class: 'card'}, area('note', it.note, 'Add note', v => save({note: v}))),
    el('div', {class: 'cap'}, 'Evidence'),
    el('div', {class: 'card'}, area('evidence', it.evidence, 'Source, result, numbers', v => save({evidence: v}))),
    el('div', {class: 'cap'}, 'Papers'), papers,
    el('div', {class: 'pfoot'},
      el('span', {class: 'mv'}, iconBtn(I.up, 'Move up', () => op({op: 'move', id: it.id, dir: -1})),
        iconBtn(I.down, 'Move down', () => op({op: 'move', id: it.id, dir: 1}))),
      el('span', {}, 'Updated ' + (it.updated || '?')),
      iconBtn(I.trash, 'Delete', () => confirm('Delete “' + it.title + '”?') &&
        op({op: 'delete', id: it.id}).then(ok => ok && go(r.view)), 'del')));
}
function paperPane(pane, r) {
  const p = paper(r.sel.id);
  if (!p) return go(r.view);
  const save = fields => op({op: 'paper_update', id: p.id, fields}, {pane: false});
  const users = data.items.filter(i => (i.papers || []).includes(p.id));
  pane.style.setProperty('--tc', 'var(--c7)');
  pane.append(paneTop(r),
    el('div', {class: 'card head'}, el('span', {class: 'ni', style: 'background:var(--c7)'}, svg(I.book)),
      area('title', p.title, 'Title', v => save({title: v}), 'ttl')),
    el('div', {class: 'card form'},
      field('Link', input('url', p.url, 'https://arxiv.org/abs/…', v => op({op: 'paper_update', id: p.id,
        fields: {url: v}}))),
      safe(p.url) ? el('div', {class: 'f'}, el('a', {class: 'open', href: p.url, target: '_blank', rel: 'noopener'},
        'Open', svg(I.out))) : null),
    el('div', {class: 'cap'}, 'Idea'),
    el('div', {class: 'card'}, area('idea', p.idea, 'The idea in a sentence or two', v => save({idea: v}))),
    el('div', {class: 'cap'}, 'Used by'),
    users.length ? el('div', {class: 'card ul'}, ...users.map(i => el('a', {class: 'ur', style: '--tc:' + tcol(i.track),
      href: '#' + r.view + '/i:' + i.id}, el('span', {class: 'dot'}), i.title)))
      : el('div', {class: 'card empty'}, 'Not linked to any item yet'),
    el('div', {class: 'pfoot'}, el('span'), iconBtn(I.trash, 'Delete paper', () =>
      confirm('Delete paper “' + p.title + '”?') &&
        op({op: 'paper_delete', id: p.id}).then(ok => ok && go(r.view)), 'del')));
}

addEventListener('hashchange', render);
wide.addEventListener('change', render);
const refit = () => document.querySelectorAll('textarea').forEach(fit);
addEventListener('resize', refit);
addEventListener('load', refit);
if (document.fonts) { document.fonts.ready.then(refit); document.fonts.addEventListener('loadingdone', refit); }
addEventListener('keydown', e => {
  if (e.key !== 'Escape' || !data) return;
  const r = route(); if (r.sel) { document.activeElement.blur(); go(r.view); }
});
load();
setInterval(() => {
  const a = document.activeElement;
  if (document.visibilityState === 'visible' && !(a && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName))) load();
}, 30000);
</script></body></html>
"""
