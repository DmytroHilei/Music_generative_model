"""
The project to-do list as a page of the web dashboard (tools/status.py --web): /todo?t=<token>, grouped by track,
open items by priority, done and rejected items in collapsible lists. Everything can be edited from the page (add,
edit, status, move, delete); the list lives in context/todo.json, which agents edit too (keep the format). Papers
(title, link, one-line idea) are a list of their own; items name theirs in 'papers' (ids), the page shows them under
the item and all of them in a 'Papers' section.

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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project To-do</title>
<style>
:root{--bg:#fafaf9;--card:#fff;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;
 --p0:#dc2626;--p1:#d97706;--p2:#0891b2;--p3:#78716c;--done:#16a34a;--rej:#a8a29e}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--card:#1c1917;--fg:#e7e5e4;--muted:#a8a29e;--line:#292524;
 --accent:#60a5fa;--p0:#f87171;--p1:#fbbf24;--p2:#22d3ee;--p3:#a8a29e;--done:#4ade80;--rej:#78716c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;
 display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;z-index:2}
header h1{font-size:17px;margin:0}
header a{color:var(--accent);text-decoration:none;font-size:14px}
#sum{color:var(--muted);font-size:13px;margin-left:auto}
main{max-width:820px;margin:0 auto;padding:8px 16px 48px}
section{margin-top:18px}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 6px;
 display:flex;align-items:center;gap:8px}
h2 .n{font-weight:400;text-transform:none;letter-spacing:0}
h2 button{margin-left:auto}
.list{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.row{display:flex;gap:8px;align-items:baseline;padding:8px 12px;border-top:1px solid var(--line);cursor:pointer}
.row:first-child{border-top:0}
.row:hover{background:color-mix(in srgb,var(--line) 40%,transparent)}
.pri{font:600 11px ui-monospace,monospace;min-width:24px;color:var(--p3)}
.pri.P0{color:var(--p0)}.pri.P1{color:var(--p1)}.pri.P2{color:var(--p2)}
.dot{width:9px;height:9px;border-radius:50%;border:1.5px solid var(--muted);flex:none;transform:translateY(-1px)}
.dot.doing{background:var(--accent);border-color:var(--accent)}
.dot.done{background:var(--done);border-color:var(--done)}
.dot.rejected{background:var(--rej);border-color:var(--rej)}
.title{flex:1;min-width:0}
.rejected .title{text-decoration:line-through;color:var(--muted)}
.cost{color:var(--muted);font-size:12px;white-space:nowrap}
.tag{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-right:4px}
.det{padding:4px 12px 12px 44px;font-size:14px;border-top:1px dashed var(--line)}
.det p{margin:4px 0;white-space:pre-wrap}
.det .k{color:var(--muted);font-size:12px}
.bar{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
button{font:inherit;font-size:13px;padding:3px 9px;border:1px solid var(--line);border-radius:6px;background:var(--card);
 color:var(--fg);cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button.del{color:var(--p0)}
details{margin-top:18px}
summary{cursor:pointer;color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.06em}
details .list{margin-top:6px}
form{display:grid;gap:6px;margin-top:8px}
form .two{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
input,select,textarea{font:inherit;font-size:14px;padding:5px 7px;border:1px solid var(--line);border-radius:6px;
 background:var(--bg);color:var(--fg);width:100%}
textarea{min-height:60px;resize:vertical}
#err{position:fixed;bottom:12px;left:16px;right:16px;max-width:620px;margin:auto;background:var(--p0);color:#fff;
 padding:8px 12px;border-radius:8px;display:none}
.empty{padding:8px 12px;color:var(--muted);font-size:13px}
.refs{margin:6px 0 2px;padding:0;list-style:none}
.refs li{margin:4px 0}
a.paper{color:var(--accent);text-decoration:none;font-weight:500}
.idea{color:var(--muted);font-size:13px}
.np{font-size:11px;color:var(--muted)}
select[multiple]{min-height:120px}
.prow{padding:8px 12px;border-top:1px solid var(--line)}
.prow:first-child{border-top:0}
.prow .bar{margin-top:4px}
</style></head>
<body>
<header><h1>Project to-do</h1><a id="runs" href="#">Runs dashboard</a><span id="sum"></span></header>
<main id="app">Loading…</main>
<div id="err"></div>
<script>
const T = new URLSearchParams(location.search).get('t') || '';
const API = '/api/todo?t=' + encodeURIComponent(T);
document.getElementById('runs').href = '/?t=' + encodeURIComponent(T);
const PRI = ['P0', 'P1', 'P2', 'P3', ''];
const ST = ['todo', 'doing', 'done', 'rejected'];
let data = null, open = new Set(), editing = null, adding = null, pEditing = null, pAdding = false;
const safe = u => /^https?:\/\//.test(u || '') ? u : undefined;
const paper = id => (data.papers || []).find(p => p.id === id);
function refs(ids) {
  const ul = el('ul', {class: 'refs'});
  for (const id of ids || []) {
    const p = paper(id); if (!p) continue;
    ul.append(el('li', {}, el('a', {class: 'paper', href: safe(p.url), target: '_blank', rel: 'noopener'}, p.title),
      p.idea ? el('div', {class: 'idea'}, p.idea) : null));
  }
  return ul;
}
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== undefined) e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined) e.append(k);
  return e;
};
function err(msg) {
  const b = document.getElementById('err'); b.textContent = msg; b.style.display = 'block';
  setTimeout(() => b.style.display = 'none', 5000);
}
async function load() {
  const r = await fetch(API); if (!r.ok) { err('load failed: ' + r.status); return; }
  data = await r.json(); render();
}
async function op(body) {
  const r = await fetch(API, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) { err(j.error || r.status); await load(); return false; }
  data = j; render(); return true;
}
const rank = it => PRI.indexOf(it.priority || '') * 1e6 + (it.order || 0);
const trackName = id => (data.tracks.find(t => t.id === id) || {name: id}).name;
function form(it, onSave, onCancel) {
  const f = el('form', {onsubmit: e => { e.preventDefault(); const fd = new FormData(f);
    onSave({...Object.fromEntries(fd), papers: fd.getAll('papers')}); }});
  const sel = (name, opts, val) => el('select', {name}, ...opts.map(([v, l]) => {
    const o = el('option', {value: v}, l); if (v === (val || '')) o.selected = true; return o; }));
  f.append(
    el('input', {name: 'title', placeholder: 'Title', value: it.title || '', required: ''}),
    el('div', {class: 'two'},
      sel('track', data.tracks.map(t => [t.id, t.name]), it.track),
      sel('priority', PRI.map(p => [p, p || '—']), it.priority),
      sel('status', ST.map(s => [s, s]), it.status || 'todo')),
    el('input', {name: 'cost', placeholder: 'Cost (e.g. ~2 h, CPU, €5)', value: it.cost || ''}),
    el('textarea', {name: 'note', placeholder: 'Note / plan'}, it.note || ''),
    el('input', {name: 'evidence', placeholder: 'Evidence / source / result', value: it.evidence || ''}),
    el('label', {class: 'k'}, 'Papers (Ctrl / long-press for several)'),
    el('select', {name: 'papers', multiple: ''}, ...(data.papers || []).map(p => {
      const o = el('option', {value: p.id}, p.title); if ((it.papers || []).includes(p.id)) o.selected = true;
      return o; })),
    el('div', {class: 'bar'}, el('button', {type: 'submit', class: 'on'}, 'Save'),
      el('button', {type: 'button', onclick: onCancel}, 'Cancel')));
  return f;
}
function row(it, showTrack) {
  const wrap = el('div', {class: it.status});
  const r = el('div', {class: 'row', onclick: () => { open.has(it.id) ? open.delete(it.id) : open.add(it.id);
    editing = null; render(); }},
    el('span', {class: 'pri ' + (it.priority || '')}, it.priority || '·'),
    el('span', {class: 'dot ' + it.status, title: it.status}),
    el('span', {class: 'title'}, showTrack ? el('span', {class: 'tag'}, trackName(it.track)) : null, it.title),
    (it.papers || []).length ? el('span', {class: 'np', title: 'papers'}, '📄' + it.papers.length) : null,
    el('span', {class: 'cost'}, it.cost || ''));
  wrap.append(r);
  if (!open.has(it.id)) return wrap;
  const d = el('div', {class: 'det'});
  if (editing === it.id) {
    d.append(form(it, f => op({op: 'update', id: it.id, fields: f}).then(ok => { if (ok) editing = null; render(); }),
      () => { editing = null; render(); }));
  } else {
    if (it.note) d.append(el('p', {}, it.note));
    if (it.evidence) d.append(el('p', {}, el('span', {class: 'k'}, 'Evidence: '), it.evidence));
    if ((it.papers || []).length) d.append(refs(it.papers));
    d.append(el('p', {class: 'k'}, 'Updated ' + (it.updated || '?')));
    d.append(el('div', {class: 'bar'},
      ...ST.map(s => el('button', {class: s === it.status ? 'on' : '',
        onclick: () => op({op: 'update', id: it.id, fields: {status: s}})}, s)),
      el('button', {onclick: () => op({op: 'move', id: it.id, dir: -1})}, '↑'),
      el('button', {onclick: () => op({op: 'move', id: it.id, dir: 1})}, '↓'),
      el('button', {onclick: () => { editing = it.id; render(); }}, 'Edit'),
      el('button', {class: 'del', onclick: () => confirm('Delete "' + it.title + '"?') &&
        op({op: 'delete', id: it.id}).then(() => open.delete(it.id))}, 'Delete')));
  }
  wrap.append(d);
  return wrap;
}
function render() {
  const app = document.getElementById('app'), keep = [...app.querySelectorAll('details')].map(d => d.open);
  app.textContent = '';
  const items = data.items, live = it => it.status === 'todo' || it.status === 'doing';
  const nOpen = items.filter(live).length, nDoing = items.filter(i => i.status === 'doing').length;
  document.getElementById('sum').textContent =
    `${nOpen} open · ${nDoing} doing · ${items.filter(i => i.status === 'done').length} done`;
  for (const t of data.tracks) {
    const its = items.filter(i => i.track === t.id && live(i)).sort((a, b) => rank(a) - rank(b));
    const doing = its.filter(i => i.status === 'doing').length;
    const sec = el('section', {}, el('h2', {}, t.name,
      el('span', {class: 'n'}, `${its.length} open` + (doing ? ` · ${doing} doing` : '')),
      el('button', {onclick: () => { adding = adding === t.id ? null : t.id; render(); }}, '+ Add')));
    const list = el('div', {class: 'list'});
    if (adding === t.id) list.append(el('div', {class: 'det'}, form({track: t.id, status: 'todo'},
      f => op({op: 'add', item: f}).then(ok => { if (ok) adding = null; render(); }),
      () => { adding = null; render(); })));
    its.forEach(i => list.append(row(i, false)));
    if (!its.length && adding !== t.id) list.append(el('div', {class: 'empty'}, 'Nothing open'));
    sec.append(list); app.append(sec);
  }
  ['done', 'rejected'].forEach((s, k) => {
    const its = items.filter(i => i.status === s)
      .sort((a, b) => data.tracks.findIndex(t => t.id === a.track) - data.tracks.findIndex(t => t.id === b.track)
        || (b.updated || '').localeCompare(a.updated || ''));
    const det = el('details', {}, el('summary', {}, `${s === 'done' ? 'Resolved' : 'Rejected'} (${its.length})`));
    if (keep[k]) det.open = true;
    const list = el('div', {class: 'list'}); its.forEach(i => list.append(row(i, true)));
    det.append(list); app.append(det);
  });
  app.append(papersSection(keep[2]));
}
function paperForm(p, onSave, onCancel) {
  const f = el('form', {onsubmit: e => { e.preventDefault(); onSave(Object.fromEntries(new FormData(f))); }});
  f.append(el('input', {name: 'title', placeholder: 'Title', value: p.title || '', required: ''}),
    el('input', {name: 'url', placeholder: 'https://arxiv.org/abs/…', value: p.url || ''}),
    el('textarea', {name: 'idea', placeholder: 'The idea in a sentence or two'}, p.idea || ''),
    el('div', {class: 'bar'}, el('button', {type: 'submit', class: 'on'}, 'Save'),
      el('button', {type: 'button', onclick: onCancel}, 'Cancel')));
  return f;
}
function papersSection(wasOpen) {
  const ps = [...(data.papers || [])].sort((a, b) => a.title.localeCompare(b.title));
  const det = el('details', {}, el('summary', {}, `Papers (${ps.length})`));
  if (wasOpen) det.open = true;
  const list = el('div', {class: 'list'});
  const add = el('div', {class: 'prow'});
  if (pAdding) add.append(paperForm({}, f => op({op: 'paper_add', paper: f}).then(ok => { if (ok) pAdding = false;
    render(); }), () => { pAdding = false; render(); }));
  else add.append(el('button', {onclick: () => { pAdding = true; render(); }}, '+ Add paper'));
  list.append(add);
  for (const p of ps) {
    const users = data.items.filter(i => (i.papers || []).includes(p.id));
    const r = el('div', {class: 'prow'});
    if (pEditing === p.id) {
      r.append(paperForm(p, f => op({op: 'paper_update', id: p.id, fields: f}).then(ok => { if (ok) pEditing = null;
        render(); }), () => { pEditing = null; render(); }));
    } else {
      r.append(el('a', {class: 'paper', href: safe(p.url), target: '_blank', rel: 'noopener'}, p.title),
        p.idea ? el('div', {class: 'idea'}, p.idea) : null,
        users.length ? el('div', {class: 'np'}, 'Used by: ' + users.map(i => i.title).join(' · ')) : null,
        el('div', {class: 'bar'}, el('button', {onclick: () => { pEditing = p.id; render(); }}, 'Edit'),
          el('button', {class: 'del', onclick: () => confirm('Delete paper "' + p.title + '"?') &&
            op({op: 'paper_delete', id: p.id})}, 'Delete')));
    }
    list.append(r);
  }
  det.append(list);
  return det;
}
load();
setInterval(() => { if (!editing && !adding && !pEditing && !pAdding && document.visibilityState === 'visible') load(); },
  30000);
</script></body></html>
"""
