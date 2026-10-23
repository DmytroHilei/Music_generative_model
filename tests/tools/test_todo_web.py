"""tools/todo_web.py operations: restore (undo of a delete), reorder (drag and drop), My Day pins."""

import copy
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('todo_web', Path(__file__).resolve().parents[2] / 'tools/todo_web.py')
todo_web = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(todo_web)


def _data():
    item = lambda i, pri, order, status='todo': {'track': 'a', 'title': i, 'priority': pri, 'status': status, 'id': i,
                                                 'order': order, 'updated': '2026-01-01'}
    return {'tracks': [{'id': 'a', 'name': 'A'}, {'id': 'b', 'name': 'B'}], 'papers': [],
            'items': [item('p0a', 'P0', 1), item('p0b', 'P0', 2), item('p1a', 'P1', 3), item('p1b', 'P1', 4),
                      item('p1c', 'P1', 5), item('done', 'P1', 6, 'done')]}


def _band(data, pri):
    return [x['id'] for x in sorted((x for x in data['items'] if x['priority'] == pri and x['status'] == 'todo'),
                                    key=lambda x: x['order'])]


def _raises(data, req):
    try:
        todo_web.apply(data, req)
    except ValueError:
        return True
    return False


def test_restore_brings_back_id_and_order():
    data = _data()
    gone = copy.deepcopy(data['items'][1])
    todo_web.apply(data, {'op': 'delete', 'id': 'p0b'})
    todo_web.apply(data, {'op': 'restore', 'item': gone})
    back = next(x for x in data['items'] if x['id'] == 'p0b')
    assert back['order'] == 2 and back['updated'] == '2026-01-01' and _band(data, 'P0') == ['p0a', 'p0b']
    assert _raises(data, {'op': 'restore', 'item': gone})  # the id exists again
    assert _raises(data, {'op': 'restore', 'item': {**gone, 'id': 'new', 'track': 'nope'}})


def test_reorder_within_band_and_to_end():
    data = _data()
    todo_web.apply(data, {'op': 'reorder', 'id': 'p1c', 'priority': 'P1', 'before': 'p1a'})
    assert _band(data, 'P1') == ['p1c', 'p1a', 'p1b']
    todo_web.apply(data, {'op': 'reorder', 'id': 'p1c', 'priority': 'P1', 'before': ''})
    assert _band(data, 'P1') == ['p1a', 'p1b', 'p1c']
    assert next(x for x in data['items'] if x['id'] == 'p1c')['updated'] == '2026-01-01'  # same priority: no update


def test_reorder_changes_priority_and_undo():
    data = _data()
    todo_web.apply(data, {'op': 'reorder', 'id': 'p1b', 'priority': 'P0', 'before': 'p0b'})
    assert _band(data, 'P0') == ['p0a', 'p1b', 'p0b'] and _band(data, 'P1') == ['p1a', 'p1c']
    assert next(x for x in data['items'] if x['id'] == 'p1b')['updated'] != '2026-01-01'
    todo_web.apply(data, {'op': 'reorder', 'id': 'p1b', 'priority': 'P1', 'before': 'p1c'})  # what the page's undo sends
    assert _band(data, 'P0') == ['p0a', 'p0b'] and _band(data, 'P1') == ['p1a', 'p1b', 'p1c']
    orders = [x['order'] for x in data['items'] if x['status'] == 'todo']
    assert len(set(orders)) == len(orders)


def test_reorder_rejects_closed_items_and_bad_priority():
    data = _data()
    assert _raises(data, {'op': 'reorder', 'id': 'done', 'priority': 'P1', 'before': ''})
    assert _raises(data, {'op': 'reorder', 'id': 'p1a', 'priority': 'P9', 'before': ''})


def test_reorder_makes_equal_orders_distinct():
    data = _data()
    for x in data['items']:
        x['order'] = 1
    todo_web.apply(data, {'op': 'reorder', 'id': 'p1c', 'priority': 'P1', 'before': 'p1a'})
    band = sorted((x for x in data['items'] if x['priority'] == 'P1' and x['status'] == 'todo'), key=lambda x: x['order'])
    assert band[0]['id'] == 'p1c' and len({x['order'] for x in band}) == 3


def test_my_day_pin_is_not_an_update():
    data = _data()
    todo_web.apply(data, {'op': 'update', 'id': 'p0a', 'fields': {'today': '2026-10-23'}})
    it = data['items'][0]
    assert it['today'] == '2026-10-23' and it['updated'] == '2026-01-01'
    todo_web.apply(data, {'op': 'update', 'id': 'p0a', 'fields': {'today': ''}})
    assert 'today' not in it and it['updated'] == '2026-01-01'
    todo_web.apply(data, {'op': 'update', 'id': 'p0a', 'fields': {'today': '2026-10-23', 'cost': '1 h'}})
    assert it['updated'] != '2026-01-01'
