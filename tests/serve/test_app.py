"""The engine's HTTP API (musicar/serve, CPU, tiny model): python tests/serve/test_app.py  (or pytest tests/)."""

import io
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run, tiny  # noqa: E402
import pretty_midi  # noqa: E402
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from musicar.data.loader import BOS_PITCH  # noqa: E402
from musicar.generation.piece import GenerationError  # noqa: E402
from musicar.serve.app import create_app  # noqa: E402
from musicar.serve.config import CONFIG, EngineConfig  # noqa: E402
from musicar.serve.jobs import wire  # noqa: E402


def engine_app(d, **kw):
    """An app on a tiny multi-instrument checkpoint (BOS / EOS, no text input) with a length-stop base preset; no
    auth unless asked, tokens in d."""
    m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2)
    args = {k: v for k, v in vars(m.config).items() if k != 'dropout'}
    torch.save(dict(model=m.state_dict(), model_args=args, iter_num=0), Path(d) / 'ckpt.pt')
    (Path(d) / 'test.toml').write_text('sample-instruments = true\nlength-stop = 1.2\ntemperature = 1.0\n')
    kw = {'auth': False, 'tokens_db': str(Path(d) / 'tokens.sqlite'), **kw}
    config = EngineConfig(checkpoint=str(Path(d) / 'ckpt.pt'), base=[str(Path(d) / 'test.toml')], device='cpu',
                          name='test', presets={'balanced': {}, 'hot': {'temperature': 1.3}}, **kw)
    return create_app(config)


def read_events(client, job_id, raw=False):
    """[(name, data)] of a job's event stream, read to its end (raw: the text)."""
    out, name = [], None
    with client.stream('GET', f'/v1/jobs/{job_id}/events') as r:
        assert r.status_code == 200 and r.headers['content-type'].startswith('text/event-stream')
        lines = list(r.iter_lines())
    if raw:
        return '\n'.join(lines)
    for line in lines:
        if line.startswith('event: '):
            name = line[7:]
        elif line.startswith('data: '):
            out.append((name, json.loads(line[6:])))
    return out


def wait(cond, timeout=10):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, 'timed out'
        time.sleep(0.02)


class SlowGenerator:
    """Stands in for Generator: after `quiet` seconds a note every 10 ms, n of them, then a GenerationError."""

    def __init__(self, quiet=0.0, n=1000):
        self.quiet, self.n = quiet, n

    def run(self, args, on_notes=None, progress=None):
        time.sleep(self.quiet)
        for i in range(self.n):
            on_notes(0, [[round(i * 0.1, 3), 0, 60, 0.1, 80]])
            time.sleep(0.01)
        raise GenerationError('slow generator done')


def test_generate_stream():
    """health; a job streams queued -> started -> notes -> done; the streamed notes of every candidate are its
    finished notes; a finished job's stream replays from the start."""
    with tempfile.TemporaryDirectory() as d:
        app = engine_app(d)
        client = TestClient(app)
        h = client.get('/v1/health').json()
        assert (h['engine'], h['loaded'], h['busy'], h['gpu_free_mb']) == ('test', False, False, None)
        r = client.post('/v1/generate', json=dict(prompt='', length_s=5, seed=3, best_of=2))
        assert r.status_code == 202 and r.json()['position'] == 0
        job_id = r.json()['job_id']
        events = read_events(client, job_id)
        names = [n for n, _ in events]
        assert names[:2] == ['queued', 'started'] and names[-1] == 'done' and 'notes' in names, names
        assert events[1][1]['seed'] == 3 and events[1][1]['settings']['max_new_tokens'] == 150
        done = events[-1][1]
        assert {c['candidate'] for c in done['candidates']} == {0, 1} and done['best'] == done['candidates'][0][
            'candidate']
        streamed = defaultdict(list)
        for n, data in events:
            if n == 'notes':
                streamed[data['candidate']] += data['notes']
        job = app.state.engine.jobs[job_id]
        for p, c in zip(job.pieces, done['candidates']):
            final = [wire(n) for n in p.notes()]
            # velocities of a length-stopped candidate are faded only at the end
            assert [n[:4] for n in streamed[p.index]] == [n[:4] for n in final] and c['n_notes'] == len(final)
        assert read_events(client, job_id) == events
        h = client.get('/v1/health').json()
        assert h['loaded'] and h['notes_per_s'] > 0


def test_midi_and_presets():
    """/midi: the best candidate by default, any candidate by number, 404 beyond; /presets and a preset's options."""
    with tempfile.TemporaryDirectory() as d:
        client = TestClient(engine_app(d))
        p = client.get('/v1/presets').json()
        assert p['default'] == 'balanced' and p['presets']['hot']['temperature'] == 1.3
        assert p['presets']['balanced']['temperature'] == 1.0 and 'cfg_scale' in p['overrides']
        job_id = client.post('/v1/generate', json=dict(length_s=5, seed=4, best_of=2, preset='hot')).json()['job_id']
        events = read_events(client, job_id)
        assert events[1][1]['settings']['temperature'] == 1.3
        done = events[-1][1]
        for candidate, query in ((done['best'], ''), (1 - done['best'], f"?candidate={1 - done['best']}")):
            r = client.get(f'/v1/jobs/{job_id}/midi{query}')
            assert r.status_code == 200 and r.headers['content-type'] == 'audio/midi'
            assert f'musicar_4_{candidate}.mid' in r.headers['content-disposition']
            notes = [n for i in pretty_midi.PrettyMIDI(io.BytesIO(r.content)).instruments for n in i.notes]
            assert len(notes) == next(c['n_notes'] for c in done['candidates'] if c['candidate'] == candidate)
        assert client.get(f'/v1/jobs/{job_id}/midi?candidate=5').status_code == 404
        assert client.post('/v1/generate', json=dict(preset='loud')).status_code == 422


def test_cancel_queue_heartbeat():
    """a full queue: 429 with Retry-After; cancel a queued job (at once) and a running one (at its next notes, its
    MIDI = the notes so far); a quiet stream gets ': ping' lines."""
    with tempfile.TemporaryDirectory() as d:
        app = engine_app(d, max_queue=1, heartbeat_s=0.1)
        engine, client = app.state.engine, TestClient(app)
        engine.generator = SlowGenerator(quiet=0.5, n=3)  # read while it is quiet: pings, then 3 notes and an error
        text = read_events(client, client.post('/v1/generate', json=dict(length_s=5)).json()['job_id'], raw=True)
        assert ': ping' in text and text.count('event: notes') == 3 and 'event: error' in text, text
        engine.generator = SlowGenerator()
        a = client.post('/v1/generate', json=dict(length_s=5)).json()['job_id']
        wait(lambda: engine.running is not None)
        b = client.post('/v1/generate', json=dict(length_s=5)).json()
        assert b['position'] == 1
        r = client.post('/v1/generate', json=dict(length_s=5))
        assert r.status_code == 429 and int(r.headers['retry-after']) >= 1
        assert client.delete(f"/v1/jobs/{b['job_id']}").json()['state'] == 'cancelled'
        assert [n for n, _ in read_events(client, b['job_id'])] == ['queued', 'cancelled']
        wait(lambda: len(engine.jobs[a].notes(0)) >= 5)
        assert client.delete(f'/v1/jobs/{a}').json()['state'] == 'cancelling'
        assert [n for n, _ in read_events(client, a)][-1] == 'cancelled'
        n = len(engine.jobs[a].notes(0))
        midi = pretty_midi.PrettyMIDI(io.BytesIO(client.get(f'/v1/jobs/{a}/midi').content))
        assert 5 <= n < 1000 and sum(len(i.notes) for i in midi.instruments) == n
        assert client.delete(f'/v1/jobs/{a}').status_code == 409


def test_errors_gpu_busy_unload_cleanup():
    """a caption for a model without text: an error event; bad requests: 422 / 404; a GPU too full to load the model:
    an error event; an idle engine unloads the model and forgets finished jobs."""
    with tempfile.TemporaryDirectory() as d:
        app = engine_app(d, idle_unload_min=0.01, keep_jobs_min=0.02)
        engine, client = app.state.engine, TestClient(app)
        job_id = client.post('/v1/generate', json=dict(prompt='a sad piano piece', length_s=5)).json()['job_id']
        events = read_events(client, job_id)
        assert events[-1][0] == 'error' and 'without text' in events[-1][1]['message'], events
        assert client.post('/v1/generate', json=dict(overrides=dict(checkpoint='x'))).status_code == 422
        assert client.post('/v1/generate', json=dict(overrides=dict(dry_key='chords'))).status_code == 422
        assert client.post('/v1/generate', json=dict(length_s=1)).status_code == 422
        assert client.get('/v1/jobs/nope/events').status_code == 404
        assert engine.generator is not None
        wait(lambda: engine.generator is None)  # 0.6 s idle
        wait(lambda: job_id not in engine.jobs)  # 1.2 s after it finished
        engine.on_gpu = True
        engine.gpu_status = lambda: dict(free_mb=500, total_mb=8000, own_mb=0, other_mb=7000)
        assert client.get('/v1/health').json()['busy_other']
        job_id = client.post('/v1/generate', json=dict(length_s=5)).json()['job_id']
        events = read_events(client, job_id)
        assert events[-1][0] == 'error' and 'busy with other work' in events[-1][1]['message'], events


def test_tokens_owners_cors():
    """auth: 401 without / with a bad or revoked token, health stays open; a job is seen only by its token; a token
    may have per_token_jobs unfinished jobs; CORS answers the listed origins only."""
    with tempfile.TemporaryDirectory() as d:
        app = engine_app(d, auth=True, per_token_jobs=1, max_queue=4)
        engine, store, client = app.state.engine, app.state.tokens, TestClient(app)
        anna, bob = store.add('anna'), store.add('bob')
        try:
            store.add('anna')
            raise AssertionError('a second active token for anna')
        except ValueError:
            pass
        auth = lambda token: {'Authorization': f'Bearer {token}'}  # noqa: E731
        assert client.get('/v1/health').status_code == 200
        for headers in ({}, auth('mus_wrong'), {'Authorization': f'Basic {anna}'}):
            r = client.get('/v1/presets', headers=headers)
            assert r.status_code == 401 and r.headers['www-authenticate'] == 'Bearer', headers
        assert client.get('/v1/presets', headers=auth(anna)).status_code == 200
        engine.generator = SlowGenerator()
        a = client.post('/v1/generate', json=dict(length_s=5), headers=auth(anna)).json()['job_id']
        r = client.post('/v1/generate', json=dict(length_s=5), headers=auth(anna))
        assert r.status_code == 429 and 'your jobs' in r.json()['detail']
        b = client.post('/v1/generate', json=dict(length_s=5), headers=auth(bob)).json()['job_id']
        for method, url in (('GET', f'/v1/jobs/{a}/events'), ('GET', f'/v1/jobs/{a}/midi'),
                            ('DELETE', f'/v1/jobs/{a}')):
            assert client.request(method, url, headers=auth(bob)).status_code == 404, url
        assert client.delete(f'/v1/jobs/{b}', headers=auth(bob)).status_code == 200
        assert client.delete(f'/v1/jobs/{a}', headers=auth(anna)).status_code == 200
        assert store.revoke('bob') and client.get('/v1/presets', headers=auth(bob)).status_code == 401
        assert {name: jobs for name, _, _, jobs, _ in store.list()} == {'anna': 1, 'bob': 1}
        pre = dict(Origin='http://localhost:3000', **{'Access-Control-Request-Method': 'POST',
                                                      'Access-Control-Request-Headers': 'authorization,content-type'})
        r = client.options('/v1/generate', headers=pre)
        assert r.status_code == 200 and r.headers['access-control-allow-origin'] == 'http://localhost:3000'
        assert 'authorization' in r.headers['access-control-allow-headers'].lower()
        r = client.options('/v1/generate', headers={**pre, 'Origin': 'https://evil.example'})
        assert 'access-control-allow-origin' not in r.headers
        r = client.get('/v1/presets', headers={'Origin': 'http://localhost:3000', **auth(anna)})
        assert r.headers['access-control-allow-origin'] == 'http://localhost:3000'
        assert 'retry-after' in r.headers['access-control-expose-headers'].lower()


def test_tokens_cli():
    # python -m musicar.serve.tokens add / list / revoke on the tokens_db of a given engine.toml
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / 'engine.toml'
        cfg.write_text(f'tokens_db = "{Path(d) / "t.sqlite"}"\n')
        root = str(Path(__file__).parents[2])
        cli = lambda *a: subprocess.run(  # noqa: E731
            [sys.executable, '-m', 'musicar.serve.tokens', *a, '--config', str(cfg)], capture_output=True, text=True,
            check=True, cwd=root)
        token = cli('add', 'guest').stdout.strip()
        assert token.startswith('mus_') and len(token) > 30
        assert 'guest' in cli('list').stdout and 'REVOKED' not in cli('list').stdout
        assert cli('revoke', 'guest').stdout.strip() == 'revoked' and 'REVOKED' in cli('list').stdout


def test_config_file():
    # the shipped engine.toml loads, its presets resolve, overrides replace its values, unknown keys are refused
    c = EngineConfig.load(CONFIG, idle_unload_min=30, max_queue=None)
    assert c.idle_unload_min == 30 and c.max_queue == 8 and set(c.presets) == {'faithful', 'balanced', 'wild'}
    try:
        EngineConfig.load(CONFIG, idle_unload=3)
        raise AssertionError('accepted an unknown setting')
    except ValueError:
        pass


def test_early_stop():
    """on_notes returning True stops generate(): the notes not generated stay 0."""
    from musicar.generation.sampling import generate
    m = tiny()
    z = torch.zeros(2, 1, dtype=torch.long)
    calls = []
    out = generate(m, z, z, z + 10, z, max_new_tokens=40, on_notes=lambda a, n: calls.append(a) or True,
                   notes_every=8, cuda_graph=False)
    assert calls == [0] and out[1][:, 9:].abs().sum() == 0 and out[1][:, 1:9].abs().sum() > 0


if __name__ == '__main__':
    run(globals())
