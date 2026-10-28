"""The engine's HTTP API (musicar/serve, CPU, tiny model): python tests/serve/test_app.py  (or pytest tests/)."""

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import run, tiny  # noqa: E402
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from musicar.data.loader import BOS_PITCH  # noqa: E402
from musicar.serve.app import create_app  # noqa: E402
from musicar.serve.jobs import wire  # noqa: E402


def engine_app(d):
    """An app on a tiny multi-instrument checkpoint (BOS / EOS, no text input) with a length-stop preset."""
    m = tiny(n_programs=129, pitch_size=BOS_PITCH + 2)
    args = {k: v for k, v in vars(m.config).items() if k != 'dropout'}
    torch.save(dict(model=m.state_dict(), model_args=args, iter_num=0), Path(d) / 'ckpt.pt')
    (Path(d) / 'test.toml').write_text('sample-instruments = true\nlength-stop = 1.2\ntemperature = 1.0\n')
    return create_app(str(Path(d) / 'ckpt.pt'), [str(Path(d) / 'test.toml')], device='cpu', name='test')


def read_events(client, job_id):
    """[(name, data)] of a job's event stream, read to its end."""
    out, name = [], None
    with client.stream('GET', f'/v1/jobs/{job_id}/events') as r:
        assert r.status_code == 200 and r.headers['content-type'].startswith('text/event-stream')
        for line in r.iter_lines():
            if line.startswith('event: '):
                name = line[7:]
            elif line.startswith('data: '):
                out.append((name, json.loads(line[6:])))
    return out


def test_generate_stream():
    """health; a job streams queued -> started -> notes -> done; the streamed notes of every candidate are its
    finished notes; a finished job's stream replays from the start."""
    with tempfile.TemporaryDirectory() as d:
        app = engine_app(d)
        client = TestClient(app)
        h = client.get('/v1/health').json()
        assert (h['engine'], h['loaded'], h['busy']) == ('test', False, False)
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
        assert client.get('/v1/health').json()['loaded']


def test_errors():
    """a caption for a model without text: an error event; bad overrides / lengths: 422; unknown job: 404."""
    with tempfile.TemporaryDirectory() as d:
        client = TestClient(engine_app(d))
        job_id = client.post('/v1/generate', json=dict(prompt='a sad piano piece', length_s=5)).json()['job_id']
        events = read_events(client, job_id)
        assert events[-1][0] == 'error' and 'without text' in events[-1][1]['message'], events
        assert client.post('/v1/generate', json=dict(overrides=dict(checkpoint='x'))).status_code == 422
        assert client.post('/v1/generate', json=dict(overrides=dict(dry_key='chords'))).status_code == 422
        assert client.post('/v1/generate', json=dict(length_s=1)).status_code == 422
        assert client.get('/v1/jobs/nope/events').status_code == 404


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
