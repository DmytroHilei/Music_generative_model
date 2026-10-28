"""
The engine's HTTP API, the contract in context/claude/app_2026-10-27.md §2:

    GET    /v1/health               engine, model, loaded, busy, queue, notes/s, GPU memory, uptime
    GET    /v1/presets              the decoding presets (faithful / balanced / wild), ranges, allowed overrides
    POST   /v1/generate             {prompt, length_s, preset, seed, best_of, overrides} -> 202 {job_id, position};
                                    429 when the queue is full
    GET    /v1/jobs/{id}/events     server-sent events of the job from its start (jobs.py), ': ping' when quiet
    GET    /v1/jobs/{id}/midi       the job's MIDI (?candidate=k), also while it runs
    DELETE /v1/jobs/{id}            cancel
    POST   /v1/admin/invites        {name: gh-<login>} -> 201 {token}: the page's GitHub sign-in (server to server,
                                    X-Admin-Key = the admin key file; off without the file)

Every route but /v1/health needs "Authorization: Bearer <token>" (tokens.py) unless auth = false; a job is seen only
by its token. Browsers may call the engine from the cors_origins pages. Run: python -m musicar.serve (settings:
config/serve/engine.toml).
"""

import asyncio
import hmac
import io
import json
import random
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from musicar.generation.piece import GenerationError, settings
from musicar.serve.config import EngineConfig
from musicar.serve.jobs import Engine, QueueFull
from musicar.serve.tokens import Revoked, TokenStore

# decoding options a request (or an engine preset) may change (generate.py names); the rest comes from config.base
OVERRIDES = {'temperature', 'top_k', 'top_p', 'min_p', 'cfg_scale', 'cfg_notes', 'cfg_end', 'cfg_linear',
             'instrument_temperature', 'dry_multiplier', 'dry_allowed', 'dry_base', 'dry_key', 'eos_ramp',
             'length_stop', 'eos_floor', 'min_notes'}
LENGTH_S, BEST_OF = (5, 300), (1, 8)
NOTES_PER_S = 30  # note budget per second of the asked length (with the length-stop margin); dense bands run ~15-25
MAX_NOTES = 8192
POLL_S = 0.05  # how often an open event stream looks for new events


class GenerateRequest(BaseModel):
    prompt: str = Field('', max_length=1000, description='caption, English or Ukrainian; empty = unconditional')
    length_s: float = Field(60.0, ge=LENGTH_S[0], le=LENGTH_S[1])
    preset: str | None = Field(None, description='an engine preset (GET /v1/presets); null = the default')
    seed: int | None = Field(None, ge=0, lt=2 ** 31, description='null = random')
    best_of: int = Field(1, ge=BEST_OF[0], le=BEST_OF[1])
    opening: bool = Field(False, description='start from a caption-matched real opening (not in the result)')
    overrides: dict[str, float | int | str | bool] = {}


class InviteRequest(BaseModel):
    name: str = Field(..., max_length=64, description='gh-<GitHub login, lowercase>')


def sse(name, data):
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def create_app(config=None):
    config = config or EngineConfig.load()
    for name, options in config.presets.items():  # bad presets fail here, not at the first request
        bad = sorted(set(options) - OVERRIDES)
        if bad:
            raise ValueError(f"engine preset {name}: {', '.join(bad)} can't be set by a preset")
        settings(config.base, **options)
    if config.default_preset not in config.presets:
        raise ValueError(f"default_preset {config.default_preset!r} is not one of {', '.join(config.presets)}")
    tokens = TokenStore(config.tokens_path) if config.auth else None
    engine = Engine(config)
    app = FastAPI(title='musicar engine', version='1')
    app.state.engine, app.state.tokens = engine, tokens
    app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins, allow_methods=['GET', 'POST', 'DELETE'],
                       allow_headers=['Authorization', 'Content-Type'],
                       expose_headers=['Retry-After', 'Content-Disposition'], max_age=600)

    def user(creds: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False))):
        """The token's name (None with auth off); 401 without a valid token. Declared as a bearer scheme, so /docs
        has an Authorize button (a phone can try the API there)."""
        if tokens is None:
            return None
        name = tokens.check(creds.credentials.strip()) if creds else None
        if name is None:
            raise HTTPException(401, 'a valid invite token is needed' if creds else 'no invite token',
                                headers={'WWW-Authenticate': 'Bearer'})
        return name

    def get_job(job_id, owner):
        job = engine.jobs.get(job_id)
        if job is None or job.owner != owner:  # another token's job looks like none
            raise HTTPException(404, 'no such job (or forgotten: finished jobs are kept '
                                     f'{config.keep_jobs_min:g} min)')
        return job

    @app.get('/v1/health')
    def health():
        return engine.health()

    @app.get('/v1/presets')
    def presets(owner=Depends(user)):
        resolved = {name: settings(config.base, **options) for name, options in config.presets.items()}
        return dict(default=config.default_preset,
                    presets={name: {k: getattr(args, k) for k in sorted(OVERRIDES)} for name, args in resolved.items()},
                    length_s=dict(min=LENGTH_S[0], max=LENGTH_S[1], default=GenerateRequest().length_s),
                    best_of=dict(min=BEST_OF[0], max=BEST_OF[1], default=1), overrides=sorted(OVERRIDES))

    @app.post('/v1/generate', status_code=202)
    def generate(req: GenerateRequest, owner=Depends(user)):
        unknown = sorted(set(req.overrides) - OVERRIDES)
        if unknown:
            raise HTTPException(422, f"overrides not allowed: {', '.join(unknown)} "
                                     f"(allowed: {', '.join(sorted(OVERRIDES))})")
        preset = req.preset or config.default_preset
        if preset not in config.presets:
            raise HTTPException(422, f"no preset {preset!r} (presets: {', '.join(config.presets)})")
        seed = req.seed if req.seed is not None else random.randrange(2 ** 31)
        try:
            if req.opening and not (req.prompt.strip() and config.opening_store):
                raise GenerationError('a real opening needs a caption' if config.opening_store
                                      else 'this engine has no opening store (opening_store)')
            # the store is the engine's (engine.toml), never a request's: retrieve is a path
            opening = dict(retrieve=config.opening_store, retrieve_strict=','.join(config.opening_strict)) \
                if req.opening else {}
            args = settings(config.base, **{**config.presets[preset], **req.overrides}, text=req.prompt.strip() or None,
                            length_seconds=req.length_s, seed=seed, best_of=req.best_of, num_samples=1,
                            batch_size=min(req.best_of, config.batch_candidates),
                            max_new_tokens=min(MAX_NOTES, int(req.length_s * NOTES_PER_S)), **opening)
        except GenerationError as e:
            raise HTTPException(422, str(e))
        try:
            job = engine.submit(args, owner)
        except QueueFull as e:
            raise HTTPException(429, str(e), headers={'Retry-After': str(e.retry_after)})
        if tokens is not None:
            tokens.count_job(owner)
        return {'job_id': job.id, 'position': job.events[0][1]['position']}

    @app.get('/v1/jobs/{job_id}/events')
    async def events(job_id: str, owner=Depends(user)):
        job = get_job(job_id, owner)

        async def stream():
            i, last = 0, time.monotonic()
            while True:
                finished = job.finished  # read before the events: the worker sets it after the last event
                while i < len(job.events):
                    yield sse(*job.events[i])
                    i, last = i + 1, time.monotonic()
                if finished:
                    return
                if time.monotonic() - last >= config.heartbeat_s:
                    yield ': ping\n\n'
                    last = time.monotonic()
                await asyncio.sleep(POLL_S)

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/v1/jobs/{job_id}/midi')
    def midi(job_id: str, candidate: int | None = None, owner=Depends(user)):
        job = get_job(job_id, owner)
        try:
            pm, c = engine.midi(job, candidate)
        except KeyError:
            raise HTTPException(404, f'job {job_id} has no candidate {candidate}')
        buf = io.BytesIO()
        pm.write(buf)
        name = f"musicar_{job.args.seed}" + (f"_{c}" if job.args.best_of > 1 else "") + ".mid"
        return Response(buf.getvalue(), media_type='audio/midi',
                        headers={'Content-Disposition': f'attachment; filename="{name}"'})

    @app.delete('/v1/jobs/{job_id}')
    def cancel(job_id: str, owner=Depends(user)):
        job = get_job(job_id, owner)
        if not engine.cancel(job):
            raise HTTPException(409, f'job already {job.state}')
        return {'job_id': job.id, 'state': 'cancelled' if job.finished else 'cancelling'}

    @app.post('/v1/admin/invites', status_code=201, include_in_schema=False)
    def invite(req: InviteRequest, x_admin_key: str | None = Header(None)):
        """A token for a GitHub user the page's server has checked (an approved access request): only gh- names, so
        the key can't mint the hand-made invites' names."""
        path = config.admin_key_path
        if tokens is None or not path.exists():
            raise HTTPException(404, 'Not Found')
        if not x_admin_key or not hmac.compare_digest(x_admin_key.strip().encode(), path.read_text().strip().encode()):
            raise HTTPException(403, 'wrong admin key')
        if not req.name.startswith('gh-'):
            raise HTTPException(422, 'only gh-<login> names are made here')
        try:
            return {'name': req.name, 'token': tokens.issue(req.name)}
        except Revoked:
            raise HTTPException(403, f'access for {req.name} was revoked')
        except ValueError as e:
            raise HTTPException(422, str(e))

    return app
