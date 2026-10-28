"""
The engine's HTTP API, step 1 of the contract in context/claude/app_2026-10-27.md §2:

    GET  /v1/health               engine, model, loaded, busy, queue length, uptime
    POST /v1/generate             {prompt, length_s, seed, best_of, overrides} -> 202 {job_id, position}
    GET  /v1/jobs/{id}/events     server-sent events of the job from its start (jobs.py)

Run: python -m musicar.serve (see __main__.py). No auth yet: bind to localhost only.
"""

import asyncio
import json
import random

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from musicar.generation.piece import GenerationError, settings
from musicar.serve.jobs import Engine

# decoding options a request may change (generate.py names); everything else comes from the engine's presets
OVERRIDES = {'temperature', 'top_k', 'top_p', 'min_p', 'cfg_scale', 'cfg_notes', 'cfg_end', 'cfg_linear',
             'instrument_temperature', 'dry_multiplier', 'dry_allowed', 'dry_base', 'dry_key', 'eos_ramp',
             'length_stop', 'min_notes'}
NOTES_PER_S = 30  # note budget per second of the asked length (with the length-stop margin); dense bands run ~15-25
MAX_NOTES = 8192
POLL_S = 0.05  # how often an open event stream looks for new events


class GenerateRequest(BaseModel):
    prompt: str = Field('', max_length=1000, description='caption, English or Ukrainian; empty = unconditional')
    length_s: float = Field(60.0, ge=5, le=300)
    seed: int | None = Field(None, ge=0, lt=2 ** 31, description='null = random')
    best_of: int = Field(1, ge=1, le=8)
    overrides: dict[str, float | int | str | bool] = {}


def sse(name, data):
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def create_app(checkpoint='checkpoints/f1', presets=('f1',), device='auto', name='laptop'):
    """presets: generate.py presets every request starts from (their checkpoint is ignored: the engine's counts)."""
    settings(presets)  # a bad preset fails here, not at the first request
    engine = Engine(checkpoint, device, name)
    app = FastAPI(title='musicar engine', version='1')
    app.state.engine = engine

    @app.get('/v1/health')
    def health():
        return engine.health()

    @app.post('/v1/generate', status_code=202)
    def generate(req: GenerateRequest):
        unknown = sorted(set(req.overrides) - OVERRIDES)
        if unknown:
            raise HTTPException(422, f"overrides not allowed: {', '.join(unknown)} "
                                     f"(allowed: {', '.join(sorted(OVERRIDES))})")
        seed = req.seed if req.seed is not None else random.randrange(2 ** 31)
        try:
            args = settings(presets, text=req.prompt.strip() or None, length_seconds=req.length_s, seed=seed,
                            best_of=req.best_of, num_samples=1,
                            max_new_tokens=min(MAX_NOTES, int(req.length_s * NOTES_PER_S)), **req.overrides)
        except GenerationError as e:
            raise HTTPException(422, str(e))
        job = engine.submit(args)
        return {'job_id': job.id, 'position': job.events[0][1]['position']}

    @app.get('/v1/jobs/{job_id}/events')
    async def events(job_id: str):
        job = engine.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, 'no such job')

        async def stream():
            i = 0
            while True:
                finished = job.finished  # read before the events: the worker sets it after the last event
                while i < len(job.events):
                    yield sse(*job.events[i])
                    i += 1
                if finished:
                    return
                await asyncio.sleep(POLL_S)

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app
