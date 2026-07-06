"""FastAPI server exposing a subset of the OpenAI completions API.

Endpoints:
  POST /v1/completions   text completion, streaming (SSE) or not
  GET  /v1/models        the loaded model
  GET  /healthz          liveness
  GET  /metrics          engine counters (prefix cache, preemptions, spec)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.async_engine import AsyncEngine
from engine.config import EngineConfig, SamplingParams


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


def _params_from(req: CompletionRequest) -> SamplingParams:
    stop = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    return SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature,
                          top_k=req.top_k, top_p=req.top_p,
                          repetition_penalty=req.repetition_penalty,
                          seed=req.seed, stop=stop, n=req.n)


def create_app(config: EngineConfig) -> FastAPI:
    engine: AsyncEngine | None = None

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal engine
        engine = await asyncio.to_thread(AsyncEngine, config)
        engine.start()
        app.state.engine = engine
        yield
        await engine.stop()

    app = FastAPI(title="inference-engine", lifespan=lifespan)

    def _completion_id() -> str:
        return "cmpl-" + uuid.uuid4().hex[:24]

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list",
                "data": [{"id": config.model, "object": "model",
                          "owned_by": "inference-engine"}]}

    @app.get("/metrics")
    async def metrics() -> dict:
        assert engine is not None
        eng = engine.engine
        stats = eng.block_manager.stats
        return {
            "num_free_blocks": eng.block_manager.num_free_blocks,
            "num_total_blocks": eng.block_manager.num_blocks,
            "prefix_cache_queries": stats.prefix_cache_queries,
            "prefix_cache_hit_tokens": stats.prefix_cache_hit_tokens,
            "cow_copies": stats.cow_copies,
            "evictions": stats.evictions,
            "preemptions": eng.scheduler.num_preemptions,
            "speculative_drafted": eng.spec_stats.drafted,
            "speculative_accepted": eng.spec_stats.accepted,
            "speculative_acceptance_rate": eng.spec_stats.acceptance_rate,
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw: Request):
        assert engine is not None
        if req.model is not None and req.model != config.model:
            raise HTTPException(404, f"model {req.model!r} not loaded "
                                     f"(serving {config.model!r})")
        try:
            seq_ids = await engine.submit(req.prompt, _params_from(req))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        cid = _completion_id()
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _sse_stream(engine, raw, cid, created, config.model, seq_ids),
                media_type="text/event-stream")

        choices = []
        for index, sid in enumerate(seq_ids):
            text, finish_reason, num_tokens = "", None, 0
            async for out in engine.stream(sid):
                text += out.text_delta
                num_tokens += len(out.new_token_ids)
                if out.finished:
                    finish_reason = out.finish_reason
            choices.append({"index": index, "text": text,
                            "finish_reason": finish_reason, "logprobs": None})
        return JSONResponse({"id": cid, "object": "text_completion",
                             "created": created, "model": config.model,
                             "choices": choices})

    return app


async def _sse_stream(engine: AsyncEngine, raw: Request, cid: str,
                      created: int, model: str, seq_ids: list[int]):
    async def one(index: int, sid: int, out_q: asyncio.Queue) -> None:
        async for out in engine.stream(sid):
            await out_q.put((index, out))
        await out_q.put((index, None))

    out_q: asyncio.Queue = asyncio.Queue()
    tasks = [asyncio.create_task(one(i, sid, out_q))
             for i, sid in enumerate(seq_ids)]
    live = len(seq_ids)
    try:
        while live > 0:
            index, out = await out_q.get()
            if out is None:
                live -= 1
                continue
            if await raw.is_disconnected():
                break
            payload = {"id": cid, "object": "text_completion",
                       "created": created, "model": model,
                       "choices": [{"index": index, "text": out.text_delta,
                                    "finish_reason": out.finish_reason,
                                    "logprobs": None}]}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        for t in tasks:
            t.cancel()
        for sid in seq_ids:
            await engine.abort(sid)
