"""Streaming HTTP API tests over an in-process ASGI transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest_asyncio

from engine.config import EngineConfig
from server.app import create_app

import pytest

CONFIG = EngineConfig(model="distilgpt2", block_size=8, num_blocks=256,
                      max_batch_size=4)

# The engine runs a background step loop, so the fixture and every test in
# this module must share one event loop.
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def client():
    app = create_app(CONFIG)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as c:
            yield c


async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_models(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "distilgpt2"


async def test_completion_non_streaming(client):
    r = await client.post("/v1/completions", json={
        "prompt": "The sky is", "max_tokens": 8, "temperature": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "distilgpt2"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] == "length"


async def collect_sse(client, payload: dict) -> tuple[list[dict], bool]:
    events: list[dict] = []
    got_done = False
    async with client.stream("POST", "/v1/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                got_done = True
                break
            events.append(json.loads(data))
    return events, got_done


async def test_streaming_matches_non_streaming(client):
    payload = {"prompt": "The sky is", "max_tokens": 8, "temperature": 0.0}
    r = await client.post("/v1/completions", json=payload)
    full = r.json()["choices"][0]["text"]
    events, done = await collect_sse(client, {**payload, "stream": True})
    assert done
    streamed = "".join(e["choices"][0]["text"] for e in events)
    assert streamed == full
    assert len(events) >= 2  # actually incremental, not one blob
    assert events[-1]["choices"][0]["finish_reason"] == "length"


async def test_concurrent_streaming_clients(client):
    prompts = ["One day", "The robot said", "In the beginning",
               "My favorite food is"]
    async def one(prompt: str) -> str:
        events, done = await collect_sse(client, {
            "prompt": prompt, "max_tokens": 10, "temperature": 0.0,
            "stream": True})
        assert done
        return "".join(e["choices"][0]["text"] for e in events)

    results = await asyncio.gather(*(one(p) for p in prompts))
    assert all(results)
    # Batched execution must not leak tokens across requests.
    for prompt, text in zip(prompts, results):
        solo = await client.post("/v1/completions", json={
            "prompt": prompt, "max_tokens": 10, "temperature": 0.0})
        assert solo.json()["choices"][0]["text"] == text


async def test_n_gives_multiple_choices(client):
    r = await client.post("/v1/completions", json={
        "prompt": "The sky is", "max_tokens": 6, "temperature": 0.9,
        "seed": 5, "n": 2})
    choices = r.json()["choices"]
    assert [c["index"] for c in choices] == [0, 1]
    assert all(c["text"] for c in choices)


async def test_seed_reproducible_over_http(client):
    payload = {"prompt": "The sky is", "max_tokens": 10,
               "temperature": 0.9, "seed": 42}
    r1 = await client.post("/v1/completions", json=payload)
    r2 = await client.post("/v1/completions", json=payload)
    assert r1.json()["choices"][0]["text"] == r2.json()["choices"][0]["text"]


async def test_wrong_model_404(client):
    r = await client.post("/v1/completions", json={
        "model": "gpt-4", "prompt": "hi", "max_tokens": 4})
    assert r.status_code == 404


async def test_oversized_request_400(client):
    r = await client.post("/v1/completions", json={
        "prompt": "hi", "max_tokens": 5000})
    assert r.status_code == 400


async def test_metrics(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["num_total_blocks"] == 256
    assert body["prefix_cache_queries"] >= 1
