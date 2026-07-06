"""Async wrapper around the synchronous engine for concurrent serving.

The blocking ``step()`` runs in a worker thread so the event loop stays free
to accept connections and stream tokens. A single lock serializes engine
mutations (step, add, abort); token deltas fan out through per-sequence
asyncio queues.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

from .config import EngineConfig, SamplingParams
from .llm_engine import LLMEngine, StepOutput


class AsyncEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.engine = LLMEngine(config)
        self._lock = threading.Lock()
        self._queues: dict[int, asyncio.Queue[StepOutput]] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while True:
            if not self.engine.has_work:
                self._wake.clear()
                await self._wake.wait()
            outputs = await asyncio.to_thread(self._locked_step)
            for out in outputs:
                queue = self._queues.get(out.seq_id)
                if queue is not None:
                    queue.put_nowait(out)

    def _locked_step(self) -> list[StepOutput]:
        with self._lock:
            return self.engine.step()

    async def submit(self, prompt: str | list[int],
                     params: SamplingParams) -> list[int]:
        """Queue a request; returns seq_ids (one per requested completion)."""
        def _add() -> list[int]:
            with self._lock:
                return self.engine.add_request(prompt, params)
        seq_ids = await asyncio.to_thread(_add)
        for sid in seq_ids:
            self._queues[sid] = asyncio.Queue()
        self._wake.set()
        return seq_ids

    async def abort(self, seq_id: int) -> None:
        def _abort() -> None:
            with self._lock:
                self.engine.abort(seq_id)
        await asyncio.to_thread(_abort)
        self._queues.pop(seq_id, None)

    async def stream(self, seq_id: int) -> AsyncIterator[StepOutput]:
        """Yield step outputs for one sequence until it finishes."""
        queue = self._queues[seq_id]
        try:
            while True:
                out = await queue.get()
                yield out
                if out.finished:
                    break
        finally:
            self._queues.pop(seq_id, None)
