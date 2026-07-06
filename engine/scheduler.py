"""Continuous batching scheduler with KV-pressure preemption."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager
from .config import EngineConfig
from .sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """One scheduling decision: prompts to prefill, sequences to decode."""

    prefill: list[Sequence] = field(default_factory=list)
    decode: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)


class Scheduler:
    """Admits, batches, and preempts sequences at token boundaries.

    Requests join the running batch whenever there is batch room and enough
    free KV blocks for their prompt. When a running sequence cannot get a
    block for its next token, the most recently arrived sequence is preempted:
    its blocks are freed and it re-enters the front of the waiting queue for
    recomputation (which the prefix cache usually makes cheap).
    """

    def __init__(self, config: EngineConfig, block_manager: BlockManager,
                 lookahead: int = 0) -> None:
        self.config = config
        self.block_manager = block_manager
        # Extra tokens each decode step may append (speculative drafting).
        self.lookahead = lookahead
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        # Admit waiting sequences while there is room.
        while self.waiting and len(self.running) < self.config.max_batch_size:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.prefill.append(seq)

        # Guarantee the whole decode batch can take its next step (the
        # capacity check is cumulative across sequences); preempt the newest
        # sequences until the rest fit.
        decode = [s for s in self.running if s not in out.prefill]
        while decode:
            needed = sum(self.block_manager.blocks_needed(s, self.lookahead)
                         for s in decode)
            if needed <= self.block_manager.num_free_blocks:
                break
            victim = max(self.running, key=lambda s: (s.arrival_time, s.seq_id))
            self._preempt(victim)
            out.preempted.append(victim)
            if victim in decode:
                decode.remove(victim)
            if victim in out.prefill:
                out.prefill.remove(victim)
        out.decode = decode
        return out

    def _preempt(self, seq: Sequence) -> None:
        self.block_manager.free_seq(seq)
        seq.num_computed = 0
        seq.num_cached_tokens = 0
        seq.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def finish_seq(self, seq: Sequence) -> None:
        self.block_manager.free_seq(seq)
        seq.status = SequenceStatus.FINISHED
        if seq in self.running:
            self.running.remove(seq)
