"""Sequence state tracked by the scheduler and block manager."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch

from .config import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(enum.Enum):
    LENGTH = "length"
    STOP = "stop"


@dataclass
class Sequence:
    """One generation stream: prompt tokens plus generated tokens.

    ``token_ids`` always holds prompt + output tokens. ``num_computed`` counts
    how many of those tokens have their KV entries written to the paged cache;
    the invariant between steps is ``num_computed == len(token_ids) - 1`` (the
    newest token has not been fed through the model yet).
    """

    seq_id: int
    token_ids: list[int]
    params: SamplingParams
    arrival_time: float = 0.0
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_computed: int = 0
    num_cached_tokens: int = 0
    finish_reason: FinishReason | None = None
    generator: torch.Generator | None = None
    num_preemptions: int = 0

    def __post_init__(self) -> None:
        if self.num_prompt_tokens == 0:
            self.num_prompt_tokens = len(self.token_ids)
        if self.generator is None and self.params.seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.params.seed)

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def num_blocks_needed(self, block_size: int) -> int:
        return (len(self.token_ids) + block_size - 1) // block_size
