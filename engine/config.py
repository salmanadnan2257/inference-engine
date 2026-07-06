"""Configuration dataclasses for the engine and per-request sampling."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """Static engine configuration, fixed for the lifetime of an engine."""

    model: str = "gpt2"
    draft_model: str | None = None
    block_size: int = 16
    num_blocks: int = 512
    max_batch_size: int = 8
    max_model_len: int = 1024
    num_speculative_tokens: int = 4
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")


@dataclass
class SamplingParams:
    """Per-request sampling parameters.

    temperature == 0.0 means greedy decoding.
    top_k == 0 disables top-k filtering; top_p == 1.0 disables nucleus filtering.
    """

    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    n: int = 1

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
        if self.n < 1:
            raise ValueError("n must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
