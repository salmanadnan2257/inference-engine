"""Shared fixtures. Model-backed fixtures are session-scoped because loading
weights dominates test time on CPU."""

from __future__ import annotations

import pytest

from engine import EngineConfig, LLMEngine


@pytest.fixture(scope="session")
def distil_engine() -> LLMEngine:
    return LLMEngine(EngineConfig(model="distilgpt2", block_size=8,
                                  num_blocks=256, max_batch_size=4))


@pytest.fixture(scope="session")
def gpt2_engine() -> LLMEngine:
    return LLMEngine(EngineConfig(model="gpt2", block_size=8, num_blocks=256,
                                  max_batch_size=4))


@pytest.fixture(scope="session")
def spec_engine() -> LLMEngine:
    return LLMEngine(EngineConfig(model="gpt2", draft_model="distilgpt2",
                                  block_size=8, num_blocks=256,
                                  max_batch_size=4, num_speculative_tokens=4))
