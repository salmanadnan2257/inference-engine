"""Token sampling: greedy, temperature, top-k, top-p, repetition penalty."""

from __future__ import annotations

import torch

from .config import SamplingParams


def adjust_logits(logits: torch.Tensor, params: SamplingParams,
                  token_ids: list[int]) -> torch.Tensor:
    """Apply repetition penalty and temperature to a 1D logits vector."""
    logits = logits.clone()
    if params.repetition_penalty != 1.0 and token_ids:
        seen = torch.tensor(sorted(set(token_ids)))
        vals = logits[seen]
        logits[seen] = torch.where(vals > 0, vals / params.repetition_penalty,
                                   vals * params.repetition_penalty)
    if not params.greedy and params.temperature != 1.0:
        logits = logits / params.temperature
    return logits


def filter_logits(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Apply top-k and top-p filtering to a 1D logits vector."""
    if params.top_k > 0 and params.top_k < logits.shape[-1]:
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Keep the smallest set of tokens whose cumulative probability
        # reaches top_p (the token that crosses the threshold stays).
        remove = cum - torch.softmax(sorted_logits, dim=-1) >= params.top_p
        remove[0] = False
        logits = logits.clone()
        logits[sorted_idx[remove]] = float("-inf")
    return logits


def probs_for(logits: torch.Tensor, params: SamplingParams,
              token_ids: list[int]) -> torch.Tensor:
    """Full sampling distribution for one position (used by speculative)."""
    logits = filter_logits(adjust_logits(logits, params, token_ids), params)
    if params.greedy:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
    return torch.softmax(logits, dim=-1)


def sample_token(logits: torch.Tensor, params: SamplingParams,
                 token_ids: list[int],
                 generator: torch.Generator | None = None) -> int:
    """Sample the next token id from a 1D logits vector."""
    logits = adjust_logits(logits, params, token_ids)
    if params.greedy:
        return int(torch.argmax(logits))
    logits = filter_logits(logits, params)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def sample_from_probs(probs: torch.Tensor,
                      generator: torch.Generator | None = None) -> int:
    if torch.count_nonzero(probs) == 1:
        return int(torch.argmax(probs))
    return int(torch.multinomial(probs, 1, generator=generator))
