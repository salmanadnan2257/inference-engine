"""Speculative decoding: a small draft model proposes, the target verifies.

Uses standard speculative sampling (Leviathan et al., 2023): the draft
proposes ``k`` tokens from its distribution q; the target scores all of them
in one batched forward pass giving p. Token i is accepted with probability
``min(1, p(x)/q(x))``; on the first rejection a replacement is sampled from
``normalize(max(p - q, 0))``. The emitted tokens are distributed exactly as if
sampled from the target model alone, for greedy and stochastic sampling alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SamplingParams
from .sampling import probs_for, sample_from_probs


@dataclass
class SpecStats:
    drafted: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


@dataclass
class SpecResult:
    """Outcome of verifying one sequence's draft."""

    tokens: list[int]        # accepted draft tokens plus the final sampled token
    num_accepted: int


def accept_reject(draft_tokens: list[int], draft_probs: list[torch.Tensor],
                  target_probs: list[torch.Tensor],
                  generator: torch.Generator | None = None,
                  stats: SpecStats | None = None) -> SpecResult:
    """Run the accept/reject loop for one sequence.

    ``target_probs`` must have ``len(draft_tokens) + 1`` entries: one for each
    drafted position plus the bonus position used when everything is accepted.
    """
    k = len(draft_tokens)
    assert len(draft_probs) == k and len(target_probs) == k + 1
    accepted: list[int] = []
    for i, tok in enumerate(draft_tokens):
        p, q = target_probs[i], draft_probs[i]
        p_tok, q_tok = float(p[tok]), float(q[tok])
        ratio = 1.0 if q_tok == 0.0 else min(1.0, p_tok / q_tok)
        u = float(torch.rand((), generator=generator)) if ratio < 1.0 else 0.0
        if u < ratio:
            accepted.append(tok)
            continue
        residual = torch.clamp(p - q, min=0.0)
        total = float(residual.sum())
        if total <= 0.0:
            # p == q at this position; any rejection here is a numerical
            # artifact, fall back to sampling from p directly.
            residual, total = p, 1.0
        final = sample_from_probs(residual / total, generator)
        if stats is not None:
            stats.drafted += k
            stats.accepted += len(accepted)
        return SpecResult(accepted + [final], len(accepted))
    bonus = sample_from_probs(target_probs[k], generator)
    if stats is not None:
        stats.drafted += k
        stats.accepted += k
    return SpecResult(accepted + [bonus], k)


def target_probs_from_logits(logits: torch.Tensor, params: SamplingParams,
                             token_ids: list[int]) -> list[torch.Tensor]:
    """Convert verify-pass logits [T, V] into per-position distributions.

    ``token_ids`` is the full context including the drafted tokens, so the
    repetition penalty sees the correct history at each verified position:
    position i predicts the token at index ``len(token_ids) - T + 1 + i``.
    """
    T = logits.shape[0]
    base = len(token_ids) - T + 1
    out = []
    for i in range(T):
        ctx = token_ids[:base + i]
        out.append(probs_for(logits[i], params, ctx))
    return out
