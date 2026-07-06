"""Speculative decoding correctness.

Two layers of evidence:
1. Greedy speculative output is exactly the target model's greedy output.
2. On a tiny random transformer with a small vocab, the empirical
   distribution of the first token emitted by a full draft-verify-accept
   round matches the target distribution (the core guarantee of
   speculative sampling).
"""

from __future__ import annotations

import torch

from engine.config import SamplingParams
from engine.model import GPT2, ModelConfig
from engine.sampling import probs_for
from engine.speculative import SpecStats, accept_reject, target_probs_from_logits

VOCAB = 13
K = 3


def tiny_model(seed: int) -> GPT2:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, n_positions=32, n_embd=16,
                      n_layer=2, n_head=2)
    model = GPT2(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.4)
    model.eval()
    return model


@torch.inference_mode()
def last_logits(model: GPT2, ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids])
    pos = torch.arange(len(ids)).unsqueeze(0)
    return model(x, pos)[0, -1]


@torch.inference_mode()
def spec_round(target: GPT2, draft: GPT2, prompt: list[int],
               params: SamplingParams, stats: SpecStats) -> int:
    """One full speculative round; returns the first emitted token."""
    ids = list(prompt)
    draft_tokens: list[int] = []
    draft_probs: list[torch.Tensor] = []
    for _ in range(K):
        q = probs_for(last_logits(draft, ids), params, ids)
        tok = int(torch.multinomial(q, 1))
        draft_tokens.append(tok)
        draft_probs.append(q)
        ids.append(tok)
    x = torch.tensor([ids])
    pos = torch.arange(len(ids)).unsqueeze(0)
    logits = target(x, pos)[0, len(prompt) - 1:]
    target_probs = target_probs_from_logits(logits, params, ids)
    return accept_reject(draft_tokens, draft_probs, target_probs,
                         stats=stats).tokens[0]


def total_variation(counts: torch.Tensor, ref: torch.Tensor) -> float:
    emp = counts / counts.sum()
    return 0.5 * float((emp - ref).abs().sum())


def test_empirical_distribution_matches_target():
    target, draft = tiny_model(0), tiny_model(1)
    prompt = [1, 2, 3, 4, 5]
    params = SamplingParams(max_tokens=8, temperature=1.0)
    ref = probs_for(last_logits(target, prompt), params, prompt)
    torch.manual_seed(42)
    stats = SpecStats()
    counts = torch.zeros(VOCAB)
    n = 4000
    for _ in range(n):
        counts[spec_round(target, draft, prompt, params, stats)] += 1
    tv = total_variation(counts, ref)
    # Expected TV for 4000 samples over 13 symbols is about 0.02.
    assert tv < 0.05, f"TV distance {tv:.4f}"
    assert 0.0 < stats.acceptance_rate < 1.0


def test_empirical_distribution_with_temperature_and_top_k():
    """The guarantee holds for transformed distributions too, as long as
    draft and target apply the same transformation."""
    target, draft = tiny_model(2), tiny_model(3)
    prompt = [6, 7, 8]
    params = SamplingParams(max_tokens=8, temperature=0.7, top_k=6)
    ref = probs_for(last_logits(target, prompt), params, prompt)
    torch.manual_seed(7)
    stats = SpecStats()
    counts = torch.zeros(VOCAB)
    for _ in range(4000):
        counts[spec_round(target, draft, prompt, params, stats)] += 1
    tv = total_variation(counts, ref)
    assert tv < 0.05, f"TV distance {tv:.4f}"


def test_identical_models_accept_everything():
    model = tiny_model(4)
    params = SamplingParams(max_tokens=8, temperature=1.0)
    stats = SpecStats()
    torch.manual_seed(11)
    for _ in range(50):
        spec_round(model, model, [1, 2, 3], params, stats)
    assert stats.acceptance_rate > 0.999


def test_greedy_accept_reject_is_exact():
    """With temperature 0 both distributions are one-hot: accepted iff the
    draft argmax equals the target argmax, and the correction token is the
    target argmax."""
    p1 = torch.zeros(5); p1[2] = 1.0
    p2 = torch.zeros(5); p2[4] = 1.0
    q = torch.zeros(5); q[2] = 1.0
    res = accept_reject([2, 2], [q, q], [p1, p2, p1])
    # First draft accepted (matches argmax), second rejected -> emits 4.
    assert res.tokens == [2, 4]
    assert res.num_accepted == 1


def test_spec_engine_greedy_matches_plain_engine(spec_engine, gpt2_engine):
    params = SamplingParams(max_tokens=24, temperature=0.0)
    prompt = "The meaning of life is"
    spec_text, spec_ids = spec_engine.generate(prompt, params)
    plain_text, plain_ids = gpt2_engine.generate(prompt, params)
    assert spec_ids == plain_ids
    assert spec_text == plain_text
    assert spec_engine.spec_stats.drafted > 0
    assert 0.0 < spec_engine.spec_stats.acceptance_rate <= 1.0


def test_spec_engine_seeded_sampling_is_deterministic(spec_engine):
    params = SamplingParams(max_tokens=16, temperature=0.8, seed=321)
    r1 = spec_engine.generate("Deep in the forest", params)
    r2 = spec_engine.generate("Deep in the forest", params)
    assert r1 == r2
