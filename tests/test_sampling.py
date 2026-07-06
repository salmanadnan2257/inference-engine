"""Sampling primitives: filtering, penalties, determinism."""

from __future__ import annotations

import torch

from engine.config import SamplingParams
from engine.sampling import adjust_logits, filter_logits, probs_for, sample_token


def test_greedy_picks_argmax():
    logits = torch.tensor([0.1, 2.0, -1.0, 1.9])
    params = SamplingParams(temperature=0.0)
    assert sample_token(logits, params, []) == 1


def test_top_k_masks_everything_else():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    params = SamplingParams(top_k=2)
    out = filter_logits(logits.clone(), params)
    assert out[0] == 5.0 and out[1] == 4.0
    assert torch.isinf(out[2:]).all()


def test_top_p_keeps_smallest_covering_set():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    out = filter_logits(logits.clone(), SamplingParams(top_p=0.7))
    # 0.5 alone < 0.7, so token 1 must survive too; tokens 2, 3 are cut.
    assert not torch.isinf(out[0]) and not torch.isinf(out[1])
    assert torch.isinf(out[2]) and torch.isinf(out[3])


def test_top_p_always_keeps_best_token():
    logits = torch.tensor([10.0, 0.0, 0.0])
    out = filter_logits(logits.clone(), SamplingParams(top_p=0.01))
    assert not torch.isinf(out[0])
    assert torch.isinf(out[1:]).all()


def test_repetition_penalty_discourages_seen_tokens():
    logits = torch.tensor([2.0, 2.0, -1.0])
    params = SamplingParams(repetition_penalty=2.0)
    out = adjust_logits(logits, params, token_ids=[0, 2])
    assert out[0] == 1.0    # positive logit divided
    assert out[1] == 2.0    # unseen token untouched
    assert out[2] == -2.0   # negative logit multiplied


def test_temperature_scales_before_softmax():
    logits = torch.tensor([1.0, 0.0])
    hot = probs_for(logits, SamplingParams(temperature=2.0), [])
    cold = probs_for(logits, SamplingParams(temperature=0.5), [])
    assert hot[0] < cold[0]  # higher temperature flattens


def test_seeded_sampling_is_deterministic():
    logits = torch.randn(100)
    params = SamplingParams(temperature=0.9, top_k=50, top_p=0.95, seed=7)
    def run() -> list[int]:
        gen = torch.Generator()
        gen.manual_seed(7)
        return [sample_token(logits, params, [], gen) for _ in range(20)]
    assert run() == run()


def test_probs_for_greedy_is_one_hot():
    logits = torch.tensor([0.5, 3.0, 1.0])
    probs = probs_for(logits, SamplingParams(temperature=0.0), [])
    assert probs.tolist() == [0.0, 1.0, 0.0]
