"""Logit parity of the from-scratch GPT-2 forward vs the transformers
reference implementation."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model import GPT2

PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "def fibonacci(n):\n    if n < 2:",
    "1 2 3 4 5 6 7 8 9",
]


@pytest.mark.parametrize("model_name", ["distilgpt2", "gpt2"])
def test_logit_parity(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mine = GPT2.from_hf(model_name)
    ref = AutoModelForCausalLM.from_pretrained(model_name,
                                               dtype=torch.float32).eval()
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt)])
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        with torch.inference_mode():
            got = mine(ids, positions)
            want = ref(ids).logits
        max_diff = (got - want).abs().max().item()
        assert max_diff < 1e-3, f"{model_name} {prompt!r}: {max_diff}"


def test_paged_forward_matches_cache_free():
    """The paged-attention path must produce the same logits as the plain
    causal path for the final position."""
    from engine.block_manager import BlockManager
    from engine.config import SamplingParams
    from engine.model import ModelRunner
    from engine.sequence import Sequence

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = GPT2.from_hf("distilgpt2")
    ids = tok.encode("Paged attention should not change the math at all")
    bm = BlockManager(64, 4, enable_prefix_caching=False)
    runner = ModelRunner(model, 64, 4)
    seq = Sequence(0, list(ids), SamplingParams(max_tokens=4))
    bm.allocate(seq)
    with torch.inference_mode():
        paged = runner.execute([seq], bm)[0, -1]
        plain = model(torch.tensor([ids]),
                      torch.arange(len(ids)).unsqueeze(0))[0, -1]
    assert (paged - plain).abs().max().item() < 1e-4
