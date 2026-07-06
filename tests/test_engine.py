"""End-to-end engine behavior: HF equivalence, continuous batching,
prefix caching, determinism, COW forks, preemption under pressure."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import EngineConfig, LLMEngine, SamplingParams

PROMPT = "The meaning of life is"
LONG_PROMPT = ("In a distant future, humanity has spread across the stars, "
               "building vast networks of cities on worlds that once seemed "
               "unreachable. The engineers who designed the first ships")


def hf_greedy(model_name: str, prompt: str, max_new: int) -> str:
    tok = AutoTokenizer.from_pretrained(model_name)
    ref = AutoModelForCausalLM.from_pretrained(model_name,
                                               dtype=torch.float32).eval()
    ids = tok.encode(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = ref.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:])


def test_greedy_matches_hf_generate(distil_engine):
    text, _ = distil_engine.generate(PROMPT,
                                     SamplingParams(max_tokens=20, temperature=0.0))
    assert text == hf_greedy("distilgpt2", PROMPT, 20)


def test_continuous_batching_matches_solo_runs(distil_engine):
    """Requests that join mid-flight must not disturb each other's tokens."""
    eng = distil_engine
    params = SamplingParams(max_tokens=12, temperature=0.0)
    solo_a, _ = eng.generate("The cat sat on", params)
    solo_b, _ = eng.generate("Once upon a time there was", params)

    a_ids = eng.add_request("The cat sat on", params)
    outs: dict[int, str] = {}
    done: set[int] = set()
    b_ids: list[int] = []
    steps = 0
    while eng.has_work:
        if steps == 3:  # b joins while a is mid-generation
            b_ids = eng.add_request("Once upon a time there was", params)
        for out in eng.step():
            outs[out.seq_id] = outs.get(out.seq_id, "") + out.text_delta
            if out.finished:
                done.add(out.seq_id)
        steps += 1
    assert done == {a_ids[0], b_ids[0]}
    assert outs[a_ids[0]] == solo_a
    assert outs[b_ids[0]] == solo_b


def test_prefix_cache_reuses_blocks_and_output_is_identical(distil_engine):
    eng = distil_engine
    params = SamplingParams(max_tokens=15, temperature=0.0)
    hits_before = eng.block_manager.stats.prefix_cache_hit_tokens
    t1, ids1 = eng.generate(LONG_PROMPT, params)
    hits_mid = eng.block_manager.stats.prefix_cache_hit_tokens
    t2, ids2 = eng.generate(LONG_PROMPT, params)
    hits_after = eng.block_manager.stats.prefix_cache_hit_tokens
    prompt_len = len(eng.tokenizer.encode(LONG_PROMPT))
    expected_full_blocks = (prompt_len - 1) // eng.config.block_size
    assert hits_after - hits_mid == expected_full_blocks * eng.config.block_size
    assert hits_after - hits_mid > hits_mid - hits_before
    assert (t1, ids1) == (t2, ids2)


def test_seeded_determinism(distil_engine):
    params = SamplingParams(max_tokens=20, temperature=0.9, top_p=0.9,
                            top_k=100, seed=1234)
    r1 = distil_engine.generate(PROMPT, params)
    r2 = distil_engine.generate(PROMPT, params)
    assert r1 == r2
    r3 = distil_engine.generate(PROMPT, SamplingParams(
        max_tokens=20, temperature=0.9, top_p=0.9, top_k=100, seed=99))
    assert r3 != r1


def test_n_completions_fork_with_cow(distil_engine):
    eng = distil_engine
    cows_before = eng.block_manager.stats.cow_copies
    params = SamplingParams(max_tokens=10, temperature=0.8, seed=7, n=2)
    seq_ids = eng.add_request(PROMPT, params)
    texts = {sid: "" for sid in seq_ids}
    finished: set[int] = set()
    while eng.has_work and len(finished) < 2:
        for out in eng.step():
            if out.seq_id in texts:
                texts[out.seq_id] += out.text_delta
                if out.finished:
                    finished.add(out.seq_id)
    assert len(finished) == 2
    assert all(texts.values())
    # The sibling forked the prompt blocks and copied on first write.
    assert eng.block_manager.stats.cow_copies > cows_before


def test_preemption_under_kv_pressure_still_completes():
    eng = LLMEngine(EngineConfig(model="distilgpt2", block_size=4,
                                 num_blocks=24, max_batch_size=4,
                                 enable_prefix_caching=False))
    params = SamplingParams(max_tokens=30, temperature=0.0)
    ids = [eng.add_request(p, params)[0]
           for p in ["One", "Two", "Three", "Four"]]
    texts = {sid: "" for sid in ids}
    finished: set[int] = set()
    while eng.has_work:
        for out in eng.step():
            texts[out.seq_id] += out.text_delta
            if out.finished:
                finished.add(out.seq_id)
    assert finished == set(ids)
    assert eng.scheduler.num_preemptions > 0
    # Preempted sequences recompute and still match an unpressured run.
    solo = LLMEngine(EngineConfig(model="distilgpt2", block_size=4,
                                  num_blocks=256, max_batch_size=4))
    for prompt, sid in zip(["One", "Two", "Three", "Four"], ids):
        want, _ = solo.generate(prompt, params)
        assert texts[sid] == want


def test_stop_string_truncates(distil_engine):
    params = SamplingParams(max_tokens=40, temperature=0.0, stop=["."])
    text, _ = distil_engine.generate(PROMPT, params)
    assert "." not in text


def test_max_model_len_rejected(distil_engine):
    try:
        distil_engine.add_request("hi", SamplingParams(max_tokens=5000))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
