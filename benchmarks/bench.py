"""Benchmark harness. All numbers are CPU-only, measured on the host in
benchmarks/results/*.json ("machine" field).

Subcommands:
  sweep        throughput + TTFT vs concurrency against a live server (SSE)
  baseline     sequential HF transformers generate() on the same workload
  speculative  in-process engine, speculative decoding on vs off
  all          run everything and write JSON results

Usage:
  python benchmarks/bench.py all --model gpt2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"

PROMPTS = [
    "The history of the printing press begins with",
    "In modern distributed systems, the hardest problem is",
    "Once upon a time in a small coastal village, a fisherman",
    "The recipe for a perfect loaf of bread starts with",
    "Quantum computing differs from classical computing because",
    "The detective examined the room carefully and noticed",
    "A gentle introduction to linear algebra should cover",
    "When the spacecraft finally reached the outer planets,",
    "The economics of renewable energy have shifted because",
    "My grandmother always told me that the secret to happiness",
    "The compiler reported an unusual error on line forty:",
    "Deep beneath the ocean surface, researchers discovered",
    "The rules of chess are simple to learn but",
    "During the industrial revolution, cities grew rapidly as",
    "The best way to train for a marathon involves",
    "In the year 2140, the last library on Earth",
]

MAX_TOKENS = 32


def machine_info() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(), "device": "cpu"}


def write_result(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"machine": machine_info(),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]


# ----------------------------------------------------------------------
# Serving sweep (HTTP + SSE against a real server process)
# ----------------------------------------------------------------------
async def _stream_one(client: httpx.AsyncClient, base: str, prompt: str) -> dict:
    payload = {"prompt": prompt, "max_tokens": MAX_TOKENS,
               "temperature": 0.0, "stream": True}
    start = time.perf_counter()
    ttft = None
    tokens = 0
    async with client.stream("POST", f"{base}/v1/completions",
                             json=payload, timeout=600.0) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - start
            tokens += 1
    return {"ttft_s": ttft, "latency_s": time.perf_counter() - start,
            "tokens": tokens}


async def _run_level(base: str, concurrency: int, num_requests: int) -> dict:
    prompts = [PROMPTS[i % len(PROMPTS)] + f" (case {i})"
               for i in range(num_requests)]
    sem = asyncio.Semaphore(concurrency)

    async def guarded(client: httpx.AsyncClient, p: str) -> dict:
        async with sem:
            return await _stream_one(client, base, p)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        results = await asyncio.gather(*(guarded(client, p) for p in prompts))
        wall = time.perf_counter() - start
    total_tokens = sum(r["tokens"] for r in results)
    ttfts = [r["ttft_s"] for r in results]
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "total_new_tokens": total_tokens,
        "wall_time_s": round(wall, 3),
        "throughput_tok_s": round(total_tokens / wall, 2),
        "ttft_mean_s": round(statistics.mean(ttfts), 3),
        "ttft_p50_s": round(percentile(ttfts, 50), 3),
        "ttft_p95_s": round(percentile(ttfts, 95), 3),
        "ttft_max_s": round(max(ttfts), 3),
    }


def start_server(model: str, port: int, extra: list[str] | None = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "server", "--model", model,
           "--port", str(port), "--num-blocks", "512",
           "--max-batch-size", "16"] + (extra or [])
    proc = subprocess.Popen(cmd, cwd=Path(__file__).parent.parent,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/healthz", timeout=2.0).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server failed to start")


def cmd_sweep(model: str) -> None:
    port = 8377
    proc = start_server(model, port)
    base = f"http://127.0.0.1:{port}"
    levels = []
    try:
        # Warm up (model load is done; prime caches and threads).
        asyncio.run(_run_level(base, 1, 2))
        for concurrency in (1, 4, 8, 16):
            level = asyncio.run(_run_level(base, concurrency,
                                           num_requests=max(8, concurrency)))
            print(level)
            levels.append(level)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    write_result("serving_sweep", {"model": model, "max_tokens": MAX_TOKENS,
                                   "temperature": 0.0, "levels": levels})


# ----------------------------------------------------------------------
# Sequential HF transformers baseline
# ----------------------------------------------------------------------
def cmd_baseline(model: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    ref = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    prompts = [PROMPTS[i % len(PROMPTS)] + f" (case {i})" for i in range(8)]
    # Warm up.
    with torch.inference_mode():
        ref.generate(tok.encode(prompts[0], return_tensors="pt"),
                     max_new_tokens=4, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    latencies = []
    total_tokens = 0
    start = time.perf_counter()
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = ref.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                               min_new_tokens=MAX_TOKENS,
                               pad_token_id=tok.eos_token_id)
        latencies.append(time.perf_counter() - t0)
        total_tokens += out.shape[1] - ids.shape[1]
    wall = time.perf_counter() - start
    result = {
        "model": model, "max_tokens": MAX_TOKENS, "num_requests": len(prompts),
        "total_new_tokens": total_tokens, "wall_time_s": round(wall, 3),
        "throughput_tok_s": round(total_tokens / wall, 2),
        "latency_mean_s": round(statistics.mean(latencies), 3),
    }
    print(result)
    write_result("hf_baseline", result)


# ----------------------------------------------------------------------
# Speculative decoding on vs off (in-process, single stream)
# ----------------------------------------------------------------------
def cmd_speculative(model: str, draft: str) -> None:
    from engine import EngineConfig, LLMEngine, SamplingParams

    prompts = PROMPTS[:4]
    params = SamplingParams(max_tokens=64, temperature=0.0)

    def run(engine: LLMEngine) -> tuple[float, int]:
        total, t0 = 0, time.perf_counter()
        for p in prompts:
            _, toks = engine.generate(p, params)
            total += len(toks)
        return time.perf_counter() - t0, total

    plain = LLMEngine(EngineConfig(model=model, num_blocks=512))
    run(plain)  # warm up
    plain_wall, plain_tokens = run(plain)

    spec = LLMEngine(EngineConfig(model=model, draft_model=draft,
                                  num_blocks=512, num_speculative_tokens=4))
    run(spec)  # warm up
    spec.spec_stats.drafted = spec.spec_stats.accepted = 0
    spec_wall, spec_tokens = run(spec)

    result = {
        "target_model": model, "draft_model": draft,
        "num_speculative_tokens": 4, "max_tokens": 64,
        "temperature": 0.0, "num_requests": len(prompts),
        "plain": {"tokens": plain_tokens, "wall_time_s": round(plain_wall, 3),
                  "throughput_tok_s": round(plain_tokens / plain_wall, 2)},
        "speculative": {"tokens": spec_tokens,
                        "wall_time_s": round(spec_wall, 3),
                        "throughput_tok_s": round(spec_tokens / spec_wall, 2),
                        "acceptance_rate": round(spec.spec_stats.acceptance_rate, 4),
                        "drafted": spec.spec_stats.drafted,
                        "accepted": spec.spec_stats.accepted},
        "speedup": round((spec_tokens / spec_wall) / (plain_tokens / plain_wall), 3),
    }
    print(result)
    write_result("speculative", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sweep", "baseline", "speculative", "all"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--draft", default="distilgpt2")
    args = parser.parse_args()
    if args.command in ("sweep", "all"):
        cmd_sweep(args.model)
    if args.command in ("baseline", "all"):
        cmd_baseline(args.model)
    if args.command in ("speculative", "all"):
        cmd_speculative(args.model, args.draft)


if __name__ == "__main__":
    main()
