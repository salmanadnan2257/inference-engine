# inference-engine

A from-scratch LLM inference and serving engine in Python + PyTorch, built to
understand how systems like vLLM actually work: paged KV cache with
copy-on-write, prefix caching, continuous batching with preemption,
speculative decoding, and token-by-token SSE streaming behind an
OpenAI-compatible HTTP API. Runs GPT-2 family models (gpt2, distilgpt2) on
CPU; Hugging Face is used only to download weights, the forward pass,
attention, cache and scheduler are all implemented here.

## Why

Serving engines are where ML meets systems programming: memory allocators,
schedulers, cache coherence, admission control. Reading vLLM's paper is one
thing; the goal here was to rebuild the core mechanisms small enough to fit
in one head, verify them against a reference implementation, and measure
them honestly on real hardware.

## Features

- Own GPT-2 forward implementation (loads HF weights into plain
  `nn.Linear`), verified against transformers to max abs logit
  diff < 1e-3 (measured ~3e-5 on distilgpt2, ~2e-4 on gpt2 in float32).
- Paged KV cache: fixed-size blocks, refcounted free lists, per-sequence
  block tables, copy-on-write on shared-block writes.
- Prefix caching: full blocks are content-hashed (hash-chained per prefix);
  identical prompt prefixes reuse cached blocks with LRU eviction.
- Continuous batching: requests join and leave the running batch at token
  boundaries; when KV blocks run out, the newest sequence is preempted and
  recomputed later.
- Speculative decoding: distilgpt2 drafts k tokens, gpt2 verifies them in
  one forward; standard accept/reject sampling that preserves the target
  distribution (tested empirically), acceptance rate reported in /metrics.
- Sampling: greedy, temperature, top-k, top-p, repetition penalty, seeded
  determinism, n > 1 completions (siblings fork prompt blocks via COW).
- Server: FastAPI, `/v1/completions` (OpenAI subset) with SSE streaming,
  `/v1/models`, `/healthz`, `/metrics`.

## Architecture

```
server/app.py        FastAPI + SSE fan-out
engine/async_engine  step loop in a worker thread, per-sequence queues
engine/llm_engine    synchronous core: prefill/decode/speculative rounds
engine/scheduler     admission, batching, preemption
engine/block_manager block metadata: free lists, prefix hashes, COW
engine/model         GPT-2 forward + paged attention (KVCache tensors)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram and
tradeoffs.

## Setup

```bash
python3 -m venv ~/venvs/inference-engine && source ~/venvs/inference-engine/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
export HF_HOME=/path/to/hf-cache   # keep weights outside the repo
```

Weights (distilgpt2 ~330 MB, gpt2 ~550 MB) download on first run. No weights
are stored in this repository.

## Usage

Serve:

```bash
python -m server --model gpt2 --draft-model distilgpt2 --port 8000
```

Stream a completion:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The meaning of life is", "max_tokens": 32,
       "temperature": 0.7, "seed": 42, "stream": true}'
```

Engine counters: `curl http://127.0.0.1:8000/metrics` (free blocks, prefix
cache hits, COW copies, preemptions, speculative acceptance rate).

Tests and benchmarks:

```bash
pytest                              # full suite
python benchmarks/bench.py all      # writes benchmarks/results/*.json
```

## Design notes

**Paged attention.** The naive KV cache reserves one contiguous tensor of
`max_len` per sequence, so memory is claimed by tokens that may never be
generated, and a long-max-tokens request blocks others even while short ones
finish. Paging borrows the OS trick: carve cache memory into fixed-size
blocks (16 tokens here), give each sequence a block *table* instead of a
region, and translate token position to physical slot on the fly. Now
allocation is on demand (waste is at most one partial block per sequence),
free blocks form a simple list (fragmentation cannot exist because every
block is the same size), and two sequences can point at the same physical
block. That last property is what makes prefix caching and n>1 forks nearly
free: sharing is a refcount bump, and only a write into a shared block forces
a copy (copy-on-write), exactly like fork(2) semantics for process memory.

**Continuous batching.** Static batching waits for a group of requests,
pads them together, and holds the batch until the slowest one finishes; the
GPU/CPU spends most of that time computing padding for finished rows.
Continuous batching reschedules at every token: the batch is whatever set of
live sequences exists *right now*, a new request can join at the next token
boundary (its prefill runs, then it decodes with everyone else), and a
finished sequence leaves immediately, releasing its blocks. The scheduler's
one hard invariant is that the entire decode batch must be able to grow by
one step; the check is cumulative across sequences (per-sequence checks admit
batches that collectively do not fit; that exact bug is pinned by a test).
When the check fails, the newest sequence is evicted and recomputed later,
which the prefix cache makes cheap.

## Benchmarks (CPU)

All numbers measured on this machine: Intel i7-1185G7 (8 threads), 32 GB
RAM, PyTorch CPU float32, greedy decoding, 32 new tokens per request.
Raw JSON with the exact settings is in `benchmarks/results/`; the harness is
`benchmarks/bench.py`. These are CPU numbers and should be read as
relative comparisons, not absolute serving performance.

**Throughput and TTFT vs concurrency** (gpt2, server + SSE, 32 tokens each,
`benchmarks/results/serving_sweep.json`):

| Concurrency | Throughput (tok/s) | TTFT mean (s) | TTFT p95 (s) |
|-------------|--------------------|---------------|--------------|
| 1           | 44.7               | 0.07          | 0.10         |
| 4           | 83.8               | 0.27          | 0.32         |
| 8           | 113.5              | 0.54          | 0.54         |
| 16          | 108.5              | 1.27          | 1.27         |

Continuous batching turns extra concurrent load into ~2.5x more aggregate
throughput up to 8 in-flight requests, then flattens as the 8-thread CPU
saturates; time-to-first-token grows roughly linearly with the queue depth,
which is the expected tradeoff.

**Batched engine vs sequential HF `generate()`** (gpt2, 8 requests, 32
tokens, greedy; `hf_baseline.json`):

| Setup                                  | Throughput (tok/s) |
|----------------------------------------|--------------------|
| transformers `generate()`, sequential  | 40.0               |
| this engine, 1 concurrent request       | 44.7               |
| this engine, 8 concurrent requests      | 113.5              |

At equal settings the engine matches HF one-at-a-time and is ~2.8x faster
once it can batch 8 requests together.

**Speculative decoding on vs off** (target gpt2, draft distilgpt2, k=4,
greedy, single stream; `speculative.json`):

| Mode                | Throughput (tok/s) | Acceptance rate |
|---------------------|--------------------|-----------------|
| plain               | 44.7               | n/a             |
| speculative         | 32.2               | 0.54            |

Speculative decoding is correct (identical greedy output, 54% of drafts
accepted) but ~0.7x slower here: on CPU the draft model's forward is not
cheap relative to the target, so the extra draft and verify passes cost more
than the target-forwards they save. Speculative decoding pays off when the
target forward dominates draft cost (large model, GPU), which this setup is
not. It stays in the codebase as a correctness-verified feature, not a CPU
speedup.

A comparison against vLLM is deliberately absent: vLLM requires a GPU and
this machine has none, so any number would be fabricated. That comparison is
future work (see below).

## Challenges

- **Prefix-cache correctness with in-place rewrites.** Speculative rollback
  and stop-token truncation both rewrite positions that a previous step had
  already hashed into the prefix cache. Leaving a stale hash pointing at a
  block whose tokens changed corrupts later lookups. The fix is to
  deregister a block's content hash the moment it is about to be rewritten
  (`BlockManager.append_slots` / `truncate`), pinned by
  `test_truncate_deregisters_rewritten_block`.
- **Cumulative vs per-sequence capacity checks.** The first scheduler checked
  each decode sequence against free blocks independently, which admits a
  batch that collectively does not fit and then crashes mid-step with an
  empty free list. The churn test caught it; the check is now summed across
  the whole batch.
- **Async test hangs.** The server's background step loop needs the fixture
  and every test to share one event loop; the default per-test loop made the
  API suite hang silently. Pinning `loop_scope="module"` fixed it (5s for
  10 tests instead of never finishing).
- **UTF-8 across token boundaries.** GPT-2 byte-level BPE can split a
  multi-byte character over two tokens, so naive per-token decoding emits
  U+FFFD. Detokenization holds back a partial tail until it decodes cleanly.

## What I learned

- Paged attention is mostly a bookkeeping problem, not a math problem: the
  attention kernel barely changes, but the allocator, refcounting, and
  hashing underneath it are where the design lives. Keeping the BlockManager
  pure-metadata (it returns copy ops, the runner touches tensors) made both
  halves testable in isolation.
- Continuous batching's throughput win is real and measurable even on CPU
  (2.5x from concurrency 1 to 8), and it comes entirely from not wasting
  compute on padding or on waiting for the slowest request.
- Speculative decoding's speedup is conditional on the draft being much
  cheaper than the target. Measuring it honestly on CPU showed it going
  backwards, which is a more useful thing to have learned than assuming the
  paper's GPU numbers transfer.
- Verifying against a reference (transformers logits to <1e-3) early made
  every later bug a systems bug, never a "did I get the model wrong" bug.

## What I'd do differently

- **The attention path is not truly paged at the kernel level.** It gathers
  each sequence's KV into a padded dense tensor and calls
  `scaled_dot_product_attention`, so the memory savings are real but the
  compute still touches padding. A fused varlen/paged kernel would fix that;
  on CPU with small models it was not the bottleneck, so I left it.
- **Prefills run one at a time.** Batching ragged prompts would cut TTFT
  under load. I skipped it because CPU prefill is already compute-bound, but
  on a GPU it would matter and the current design would need a flattened
  varlen prefill path.
- **One coarse engine lock.** Fine for CPU where the forward dominates, but
  it serializes all engine mutations; a real deployment would want to overlap
  prefill and decode and pipeline the sampler.
- **Preemption is recompute-only.** With a real device/host split, swapping
  KV blocks to host memory would beat recomputing long prompts. On CPU there
  is no split, so recompute was the honest choice.
- **Benchmarks are single-machine and CPU-only.** They show the right
  relative shapes but say nothing about absolute serving performance or GPU
  behavior, and there is no comparison against a production engine like vLLM
  (needs a GPU). That is the obvious next step.
