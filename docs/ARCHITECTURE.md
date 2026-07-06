# Architecture

The engine is four layers. Requests flow down, tokens flow back up.

```
                 HTTP clients (curl, httpx, OpenAI SDKs)
                        |  POST /v1/completions (SSE)
  +---------------------v----------------------------------------+
  |  server/app.py           FastAPI, OpenAI-compatible surface  |
  |  - request validation (pydantic)                             |
  |  - SSE fan-out, one asyncio.Queue per sequence               |
  +---------------------v----------------------------------------+
  |  engine/async_engine.py  AsyncEngine                         |
  |  - background step loop, blocking work in a worker thread    |
  |  - one lock serializes step()/add_request()/abort()          |
  +---------------------v----------------------------------------+
  |  engine/llm_engine.py    LLMEngine (synchronous core)        |
  |                                                              |
  |   +------------------+       +---------------------------+   |
  |   | Scheduler        |       | BlockManager              |   |
  |   | waiting deque    |<----->| free list + refcounts     |   |
  |   | running list     |       | prefix hash table (LRU)   |   |
  |   | preemption       |       | COW copy ops              |   |
  |   +------------------+       +---------------------------+   |
  |            |   token ids, block tables, slot mappings        |
  |   +--------v--------------------------------------------+    |
  |   | ModelRunner + GPT2 (engine/model.py)                |    |
  |   | paged attention over KVCache tensors                |    |
  |   | [num_blocks * block_size, heads, head_dim] / layer  |    |
  |   +-----------------------------------------------------+    |
  |                                                              |
  |   Speculative path: a second (draft) GPT2 + BlockManager     |
  |   shadows each sequence; the target verifies k drafts in     |
  |   one forward (engine/speculative.py).                       |
  +--------------------------------------------------------------+
```

## Engine step

`LLMEngine.step()` advances every live sequence by one token boundary:

1. `Scheduler.schedule()` admits waiting prompts while there is batch room
   and KV space, then checks (cumulatively, not per sequence) that the whole
   decode batch can grow by one step. If not, the newest sequence is
   preempted: blocks freed, sequence requeued at the front of the waiting
   queue for recomputation.
2. Prefills run one at a time (variable prompt lengths), each producing its
   first token. `n > 1` siblings fork the parent's block table here instead
   of re-running the prompt.
3. All decoding sequences run as one batched forward (each contributes one
   token position; padding only in the attention key dimension).
4. Sampling, incremental detokenization (with UTF-8 holdback), stop-string
   and EOS handling, and stream fan-out.

## Paged KV cache

KV entries live in per-layer tensors shaped
`[num_blocks * block_size, num_heads, head_dim]`, so a flat slot id addresses
one token. Sequences own logical block tables (lists of block ids); position
`p` maps to slot `table[p // block_size] * block_size + p % block_size`.
Consequences:

- No contiguous reservation per sequence, so no external fragmentation and no
  up-front allocation for `max_tokens`. Waste is bounded by one partial block
  per sequence.
- Blocks are refcounted. Prefix caching and forks share physical blocks by
  bumping refcounts; a write into a shared block triggers copy-on-write. The
  BlockManager only records metadata and returns `(src, dst)` copy pairs; the
  ModelRunner applies them to the tensors.
- Full blocks are content-hashed (SHA-256 over the hash-chain of the prefix
  plus the block's token ids, so identical token windows with different
  prefixes never collide). Freed hashed blocks go to an LRU "evictable" pool
  instead of the free list: a later request with the same prompt prefix
  re-references them and skips recomputing those positions entirely.

## Scheduling and preemption

Admission is FIFO. Preemption picks the victim with the newest arrival time,
frees all its blocks, and requeues it for full recomputation (recompute
rather than swap: there is no second memory tier here, and on re-admission
the prefix cache usually restores most of the prompt for free). The capacity
check before a decode step sums worst-case block demand (growth plus
potential COW copies) across the whole batch; checking sequences one at a
time admits batches that collectively do not fit, which is exactly the bug
class the churn test in `tests/test_scheduler.py` covers.

## Speculative decoding

The draft model (distilgpt2) keeps its own paged cache and a shadow sequence
per target sequence. Each round: draft k tokens autoregressively, then one
target forward over k+1 positions scores all of them. Acceptance follows
standard speculative sampling: accept draft token x with probability
`min(1, p(x)/q(x))`; on the first rejection sample from
`normalize(max(p - q, 0))` and roll back both caches (block-table truncation,
plus deregistering any prefix-cache hash for a partially rewritten block).
The emitted stream is distributed exactly as target-only sampling; with
temperature 0 both distributions are one-hot and the scheme degenerates to
"accept while the draft argmax equals the target argmax".

Both models transform logits identically (temperature, top-k/top-p,
repetition penalty) before proposing/verifying, so the guarantee applies to
the transformed distributions.

## Design tradeoffs

- CPU first. Attention gathers per-sequence K/V into a padded dense tensor
  and calls `scaled_dot_product_attention`; on a GPU you would write a fused
  paged-attention kernel instead. The bookkeeping layers above would not
  change, which is the point of keeping them separate.
- One engine lock, step-level granularity. Simple to reason about, and the
  model forward dominates step time on CPU anyway. Finer-grained scheduling
  (overlapping prefill with decode) would need a different execution model.
- Prefills run unbatched. Batching ragged prompts needs either padding waste
  or a flattened varlen attention path; with CPU prefill already
  compute-bound, the added complexity bought little here.
- Recompute-only preemption. Swap-to-host is the other option; on CPU the
  "device" and host are the same memory, so swapping would be a no-op with
  extra bookkeeping.
- The draft cache runs without prefix caching so its block usage is always a
  subset of the target's; one capacity check then covers both caches.
