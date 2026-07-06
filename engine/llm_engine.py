"""Synchronous engine core: owns the scheduler, executors, and sampling.

Drive it by calling :meth:`LLMEngine.step` in a loop; each call advances every
running sequence by one token boundary (one token normally, up to ``k + 1``
tokens with speculative decoding).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .block_manager import BlockManager
from .config import EngineConfig, SamplingParams
from .model import GPT2, ModelRunner
from .sampling import probs_for, sample_from_probs, sample_token
from .scheduler import Scheduler
from .sequence import FinishReason, Sequence, SequenceStatus
from .speculative import SpecStats, accept_reject, target_probs_from_logits

EOS_TOKEN_ID = 50256


@dataclass
class StepOutput:
    """Tokens produced for one sequence during one engine step."""

    seq_id: int
    new_token_ids: list[int]
    text_delta: str
    finished: bool
    finish_reason: str | None


@dataclass
class _SeqState:
    """Engine-side per-sequence state not owned by the scheduler."""

    emitted_text: str = ""


class Detokenizer:
    """Incremental detokenization with UTF-8 holdback.

    GPT-2 byte-level BPE can split a multi-byte character across tokens;
    decoding a partial character yields U+FFFD, so emission waits until the
    decoded tail is clean.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def delta(self, seq: Sequence, state: _SeqState) -> str:
        text = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta = text[len(state.emitted_text):]
        state.emitted_text = text
        return delta


class LLMEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = GPT2.from_hf(config.model)
        self.block_manager = BlockManager(config.num_blocks, config.block_size,
                                          config.enable_prefix_caching)
        self.runner = ModelRunner(self.model, config.num_blocks, config.block_size)
        lookahead = 0
        self.draft_runner: ModelRunner | None = None
        self.draft_bm: BlockManager | None = None
        if config.draft_model:
            draft = GPT2.from_hf(config.draft_model)
            self.draft_runner = ModelRunner(draft, config.num_blocks, config.block_size)
            # No prefix caching for the draft: freed blocks return to the free
            # list immediately, so draft usage never exceeds target usage and
            # a target-side capacity check covers both caches.
            self.draft_bm = BlockManager(config.num_blocks, config.block_size,
                                         enable_prefix_caching=False)
            lookahead = config.num_speculative_tokens
        self.scheduler = Scheduler(config, self.block_manager, lookahead)
        self.detokenizer = Detokenizer(self.tokenizer)
        self.seq_states: dict[int, _SeqState] = {}
        self.shadows: dict[int, Sequence] = {}
        self._pending_forks: dict[int, list[Sequence]] = {}
        self.spec_stats = SpecStats()
        self._next_seq_id = 0

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    def add_request(self, prompt: str | list[int], params: SamplingParams) -> list[int]:
        """Queue a request; returns one seq_id per requested completion (n)."""
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        max_len = len(token_ids) + params.max_tokens
        if max_len > self.config.max_model_len:
            raise ValueError(
                f"prompt + max_tokens = {max_len} exceeds max_model_len "
                f"{self.config.max_model_len}")
        worst_blocks = (max_len + self.config.num_speculative_tokens
                        + self.config.block_size - 1) // self.config.block_size
        if worst_blocks > self.config.num_blocks:
            raise ValueError("request cannot fit in KV cache even when alone")
        seq_ids = []
        parent: Sequence | None = None
        for i in range(params.n):
            seq = Sequence(self._next_seq_id, list(token_ids), params,
                           arrival_time=time.monotonic())
            if params.seed is not None and i > 0:
                seq.generator = torch.Generator()
                seq.generator.manual_seed(params.seed + i)
            self._next_seq_id += 1
            self.seq_states[seq.seq_id] = _SeqState()
            seq_ids.append(seq.seq_id)
            if i == 0:
                parent = seq
                self.scheduler.add(seq)
            else:
                # Siblings fork off the parent's prompt blocks after its
                # prefill (copy-on-write); they never re-run the prompt.
                self._pending_forks.setdefault(parent.seq_id, []).append(seq)
        return seq_ids

    def abort(self, seq_id: int) -> None:
        for child in self._pending_forks.pop(seq_id, []):
            child.status = SequenceStatus.FINISHED
            self.seq_states.pop(child.seq_id, None)
        for seq in list(self.scheduler.running):
            if seq.seq_id == seq_id:
                self._finish(seq, FinishReason.STOP)
                return
        for seq in list(self.scheduler.waiting):
            if seq.seq_id == seq_id:
                self.scheduler.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED
                self.seq_states.pop(seq_id, None)
                return

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(self) -> list[StepOutput]:
        """Advance all running sequences by one token boundary."""
        decision = self.scheduler.schedule()
        for seq in decision.preempted:
            self._drop_shadow(seq)
        outputs: list[StepOutput] = []

        for seq in decision.prefill:
            logits = self.runner.execute([seq], self.block_manager)
            for child in self._pending_forks.pop(seq.seq_id, []):
                tok = sample_token(logits[0, -1], child.params, child.token_ids,
                                   child.generator)
                if len(self.scheduler.running) < self.config.max_batch_size:
                    self.block_manager.fork(seq, child)
                    child.append_token(tok)
                    child.status = SequenceStatus.RUNNING
                    self.scheduler.running.append(child)
                    outputs.append(self._postprocess(child, [tok]))
                else:
                    # No batch room: requeue as an ordinary request; the
                    # prefix cache makes its prefill nearly free.
                    self.scheduler.add(child)
            token = sample_token(logits[0, -1], seq.params, seq.token_ids,
                                 seq.generator)
            seq.append_token(token)
            outputs.append(self._postprocess(seq, [token]))

        decode = [s for s in decision.decode if not s.is_finished]
        if decode:
            if self.draft_runner is not None:
                outputs.extend(self._speculative_decode(decode))
            else:
                outputs.extend(self._decode(decode))
        return outputs

    def _decode(self, seqs: list[Sequence]) -> list[StepOutput]:
        copies = []
        for seq in seqs:
            copies += self.block_manager.append_slots(seq)
        self.runner.apply_copies(copies)
        logits = self.runner.execute(seqs, self.block_manager)
        outputs = []
        for i, seq in enumerate(seqs):
            token = sample_token(logits[i, -1], seq.params, seq.token_ids,
                                 seq.generator)
            seq.append_token(token)
            outputs.append(self._postprocess(seq, [token]))
        return outputs

    # ------------------------------------------------------------------
    # Speculative decoding
    # ------------------------------------------------------------------
    def _shadow(self, seq: Sequence) -> Sequence:
        """Draft-side mirror of a target sequence (own cache bookkeeping)."""
        shadow = self.shadows.get(seq.seq_id)
        if shadow is None or not shadow.block_table:
            shadow = Sequence(seq.seq_id, list(seq.token_ids), seq.params)
            shadow.num_prompt_tokens = seq.num_prompt_tokens
            assert self.draft_bm is not None
            if not self.draft_bm.can_allocate(shadow):
                raise RuntimeError("draft KV cache exhausted")
            self.draft_bm.allocate(shadow)
            self.shadows[seq.seq_id] = shadow
        else:
            shadow.token_ids = list(seq.token_ids)
            shadow.num_computed = min(shadow.num_computed, len(shadow) - 1)
        return shadow

    def _drop_shadow(self, seq: Sequence) -> None:
        shadow = self.shadows.pop(seq.seq_id, None)
        if shadow is not None and self.draft_bm is not None:
            self.draft_bm.free_seq(shadow)

    def _speculative_decode(self, seqs: list[Sequence]) -> list[StepOutput]:
        assert self.draft_runner is not None and self.draft_bm is not None
        # All sequences in the batch must draft the same number of tokens so
        # the verify pass stays a single rectangular forward.
        k = min(self.config.num_speculative_tokens,
                min(self.config.max_model_len - len(s) for s in seqs),
                min(s.num_prompt_tokens + s.params.max_tokens - len(s)
                    for s in seqs))
        if k < 1:
            return self._decode(seqs)
        shadows = [self._shadow(s) for s in seqs]
        generators = {s.seq_id: s.generator for s in seqs}

        # Draft k tokens autoregressively. Shadows that need to catch up
        # (fresh or after rollback) run alone; steady-state shadows batch.
        # Draft sampling draws from the same generator stream as the later
        # accept/reject pass, keeping seeded runs deterministic while the
        # draws stay independent.
        draft_tokens: dict[int, list[int]] = {s.seq_id: [] for s in seqs}
        draft_probs: dict[int, list[torch.Tensor]] = {s.seq_id: [] for s in seqs}
        for _ in range(k):
            groups: dict[int, list[Sequence]] = {}
            for sh in shadows:
                groups.setdefault(len(sh) - sh.num_computed, []).append(sh)
            for group in groups.values():
                copies = []
                for sh in group:
                    copies += self.draft_bm.append_slots(sh)
                self.draft_runner.apply_copies(copies)
                logits = self.draft_runner.execute(group, self.draft_bm)
                for i, sh in enumerate(group):
                    q = probs_for(logits[i, -1], sh.params, sh.token_ids)
                    tok = sample_from_probs(q, generators[sh.seq_id])
                    draft_tokens[sh.seq_id].append(tok)
                    draft_probs[sh.seq_id].append(q)
                    sh.append_token(tok)

        # Verify all drafts in one batched target forward (T = k + 1).
        copies = []
        for seq in seqs:
            seq.token_ids.extend(draft_tokens[seq.seq_id])
            copies += self.block_manager.append_slots(seq)
        self.runner.apply_copies(copies)
        logits = self.runner.execute(seqs, self.block_manager)

        outputs = []
        for i, (seq, shadow) in enumerate(zip(seqs, shadows)):
            base_len = len(seq) - k  # length before drafts were appended
            target_probs = target_probs_from_logits(logits[i], seq.params,
                                                    seq.token_ids)
            result = accept_reject(draft_tokens[seq.seq_id],
                                   draft_probs[seq.seq_id], target_probs,
                                   seq.generator, self.spec_stats)
            new_len = base_len + result.num_accepted
            seq.token_ids = seq.token_ids[:new_len] + [result.tokens[-1]]
            seq.num_computed = new_len
            self.block_manager.truncate(seq, len(seq.token_ids))
            shadow.token_ids = list(seq.token_ids)
            shadow.num_computed = min(shadow.num_computed, new_len)
            self.draft_bm.truncate(shadow, len(shadow.token_ids))
            outputs.append(self._postprocess(seq, result.tokens))
        return outputs

    # ------------------------------------------------------------------
    # Finishing and detokenization
    # ------------------------------------------------------------------
    def _postprocess(self, seq: Sequence, new_tokens: list[int]) -> StepOutput:
        state = self.seq_states[seq.seq_id]
        finish: FinishReason | None = None

        if EOS_TOKEN_ID in new_tokens:
            cut = new_tokens.index(EOS_TOKEN_ID)
            drop = len(new_tokens) - cut
            self._truncate_tail(seq, drop)
            new_tokens = new_tokens[:cut]
            finish = FinishReason.STOP
        if seq.num_output_tokens >= seq.params.max_tokens:
            drop = seq.num_output_tokens - seq.params.max_tokens
            if drop > 0:
                self._truncate_tail(seq, drop)
                new_tokens = new_tokens[:len(new_tokens) - drop]
            finish = finish or FinishReason.LENGTH

        delta = self.detokenizer.delta(seq, state) if new_tokens else ""
        if seq.params.stop and delta:
            emitted_before = state.emitted_text[:len(state.emitted_text) - len(delta)]
            for stop in seq.params.stop:
                idx = (emitted_before + delta).find(stop)
                if idx != -1:
                    delta = (emitted_before + delta)[len(emitted_before):idx]
                    state.emitted_text = emitted_before + delta
                    finish = FinishReason.STOP
                    break

        if finish is not None:
            self._finish(seq, finish)
        return StepOutput(seq.seq_id, new_tokens, delta,
                          finish is not None,
                          finish.value if finish else None)

    def _truncate_tail(self, seq: Sequence, drop: int) -> None:
        seq.token_ids = seq.token_ids[:len(seq.token_ids) - drop]
        seq.num_computed = min(seq.num_computed, len(seq.token_ids) - 1)
        self.block_manager.truncate(seq, len(seq.token_ids))

    def _finish(self, seq: Sequence, reason: FinishReason) -> None:
        seq.finish_reason = reason
        self.scheduler.finish_seq(seq)
        self._drop_shadow(seq)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def generate(self, prompt: str | list[int],
                 params: SamplingParams) -> tuple[str, list[int]]:
        """Blocking single-request generation (tests, benchmarks)."""
        seq_ids = self.add_request(prompt, params)
        texts: dict[int, str] = {sid: "" for sid in seq_ids}
        tokens: dict[int, list[int]] = {sid: [] for sid in seq_ids}
        pending = set(seq_ids)
        while pending and self.has_work:
            for out in self.step():
                if out.seq_id in pending:
                    texts[out.seq_id] += out.text_delta
                    tokens[out.seq_id].extend(out.new_token_ids)
                    if out.finished:
                        pending.discard(out.seq_id)
        sid = seq_ids[0]
        return texts[sid], tokens[sid]
