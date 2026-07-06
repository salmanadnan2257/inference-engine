"""Paged KV cache bookkeeping: blocks, free lists, prefix cache, copy-on-write.

The block manager owns only metadata. Physical KV tensors live in
``engine.cache.KVCache``; the manager hands out block ids and tells the model
runner which physical copies to perform (for COW) via lists of
``(src_block, dst_block)`` pairs.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass

from .sequence import Sequence

CopyOp = tuple[int, int]


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    # Hash over (prefix chain, token ids) once the block is full, else None.
    content_hash: bytes | None = None


def _hash_block(prev_hash: bytes, token_ids: list[int]) -> bytes:
    h = hashlib.sha256(prev_hash)
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return h.digest()


@dataclass
class BlockManagerStats:
    prefix_cache_queries: int = 0
    prefix_cache_hit_tokens: int = 0
    cow_copies: int = 0
    evictions: int = 0


class BlockManager:
    """Allocator for fixed-size KV cache blocks with prefix caching and COW."""

    def __init__(self, num_blocks: int, block_size: int,
                 enable_prefix_caching: bool = True) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids: deque[int] = deque(range(num_blocks))
        # ref_count == 0 blocks that still hold reusable cached content, in
        # LRU order (front = evict first).
        self.evictable: OrderedDict[int, None] = OrderedDict()
        self.hash_to_block: dict[bytes, int] = {}
        self.stats = BlockManagerStats()

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids) + len(self.evictable)

    def _pop_free_block(self) -> int:
        if self.free_ids:
            bid = self.free_ids.popleft()
        else:
            bid, _ = self.evictable.popitem(last=False)
            self._deregister(bid)
            self.stats.evictions += 1
        block = self.blocks[bid]
        block.ref_count = 1
        return bid

    def _deregister(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.content_hash is not None:
            if self.hash_to_block.get(block.content_hash) == block_id:
                del self.hash_to_block[block.content_hash]
            block.content_hash = None

    def _ref_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.evictable.pop(block_id, None)
        block.ref_count += 1

    def _unref_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError(f"double free of block {block_id}")
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.content_hash is not None and self.enable_prefix_caching:
                self.evictable[block_id] = None
            else:
                block.content_hash = None
                self.free_ids.append(block_id)

    # ------------------------------------------------------------------
    # Prefix cache lookup
    # ------------------------------------------------------------------
    def _block_hashes(self, token_ids: list[int], num_full_blocks: int) -> list[bytes]:
        hashes: list[bytes] = []
        prev = b""
        for i in range(num_full_blocks):
            prev = _hash_block(prev, token_ids[i * self.block_size:(i + 1) * self.block_size])
            hashes.append(prev)
        return hashes

    def _match_prefix(self, token_ids: list[int]) -> list[int]:
        """Longest run of already-cached full blocks for this token prefix.

        Never matches every token of the prompt: the model must compute at
        least one position to produce logits, so matching is capped at
        ``len(token_ids) - 1`` tokens (rounded down to a block boundary).
        """
        if not self.enable_prefix_caching:
            return []
        max_match_tokens = len(token_ids) - 1
        num_full = max_match_tokens // self.block_size
        matched: list[int] = []
        for h in self._block_hashes(token_ids, num_full):
            bid = self.hash_to_block.get(h)
            if bid is None:
                break
            matched.append(bid)
        return matched

    # ------------------------------------------------------------------
    # Sequence-level operations
    # ------------------------------------------------------------------
    def can_allocate(self, seq: Sequence) -> bool:
        matched = self._match_prefix(seq.token_ids)
        matched_evictable = sum(1 for bid in matched if bid in self.evictable)
        needed = seq.num_blocks_needed(self.block_size) - len(matched)
        return self.num_free_blocks - matched_evictable >= needed

    def allocate(self, seq: Sequence) -> None:
        """Build the block table for a prompt, reusing cached prefix blocks."""
        if seq.block_table:
            raise RuntimeError(f"seq {seq.seq_id} already has a block table")
        if not self.can_allocate(seq):
            raise RuntimeError("insufficient free blocks; call can_allocate first")
        matched = self._match_prefix(seq.token_ids)
        self.stats.prefix_cache_queries += 1
        self.stats.prefix_cache_hit_tokens += len(matched) * self.block_size
        table: list[int] = []
        for bid in matched:
            self._ref_block(bid)
            table.append(bid)
        total = seq.num_blocks_needed(self.block_size)
        num_full = len(seq.token_ids) // self.block_size
        hashes = self._block_hashes(seq.token_ids, num_full) if self.enable_prefix_caching else []
        for idx in range(len(matched), total):
            bid = self._pop_free_block()
            table.append(bid)
            # Prompt blocks that will be completely filled by the upcoming
            # prefill become immediately reusable by later requests.
            if idx < num_full and self.enable_prefix_caching:
                self._register(bid, hashes[idx])
        seq.block_table = table
        seq.num_cached_tokens = len(matched) * self.block_size
        seq.num_computed = seq.num_cached_tokens

    def _register(self, block_id: int, content_hash: bytes) -> None:
        if content_hash in self.hash_to_block:
            return
        self.blocks[block_id].content_hash = content_hash
        self.hash_to_block[content_hash] = block_id

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        return self.num_free_blocks >= self.blocks_needed(seq, num_new_tokens)

    def blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        """Worst-case free blocks required to append and compute
        ``num_new_tokens`` more tokens for this sequence."""
        new_len = len(seq.token_ids) + num_new_tokens
        total = (new_len + self.block_size - 1) // self.block_size
        growth = max(0, total - len(seq.block_table))
        # Worst case, every already-owned block in the write region is shared
        # and needs a COW copy.
        first_write_block = seq.num_computed // self.block_size
        cow = sum(1 for i in range(first_write_block, len(seq.block_table))
                  if self.blocks[seq.block_table[i]].ref_count > 1)
        return growth + cow

    def append_slots(self, seq: Sequence) -> list[CopyOp]:
        """Ensure the block table covers ``seq.token_ids`` and resolve COW.

        Call after appending new token ids to the sequence and before the
        model writes KV entries for positions ``[seq.num_computed, len(seq))``.
        Returns physical block copies the model runner must perform.
        """
        copies: list[CopyOp] = []
        first_write_block = seq.num_computed // self.block_size
        for i in range(first_write_block, len(seq.block_table)):
            bid = seq.block_table[i]
            if self.blocks[bid].ref_count > 1:
                new_bid = self._pop_free_block()
                copies.append((bid, new_bid))
                seq.block_table[i] = new_bid
                self._unref_block(bid)
                self.stats.cow_copies += 1
            elif self.blocks[bid].content_hash is not None:
                # About to rewrite a registered block in place (possible after
                # a speculative rollback); its cached identity no longer holds.
                self._deregister(bid)
        total = seq.num_blocks_needed(self.block_size)
        while len(seq.block_table) < total:
            seq.block_table.append(self._pop_free_block())
        if self.enable_prefix_caching:
            num_full = len(seq.token_ids) // self.block_size
            hashes = self._block_hashes(seq.token_ids, num_full)
            for i in range(first_write_block, num_full):
                bid = seq.block_table[i]
                if self.blocks[bid].content_hash is None:
                    self._register(bid, hashes[i])
        return copies

    def truncate(self, seq: Sequence, new_num_tokens: int) -> None:
        """Drop trailing blocks after a speculative rollback.

        The caller trims ``seq.token_ids``/``seq.num_computed`` itself; this
        only releases block table entries past the new length.
        """
        keep = (new_num_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) > keep:
            self._unref_block(seq.block_table.pop())

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Share the parent's blocks with a child sequence (COW semantics)."""
        if child.block_table:
            raise RuntimeError("child already has a block table")
        for bid in parent.block_table:
            self._ref_block(bid)
        child.block_table = list(parent.block_table)
        child.num_computed = parent.num_computed
        child.num_cached_tokens = parent.num_cached_tokens

    def free_seq(self, seq: Sequence) -> None:
        for bid in reversed(seq.block_table):
            self._unref_block(bid)
        seq.block_table = []

    def slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        """Flat physical slot indices for token positions [start, end)."""
        return [seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                for p in range(start, end)]
