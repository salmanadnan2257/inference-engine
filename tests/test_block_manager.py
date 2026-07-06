"""Block manager: allocation, fragmentation, prefix cache, COW, eviction."""

from __future__ import annotations

import pytest

from engine.block_manager import BlockManager
from engine.config import SamplingParams
from engine.sequence import Sequence


def make_seq(seq_id: int, tokens: list[int]) -> Sequence:
    return Sequence(seq_id, list(tokens), SamplingParams(max_tokens=8))


def simulate_compute(bm: BlockManager, seq: Sequence) -> None:
    """Mark the sequence's tail tokens as computed (what the runner does)."""
    seq.num_computed = len(seq.token_ids)


class TestAllocation:
    def test_basic_allocate_and_free(self):
        bm = BlockManager(8, 4, enable_prefix_caching=False)
        seq = make_seq(0, list(range(10)))  # 10 tokens -> 3 blocks
        assert bm.can_allocate(seq)
        bm.allocate(seq)
        assert len(seq.block_table) == 3
        assert bm.num_free_blocks == 5
        bm.free_seq(seq)
        assert bm.num_free_blocks == 8
        assert seq.block_table == []

    def test_oom_rejected(self):
        bm = BlockManager(2, 4, enable_prefix_caching=False)
        seq = make_seq(0, list(range(12)))  # needs 3 blocks, only 2 exist
        assert not bm.can_allocate(seq)
        with pytest.raises(RuntimeError):
            bm.allocate(seq)

    def test_fragmented_free_list_serves_one_sequence(self):
        """Freed non-contiguous blocks are all reusable; no external
        fragmentation is possible with fixed-size blocks."""
        bm = BlockManager(6, 4, enable_prefix_caching=False)
        seqs = [make_seq(i, list(range(8))) for i in range(3)]  # 2 blocks each
        for s in seqs:
            bm.allocate(s)
        bm.free_seq(seqs[0])
        bm.free_seq(seqs[2])  # free blocks {0,1,4,5}: non-contiguous
        big = make_seq(9, list(range(16)))  # needs 4 blocks
        assert bm.can_allocate(big)
        bm.allocate(big)
        assert sorted(big.block_table) == [0, 1, 4, 5]
        assert bm.num_free_blocks == 0

    def test_double_free_raises(self):
        bm = BlockManager(4, 4, enable_prefix_caching=False)
        seq = make_seq(0, list(range(4)))
        bm.allocate(seq)
        table = list(seq.block_table)
        bm.free_seq(seq)
        seq.block_table = table
        with pytest.raises(RuntimeError):
            bm.free_seq(seq)


class TestAppend:
    def test_append_grows_at_block_boundary(self):
        bm = BlockManager(8, 4, enable_prefix_caching=False)
        seq = make_seq(0, list(range(4)))  # exactly one full block
        bm.allocate(seq)
        simulate_compute(bm, seq)
        seq.append_token(99)
        assert bm.can_append(seq)
        copies = bm.append_slots(seq)
        assert copies == []
        assert len(seq.block_table) == 2

    def test_can_append_false_when_exhausted(self):
        bm = BlockManager(1, 4, enable_prefix_caching=False)
        seq = make_seq(0, list(range(4)))
        bm.allocate(seq)
        simulate_compute(bm, seq)
        seq.append_token(99)
        assert not bm.can_append(seq)


class TestCopyOnWrite:
    def test_fork_shares_then_copies(self):
        bm = BlockManager(8, 4, enable_prefix_caching=False)
        parent = make_seq(0, list(range(6)))  # blocks: 1 full + 1 partial
        bm.allocate(parent)
        simulate_compute(bm, parent)
        child = make_seq(1, list(range(6)))
        bm.fork(parent, child)
        assert child.block_table == parent.block_table
        assert bm.blocks[parent.block_table[1]].ref_count == 2
        assert bm.num_free_blocks == 6  # sharing allocated nothing

        # Child writes into the shared partial block: must trigger a copy.
        child.append_token(50)
        copies = bm.append_slots(child)
        assert len(copies) == 1
        src, dst = copies[0]
        assert src == parent.block_table[1]
        assert child.block_table[1] == dst
        assert child.block_table[1] != parent.block_table[1]
        assert child.block_table[0] == parent.block_table[0]  # full block still shared
        assert bm.blocks[parent.block_table[1]].ref_count == 1
        assert bm.stats.cow_copies == 1

        # Parent then writes its own partial block: no copy needed anymore.
        parent.append_token(60)
        assert bm.append_slots(parent) == []

    def test_can_append_accounts_for_cow(self):
        bm = BlockManager(2, 4, enable_prefix_caching=False)
        parent = make_seq(0, list(range(6)))
        bm.allocate(parent)  # uses both blocks
        simulate_compute(bm, parent)
        child = make_seq(1, list(range(6)))
        bm.fork(parent, child)
        child.append_token(1)
        assert not bm.can_append(child)  # COW would need a free block


class TestPrefixCache:
    def test_full_blocks_reused(self):
        bm = BlockManager(16, 4)
        a = make_seq(0, list(range(10)))  # 2 full blocks + 1 partial
        bm.allocate(a)
        table_a = list(a.block_table)
        simulate_compute(bm, a)
        bm.free_seq(a)

        b = make_seq(1, list(range(10)))
        bm.allocate(b)
        assert b.num_cached_tokens == 8
        assert b.block_table[:2] == table_a[:2]  # same physical blocks
        assert b.block_table[2] != table_a[2]    # partial block not shared
        assert bm.stats.prefix_cache_hit_tokens == 8

    def test_shared_prefix_between_live_sequences(self):
        bm = BlockManager(16, 4)
        a = make_seq(0, list(range(8)) + [100, 101])
        bm.allocate(a)
        simulate_compute(bm, a)
        b = make_seq(1, list(range(8)) + [200, 201])  # same first 8 tokens
        bm.allocate(b)
        assert b.num_cached_tokens == 8
        assert b.block_table[:2] == a.block_table[:2]
        assert bm.blocks[a.block_table[0]].ref_count == 2

    def test_no_match_capped_below_full_prompt(self):
        """A prompt fully covered by cached blocks still recomputes its last
        block so the model has at least one position to produce logits."""
        bm = BlockManager(16, 4)
        a = make_seq(0, list(range(8)))
        bm.allocate(a)
        simulate_compute(bm, a)
        b = make_seq(1, list(range(8)))
        bm.allocate(b)
        assert b.num_cached_tokens == 4  # not 8

    def test_eviction_lru(self):
        bm = BlockManager(4, 4)
        a = make_seq(0, list(range(8)))
        bm.allocate(a)
        simulate_compute(bm, a)
        bm.free_seq(a)  # both full blocks become evictable, still cached
        assert bm.num_free_blocks == 4
        b = make_seq(1, [500 + i for i in range(16)])  # needs all 4 blocks
        bm.allocate(b)
        assert bm.stats.evictions == 2
        # a's cached hashes were dropped; only b's 4 full blocks remain.
        assert len(bm.hash_to_block) == 4
        c = make_seq(2, list(range(8)))
        assert not bm.can_allocate(c)  # everything is held by b

    def test_truncate_deregisters_rewritten_block(self):
        """After a speculative rollback, a full block that is about to be
        partially rewritten must not stay in the hash table."""
        bm = BlockManager(8, 4)
        seq = make_seq(0, list(range(8)))
        bm.allocate(seq)
        simulate_compute(bm, seq)
        assert len(bm.hash_to_block) == 2
        # Roll back to 6 tokens and rewrite position 6 with a new token.
        seq.token_ids = seq.token_ids[:6]
        seq.num_computed = 6
        bm.truncate(seq, 6)
        seq.append_token(99)
        seq.num_computed = 6  # position 6 pending recompute
        bm.append_slots(seq)
        assert len(bm.hash_to_block) == 1  # second block deregistered


class TestTruncate:
    def test_truncate_frees_tail_blocks(self):
        bm = BlockManager(8, 4, enable_prefix_caching=False)
        seq = make_seq(0, list(range(14)))  # 4 blocks
        bm.allocate(seq)
        assert bm.num_free_blocks == 4
        seq.token_ids = seq.token_ids[:5]
        bm.truncate(seq, 5)
        assert len(seq.block_table) == 2
        assert bm.num_free_blocks == 6
