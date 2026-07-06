"""Scheduler: admission, batching bounds, preemption under KV pressure."""

from __future__ import annotations

from engine.block_manager import BlockManager
from engine.config import EngineConfig, SamplingParams
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SequenceStatus


def make_scheduler(num_blocks: int = 8, block_size: int = 4,
                   max_batch: int = 4) -> Scheduler:
    cfg = EngineConfig(model="gpt2", block_size=block_size,
                       num_blocks=num_blocks, max_batch_size=max_batch,
                       enable_prefix_caching=False)
    bm = BlockManager(num_blocks, block_size, enable_prefix_caching=False)
    return Scheduler(cfg, bm)


def make_seq(seq_id: int, num_tokens: int, arrival: float) -> Sequence:
    return Sequence(seq_id, list(range(num_tokens)),
                    SamplingParams(max_tokens=32), arrival_time=arrival)


def simulate_step(sched: Scheduler, out) -> None:
    """Mimic what the engine does after a scheduling decision."""
    for seq in out.prefill:
        seq.num_computed = len(seq.token_ids)
        seq.append_token(0)
    for seq in out.decode:
        sched.block_manager.append_slots(seq)
        seq.num_computed = len(seq.token_ids)
        seq.append_token(0)


def test_admission_respects_batch_size():
    sched = make_scheduler(num_blocks=32, max_batch=2)
    for i in range(4):
        sched.add(make_seq(i, 4, arrival=float(i)))
    out = sched.schedule()
    assert len(out.prefill) == 2
    assert len(sched.waiting) == 2
    assert all(s.status == SequenceStatus.RUNNING for s in out.prefill)


def test_admission_respects_kv_capacity():
    sched = make_scheduler(num_blocks=3, max_batch=4)
    sched.add(make_seq(0, 8, arrival=0.0))   # 2 blocks
    sched.add(make_seq(1, 8, arrival=1.0))   # would need 2 more
    out = sched.schedule()
    assert [s.seq_id for s in out.prefill] == [0]
    assert len(sched.waiting) == 1


def test_join_at_token_boundary():
    sched = make_scheduler(num_blocks=32, max_batch=4)
    sched.add(make_seq(0, 4, arrival=0.0))
    simulate_step(sched, sched.schedule())
    # Request 1 arrives while 0 is mid-generation; next step batches both.
    sched.add(make_seq(1, 4, arrival=1.0))
    out = sched.schedule()
    assert [s.seq_id for s in out.prefill] == [1]
    assert [s.seq_id for s in out.decode] == [0]


def test_preemption_evicts_newest_and_requeues():
    sched = make_scheduler(num_blocks=4, block_size=4, max_batch=4)
    a = make_seq(0, 7, arrival=0.0)  # 2 blocks, one slot left in block 2
    b = make_seq(1, 7, arrival=1.0)  # 2 blocks
    sched.add(a)
    sched.add(b)
    simulate_step(sched, sched.schedule())  # both prefilled, all blocks used
    simulate_step(sched, sched.schedule())  # both decode into their last slot
    # Both now hold 9 tokens and need a third block, but none are free:
    # b (newest) must be preempted so a can continue.
    out = sched.schedule()
    assert [s.seq_id for s in out.preempted] == [1]
    assert [s.seq_id for s in out.decode] == [0]
    assert b.status == SequenceStatus.WAITING
    assert b.block_table == []
    assert b.num_computed == 0
    assert b.num_preemptions == 1
    assert sched.waiting[0] is b  # requeued at the front
    simulate_step(sched, out)
    assert a.num_computed == 9


def test_preempted_sequence_resumes_and_finishes():
    sched = make_scheduler(num_blocks=4, block_size=4, max_batch=4)
    a = make_seq(0, 7, arrival=0.0)
    b = make_seq(1, 7, arrival=1.0)
    sched.add(a)
    sched.add(b)
    simulate_step(sched, sched.schedule())
    simulate_step(sched, sched.schedule())
    simulate_step(sched, sched.schedule())  # b preempted, a decodes
    assert b.status == SequenceStatus.WAITING
    sched.finish_seq(a)                     # a's blocks are released
    out = sched.schedule()                  # b re-admitted as a prefill
    assert [s.seq_id for s in out.prefill] == [1]
    # b kept its full token history (prompt + the two tokens it generated).
    assert len(b.token_ids) == 9


def test_scheduler_invariants_under_churn():
    """Block accounting stays exact across admissions, preemptions, finishes."""
    sched = make_scheduler(num_blocks=8, block_size=4, max_batch=3)
    bm = sched.block_manager
    for i in range(6):
        sched.add(make_seq(i, 5, arrival=float(i)))
    for _ in range(300):
        out = sched.schedule()
        simulate_step(sched, out)
        used = sum(len(s.block_table) for s in sched.running)
        assert used + bm.num_free_blocks == bm.num_blocks
        for s in sched.running:
            assert len(s.block_table) * bm.block_size >= s.num_computed
        for s in sched.running:
            if len(s.token_ids) >= 12:
                sched.finish_seq(s)
        if not sched.has_work:
            break
    assert not sched.has_work
    assert bm.num_free_blocks == bm.num_blocks
