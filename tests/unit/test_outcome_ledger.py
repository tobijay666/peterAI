"""Unit tests for OutcomeLedger and related B1, B2, B4 features."""
import pytest
from unittest import mock
from core.clock import FakeClock
from memory.outcomes import OutcomeLedger
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory


def test_outcome_ledger_record_and_get(temp_dir, fake_clock):
    db_path = str(temp_dir / "graph.db")
    # Init schema first
    gm = GraphMemory(db_path, clock=fake_clock)
    gm.close()
    
    ledger = OutcomeLedger(db_path, clock=fake_clock)
    row_id = ledger.record(
        task_id="t1", level="task", agent="peter", status="success",
        memory_ids=["m1", "m2"]
    )
    assert row_id is not None
    
    outcomes = ledger.get_task_outcomes("t1")
    assert len(outcomes) == 1
    assert outcomes[0]["task_id"] == "t1"
    assert outcomes[0]["status"] == "success"
    
    cited = ledger.get_cited_memory_ids("t1")
    assert set(cited) == {"m1", "m2"}


def test_outcome_ledger_rate_and_stats(temp_dir, fake_clock):
    db_path = str(temp_dir / "graph.db")
    gm = GraphMemory(db_path, clock=fake_clock)
    gm.close()
    
    ledger = OutcomeLedger(db_path, clock=fake_clock)
    ledger.record(task_id="t1", level="task", agent="peter", status="success", latency_ms=100)
    ledger.record(task_id="t2", level="task", agent="peter", status="failure", latency_ms=200)
    
    ledger.rate_task("t1", rating=1)
    
    stats = ledger.stats(group_by="agent")
    assert len(stats) == 1
    s = stats[0]
    assert s["key"] == "peter"
    assert s["total"] == 2
    assert s["successes"] == 1
    assert s["avg_latency_ms"] == 150.0


def test_v_skill_stats(temp_dir, fake_clock):
    db_path = str(temp_dir / "graph.db")
    gm = GraphMemory(db_path, clock=fake_clock)
    
    ledger = OutcomeLedger(db_path, clock=fake_clock)
    # Task 1: success, rating 1 -> success
    ledger.record(task_id="t1", level="task", agent="peter", status="success", skill_id="s1")
    ledger.rate_task("t1", rating=1)
    
    # Task 2: success, rating -1 -> failure
    ledger.record(task_id="t2", level="task", agent="peter", status="success", skill_id="s1")
    ledger.rate_task("t2", rating=-1)
    
    # Task 3: failure, no rating -> failure
    ledger.record(task_id="t3", level="task", agent="peter", status="failure", skill_id="s1")
    
    with gm._get_connection() as conn:
        row = conn.execute("SELECT * FROM v_skill_stats WHERE skill_id = 's1'").fetchone()
        assert row["uses"] == 3
        assert row["successes"] == 1
        
    gm.close()


def test_outcome_ledger_non_fatal_error(temp_dir, fake_clock):
    # Simulate an error (e.g. read-only DB or missing table)
    db_path = str(temp_dir / "graph.db")
    # Do NOT init GraphMemory, so tables don't exist
    ledger = OutcomeLedger(db_path, clock=fake_clock)
    
    # Should not raise exception
    row_id = ledger.record(task_id="t1", level="task", agent="p", status="success")
    assert row_id is None
    
    row_id2 = ledger.rate_task("t1", 1)
    assert row_id2 is None
    
    assert ledger.stats() == []
    assert ledger.get_task_outcomes("t1") == []


def test_adjust_importance(temp_dir, fake_clock):
    chroma_dir = temp_dir / "chroma"
    vm = VectorMemory(persist_directory=str(chroma_dir), clock=fake_clock)
    
    # Normal record
    doc_id = vm.store("peter", text="test", importance=0.5, pinned=False)
    # Pinned record
    pinned_id = vm.store("peter", text="test2", importance=0.8, pinned=True)
    
    new_val = vm.adjust_importance("peter", doc_id, 0.3)
    assert new_val == 0.8
    
    # Clamp to 1.0
    val_clamped = vm.adjust_importance("peter", doc_id, 0.5)
    assert val_clamped == 1.0
    
    # Pinned should be skipped
    pinned_val = vm.adjust_importance("peter", pinned_id, -0.5)
    assert pinned_val is None
