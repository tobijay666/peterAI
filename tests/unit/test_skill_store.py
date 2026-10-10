import json
import pytest
from pathlib import Path
from memory.skills import SkillStore
from memory.graph_store import GraphMemory
from core.clock import FakeClock

@pytest.fixture
def temp_graph(tmp_path):
    db_path = tmp_path / "test_graph.db"
    graph = GraphMemory(database_path=str(db_path), clock=FakeClock())
    yield graph
    graph.close()

@pytest.fixture
def skill_store(temp_graph, tmp_path):
    config_path = tmp_path / "evolution.yaml"
    config_content = """
skills:
  promote_min_successes: 3
  promote_min_success_rate: 0.80
  deprecate_window: 5
  deprecate_below_rate: 0.50
  deprecate_after_consecutive_failures: 2
  deprecate_after_bad_ratings: 3
"""
    config_path.write_text(config_content)
    return SkillStore(graph=temp_graph, config_path=str(config_path), clock=FakeClock())


def test_create_candidate(skill_store):
    skill_id = skill_store.create_candidate("trigger phrase", [{"plan": "step"}], ["task1"])
    assert skill_id is not None
    node = skill_store.graph.get_node(skill_id)
    assert node["kind"] == "skill"
    assert node["data"]["status"] == "candidate"
    assert "task1" in skill_store.graph.get_skill_source_tasks(skill_id)


def test_evaluate_promotion(skill_store):
    skill_id = skill_store.create_candidate("t", [], [])
    
    # Not enough successes
    assert not skill_store.evaluate_promotion(skill_id, {"uses": 2, "successes": 2})
    assert skill_store.get_skill(skill_id)["status"] == "candidate"
    
    # Not high enough rate (2 successes out of 3 uses = 0.66)
    assert not skill_store.evaluate_promotion(skill_id, {"uses": 3, "successes": 2})
    assert skill_store.get_skill(skill_id)["status"] == "candidate"
    
    # Exactly 3 successes, 80% rate (4 uses, 3 successes = 0.75 which is < 0.80) -> no
    assert not skill_store.evaluate_promotion(skill_id, {"uses": 4, "successes": 3})
    
    # 3 successes, 100% rate (3 uses, 3 successes = 1.0) -> yes
    assert skill_store.evaluate_promotion(skill_id, {"uses": 3, "successes": 3})
    assert skill_store.get_skill(skill_id)["status"] == "active"


def test_evaluate_deprecation(skill_store):
    # Setup active skill
    sid1 = skill_store.create_candidate("t1", [], [])
    skill_store._set_status(sid1, "candidate", "active")
    
    # Under window, shouldn't deprecate for rate yet, unless max consec fail or bad ratings
    assert not skill_store.evaluate_deprecation(sid1, {"uses": 4, "successes": 1})
    assert skill_store.get_skill(sid1)["status"] == "active"

    # Rate < 0.50 over window=5
    assert skill_store.evaluate_deprecation(sid1, {"uses": 5, "successes": 2})
    assert skill_store.get_skill(sid1)["status"] == "deprecated"

    # Setup another
    sid2 = skill_store.create_candidate("t2", [], [])
    skill_store._set_status(sid2, "candidate", "active")
    
    # 2 consecutive failures
    assert skill_store.evaluate_deprecation(sid2, {"uses": 2, "successes": 0}, consecutive_failures=2)
    assert skill_store.get_skill(sid2)["status"] == "deprecated"

    # Setup another
    sid3 = skill_store.create_candidate("t3", [], [])
    skill_store._set_status(sid3, "candidate", "active")
    
    # 3 bad ratings
    assert skill_store.evaluate_deprecation(sid3, {"uses": 3, "successes": 3}, bad_ratings=3)
    assert skill_store.get_skill(sid3)["status"] == "deprecated"


def test_match(skill_store):
    sid1 = skill_store.create_candidate("hello world", [], [])
    skill_store._set_status(sid1, "candidate", "active")
    
    sid2 = skill_store.create_candidate("world of warcraft", [], [])
    skill_store._set_status(sid2, "candidate", "active")

    sid3 = skill_store.create_candidate("hello there", [], [])
    # sid3 stays candidate

    sid4 = skill_store.create_candidate("unrelated skill", [], [])
    skill_store._set_status(sid4, "candidate", "deprecated")

    # Match "hello" -> should match sid1, but not sid3 (candidate), not sid4 (deprecated)
    results = skill_store.match("hello", k=5)
    assert len(results) == 1
    assert results[0]["skill_id"] == sid1
    
    results = skill_store.match("world", k=5)
    assert len(results) == 2
    # both sid1 and sid2 have "world", scores might be tied or sorted
    ids = [r["skill_id"] for r in results]
    assert sid1 in ids
    assert sid2 in ids


def test_supersede(skill_store):
    sid_old = skill_store.create_candidate("old", [], [])
    skill_store._set_status(sid_old, "candidate", "active")
    
    sid_new = skill_store.create_candidate("new", [], [])
    
    skill_store.supersede(sid_old, sid_new)
    
    assert skill_store.get_skill(sid_old)["status"] == "deprecated"
    
    # Check lineage edge
    # supersedes edge: sid_new -> supersedes -> sid_old
    # Let's just find edges from new to old
    edges = skill_store.graph._get_connection().execute(
        "SELECT relation FROM edges WHERE src = ? AND dst = ?", (sid_new, sid_old)
    ).fetchall()
    assert len(edges) == 1
    assert edges[0]["relation"] == "supersedes"


def test_config_reload(skill_store, tmp_path):
    sid = skill_store.create_candidate("test", [], [])
    
    # By default, min_suc=3.
    # 2 successes should fail
    assert not skill_store.evaluate_promotion(sid, {"uses": 2, "successes": 2})
    
    # Modify config
    config_path = Path(skill_store.config_path)
    config_path.write_text("""
skills:
  promote_min_successes: 2
  promote_min_success_rate: 0.80
""")
    skill_store.reload_config()
    
    # Now 2 successes should pass
    assert skill_store.evaluate_promotion(sid, {"uses": 2, "successes": 2})
    assert skill_store.get_skill(sid)["status"] == "active"
