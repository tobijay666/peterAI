"""Stage 0 — events reach the right user, thread-safe WS push, graph hygiene, config."""
from __future__ import annotations

import asyncio
import threading

import pytest

from tests.cognitive.conftest import of_type


def test_vector_store_events_carry_user_id(stores, events):
    vector, _ = stores
    rid = vector.store("peter", "the user likes henna patterns")
    vector.recall("peter", "henna")
    store_ev = of_type(events, "store")
    recall_ev = of_type(events, "recall")
    assert store_ev and store_ev[-1]["user_id"] == "u-test"
    assert recall_ev and recall_ev[-1]["user_id"] == "u-test"
    assert rid in recall_ev[-1]["memory_ids"]


def test_every_event_has_schema_version(events):
    from core.events import emit, EVENT_SCHEMA_VERSION
    emit("turn_start", user_id="u-test", agent="peter")
    assert events[-1]["v"] == EVENT_SCHEMA_VERSION == 1


def test_event_schema_lists_cognitive_events():
    from core.events import MIND_EVENTS, EVENT_TOPICS
    for t in ("traverse", "node_touched", "gap_detected", "node_spawned", "edge_drawn",
              "curiosity_added", "sleep_start", "embedding_migrated", "node_fading",
              "node_pruned", "sleep_end"):
        assert t in MIND_EVENTS
        assert any(topic.startswith(f"mind.{t} ") for topic in EVENT_TOPICS)


def test_web_user_queue_receives_store_event(stores, monkeypatch):
    """S0.2 done-when: a web user's WebSocket queue gets store + recall events."""
    import server.ws as ws
    monkeypatch.setattr(ws, "_queues", {})
    monkeypatch.setattr(ws, "_loop", None)
    vector, _ = stores
    vector.store("peter", "mehndi is another name for henna")
    vector.recall("peter", "mehndi")
    q = ws.get_user_queue("u-test")
    types = []
    while not q.empty():
        types.append(q.get_nowait()["type"])
    assert "store" in types and "recall" in types
    assert "local" not in ws._queues or ws._queues["local"].empty()


def test_push_event_from_worker_thread_is_marshalled_to_loop(monkeypatch):
    """S0.3: pushes from other threads go through call_soon_threadsafe."""
    import server.ws as ws
    monkeypatch.setattr(ws, "_queues", {})

    async def scenario():
        ws.set_event_loop(asyncio.get_running_loop())
        q = ws.get_user_queue("alice")
        t = threading.Thread(target=lambda: [ws.push_event("alice", {"type": "x", "i": i}) for i in range(20)])
        t.start()
        t.join()
        got = []
        for _ in range(20):
            got.append(await asyncio.wait_for(q.get(), timeout=2))
        return got

    got = asyncio.run(scenario())
    assert [e["i"] for e in got] == list(range(20))
    monkeypatch.setattr(ws, "_loop", None)


def test_spawned_by_is_a_valid_relation_and_migrate_schema_exists(stores):
    from memory.graph_store import VALID_RELATIONS
    _, graph = stores
    assert "spawned_by" in VALID_RELATIONS
    graph.migrate_schema()  # used by scripts/gen_docs.py; must exist and be idempotent
    graph.migrate_schema()


def test_delegation_task_nodes_do_not_collide(stores):
    """S0.4: two delegations with the same first 20 chars create two task nodes."""
    import uuid
    from unittest.mock import MagicMock
    from agents.peter import PeterAgent
    from core.interfaces import ToolResult

    vector, graph = stores
    tools = MagicMock()
    registered = {}
    tools.register.side_effect = lambda name, fn, schema: registered.__setitem__(name, fn)
    peter = PeterAgent(router=MagicMock(), vector=vector, graph=graph, guardrails=MagicMock(), tools=tools)
    spec = MagicMock()
    spec.run.return_value = "done"
    peter.set_specialists({"researcher": spec})
    registered["delegate_to_researcher"](query="Research the history of henna in India")
    registered["delegate_to_researcher"](query="Research the history of henna in Morocco")
    tasks = graph.get_nodes_by_kind("task")
    assert len(tasks) == 2


def test_cognition_config_defaults_and_override(tmp_path):
    from core.cognition_config import load_cognition_config, set_cognition_config, cognition_config
    cfg = load_cognition_config(tmp_path / "missing.yaml")
    assert cfg["sleep"]["max_new_collections"] == 5
    (tmp_path / "bad.yaml").write_text(":::: not yaml [", encoding="utf-8")
    assert load_cognition_config(tmp_path / "bad.yaml")["usage"]["half_life_days"] == 14
    set_cognition_config({"sleep": {"max_new_collections": 2}})
    assert cognition_config()["sleep"]["max_new_collections"] == 2
    assert cognition_config()["sleep"]["prune"]["grace_days"] == 7  # deep-merged defaults


def test_repo_config_file_parses():
    from core.cognition_config import load_cognition_config, CONFIG_PATH
    assert CONFIG_PATH.exists()
    cfg = load_cognition_config()
    assert cfg["enabled"] is True
    assert cfg["sleep"]["max_new_collections"] == 5


def test_config_path_env_override(tmp_path, monkeypatch):
    from core.cognition_config import load_cognition_config
    alt = tmp_path / "alt.yaml"
    alt.write_text("research:\n  enabled: false\n", encoding="utf-8")
    monkeypatch.setenv("PETER_COGNITION_CONFIG", str(alt))
    cfg = load_cognition_config()
    assert cfg["research"]["enabled"] is False and cfg["sleep"]["max_new_collections"] == 5
