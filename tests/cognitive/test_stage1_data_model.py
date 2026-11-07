"""Stage 1 — schema v3, usage score, dynamic collections, cycle budget, entity index."""
from __future__ import annotations

import sqlite3

import pytest

from memory.graph_store import GraphMemory
from memory.knowledge import KnowledgeStore
from memory.usage import UsageTracker, usage_score, percentile_ranks
from memory.vector_store import BudgetExceeded, CollectionBudget, VectorMemory
from tests.cognitive.conftest import DAY, of_type


# ---------------------------------------------------------------- schema v3

def test_schema_v3_columns_and_tables(stores):
    _, graph = stores
    with graph._get_connection() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)")}
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = conn.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"]
    assert {"access_count", "last_accessed", "usage_score", "collection", "status",
            "confidence", "pos_x", "pos_y", "pos_z"} <= cols
    assert {"node_aliases", "collections", "curiosity_queue"} <= tables
    assert version == 3


def test_v3_migration_is_idempotent_and_keeps_old_nodes(tmp_path, clock):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL, data TEXT,"
                 " created_at REAL NOT NULL, last_used REAL NOT NULL, importance REAL DEFAULT 0.5, pinned INTEGER DEFAULT 0)")
    conn.execute("INSERT INTO nodes (id, kind, label, created_at, last_used) VALUES ('old', 'task', 'x', 0, 0)")
    conn.commit()
    conn.close()
    g1 = GraphMemory(str(db), clock=clock)
    g1.close()
    g2 = GraphMemory(str(db), clock=clock)
    node = g2.get_node("old")
    assert node["label"] == "x" and node["access_count"] == 0 and node["status"] == "active"
    with g2._get_connection() as c:
        rows = c.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    assert [r["version"] for r in rows] == [2, 3]
    g2.close()


def test_aliases_resolve_and_cascade_on_delete(stores):
    _, graph = stores
    graph.upsert_node("ent_henna", "entity", "Henna")
    graph.add_alias("ent_henna", "Henna")
    graph.add_alias("ent_henna", "  MEHNDI ")
    assert graph.resolve_alias("mehndi") == ["ent_henna"]
    assert graph.resolve_alias("Henna!") == ["ent_henna"]
    graph.delete_node("ent_henna")
    assert graph.resolve_alias("mehndi") == []


# ---------------------------------------------------------------- usage score

def test_usage_score_formula():
    assert usage_score(0, 0, 0.5, 0, 14) == 0.0
    s1 = usage_score(1, 0, 0.5, 0, 14)
    s10 = usage_score(10, 0, 0.5, 0, 14)
    assert s10 > s1 > 0
    later = usage_score(10, 0, 0.5, 14 * DAY, 14)
    assert later == pytest.approx(s10 * 2.718281828 ** -1, rel=1e-6)


def test_touch_increments_and_scores_fall_with_time(stores, clock, events):
    _, graph = stores
    graph.upsert_node("ent_a", "entity", "A")
    tracker = UsageTracker(graph, clock=clock)
    n1 = tracker.touch("ent_a")
    n2 = tracker.touch("ent_a")
    assert n1["access_count"] == 1 and n2["access_count"] == 2
    fresh = tracker.score_node(graph.get_node("ent_a"))
    clock.advance(days=30)
    stale = tracker.score_node(graph.get_node("ent_a"))
    assert stale < fresh
    touched = of_type(events, "node_touched")
    assert len(touched) == 2 and touched[-1]["access_count"] == 2 and "style" in touched[-1]


def test_percentile_ranks():
    assert percentile_ranks({"a": 1.0}) == {"a": 0.5}
    p = percentile_ranks({"a": 0.0, "b": 1.0, "c": 2.0})
    assert p == {"a": 0.0, "b": 0.5, "c": 1.0}
    ties = percentile_ranks({"a": 1.0, "b": 1.0})
    assert ties["a"] == ties["b"] == 0.5


def test_touch_restores_fading_node(stores, clock):
    _, graph = stores
    graph.upsert_node("ent_f", "entity", "F")
    graph.set_node_fields("ent_f", status="fading", fading_since=clock.now())
    UsageTracker(graph, clock=clock).touch("ent_f")
    node = graph.get_node("ent_f")
    assert node["status"] == "active" and node["fading_since"] is None


# ---------------------------------------------------------------- collections

def test_dynamic_collections_are_listed_everywhere(stores):
    vector, _ = stores
    assert vector.create_collection("k_henna") is True
    assert vector.create_collection("k_henna") is False  # reopen is free
    vector.store("k_henna", "Henna is a dye from Lawsonia inermis")
    assert "k_henna" in vector.list_knowledge_namespaces()
    ns = vector.list_namespaces()
    assert ns[:7] == VectorMemory.DEFAULT_NAMESPACES and "k_henna" in ns
    assert VectorMemory.INDEX_NAMESPACE not in vector.list_knowledge_namespaces()
    assert vector.drop_collection("k_henna") is True
    assert "k_henna" not in vector.list_namespaces()
    assert vector.drop_collection("peter") is False  # agent namespaces are protected


def test_collection_names_are_chroma_safe():
    assert VectorMemory.knowledge_namespace("Wedding Cultures!") == "k_wedding_cultures"
    long = VectorMemory.knowledge_namespace("x" * 200)
    assert len("peter_" + long) <= 63
    assert VectorMemory.knowledge_namespace("???") == "k_topic"


def test_budget_refuses_the_sixth_collection(stores):
    vector, _ = stores
    budget = CollectionBudget(max_new=5)
    for i in range(5):
        vector.create_collection(f"k_topic_{i}", budget=budget)
    assert budget.remaining == 0
    with pytest.raises(BudgetExceeded):
        vector.create_collection("k_topic_5", budget=budget)
    assert not vector.has_collection("k_topic_5")
    vector.create_collection("k_topic_0", budget=budget)  # existing: no cost, no error


def test_store_with_precomputed_embedding_and_centroid(stores):
    vector, _ = stores
    vector.create_collection("k_a")
    emb = vector.embed(["henna dye plant"])[0]
    rid = vector.store("k_a", "copied record", embedding=emb, silent=True)
    rec = vector.get_records("k_a", [rid])[0]
    assert rec["embedding"] == pytest.approx(emb, abs=1e-5)
    c = vector.centroid("k_a")
    assert len(c) == 384


def test_entity_index_finds_by_meaning(stores):
    vector, _ = stores
    vector.index_entity("ent_henna", "Henna body art", "entity")
    vector.index_entity("ent_saree", "Saree draping", "entity")
    hits = vector.search_entities("henna art designs", n_results=2)
    assert hits[0]["node_id"] == "ent_henna"
    assert hits[0]["score"] > hits[1]["score"]


def test_knowledge_store_binds_node_collection_alias_index(stores, events):
    vector, graph = stores
    ks = KnowledgeStore(vector, graph)
    node, created = ks.ensure_node("Henna", aliases=["mehndi"])
    assert created and node["collection"] == "k_henna" and node["access_count"] == 1
    assert graph.get_collection("k_henna")["node_id"] == "ent_henna"
    assert graph.resolve_alias("mehndi") == ["ent_henna"]
    assert vector.search_entities("henna")[0]["node_id"] == "ent_henna"
    again, created2 = ks.ensure_node("henna")
    assert not created2 and again["id"] == "ent_henna"
    assert of_type(events, "node_spawned")[0]["collection"] == "k_henna"
    assert ks.remove_node("ent_henna")
    assert not vector.has_collection("k_henna") and graph.get_node("ent_henna") is None
    assert of_type(events, "node_pruned")


def test_ensure_node_respects_budget_atomically(stores):
    vector, graph = stores
    ks = KnowledgeStore(vector, graph)
    budget = CollectionBudget(max_new=1)
    ks.ensure_node("A", budget=budget)
    with pytest.raises(BudgetExceeded):
        ks.ensure_node("B", budget=budget)
    assert graph.get_node("ent_b") is None  # nothing half-made


def test_legacy_graph_prune_leaves_cognitive_nodes_alone(stores, clock):
    _, graph = stores
    graph.upsert_node("ent_x", "entity", "X", importance=0.0)
    graph.upsert_node("t1", "task", "old task", importance=0.0)
    clock.advance(days=60)
    graph.prune(max_age_days=30, min_importance=0.3)
    assert graph.get_node("ent_x") is not None
    assert graph.get_node("t1") is None


def test_graph_connections_are_closed(stores):
    """`with graph._get_connection()` must close the connection (Windows file locks, ResourceWarnings)."""
    import gc
    import warnings
    _, graph = stores
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for i in range(20):
            graph.upsert_node(f"n{i}", "entity", f"N{i}")
            graph.add_alias(f"n{i}", f"alias {i}")
            graph.get_node(f"n{i}")
        graph.all_nodes()
        gc.collect()
    unclosed = [w for w in caught if "unclosed database" in str(w.message)]
    assert unclosed == []
