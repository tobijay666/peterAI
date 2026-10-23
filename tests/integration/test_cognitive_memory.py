"""End-to-end: the whole cognitive-memory spec in one scenario (Stage 8, S8.1).

Offline and deterministic: WordHashEmbedder, FakeClock, FakeLLMRouter, a fake web.

    1. "what is henna?"   -> gap_detected -> research -> node 'henna' + its own cited collection
    2. ask again          -> access_count 2, traverse event, no new research
    3. curiosity queue    -> 3–5 henna questions
    4. sleep              -> abstraction over henna / saree / poruwa, abstracts edges,
                             copied embeddings, <= 5 new collections
    5. 45 idle days       -> an unused topic fades, a later cycle prunes it; the snapshot restores it
    6. pinned node        -> survives everything
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import engine.evolution as evolution
from agents.base import BaseAgent
from core.clock import FakeClock
from core.cognition_config import set_cognition_config
from engine.curiosity import CuriosityEngine
from engine.evolution import EvolutionCoordinator
from engine.research_pipeline import ResearchPipeline
from memory.cognition import CognitiveMemory
from memory.graph_store import GraphMemory
from memory.mind_view import graph_snapshot, node_detail
from memory.recall import ContextBuilder
from memory.snapshot import restore_snapshot, snapshot_manager_for
from memory.vector_store import VectorMemory
from tests.cognitive.fake_web import FakeWeb
from tests.fakes.fake_llm import FakeLLMRouter
from tests.fakes.word_embedder import WordHashEmbedder

pytestmark = pytest.mark.integration
DAY = 86400.0


def reply(text):
    return {"content": text, "tool_calls": [], "usage": {"total_tokens": 1}, "provider": "fake", "model": "f"}


class _Peter(BaseAgent):
    pass


@pytest.fixture
def world(tmp_path, monkeypatch):
    import core.events as ev
    events = []
    monkeypatch.setattr(ev, "_write_event_log", lambda uid, payload: events.append(payload))
    monkeypatch.setattr(evolution, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    set_cognition_config({
        "resolver": {"use_llm": False},
        "sleep": {"abstraction": {"min_similarity": 0.24, "max_similarity": 0.80, "migrate_min_similarity": 0.2}},
    })
    clock = FakeClock(start=1_000 * DAY)
    vector = VectorMemory(str(tmp_path / "chroma"), clock=clock, user_id="anton", embedding_function=WordHashEmbedder())
    graph = GraphMemory(str(tmp_path / "graph.db"), clock=clock, user_id="anton")
    cog = CognitiveMemory(vector, graph, router=None, async_research=False)
    web = FakeWeb()
    cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=web.search, fetch_fn=web.fetch)
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver, router=None)

    tools = MagicMock()
    tools.get_schemas.return_value = None
    gr = MagicMock()
    gr.check_input.return_value = SimpleNamespace(allowed=True)
    gr.check_output.return_value = SimpleNamespace(allowed=True)
    peter = _Peter(agent_id="peter", name="Peter", system_prompt="sys", policy=[], tool_names=[], max_steps=2,
                   router=FakeLLMRouter([reply("Henna is a plant dye [1].\n\nSources:\n[1] https://www.plants.example/henna")],
                                        loop=True),
                   vector=vector, graph=graph, guardrails=gr, tools=tools)
    peter._context_builder = ContextBuilder(vector, graph, cognition=cog)
    peter._user_id = "anton"

    def sleep():
        ledger = MagicMock()
        ledger._conn.side_effect = Exception("no ledger")
        skills = MagicMock()
        skills.list_skills.return_value = []
        return EvolutionCoordinator(vector=vector, graph=graph, router=None, guardrails=MagicMock(), ledger=ledger,
                                    skill_store=skills, consolidation_config={"min_age_hours": 10_000},
                                    mode="active", clock=clock, cognition=cog,
                                    snapshot_manager=snapshot_manager_for(vector, graph, clock=clock)).run_cycle()

    yield SimpleNamespace(cog=cog, peter=peter, web=web, clock=clock, events=events, sleep=sleep,
                          vector=vector, graph=graph, tmp=tmp_path)
    graph.close()


def types(events, t):
    return [e for e in events if e["type"] == t]


def test_henna_end_to_end(world):
    w = world
    cog, peter, graph, vector = w.cog, w.peter, w.graph, w.vector

    # 1. Unknown topic: gap -> research -> node + cited collection -> cited answer
    answer = peter.run("what is henna?")
    assert types(w.events, "gap_detected")[0]["term"] == "henna"
    node = graph.get_node("ent_henna")
    assert node["collection"] == "k_henna" and vector.count("k_henna") >= 3
    assert all(r["metadata"]["source_url"].startswith("https://") for r in vector.get_all_records("k_henna"))
    assert "[1]" in answer and "Sources:" in answer
    searches_after_first = len(w.web.searches)

    # 2. Ask again: graph hit, activation counter increments, traversal emitted, no new research
    before = graph.get_node("ent_henna")["access_count"]
    peter.run("what is henna?")
    assert graph.get_node("ent_henna")["access_count"] == before + 1
    assert len(w.web.searches) == searches_after_first
    assert types(w.events, "traverse")[-1]["path"][0] == "ent_henna"

    # 3. Curiosity queue after the turn
    items = cog.after_turn("what is henna?", answer, turn_id="t1")
    assert 3 <= len(items) <= 5
    assert all(i["status"] == "pending" for i in cog.curiosity.list())

    # 4. More topics, then sleep: abstraction bridges the wedding topics (cap respected)
    peter.run("tell me about saree")
    peter.run("what is the poruwa ceremony?")
    node_s = cog.knowledge.find_node("saree")
    node_p = cog.knowledge.find_node("poruwa ceremony")
    assert node_s and node_p, "both topics were researched into their own nodes"
    for n, extra in ((node_s, "Brides wear a saree at Sri Lankan weddings with henna on their hands"),
                     (node_p, "The poruwa wedding ceremony follows the bridal henna night")):
        cog.knowledge.add_record(n, extra, {"source_url": "https://weddings.example/x"})
    cog.knowledge.add_record(graph.get_node("ent_henna"), "Bridal henna is applied before Sri Lankan weddings",
                             {"source_url": "https://weddings.example/y"})
    report = w.sleep()
    sleep = report["sleep"]
    assert not report["errors"], report["errors"]
    assert len(sleep["collections_created"]) <= 5
    assert sleep["abstractions"], sleep
    abs_node = graph.get_node(sleep["abstractions"][0]["node_id"])
    kids = {e["dst"] for e in graph.edges_of(abs_node["id"]) if e["relation"] == "abstracts"}
    assert {"ent_henna", node_s["id"], node_p["id"]} <= kids
    assert sleep["migrated"] > 0 and types(w.events, "embedding_migrated")
    # recall now also traverses the abstraction edge
    res, _ = cog.recall("henna")
    assert abs_node["id"] in res.path and res.edges

    # 5. Forgetting: an unused topic fades, then is pruned after the grace period; snapshot restores it
    cog.knowledge.ensure_node("Obscure topic")
    obscure = graph.get_node("ent_obscure_topic")
    cog.knowledge.add_record(obscure, "zyxx barnacle trivia nobody asks about")
    graph.set_node_fields("ent_henna", pinned=True)                                   # 6. pinned
    keep = [node_s["id"], node_p["id"], abs_node["id"]]
    w.clock.advance(days=45)
    for nid in keep:
        cog.usage.touch(nid)
    c1 = w.sleep()
    assert "ent_obscure_topic" in c1["sleep"]["faded"]
    w.clock.advance(days=8)
    for nid in keep:
        cog.usage.touch(nid)
    c2 = w.sleep()
    assert "ent_obscure_topic" in c2["sleep"]["pruned"]
    assert graph.get_node("ent_obscure_topic") is None and not vector.has_collection("k_obscure_topic")
    assert types(w.events, "node_pruned")
    snaps = sorted((w.tmp / "snapshots").iterdir())
    restore_snapshot(snaps[-1], w.tmp / "restored_chroma", w.tmp / "restored.db")
    restored = GraphMemory(str(w.tmp / "restored.db"), clock=w.clock)
    assert restored.get_node("ent_obscure_topic") is not None
    restored.close()

    # 6. The pinned node survived (it was never touched in those 53 days)
    assert graph.get_node("ent_henna") is not None

    # Mind view reflects all of it
    snap = graph_snapshot(cog)
    ids = {n["id"] for n in snap["nodes"]}
    assert {"ent_henna", node_s["id"], node_p["id"], abs_node["id"]} <= ids
    assert "ent_obscure_topic" not in ids
    detail = node_detail(cog, "ent_henna")
    assert detail["pinned"] and detail["records"] and detail["sources"]
