"""Stage 2 — entity resolver, graph-first recall, sufficiency, ContextBuilder + turn wiring."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.cognition_config import set_cognition_config
from memory.cognition import CognitiveMemory, GapOutcome
from memory.entity_resolver import EntityResolver, heuristic_entities
from memory.recall import ContextBuilder
from tests.cognitive.conftest import of_type
from tests.fakes.fake_llm import FakeLLMRouter


def reply(text):
    return {"content": text, "tool_calls": [], "usage": {"total_tokens": 1}, "provider": "fake", "model": "f"}


def seed_henna(cog: CognitiveMemory, n: int = 3):
    node, _ = cog.knowledge.ensure_node("Henna", aliases=["mehndi"])
    facts = [
        "Henna dye comes from the leaves of the Lawsonia inermis plant.",
        "Henna paste stains skin a red-orange colour that darkens over two days.",
        "Henna designs are applied to brides before weddings in South Asia.",
    ][:n]
    for i, f in enumerate(facts):
        cog.knowledge.add_record(node, f, {"source_url": f"https://example.org/henna{i}",
                                            "source_title": f"Henna page {i}"})
    return node


# ---------------------------------------------------------------- extraction / resolution

def test_heuristic_entities():
    assert heuristic_entities("what is henna?") == ["henna"]
    assert heuristic_entities("hi there") == []
    assert "Sri Lankan weddings" in heuristic_entities("how is henna used in Sri Lankan weddings")


def test_llm_extraction_and_fallback(stores):
    vector, graph = stores
    r = EntityResolver(vector, graph, router=FakeLLMRouter([reply('{"entities": ["henna", "bridal mehndi"]}')]))
    assert r.extract("tell me about henna for brides") == ["henna", "bridal mehndi"]
    bad = EntityResolver(vector, graph, router=FakeLLMRouter([reply("I think it's henna")]))
    assert bad.extract("what is henna?") == ["henna"]   # unusable reply -> heuristic


def test_resolver_alias_index_and_unmatched(stores):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph)
    seed_henna(cog)
    res = cog.resolver.resolve("q", terms=["mehndi", "henna body art", "saree"])
    vias = {m.term: m.via for m in res.matched}
    assert vias["mehndi"] == "alias"
    assert vias["henna body art"] == "index"
    assert res.unmatched == ["saree"]
    assert [n["id"] for n in res.nodes] == ["ent_henna"]


# ---------------------------------------------------------------- graph recall

def test_known_entity_twice_counts_two_and_returns_its_records(stores, events):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph)
    seed_henna(cog)
    vector.store("peter", "Unrelated note about invoices", silent=True)
    base = graph.get_node("ent_henna")["access_count"]  # 1 = baseline activation at creation
    r1, _ = cog.recall("what is henna?")
    r2, v2 = cog.recall("what is henna?")
    assert graph.get_node("ent_henna")["access_count"] == base + 2
    assert r2.hits and all(h["node_id"] == "ent_henna" for h in r2.hits)
    assert r2.hits[0]["metadata"]["source_url"].startswith("https://example.org/")
    assert v2.sufficient
    tr = of_type(events, "traverse")
    assert tr[-1]["path"] == ["ent_henna"] and tr[-1]["memory_ids"]


def test_neighbour_abstraction_is_traversed_at_lower_weight(stores, events):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph)
    henna = seed_henna(cog)
    abs_node, _ = cog.knowledge.ensure_node("Wedding cultures", kind="abstraction")
    cog.knowledge.add_record(abs_node, "Henna nights are part of many wedding cultures.")
    eid, _ = cog.knowledge.link(abs_node["id"], henna["id"], "abstracts")
    result, _ = cog.recall("henna")
    assert result.path == ["ent_henna", "abs_wedding_cultures"]
    assert result.edges == [eid]
    abs_hits = [h for h in result.hits if h["node_id"] == "abs_wedding_cultures"]
    assert abs_hits and abs_hits[0]["score"] == pytest.approx(abs_hits[0]["raw_score"] * 0.6)
    # neighbour is visited, not activated
    assert graph.get_node("abs_wedding_cultures")["access_count"] == 1


# ---------------------------------------------------------------- sufficiency

def test_sufficiency_rules(stores):
    vector, graph = stores
    set_cognition_config({"resolver": {"use_llm": False}})
    cog = CognitiveMemory(vector, graph)
    _, v = cog.recall("what is henna?")
    assert not v.sufficient and v.reason == "no_node" and v.missing_aspects == ["henna"]
    seed_henna(cog)
    _, v = cog.recall("henna")
    assert v.sufficient and v.reason == "rules_passed" and v.secondary_gaps == []
    _, v = cog.recall("hi there")
    assert v.sufficient and v.reason == "no_topic"
    # known topic + an unknown term, strong evidence, no judge available: answer, queue the rest
    set_cognition_config({"resolver": {"use_llm": False}, "sufficiency": {"min_score": 0.0, "min_records": 1}})
    _, v = cog.recall("henna and saree draping")
    assert v.sufficient and v.secondary_gaps == ["saree draping"]
    # known topic but its records don't cover the question: research it, unknown terms join the search
    set_cognition_config({"resolver": {"use_llm": False}, "sufficiency": {"min_records": 10}})
    result, v = cog.recall("henna and saree draping")
    assert not v.sufficient and v.reason == "weak_evidence" and v.missing_aspects == ["saree draping"]
    assert cog.gap_topics(result, v) == ["Henna"]


def test_llm_judge_only_asked_in_the_ambiguous_case(stores):
    vector, graph = stores
    set_cognition_config({"sufficiency": {"min_score": 0.0, "min_records": 1}, "resolver": {"use_llm": False}})
    router = FakeLLMRouter([reply('{"sufficient": false, "missing": ["saree draping"]}')])
    cog = CognitiveMemory(vector, graph, router=router)
    seed_henna(cog)
    calls_before = len(router.calls)
    _, v = cog.recall("henna")                          # everything known: no judge call
    assert v.sufficient and len(router.calls) == calls_before
    result, v = cog.recall("henna and saree draping")   # unknown term with strong evidence: judge decides
    assert not v.sufficient and v.reason == "llm_judge" and v.missing_aspects == ["saree draping"]
    assert cog.gap_topics(result, v) == ["saree draping"]   # the judge named the unknown term -> new topic


def test_record_fallback_finds_subtopic_facts_stored_under_a_parent(stores):
    vector, graph = stores
    set_cognition_config({"resolver": {"use_llm": False}})
    cog = CognitiveMemory(vector, graph)
    node, _ = cog.knowledge.ensure_node("Poruwa ceremony")
    cog.knowledge.add_record(node, "The nekath auspicious time is set by an astrologer for each wedding ritual.")
    result, v = cog.recall("what is the nekath auspicious time?")
    assert [n["id"] for n in result.matched_nodes] == ["ent_poruwa_ceremony"]
    assert result.resolution.matched[0].via == "records"
    assert result.hits and "nekath" in result.hits[0]["document"]


# ---------------------------------------------------------------- ContextBuilder + turn

def test_context_builder_injects_numbered_cited_knowledge(stores):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph)
    seed_henna(cog)
    cb = ContextBuilder(vector, graph, cognition=cog)
    rc = cb.build("what is henna?", namespace="peter")
    assert "[KNOWLEDGE" in rc.memory_text and "[1]" in rc.memory_text
    assert "source: Henna page" in rc.memory_text and "[DATA END]" in rc.memory_text
    assert rc.verdict.sufficient and rc.graph_result.matched_nodes[0]["id"] == "ent_henna"
    before = graph.get_node("ent_henna")["access_count"]
    rc2 = cb.build("what is henna?", namespace="researcher")  # specialists don't re-activate nodes
    assert rc2.graph_result is None
    assert graph.get_node("ent_henna")["access_count"] == before


def test_disabled_flag_falls_back_to_old_recall(stores):
    vector, graph = stores
    set_cognition_config({"enabled": False})
    cog = CognitiveMemory(vector, graph)
    seed_henna(cog)
    rc = ContextBuilder(vector, graph, cognition=cog).build("henna", namespace="peter")
    assert rc.graph_result is None and "[KNOWLEDGE" not in rc.memory_text


class _SyncResearch:
    """Minimal stand-in for ResearchPipeline: stores two cited records for the topic."""

    def __init__(self, cog):
        self.cog = cog
        self.calls = []

    def research(self, topic, question="", missing_aspects=None, turn_id=""):
        self.calls.append(topic)
        node, _ = self.cog.knowledge.ensure_node(topic)
        for i in range(2):
            self.cog.knowledge.add_record(node, f"{topic} fact number {i} with useful detail about {topic}",
                                          {"source_url": f"https://src{i}.example/{topic}"})
        return SimpleNamespace(records_stored=2)

    def answer(self, q):
        return "answer"


def _peter(vector, graph, cb, script):
    from agents.base import BaseAgent

    class P(BaseAgent):
        pass

    tools = MagicMock()
    tools.get_schemas.return_value = None
    gr = MagicMock()
    gr.check_input.return_value = SimpleNamespace(allowed=True)
    gr.check_output.return_value = SimpleNamespace(allowed=True)
    p = P(agent_id="peter", name="Peter", system_prompt="sys", policy=[], tool_names=[], max_steps=3,
          router=FakeLLMRouter(script), vector=vector, graph=graph, guardrails=gr, tools=tools)
    p._context_builder = cb
    p._user_id = "u-test"
    return p


def test_unknown_entity_triggers_gap_and_sync_research(stores, events):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph, async_research=False)
    cog.research = _SyncResearch(cog)
    cb = ContextBuilder(vector, graph, cognition=cog)
    peter = _peter(vector, graph, cb, [reply("Henna is a plant dye [1].")])
    out = peter.run("what is henna?")
    assert out == "Henna is a plant dye [1]."
    assert of_type(events, "gap_detected")[0]["term"] == "henna"
    assert cog.research.calls == ["henna"]
    sys_msgs = [m.content for m in peter.router.calls[0]["messages"] if m.role == "system"]
    assert any("[KNOWLEDGE" in c for c in sys_msgs)          # re-read after research
    assert any("cite them inline as [n]" in c for c in sys_msgs)
    assert graph.get_node("ent_henna")["access_count"] == 1  # the re-read did not double-count


def test_async_gap_tells_model_research_is_running(stores, events):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph, async_research=True)
    cog.research = _SyncResearch(cog)
    cb = ContextBuilder(vector, graph, cognition=cog)
    peter = _peter(vector, graph, cb, [reply("Looking into henna for you, Sir.")])
    peter.run("what is henna?")
    cog.last_research_thread.join(timeout=10)
    sys_msgs = [m.content for m in peter.router.calls[0]["messages"] if m.role == "system"]
    assert any("[RESEARCH IN PROGRESS]" in c for c in sys_msgs)
    assert of_type(events, "research_answer")  # follow-up delivered after background research
    assert graph.get_node("ent_henna") is not None


def test_gap_without_research_pipeline_is_harmless(stores, events):
    vector, graph = stores
    cog = CognitiveMemory(vector, graph)
    result, verdict = cog.recall("what is henna?")
    out = cog.handle_gap("what is henna?", result, verdict)
    assert isinstance(out, GapOutcome) and out.mode == "skipped"
    assert of_type(events, "gap_detected")
