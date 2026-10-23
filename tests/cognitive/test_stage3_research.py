"""Stage 3 — on-demand research: claims, screening, corroboration, node + collection, cited answer."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.cognition_config import set_cognition_config
from engine.research_pipeline import ResearchPipeline, domain_of
from memory.cognition import CognitiveMemory
from memory.recall import ContextBuilder
from memory.vector_store import CollectionBudget
from tests.cognitive.conftest import of_type
from tests.cognitive.fake_web import FakeWeb
from tests.fakes.fake_llm import FakeLLMRouter


def reply(text):
    return {"content": text, "tool_calls": [], "usage": {"total_tokens": 1}, "provider": "fake", "model": "f"}


@pytest.fixture
def cog(stores):
    vector, graph = stores
    set_cognition_config({"resolver": {"use_llm": False}})
    c = CognitiveMemory(vector, graph, async_research=False)
    web = FakeWeb()
    c.research = ResearchPipeline(c.knowledge, router=None, search_fn=web.search, fetch_fn=web.fetch)
    c.web = web
    return c


def test_query_building_and_page_picking(cog):
    rp = cog.research
    qs = rp.build_queries("henna", "is black henna safe?", ["safety"])
    assert qs == ["henna", "henna safety", "is black henna safe?"]
    results = [{"url": "https://a.example/1"}, {"url": "https://a.example/2"}, {"url": "https://www.b.example/x"},
               {"url": "https://c.example/y"}, {"url": "https://d.example/z"}, {"url": "https://e.example/q"}]
    picked = rp.pick_pages(results)
    assert [domain_of(p["url"]) for p in picked] == ["a.example", "b.example", "c.example", "d.example"]


def test_claim_extraction_llm_and_heuristic(cog):
    rp = cog.research
    rp.router = FakeLLMRouter([reply("1. Henna comes from Lawsonia inermis leaves.\n- short\n* Henna stains skin orange-red.")])
    assert rp.extract_claims("henna", "page text") == ["Henna comes from Lawsonia inermis leaves.",
                                                       "Henna stains skin orange-red."]
    rp.router = None
    claims = rp.extract_claims("saree", "A saree is a long drape of cloth worn in South Asia. Unrelated sentence here about cars.")
    assert claims == ["A saree is a long drape of cloth worn in South Asia."]


def test_research_creates_cited_collection_with_corroboration(cog, events):
    out = cog.research.research("henna", question="what is henna?")
    vector, graph = cog.vector, cog.graph
    assert out.node_id == "ent_henna" and out.created_node and out.pages_fetched == 3
    assert out.claims_dropped == 1                      # the injection sentence was screened out
    node = graph.get_node("ent_henna")
    assert node["collection"] == "k_henna"
    recs = vector.get_all_records("k_henna")
    assert len(recs) >= 3 and out.records_stored == len(recs)
    assert all(r["metadata"]["source_url"].startswith("https://") for r in recs)
    assert not any("Ignore previous" in r["document"] for r in recs)
    merged = [r for r in recs if r["metadata"]["domains"] == 2]
    assert len(merged) == 1                             # same claim on two domains -> one record
    m = merged[0]["metadata"]
    assert m["confidence"] == 0.6 and m["trust"] == "tool_verified"
    assert len(json.loads(m["sources"])) == 2
    single = [r for r in recs if r["metadata"]["domains"] == 1]
    assert all(r["metadata"]["trust"] == "untrusted_web" and r["metadata"]["confidence"] == 0.4 for r in single)
    sources = graph.get_nodes_by_kind("source")
    assert len(sources) == 3
    assert {e["relation"] for e in graph.edges_of("ent_henna")} == {"cites"}
    assert graph.get_collection("k_henna")["record_count"] == len(recs)
    for t in ("research_start", "research_step", "research_done", "node_spawned", "edge_drawn"):
        assert of_type(events, t), t


def test_research_time_budget(cog):
    ticks = iter([0, 0, 100, 100, 100, 100, 100, 100])
    cog.research._monotonic = lambda: next(ticks, 100)
    out = cog.research.research("henna")
    assert any("time budget" in e for e in out.errors)


def test_budget_exhausted_uses_nearest_node_or_defers(cog):
    rp = cog.research
    spent = CollectionBudget(max_new=0)
    out = rp.research("henna", budget=spent)
    assert out.deferred and out.records_stored == 0
    cog.knowledge.ensure_node("Henna art")
    out2 = rp.research("henna", budget=CollectionBudget(max_new=0))
    assert not out2.deferred and out2.node_id == "ent_henna_art"


def test_answer_cites_sources(cog):
    cog.research.research("henna")
    cog.research.router = FakeLLMRouter([reply("Henna comes from a plant [1] and stains skin [2].")])
    ans = cog.research.answer("what is henna?")
    assert "[1]" in ans and "Sources:" in ans and "https://" in ans
    cog.research.router = None
    stitched = cog.research.answer("what is henna?")
    assert "[1]" in stitched and "Sources:" in stitched
    assert cog.research.answer("tell me about quasars") .startswith("I couldn't find")


def test_done_when_what_is_henna_end_to_end(cog, events):
    """Stage 3 done-when: 'what is henna?' creates node henna, collection k_henna with >= 3 cited
    records, and the answer the model sees carries numbered citations."""
    from agents.base import BaseAgent

    class P(BaseAgent):
        pass

    tools = MagicMock()
    tools.get_schemas.return_value = None
    gr = MagicMock()
    gr.check_input.return_value = SimpleNamespace(allowed=True)
    gr.check_output.return_value = SimpleNamespace(allowed=True)
    cb = ContextBuilder(cog.vector, cog.graph, cognition=cog)
    peter = P(agent_id="peter", name="Peter", system_prompt="sys", policy=[], tool_names=[], max_steps=2,
              router=FakeLLMRouter([reply("Henna is a plant dye [1].\n\nSources:\n[1] https://www.plants.example/henna")]),
              vector=cog.vector, graph=cog.graph, guardrails=gr, tools=tools)
    peter._context_builder = cb
    peter._user_id = "u-test"
    out = peter.run("what is henna?")
    assert cog.graph.get_node("ent_henna")["collection"] == "k_henna"
    assert cog.vector.count("k_henna") >= 3
    knowledge_msg = [m.content for m in peter.router.calls[0]["messages"] if "[KNOWLEDGE" in (m.content or "")]
    assert knowledge_msg and "[1]" in knowledge_msg[0] and "source:" in knowledge_msg[0]
    assert "[1]" in out and "Sources:" in out


def test_claims_relevant_to_the_question_are_kept(cog):
    """Regression: a fact found while researching a question under a parent topic must not be
    dropped just because it doesn't repeat the topic's name."""
    rp = cog.research
    text = "Sweets served at a sinhala wedding include kavum, kokis and aluwa. Unrelated line about trains."
    assert rp.extract_claims("kiribath", text) == []
    assert rp.extract_claims("kiribath", text, "What sweets are served at a sinhala wedding?") == [
        "Sweets served at a sinhala wedding include kavum, kokis and aluwa."]
