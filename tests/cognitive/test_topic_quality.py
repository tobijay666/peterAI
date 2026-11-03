"""Regression: "whats the main storyline in the picture of dorian gray by oscar wilde?"

What went wrong on 3 Nov 2026 (data/users/u1, peter_trace.log):
  1. gpt-oss-120b spent its 80-token budget reasoning, so the topic-extraction reply was
     empty and the crude fallback produced topics 'whats', 'main storyline', 'picture'.
  2. Each of those was researched separately; 'picture' found Pixabay / Google Images.
  3. "The source text does not contain any information about ..." was stored as a fact,
     and because two image sites produced it, it was marked cross-checked (tool_verified).
  4. Groq rejected one tool call ("context": null) and the main model was blacklisted.
These tests pin every one of those fixes.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.cognition_config import set_cognition_config
from engine.curiosity import CuriosityEngine
from engine.research_pipeline import ResearchPipeline
from memory.cognition import CognitiveMemory
from memory.entity_resolver import EntityResolver, heuristic_aspect, heuristic_entities
from tests.cognitive.conftest import of_type
from tests.fakes.fake_llm import FakeLLMRouter

QUESTION = "whats the main storyline in the picture of dorian gray by oscar wilde?"

WIKI = "https://en.wikipedia.org/wiki/The_Picture_of_Dorian_Gray"
SPARK = "https://www.sparknotes.com/lit/doriangray/summary/"
WILDE = "https://en.wikipedia.org/wiki/Oscar_Wilde"
PAGES = {
    WIKI: ("The Picture of Dorian Gray - Wikipedia",
           "The Picture of Dorian Gray is an 1890 novel by Oscar Wilde. Dorian Gray wishes that his portrait "
           "would age instead of him, and the wish comes true. Dorian Gray pursues a life of pleasure while "
           "the painting records every sin. The novel explores the duality of a public face and a hidden self."),
    SPARK: ("The Picture of Dorian Gray: Plot Summary | SparkNotes",
            "Basil Hallward paints a portrait of Dorian Gray. Lord Henry persuades Dorian Gray that youth and "
            "beauty are all that matter. Dorian Gray murders Basil Hallward when Basil sees the corrupted portrait."),
    WILDE: ("Oscar Wilde - Wikipedia",
            "Oscar Wilde was an Irish poet and playwright. Oscar Wilde wrote The Picture of Dorian Gray, "
            "his only novel, published in 1890."),
    "https://pixabay.com/": ("6.4 million+ Stunning Free Images to Use Anywhere - Pixabay",
                             "Free stock photos and pictures. Download royalty-free images."),
    "https://images.google.com/": ("Google Images", "Search the world's images."),
    "https://www.pexels.com/search/picture/": ("Picture Photos, Download The BEST Free Picture Stock Photos & HD …",
                                               "Download and use 100,000+ picture stock photos for free."),
    "https://www.vecteezy.com/free-photos/picture": ("Picture Stock Photos, Images and Backgrounds - Vecteezy",
                                                     "Browse picture photos. Free download."),
}
IMAGE_SITES = [u for u in PAGES if u not in (WIKI, SPARK, WILDE)]


class Web:
    """Search returns the image junk first (as DuckDuckGo did), real pages after."""

    def __init__(self, junk_only: bool = False):
        self.junk_only = junk_only
        self.fetched = []
        self.searched = []

    def search(self, q, n):
        self.searched.append(q)
        urls = list(IMAGE_SITES)
        if not self.junk_only:
            urls += [WILDE] if "wilde" in q.lower() and "dorian" not in q.lower() else [WIKI, SPARK]
        return [{"title": PAGES[u][0], "url": u, "snippet": PAGES[u][1][:60]} for u in urls][:n]

    def fetch(self, url):
        self.fetched.append(url)
        return PAGES[url][1]


class ClaimLLM:
    """Stands in for the model during claim extraction: real claims for real pages,
    the 'source text does not contain' sentence for anything else."""

    def complete(self, messages, **kw):
        text = messages[-1].content
        sents = [s.strip() + "." for s in text.split("SOURCE TEXT:\n")[-1].split(".") if "Dorian" in s or "Wilde" in s]
        if sents:
            return {"content": "\n".join(sents)}
        return {"content": "The source text does not contain any information about The Picture of Dorian Gray.\n"
                           "The source text does not provide details about the plot of the novel."}


def reply(text):
    return {"content": text, "tool_calls": [], "usage": {"total_tokens": 1}}


# ---------------------------------------------------------------- 1. topic extraction

def test_heuristic_never_returns_question_words_or_aspects():
    assert heuristic_entities(QUESTION) == ["picture of dorian gray", "oscar wilde"]
    assert heuristic_aspect(QUESTION) == "main storyline"
    assert heuristic_entities("hi whats up?") == []
    assert heuristic_entities('summary of "The Lord of the Rings"') == ["Lord of the Rings"]
    assert heuristic_entities("henna and saree draping") == ["henna", "saree draping"]


def test_llm_extraction_survives_reasoning_text_and_returns_aspect_and_category(stores):
    vector, graph = stores
    raw = ('We need the main entity. Maybe {"note": 1}. Final answer: {"entities": ["The Picture of Dorian Gray", '
           '"Oscar Wilde", "whats"], "aspect": "plot summary", "category": "Victorian literature"}')
    r = EntityResolver(vector, graph, router=FakeLLMRouter([reply(raw)]))
    d = r.extract_details(QUESTION)
    assert d["entities"] == ["The Picture of Dorian Gray", "Oscar Wilde"]
    assert (d["aspect"], d["category"], d["via"]) == ("plot summary", "Victorian literature", "llm")
    assert r.router.calls[0]["messages"][0].content.startswith("Identify what")


def test_empty_reply_from_a_reasoning_model_falls_back_sensibly(stores):
    vector, graph = stores
    r = EntityResolver(vector, graph, router=FakeLLMRouter([reply("")]))
    d = r.extract_details(QUESTION)
    assert d["via"] == "heuristic" and d["entities"] == ["picture of dorian gray", "oscar wilde"]


def test_extraction_budget_is_big_enough_for_reasoning_models(stores):
    vector, graph = stores
    seen = {}

    class R:
        def complete(self, messages, **kw):
            seen.update(kw)
            return reply('{"entities": ["henna"]}')

    EntityResolver(vector, graph, router=R()).extract("what is henna?")
    assert seen["max_tokens"] >= 500


# ---------------------------------------------------------------- 2-3. research quality

@pytest.fixture
def cog(stores):
    vector, graph = stores
    set_cognition_config({"resolver": {"use_llm": False}})
    c = CognitiveMemory(vector, graph, async_research=False)
    c.web = Web()
    c.research = ResearchPipeline(c.knowledge, router=ClaimLLM(), search_fn=c.web.search, fetch_fn=c.web.fetch)
    c.curiosity = CuriosityEngine(c.knowledge, resolver=c.resolver, router=None)
    return c


def test_image_and_home_pages_are_never_fetched(cog):
    picked = cog.research.pick_pages(cog.web.search("The Picture of Dorian Gray", 10), "The Picture of Dorian Gray")
    assert [p["url"] for p in picked] == [WIKI, SPARK]


def test_meta_claims_are_never_stored_and_never_marked_verified(cog):
    out = cog.research.research("The Picture of Dorian Gray", question=QUESTION, missing_aspects=["main storyline"])
    docs = [r["document"] for r in cog.vector.get_all_records("k_the_picture_of_dorian_gray")]
    assert out.records_stored >= 3 and docs
    assert not any("source text" in d.lower() for d in docs)
    assert all("Dorian" in d or "Wilde" in d or "Basil" in d for d in docs)
    assert set(cog.web.fetched) <= {WIKI, SPARK}
    assert cog.web.searched[0] == "The Picture of Dorian Gray main storyline"   # most specific query first


def test_research_that_finds_nothing_relevant_leaves_no_empty_topic(stores, events):
    vector, graph = stores
    c = CognitiveMemory(vector, graph, async_research=False)
    web = Web(junk_only=True)
    c.research = ResearchPipeline(c.knowledge, router=ClaimLLM(), search_fn=web.search, fetch_fn=web.fetch)
    out = c.research.research("The Picture of Dorian Gray", question=QUESTION)
    assert out.records_stored == 0 and out.node_id is None
    assert graph.get_node("ent_the_picture_of_dorian_gray") is None
    assert not vector.has_collection("k_the_picture_of_dorian_gray")
    assert of_type(events, "node_pruned")      # the mind view dissolves the provisional node
    assert web.fetched == []


def test_the_question_creates_the_right_nodes_and_logical_links(stores, tmp_path, events, monkeypatch):
    """End to end: one book node with real facts, the author queued and then researched
    during sleep and linked related_to, the book filed under its field."""
    import engine.evolution as evolution
    from engine.evolution import EvolutionCoordinator
    from memory.snapshot import snapshot_manager_for

    vector, graph = stores
    monkeypatch.setattr(evolution, "JOURNAL_PATH", tmp_path / "j.jsonl")
    set_cognition_config({"sufficiency": {"use_llm_judge": False}})
    extraction = reply('{"entities": ["The Picture of Dorian Gray", "Oscar Wilde"], "aspect": "plot summary", '
                       '"category": "Victorian literature"}')

    class Router(ClaimLLM):
        def complete(self, messages, **kw):
            if messages[0].content.startswith("Identify what"):
                return extraction
            return super().complete(messages, **kw)

    router = Router()
    c = CognitiveMemory(vector, graph, router=router, async_research=False)
    web = Web()
    c.research = ResearchPipeline(c.knowledge, router=router, search_fn=web.search, fetch_fn=web.fetch)
    c.curiosity = CuriosityEngine(c.knowledge, resolver=c.resolver, router=None)

    result, verdict = c.recall(QUESTION)
    assert verdict.reason == "no_node"
    out = c.handle_gap(QUESTION, result, verdict)
    assert out.topics == ["The Picture of Dorian Gray"]                     # only the main topic now
    labels = {n["label"] for n in graph.all_nodes(["entity", "abstraction"])}
    assert labels == {"The Picture of Dorian Gray", "Victorian literature"}
    assert not labels & {"whats", "picture", "main storyline"}
    hub = c.knowledge.find_node("Victorian literature")
    assert any(e["relation"] == "abstracts" and e["dst"] == "ent_the_picture_of_dorian_gray"
               for e in graph.edges_of(hub["id"]))
    queued = c.curiosity.list()
    wilde_q = [q for q in queued if q["topic"] == "Oscar Wilde"]
    assert wilde_q and wilde_q[0]["topic_node_id"] == "ent_the_picture_of_dorian_gray"

    # Sleep researches the author and links the two
    ledger = MagicMock()
    ledger._conn.side_effect = Exception("no ledger")
    skills = MagicMock()
    skills.list_skills.return_value = []
    EvolutionCoordinator(vector=vector, graph=graph, router=None, guardrails=MagicMock(), ledger=ledger,
                         skill_store=skills, consolidation_config={"min_age_hours": 1e9}, mode="active",
                         clock=c._clock, cognition=c,
                         snapshot_manager=snapshot_manager_for(vector, graph, clock=c._clock)).run_cycle()
    wilde = c.knowledge.find_node("Oscar Wilde")
    assert wilde is not None and vector.count(wilde["collection"]) >= 1
    assert any(e["relation"] == "related_to" and {e["src"], e["dst"]} == {wilde["id"], "ent_the_picture_of_dorian_gray"}
               for e in graph.edges_of(wilde["id"]))


def test_known_topics_asked_together_get_linked(cog):
    a, _ = cog.knowledge.ensure_node("henna")
    b, _ = cog.knowledge.ensure_node("saree")
    for n, t in ((a, "Henna is a plant dye used at weddings."), (b, "A saree is a draped garment worn at weddings.")):
        cog.knowledge.add_record(n, t)
    cog.recall("henna and saree")
    assert any(e["relation"] == "related_to" for e in cog.graph.edges_of("ent_henna"))


# ---------------------------------------------------------------- 4. model handling

def test_one_rejected_tool_call_does_not_blacklist_the_model():
    from llm.providers.base import _is_request_specific_error, _strip_thinking
    err = Exception("Error code: 400 - {'error': {'message': 'Tool call validation failed: parameters for tool "
                    "delegate_to_researcher did not match schema', 'type': 'invalid_request_error', "
                    "'code': 'tool_use_failed'}}")
    assert _is_request_specific_error(err)
    assert not _is_request_specific_error(TimeoutError("Request timed out."))
    assert not _is_request_specific_error(Exception("Error code: 404 - model not found"))
    assert _strip_thinking("<think>plan</think>{\"entities\": []}") == '{"entities": []}'
    assert _strip_thinking("<think>still thinking") == ""


def test_delegate_tool_accepts_null_context(stores):
    from agents.peter import PeterAgent
    vector, graph = stores
    tools = MagicMock()
    schemas = {}
    tools.register.side_effect = lambda name, fn, schema: schemas.__setitem__(name, schema)
    PeterAgent(router=MagicMock(), vector=vector, graph=graph, guardrails=MagicMock(), tools=tools)
    assert schemas["delegate_to_researcher"]["parameters"]["properties"]["context"]["type"] == ["string", "null"]
