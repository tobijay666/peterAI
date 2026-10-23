"""Stage 4 — curiosity queue: generation, dedupe, priority, persistence, graph presence, API."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.cognition_config import set_cognition_config
from engine.curiosity import CuriosityEngine
from memory.cognition import CognitiveMemory
from memory.graph_store import GraphMemory
from tests.cognitive.conftest import of_type
from tests.fakes.fake_llm import FakeLLMRouter


def reply(text):
    return {"content": text, "tool_calls": [], "usage": {"total_tokens": 1}, "provider": "fake", "model": "f"}


QUESTIONS = (
    "How is bridal henna applied before Sri Lankan weddings?\n"
    "How is bridal henna applied before Sri Lankan weddings today?\n"   # near-twin of the first
    "Is black henna safe for skin?\n"
    "What plants besides henna are used as natural skin dyes?\n"
    "not a question\n"
)


@pytest.fixture
def setup(stores):
    vector, graph = stores
    set_cognition_config({"resolver": {"use_llm": False}})
    cog = CognitiveMemory(vector, graph)
    node, _ = cog.knowledge.ensure_node("Henna")
    cog.knowledge.add_record(node, "Henna is a plant dye.")
    eng = CuriosityEngine(cog.knowledge, resolver=cog.resolver, router=FakeLLMRouter([reply(QUESTIONS)]))
    cog.curiosity = eng
    return cog, eng


def test_turn_yields_3_to_5_non_duplicate_items(setup, events):
    cog, eng = setup
    items = cog.after_turn("what is henna?", "Henna is a plant dye.", turn_id="t1")
    qs = [i["question"] for i in items]
    assert len(qs) == 3  # 4 valid proposals, one near-twin dropped (spec: 3–5 per turn)
    assert "How is bridal henna applied before Sri Lankan weddings today?" not in qs  # twin dropped
    assert all(q.endswith("?") for q in qs)
    assert all(i["topic_node_id"] == "ent_henna" for i in items)
    assert len(of_type(events, "curiosity_added")) == len(items)


def test_curiosity_nodes_are_in_the_graph(setup):
    cog, eng = setup
    items = cog.after_turn("what is henna?", "")
    for it in items:
        node = cog.graph.get_node(f"cur_{it['id']}")
        assert node["kind"] == "curiosity"
        assert any(e["relation"] == "curious_about" and e["dst"] == "ent_henna"
                   for e in cog.graph.edges_of(node["id"]))


def test_queue_survives_restart(setup, tmp_path, clock):
    cog, eng = setup
    items = cog.after_turn("what is henna?", "")
    reopened = GraphMemory(database_path=str(cog.graph.db_path), clock=clock, user_id="u-test")
    rows = reopened._get_connection().execute("SELECT id FROM curiosity_queue WHERE status='pending'").fetchall()
    assert {r["id"] for r in rows} == {i["id"] for i in items}
    reopened.close()


def test_dismiss_blocks_twins_and_boost_reorders(setup):
    cog, eng = setup
    first = cog.after_turn("what is henna?", "")
    target = next(i for i in first if "bridal henna" in i["question"])
    eng.dismiss(target["id"])
    assert eng.get(target["id"])["status"] == "dismissed"
    assert cog.graph.get_node(f"cur_{target['id']}") is None
    # The same proposal arrives again on the next turn: the dismissed twin must not come back
    eng.router = FakeLLMRouter([reply("How is bridal henna applied before Sri Lankan weddings, exactly?\n")])
    again = eng.generate("henna for brides", "")
    assert again == []
    pending = eng.list()
    last = pending[-1]
    eng.boost(last["id"], delta=5)
    assert eng.list()[0]["id"] == last["id"]


def test_priority_formula_prefers_gaps_and_used_topics(setup, clock):
    cog, eng = setup
    eng.router = FakeLLMRouter([reply("What is the history of saree draping?\n")])
    gap_items = eng.generate("tell me about saree draping", "")  # 'saree draping' has no node -> gap
    assert gap_items and gap_items[0]["gap"] == 1 and gap_items[0]["topic_node_id"] is None
    eng.router = FakeLLMRouter([reply("Which regions grow the best henna crops?\n")])
    known = eng.generate("henna", "")
    assert known[0]["gap"] == 0
    assert gap_items[0]["priority"] > known[0]["priority"]
    # recency: the same item ranks lower as it ages
    p_now = eng.effective_priority(known[0])
    clock.advance(days=14)
    assert eng.effective_priority(known[0]) < p_now


def test_question_already_covered_by_a_collection_is_skipped(setup):
    cog, eng = setup
    cog.knowledge.ensure_node("Black henna safety")
    eng.router = FakeLLMRouter([reply("Black henna safety?\n")])
    assert eng.generate("henna", "") == []


def test_template_fallback_without_router(setup):
    cog, eng = setup
    eng.router = None
    items = eng.generate("what is henna?", "")
    assert len(items) == 4 and all("Henna" in i["question"] for i in items)


def test_pop_and_complete_lifecycle(setup):
    cog, eng = setup
    cog.after_turn("what is henna?", "")
    top = eng.pop_top(1)[0]
    assert top["status"] == "researching" and top["attempts"] == 1
    eng.complete(top["id"], ok=False)
    assert eng.get(top["id"])["status"] == "pending"
    for _ in range(2):
        for popped in eng.pop_top(10):
            eng.complete(popped["id"], ok=False)
    assert eng.get(top["id"])["status"] == "dropped"
    other = eng.pop_top(1)[0]
    eng.complete(other["id"], ok=True)
    assert eng.get(other["id"])["status"] == "done"
    assert cog.graph.get_node(f"cur_{other['id']}") is None


def test_pending_cap(setup):
    cog, eng = setup
    set_cognition_config({"resolver": {"use_llm": False}, "curiosity": {"max_pending": 2}})
    eng.router = None
    eng.generate("what is henna?", "")
    assert len(eng.list()) == 2


def test_gap_item_added_then_boosted(setup):
    cog, eng = setup
    node = cog.graph.get_node("ent_henna")
    a = eng.add_gap_item(node, "few_records")
    b = eng.add_gap_item(node, "few_records")
    assert a["id"] == b["id"] and b["priority"] > a["priority"]


def test_curiosity_api(setup, monkeypatch):
    from fastapi.testclient import TestClient
    from server.app import app
    from server.auth import get_current_user
    from core.user_context import UserContext
    import server.routes.curiosity_routes as routes

    cog, eng = setup
    cog.after_turn("what is henna?", "")
    peter = SimpleNamespace(cognition=cog)
    monkeypatch.setattr(routes.pool, "get_or_create", lambda user: (peter, None, None, None))
    app.dependency_overrides[get_current_user] = lambda: UserContext(user_id="u-test")
    try:
        client = TestClient(app)
        r = client.get("/api/curiosity")
        assert r.status_code == 200 and len(r.json()["items"]) >= 3
        first = r.json()["items"][0]["id"]
        assert client.post(f"/api/curiosity/{first}", json={"action": "dismiss"}).json()["item"]["status"] == "dismissed"
        assert client.post(f"/api/curiosity/{first}", json={"action": "nope"}).status_code == 400
        assert client.post("/api/curiosity/missing", json={"action": "boost"}).status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_gap_questions_remember_their_topic_term(setup):
    """Regression: background research must use the gap term, not a phrase guessed from the question."""
    cog, eng = setup
    eng.router = None
    items = eng.generate("tell me about the kandyan saree", "")
    assert items and all(i["topic"] == "kandyan saree" and i["gap"] == 1 for i in items)
    assert eng.topic_for(items[1]) == "kandyan saree"      # e.g. "How is kandyan saree used or practised today?"
    known = eng.add_gap_item(cog.graph.get_node("ent_henna"), "few_records")
    assert eng.topic_for(known) == "Henna"
