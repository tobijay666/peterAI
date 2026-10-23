"""Stage 6 — visual style rules, stable layout, graph snapshot, node inspection, API."""
from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

from core.cognition_config import set_cognition_config
from engine.curiosity import CuriosityEngine
from memory.cognition import CognitiveMemory
from memory.mind_view import graph_snapshot, node_detail
from memory.visual_style import visual_style
from tests.cognitive.conftest import of_type

TOPICS = {
    "Henna": ["Bridal henna is applied before Sri Lankan weddings", "Henna dye comes from Lawsonia leaves"],
    "Saree": ["The Kandyan saree is the bridal dress at Sri Lankan weddings", "A saree is a long drape of cloth"],
    "Invoices": ["Invoices must be paid within thirty days", "Late invoice payments incur a fee"],
}


@pytest.fixture
def cog(stores):
    vector, graph = stores
    set_cognition_config({"resolver": {"use_llm": False}})
    c = CognitiveMemory(vector, graph)
    c.curiosity = CuriosityEngine(c.knowledge, resolver=c.resolver, router=None)
    for label, recs in TOPICS.items():
        node, _ = c.knowledge.ensure_node(label)
        for r in recs:
            c.knowledge.add_record(node, r, {"source_url": f"https://{label.lower()}.example/a", "confidence": 0.6})
        c.knowledge.refresh_stats(node)
    return c


# ---------------------------------------------------------------- style rules

def test_tiers_match_the_spec():
    hi, mid, lo, fade = visual_style(0.95), visual_style(0.5), visual_style(0.1), visual_style(0.9, "fading")
    assert (hi["tier"], mid["tier"], lo["tier"], fade["tier"]) == ("high", "moderate", "low", "fading")
    assert hi["scale"] > mid["scale"] > lo["scale"] > fade["scale"]       # bigger when used
    assert hi["emissive"] >= 1.5 > mid["emissive"] > lo["emissive"]       # glows (bloom) when used
    assert lo["saturation"] < mid["saturation"] <= hi["saturation"]       # dull when unused
    assert lo["opacity"] < mid["opacity"] <= hi["opacity"]                # faint when unused


@settings(max_examples=300, deadline=None)
@given(st.floats(0, 1), st.floats(0, 1))
def test_style_is_monotonic_in_usage(a, b):
    lo, hi = sorted((a, b))
    s_lo, s_hi = visual_style(lo), visual_style(hi)
    for k in ("scale", "emissive", "saturation", "opacity"):
        assert s_hi[k] >= s_lo[k] - 1e-9, (k, lo, hi, s_lo, s_hi)


@settings(max_examples=100, deadline=None)
@given(st.floats(0, 1))
def test_fading_is_never_brighter_than_low(p):
    f, low = visual_style(p, "fading"), visual_style(0.0)
    assert f["scale"] <= low["scale"] and f["opacity"] <= low["opacity"] and f["emissive"] <= low["emissive"]


# ---------------------------------------------------------------- layout

def test_positions_are_stable_and_new_nodes_do_not_reshuffle(cog):
    first = {n["id"]: n["position"] for n in graph_snapshot(cog)["nodes"]}
    again = {n["id"]: n["position"] for n in graph_snapshot(cog)["nodes"]}
    assert first == again
    node, _ = cog.knowledge.ensure_node("Poruwa")
    cog.knowledge.add_record(node, "The poruwa ceremony is a Sri Lankan wedding ritual")
    after = {n["id"]: n["position"] for n in graph_snapshot(cog)["nodes"]}
    for nid, pos in first.items():
        assert after[nid] == pos
    assert "ent_poruwa" in after


def test_similar_topics_sit_closer_than_unrelated(cog):
    pos = {n["id"]: n["position"] for n in graph_snapshot(cog)["nodes"]}

    def d(a, b):
        return sum((x - y) ** 2 for x, y in zip(pos[a], pos[b])) ** 0.5

    assert d("ent_henna", "ent_saree") < d("ent_henna", "ent_invoices")


def test_refit_moves_are_emitted(cog, events):
    graph_snapshot(cog)
    moves = cog.layout.refit()
    assert {m["node_id"] for m in moves} == {"ent_henna", "ent_saree", "ent_invoices"}


# ---------------------------------------------------------------- snapshot + detail

def test_graph_snapshot_shape(cog):
    cog.curiosity.generate("what is henna?", "")
    for _ in range(5):
        cog.usage.touch("ent_henna")
    snap = graph_snapshot(cog)
    nodes = {n["id"]: n for n in snap["nodes"]}
    assert nodes["ent_henna"]["tier"] == "high" and nodes["ent_henna"]["access_count"] == 6
    assert nodes["ent_henna"]["style"]["scale"] > nodes["ent_invoices"]["style"]["scale"]
    assert nodes["ent_henna"]["record_count"] == 2 and nodes["ent_henna"]["collection"] == "k_henna"
    cur = [n for n in snap["nodes"] if n["kind"] == "curiosity"]
    assert cur and all(n["tier"] == "curiosity" for n in cur)
    assert any(e["relation"] == "curious_about" for e in snap["edges"])
    assert not any(n["kind"] == "source" for n in snap["nodes"])
    hx = nodes["ent_henna"]["position"]
    cx = cur[0]["position"]
    assert sum((a - b) ** 2 for a, b in zip(hx, cx)) ** 0.5 < 3.0          # curiosity orbits its topic
    assert snap["stats"]["by_kind"]["entity"] == 3


def test_node_detail_has_everything_the_panel_needs(cog):
    cog.curiosity.generate("what is henna?", "")
    d = node_detail(cog, "ent_henna")
    assert d["usage"]["access_count"] == 1 and d["usage"]["score"] > 0
    assert d["collection"] == {"name": "k_henna", "record_count": 2}
    r = d["records"][0]
    assert r["sources"][0]["url"] == "https://henna.example/a" and r["confidence"] == 0.6
    assert len(r["embedding_preview"]) == 8 and r["embedding_dims"] == 384 and r["embedding_norm"] == pytest.approx(1, abs=1e-3)
    assert "henna" in d["aliases"] and d["curiosity"] and d["sources"] == ["https://henna.example/a"]
    assert node_detail(cog, "nope") is None


def test_snapshot_of_2000_nodes_under_500ms(cog):
    graph = cog.graph
    now = cog._clock.now()
    with graph._get_connection() as conn:
        for i in range(2000):
            conn.execute(
                "INSERT INTO nodes (id, kind, label, data, created_at, last_used, importance, pinned, access_count,"
                " last_accessed, status, pos_x, pos_y, pos_z) VALUES (?, 'entity', ?, '{}', ?, ?, 0.5, 0, ?, ?, 'active', ?, ?, ?)",
                (f"bulk_{i}", f"Topic {i}", now, now, i % 17, now, float(i % 50), float(i % 7), float(i % 11)),
            )
        for i in range(1999):
            conn.execute("INSERT INTO edges (id, src, dst, relation, data, created_at) VALUES (?, ?, ?, 'related_to', '{}', ?)",
                         (uuid.uuid4().hex, f"bulk_{i}", f"bulk_{i + 1}", now))
        conn.commit()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        snap = graph_snapshot(cog)
        best = min(best, time.perf_counter() - t0)
    assert snap["stats"]["nodes"] >= 2003 and snap["stats"]["edges"] >= 1999
    assert best < 0.5, f"snapshot took {best:.3f}s"


# ---------------------------------------------------------------- API

@pytest.fixture
def client(cog, monkeypatch):
    from fastapi.testclient import TestClient
    from server.app import app
    from server.auth import get_current_user
    from core.user_context import UserContext
    import server.routes.mind_routes as routes

    peter = SimpleNamespace(cognition=cog, vector=cog.vector, graph=cog.graph)
    monkeypatch.setattr(routes.pool, "get_or_create", lambda user: (peter, None, None, None))
    app.dependency_overrides[get_current_user] = lambda: UserContext(user_id="u-test")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_snapshot_node_and_sleep(client, monkeypatch):
    r = client.get("/api/mind/snapshot")
    assert r.status_code == 200 and r.json()["view"] == "graph" and r.json()["nodes"]
    r = client.get("/api/mind/snapshot?view=records")
    assert r.json()["view"] == "records" and r.json()["points"]
    assert client.get("/api/mind/node/ent_henna").json()["collection"]["name"] == "k_henna"
    assert client.get("/api/mind/node/missing").status_code == 404

    import server.scheduler as sched
    ran = {}
    monkeypatch.setattr(sched, "run_cycle_for_user",
                        lambda uid, reason, mode, should_pause: ran.update(uid=uid, reason=reason, mode=mode))
    r = client.post("/api/mind/sleep", json={"mode": "active"})
    assert r.json()["started"] is True
    for _ in range(50):
        if ran:
            break
        time.sleep(0.02)
    assert ran == {"uid": "u-test", "reason": "manual", "mode": "active"}
    assert client.post("/api/mind/sleep", json={"mode": "bogus"}).status_code == 400
