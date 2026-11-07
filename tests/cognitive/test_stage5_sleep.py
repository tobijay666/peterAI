"""Stage 5 — sleep cycle v2: order, audit/fade/prune, abstraction + migration, cap of 5,
gap scan + background research, shadow mode, pause, idle trigger, per-user snapshots."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import engine.evolution as evolution
from core.cognition_config import set_cognition_config
from engine.curiosity import CuriosityEngine
from engine.evolution import EvolutionCoordinator
from engine.research_pipeline import ResearchPipeline
from memory.cognition import CognitiveMemory
from memory.graph_store import GraphMemory
from memory.snapshot import restore_snapshot, snapshot_manager_for
from server.idle import IdleTracker
from tests.cognitive.conftest import DAY, of_type
from tests.cognitive.fake_web import FakeWeb

WEDDING = {
    "Henna": ["Bridal henna is applied before Sri Lankan weddings",
              "Henna dye comes from the Lawsonia plant leaves",
              "Henna designs for the bride decorate hands at weddings"],
    "Saree": ["The Kandyan saree is the bridal dress at Sri Lankan weddings",
              "A saree is a long drape of cloth",
              "Brides wear a saree with gold jewellery at weddings"],
    "Poruwa": ["The poruwa ceremony is the Sri Lankan wedding ritual",
               "The bride and groom stand on the poruwa platform",
               "Elders tie the couple's fingers with gold thread at weddings"],
    "Invoices": ["Invoices must be paid within thirty days",
                 "Late invoice payments incur a fee",
                 "Send the invoice to the accounts team"],
}

TEST_CFG = {
    "resolver": {"use_llm": False},
    "sleep": {"abstraction": {"min_similarity": 0.24, "max_similarity": 0.80, "migrate_min_similarity": 0.2}},
}


@pytest.fixture
def cog(stores, tmp_path, monkeypatch):
    vector, graph = stores
    set_cognition_config(TEST_CFG)
    monkeypatch.setattr(evolution, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    c = CognitiveMemory(vector, graph, router=None, async_research=False)
    web = FakeWeb()
    c.research = ResearchPipeline(c.knowledge, router=None, search_fn=web.search, fetch_fn=web.fetch)
    c.curiosity = CuriosityEngine(c.knowledge, resolver=c.resolver, router=None)
    c.web = web
    return c


def seed(cog, topics):
    for label, recs in topics.items():
        node, _ = cog.knowledge.ensure_node(label)
        for r in recs:
            cog.knowledge.add_record(node, r, {"source_url": f"https://{label.lower()}.example/{len(r)}"})
        cog.knowledge.refresh_stats(node)


def coordinator(cog, tmp_path, mode="active", should_pause=None):
    ledger = MagicMock()
    ledger._conn.side_effect = Exception("no ledger in this test")
    skills = MagicMock()
    skills.list_skills.return_value = []
    return EvolutionCoordinator(
        vector=cog.vector, graph=cog.graph, router=None, guardrails=MagicMock(), ledger=ledger,
        skill_store=skills, consolidation_config={"min_age_hours": 10_000}, mode=mode, clock=cog._clock,
        cognition=cog, snapshot_manager=snapshot_manager_for(cog.vector, cog.graph, clock=cog._clock),
        should_pause=should_pause,
    )


def phases(events):
    return [e["name"] for e in of_type(events, "sleep_phase")]


# ---------------------------------------------------------------- order + abstraction

def test_cycle_order_events_and_journal(cog, tmp_path, events):
    seed(cog, WEDDING)
    report = coordinator(cog, tmp_path).run_cycle(reason="manual")
    # "maintenance" (hierarchy repair, stale copies, counts, vector weights/links) runs after prune
    assert phases(events) == ["usage_audit", "abstraction", "merge", "fill", "gap_scan",
                              "background_research", "prune", "maintenance"]
    assert of_type(events, "sleep_start")[0]["reason"] == "manual"
    assert of_type(events, "sleep_end")[0]["report"]["cycle_id"] == report["cycle_id"]
    journal = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
    assert '"step": "sleep_cycle"' in journal
    assert not report["errors"], report["errors"]


def test_abstraction_bridges_wedding_topics_and_copies_embeddings(cog, tmp_path, events):
    seed(cog, WEDDING)
    report = coordinator(cog, tmp_path).run_cycle()
    abstractions = report["sleep"]["abstractions"]
    assert len(abstractions) == 1
    a = abstractions[0]
    assert set(a["members"]) == {"ent_henna", "ent_saree", "ent_poruwa"}   # invoices stay out
    node = cog.graph.get_node(a["node_id"])
    assert node["kind"] == "abstraction" and node["collection"].startswith("k_")
    kids = {e["dst"] for e in cog.graph.edges_of(node["id"]) if e["relation"] == "abstracts"}
    assert kids == set(a["members"])
    assert any(e["relation"] == "bridges" for e in cog.graph.edges_of("ent_saree"))
    copied = [r for r in cog.vector.get_all_records(node["collection"]) if r["metadata"].get("origin_id")]
    assert a["migrated"] == len(copied) > 0   # (gap research may add more, non-migrated records)
    first = copied[0]["metadata"]
    src = cog.vector.get_records(first["origin_collection"], [first["origin_id"]])[0]
    assert copied[0]["embedding"] == pytest.approx(src["embedding"], abs=1e-5)   # D1: copied, not re-embedded
    assert cog.vector.count("k_henna") == 3                                     # sources keep their records
    assert of_type(events, "embedding_migrated") and of_type(events, "edge_drawn")
    # a second cycle does not make the same abstraction again
    again = coordinator(cog, tmp_path).run_cycle()
    assert again["sleep"]["abstractions"] == []


def test_abstraction_name_from_llm(cog, tmp_path):
    from tests.fakes.fake_llm import FakeLLMRouter
    seed(cog, WEDDING)
    cog.router = FakeLLMRouter([{"content": "Wedding cultures", "tool_calls": []}], loop=True)
    report = coordinator(cog, tmp_path).run_cycle()
    assert report["sleep"]["abstractions"][0]["node_id"] == "abs_wedding_cultures"


def test_cap_of_five_new_collections_per_cycle(cog, tmp_path):
    set_cognition_config({**TEST_CFG, "sleep": {"abstraction": {"min_similarity": 0.4, "max_similarity": 0.95,
                                                                 "migrate_min_similarity": 0.2},
                                                 "research_items_per_cycle": 0}})
    topics = {}
    for i in range(7):
        w, x = f"word{i}alpha", f"word{i}beta"
        topics[f"Pair {i} one"] = [f"{w} {x} gamma{i}one", f"{w} {x} delta{i}one"]
        topics[f"Pair {i} two"] = [f"{w} {x} gamma{i}two", f"{w} {x} delta{i}two"]
    seed(cog, topics)
    r1 = coordinator(cog, tmp_path).run_cycle()
    assert len(r1["sleep"]["collections_created"]) == 5
    assert len(cog.graph.get_nodes_by_kind("abstraction")) == 5
    r2 = coordinator(cog, tmp_path).run_cycle()
    assert len(r2["sleep"]["collections_created"]) == 2
    assert len(cog.graph.get_nodes_by_kind("abstraction")) == 7


# ---------------------------------------------------------------- forgetting

def test_stale_node_fades_then_is_pruned_and_snapshot_restores_it(cog, tmp_path, clock, events):
    seed(cog, WEDDING)
    seed(cog, {"Obscure topic": ["zyxx quantum barnacle fact", "zyxx barnacle trivia"]})
    clock.advance(days=45)
    for nid in ("ent_henna", "ent_saree", "ent_poruwa", "ent_invoices"):
        cog.usage.touch(nid)
    c1 = coordinator(cog, tmp_path).run_cycle()
    assert "ent_obscure_topic" in c1["sleep"]["faded"]
    assert cog.graph.get_node("ent_obscure_topic")["status"] == "fading"
    assert of_type(events, "node_fading")
    clock.advance(days=1)
    c2 = coordinator(cog, tmp_path).run_cycle()                       # inside the 7-day grace period
    assert c2["sleep"]["pruned"] == [] and cog.graph.get_node("ent_obscure_topic")
    for nid in ("ent_henna", "ent_saree", "ent_poruwa", "ent_invoices"):
        cog.usage.touch(nid)
    clock.advance(days=8)
    for nid in ("ent_henna", "ent_saree", "ent_poruwa", "ent_invoices"):
        cog.usage.touch(nid)
    snaps_before = cog.graph.db_path.parent / "snapshots"
    c3 = coordinator(cog, tmp_path).run_cycle()
    assert c3["sleep"]["pruned"] == ["ent_obscure_topic"]
    assert cog.graph.get_node("ent_obscure_topic") is None
    assert not cog.vector.has_collection("k_obscure_topic")
    assert all(h["node_id"] != "ent_obscure_topic" for h in cog.vector.search_entities("obscure", 5))
    assert of_type(events, "node_pruned")[-1]["node_id"] == "ent_obscure_topic"
    # D6: the pre-cycle snapshot brings it back
    snap = sorted(snaps_before.iterdir())[-1]
    restored = tmp_path / "restored.db"
    restore_snapshot(snap, tmp_path / "restored_chroma", restored)
    g = GraphMemory(str(restored), clock=clock)
    assert g.get_node("ent_obscure_topic") is not None
    g.close()


def test_touched_fading_node_survives(cog, tmp_path, clock):
    seed(cog, {**WEDDING, "Rare topic": ["qwvv lighthouse keeper fact", "qwvv lighthouse lamp"]})
    clock.advance(days=45)
    for nid in ("ent_henna", "ent_saree", "ent_poruwa", "ent_invoices"):
        cog.usage.touch(nid)
    coordinator(cog, tmp_path).run_cycle()
    assert cog.graph.get_node("ent_rare_topic")["status"] == "fading"
    cog.usage.touch("ent_rare_topic")                                     # the user asks about it again
    assert cog.graph.get_node("ent_rare_topic")["status"] == "active"
    clock.advance(days=10)
    report = coordinator(cog, tmp_path).run_cycle()
    assert "ent_rare_topic" not in report["sleep"]["pruned"]
    assert cog.graph.get_node("ent_rare_topic") is not None


def test_pinned_and_bridge_nodes_are_never_pruned(cog, tmp_path, clock):
    # no shared words, so no abstraction can connect them: Hub stays the only link
    seed(cog, {"Pinned fact": ["ppqq zzaa"], "Hub": ["hhjj yybb"], "Left": ["llkk xxcc"], "Right": ["rrmm wwdd"]})
    cog.graph.set_node_fields("ent_pinned_fact", pinned=True)
    cog.knowledge.link("ent_left", "ent_hub", "related_to")
    cog.knowledge.link("ent_hub", "ent_right", "related_to")
    for _ in range(3):
        clock.advance(days=40)
        for nid in ("ent_left", "ent_right"):
            cog.usage.touch(nid)
        coordinator(cog, tmp_path).run_cycle()
    assert cog.graph.get_node("ent_pinned_fact") is not None
    assert cog.graph.get_node("ent_hub") is not None                     # only link between Left and Right


# ---------------------------------------------------------------- gaps + research

def test_gap_scan_queues_and_background_research_fills(cog, tmp_path, events):
    seed(cog, {"Saree": ["A saree is a long drape of cloth"]})             # 1 record, no edges: weak
    report = coordinator(cog, tmp_path).run_cycle()
    assert "ent_saree" in report["sleep"]["gaps_flagged"]
    researched = report["sleep"]["researched"]
    assert researched and researched[0]["topic"] == "Saree" and researched[0]["records"] > 0
    assert cog.vector.count("k_saree") > 1                                 # gap filled in the same collection
    assert "k_saree" not in report["sleep"]["collections_created"]
    done = cog.curiosity.list(status="done")
    assert done and done[0]["topic_node_id"] == "ent_saree"


def test_shadow_mode_reports_but_changes_nothing(cog, tmp_path, clock):
    seed(cog, WEDDING)
    seed(cog, {"Obscure topic": ["zyxx quantum barnacle fact"]})
    clock.advance(days=45)
    for nid in ("ent_henna", "ent_saree", "ent_poruwa", "ent_invoices"):
        cog.usage.touch(nid)
    before_nodes = {n["id"] for n in cog.graph.all_nodes()}
    before_colls = set(cog.vector.list_knowledge_namespaces())
    report = coordinator(cog, tmp_path, mode="shadow").run_cycle()
    s = report["sleep"]
    assert s["dry_run"] and "ent_obscure_topic" in s["faded"] and s["abstractions"]
    assert {n["id"] for n in cog.graph.all_nodes()} == before_nodes
    assert set(cog.vector.list_knowledge_namespaces()) == before_colls
    assert cog.graph.get_node("ent_obscure_topic")["status"] == "active"
    assert cog.curiosity.list() == []


def test_new_message_pauses_the_cycle(cog, tmp_path, events):
    seed(cog, WEDDING)
    calls = {"n": 0}

    def should_pause():
        calls["n"] += 1
        return calls["n"] > 1          # user returns after the first phase started

    report = coordinator(cog, tmp_path, should_pause=should_pause).run_cycle()
    assert report["sleep"]["paused"] is True
    assert phases(events) == ["usage_audit"]
    assert of_type(events, "sleep_end")


# ---------------------------------------------------------------- idle trigger + snapshots

def test_idle_tracker_due_and_pause():
    t = {"now": 0.0}
    tr = IdleTracker(monotonic=lambda: t["now"])
    assert not tr.due("u")                       # never active: nothing to do
    tr.mark_active("u")
    t["now"] += 60 * 19
    assert not tr.due("u")
    t["now"] += 60 * 2
    assert tr.due("u")
    seen = {}

    def runner(uid, reason, mode, should_pause):
        seen["before"] = should_pause()
        t["now"] += 1
        tr.mark_active(uid)            # a message arrives mid-sleep
        seen["after"] = should_pause()
        return "ok"

    t["now"] += 1
    assert tr.run_sleep("u", runner) == "ok"
    assert seen == {"before": False, "after": True}
    t["now"] += 60 * 25
    assert not tr.due("u")                       # slept less than min_hours_between ago
    t["now"] += 6 * 3600
    assert tr.due("u")


def test_per_user_snapshot_location(cog, tmp_path):
    seed(cog, {"Henna": WEDDING["Henna"]})
    mgr = snapshot_manager_for(cog.vector, cog.graph, clock=cog._clock)
    assert mgr.snapshot_dir == Path(cog.graph.db_path).parent / "snapshots"
    assert mgr.max_snapshots == 7
    path = mgr.take_snapshot()
    assert (path / "peter_graph.db").exists() and (path / "chroma").exists()


def test_zero_timestamps_are_not_treated_as_missing(cog, tmp_path, clock):
    """Regression: fading_since=0 / last_accessed=0 are real (very old) times, not 'unset'."""
    seed(cog, {"Ancient": ["vvzz ancient record"], **WEDDING})
    for nid in ("ent_henna", "ent_saree", "ent_poruwa", "ent_invoices"):
        cog.usage.touch(nid)
    cog.graph.set_node_fields("ent_ancient", status="fading", fading_since=0.0, last_accessed=0.0)
    report = coordinator(cog, tmp_path).run_cycle()
    assert "ent_ancient" in report["sleep"]["pruned"]


def test_bare_topic_is_searched_once_per_cycle(cog, tmp_path):
    """Each question gets its own search, but the bare topic name is searched only once per cycle."""
    set_cognition_config({**TEST_CFG, "sleep": {"research_items_per_cycle": 3}})
    for q in ("How is kandyan saree draped?", "Who wears the kandyan saree?", "Kandyan saree history?"):
        cog.curiosity.add(q, None, 1.0, gap=True, topic="saree")
    report = coordinator(cog, tmp_path).run_cycle()
    assert [r["topic"] for r in report["sleep"]["researched"]] == ["saree", "saree", "saree"]
    assert sum(1 for s in cog.web.searches if s == "saree") == 1
    assert sum(1 for s in cog.web.searches if s.endswith("?")) == 3
