"""Regression tests for the audit findings (one test or more per item).

Each test reproduces a scenario from the auditor's scripts and asserts the fixed
behaviour. Deterministic and offline (WordHashEmbedder, FakeClock, fake web/LLM).
"""
from __future__ import annotations

import json

import pytest

from core.cognition_config import set_cognition_config
from memory.cognition import CognitiveMemory
from memory.vector_store import CollectionBudget
from tests.cognitive.conftest import DAY, of_type

OSCAR = {
    "https://en.wikipedia.org/wiki/Oscar_Wilde":
        "Oscar Wilde was an Irish poet and playwright. Oscar Wilde wrote The Picture of Dorian Gray in 1890.",
    "https://www.britannica.com/biography/Oscar-Wilde":
        "Oscar Wilde was born in Dublin in 1854. Oscar Wilde died in Paris in 1900.",
}


def _oscar_search(q, n):
    return [{"url": u, "title": "Oscar Wilde", "snippet": "Oscar Wilde"} for u in OSCAR]


def _cog(stores, router=None, **cfg):
    set_cognition_config({"resolver": {"use_llm": False}, **cfg})
    vector, graph = stores
    return CognitiveMemory(vector, graph, router=router, async_research=False)


# ------------------------------------------------------------------ 1. budget fallback

def test_budget_exhausted_research_defers_instead_of_mislabelling(stores):
    from engine.research_pipeline import ResearchPipeline
    cog = _cog(stores)
    hen, _ = cog.knowledge.ensure_node("Henna")
    cog.knowledge.add_record(hen, "Henna is a plant dye used for body art.")
    rp = ResearchPipeline(cog.knowledge, router=None, search_fn=_oscar_search, fetch_fn=lambda u: OSCAR[u])
    out = rp.research("Oscar Wilde", budget=CollectionBudget(max_new=0))
    assert out.deferred is True and out.node_id is None and out.records_stored == 0
    docs = [r["document"] for r in cog.vector.get_all_records(hen["collection"])]
    assert docs == ["Henna is a plant dye used for body art."]          # nothing filed under Henna


def test_budget_exhausted_research_still_uses_the_same_topic(stores):
    from engine.research_pipeline import ResearchPipeline
    cog = _cog(stores)
    ow, _ = cog.knowledge.ensure_node("Oscar Wilde")
    rp = ResearchPipeline(cog.knowledge, router=None, search_fn=_oscar_search, fetch_fn=lambda u: OSCAR[u])
    out = rp.research("Oscar Wilde", budget=CollectionBudget(max_new=0))
    assert out.node_id == ow["id"] and out.records_stored > 0 and not out.deferred


def test_deferred_curiosity_item_stays_pending_without_spending_an_attempt(stores):
    from engine.curiosity import CuriosityEngine
    from engine.research_pipeline import ResearchPipeline
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores, sleep={"max_new_collections": 0})
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver)
    cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=_oscar_search,
                                    fetch_fn=lambda u: OSCAR[u])
    item = cog.curiosity.add("Who was Oscar Wilde?", None, 0.9, topic="Oscar Wilde")
    s = SleepCycle(cog)
    s.research()
    it = cog.curiosity.get(item["id"])
    assert it["status"] == "pending" and it["attempts"] == 0
    assert s.report.researched[0]["deferred"] is True


# ------------------------------------------------------------------ 2. per-user memory tools

def _user_stores(tmp_path, clock, user):
    from memory.graph_store import GraphMemory
    from memory.vector_store import VectorMemory
    from tests.fakes.word_embedder import WordHashEmbedder
    v = VectorMemory(str(tmp_path / user / "chroma"), clock=clock, user_id=user, embedding_function=WordHashEmbedder())
    g = GraphMemory(str(tmp_path / user / "graph.db"), clock=clock, user_id=user)
    return v, g


def test_memory_tools_are_bound_per_user(tmp_path, clock):
    from tools.memory_tools import init_memory_tools, register_memory_tools
    from tools.registry import ToolRegistry
    import tools.memory_tools as mt
    av, ag = _user_stores(tmp_path, clock, "alice")
    bv, bg = _user_stores(tmp_path, clock, "bob")
    alice_tools = init_memory_tools(av, ag)       # bootstrap_for_user(alice)
    init_memory_tools(bv, bg)                     # bootstrap_for_user(bob) later
    assert alice_tools.memory_store("peter", "Alice's bank PIN hint: blue").success
    assert av.count("peter") == 1 and bv.count("peter") == 0
    assert not [n for n in vars(mt) if n.startswith("_vector") or n.startswith("_graph")]   # no globals
    reg = ToolRegistry(guardrails=None)
    register_memory_tools(reg, alice_tools)
    assert reg.tools["memory_store"].__self__ is alice_tools          # the registry holds Alice's tools


def test_memory_store_into_a_topic_collection_goes_through_the_node(stores):
    from tools.memory_tools import init_memory_tools
    cog = _cog(stores)
    tools = init_memory_tools(cog.vector, cog.graph, knowledge=cog.knowledge)
    n, _ = cog.knowledge.ensure_node("Henna")
    assert tools.memory_store(n["collection"], "Henna is used at weddings.").success
    rec = cog.vector.get_all_records(n["collection"])[0]
    assert rec["metadata"]["node_id"] == n["id"]
    assert cog.graph.get_collection(n["collection"])["record_count"] == 1
    assert not tools.memory_store("k_no_such_topic", "x").success
    assert not tools.memory_store("index_entities", "x").success


# ------------------------------------------------------------------ 3. hubs and subjects are used with their topics

def _ctx(cog, label, category, subject, facts=()):
    from memory.entity_resolver import Match, Resolution
    from memory.graph_recall import GraphRecallResult
    node, _ = cog.knowledge.ensure_node(label)
    for f in facts:
        cog.knowledge.add_record(node, f)
    cog.knowledge.refresh_stats(node)
    res = Resolution(terms=[label], matched=[Match(label, node, "alias", 1.0)], category=category, subject=subject)
    cog.link_context(GraphRecallResult(query=label, resolution=res), node["id"])
    return cog.graph.get_node(node["id"])


def test_hub_and_subject_survive_while_their_topic_is_used(stores, clock):
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    hen = _ctx(cog, "Henna", "Body art", "Culture", ["Henna is a dye from the henna plant used for body art.",
                                                    "Henna paste stains skin orange to brown for weeks."])
    _ctx(cog, "Mehndi", "", "Culture", ["Mehndi is the application of henna paste in weddings."])
    for _ in range(60):
        clock.advance(DAY)
        cog.graph_recall.recall("henna dye")
    hub = cog.knowledge.find_node("Body art")
    culture = cog.knowledge.find_node("Culture")
    scored = cog.usage.recompute_all(persist=False)
    hen_score = scored[hen["id"]]["score"]
    # usage rolls up the route (score propagation; visiting a hub is not counted as touching it)
    assert scored[hub["id"]]["score"] == pytest.approx(0.9 * hen_score)
    assert scored[culture["id"]]["score"] == pytest.approx(0.81 * hen_score)
    assert scored[hub["id"]]["percentile"] > scored[cog.knowledge.find_node("Mehndi")["id"]]["percentile"]
    for _ in range(3):                                                      # several audit/prune cycles
        SleepCycle(cog).audit()
        clock.advance(8 * DAY)
        cog.graph_recall.recall("henna dye")
        SleepCycle(cog).prune()
    assert cog.graph.get_node(hub["id"]) and cog.graph.get_node(culture["id"])
    assert cog.hierarchy.route_to(hen["id"])[0] == [culture["id"], hub["id"], hen["id"]]


def test_subjects_and_hubs_with_live_children_are_never_prune_candidates(stores, clock):
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    _ctx(cog, "Saree", "Clothing", "Culture", ["A saree is a draped garment worn in South Asia."])
    clock.advance(400 * DAY)                                              # nothing touched for a long time
    cands, _ = SleepCycle(cog)._prune_candidates()
    ids = {n["id"] for n in cands}
    assert cog.knowledge.find_node("Culture")["id"] not in ids
    assert cog.knowledge.find_node("Clothing")["id"] not in ids          # still abstracts Saree
    assert cog.knowledge.find_node("Saree")["id"] in ids                  # the leaf itself may fade


# ------------------------------------------------------------------ 18. upsert_node keeps what it is not given

def test_raw_upsert_does_not_clobber_role_pin_or_importance(stores):
    _, graph = stores
    graph.upsert_node("abs_investing", kind="abstraction", label="Investing", data={"role": "subject"})
    graph.set_node_fields("abs_investing", pinned=True, importance=0.9)
    graph.upsert_node("abs_investing", kind="abstraction", label="Investing")     # what any raw upsert does
    n = graph.get_node("abs_investing")
    assert n["data"] == {"role": "subject"} and n["pinned"] is True and n["importance"] == pytest.approx(0.9)
    graph.upsert_node("abs_investing", kind="abstraction", label="Investing", data={"role": "x"},
                      importance=0.2, pinned=False)                               # explicit values still apply
    n = graph.get_node("abs_investing")
    assert n["data"] == {"role": "x"} and n["pinned"] is False and n["importance"] == pytest.approx(0.2)
    graph.upsert_node("fresh", kind="entity", label="Fresh")
    f = graph.get_node("fresh")
    assert f["data"] == {} and f["importance"] == pytest.approx(0.5) and f["pinned"] is False


# ------------------------------------------------------------------ 4. unique, stable ids and collections

def test_symbols_and_truncation_never_collide(stores):
    cog = _cog(stores)
    k = cog.knowledge
    ids = {}
    for lab in ["C", "C++", "C#"]:
        n, created = k.ensure_node(lab)
        assert created, lab
        ids[lab] = n
    assert len({n["id"] for n in ids.values()}) == 3 and len({n["collection"] for n in ids.values()}) == 3
    assert ids["C"]["id"] == "ent_c" and ids["C"]["collection"] == "k_c"        # plain labels unchanged
    for lab, n in ids.items():
        assert k.find_node(lab)["id"] == n["id"]
        assert cog.resolver.resolve_term(lab).node["id"] == n["id"]
    l1 = "International Monetary Fund lending programmes in Asia"
    l2 = "International Monetary Fund lending programmes in Africa"
    n1, _ = k.ensure_node(l1)
    n2, c2 = k.ensure_node(l2)
    assert c2 and n1["id"] != n2["id"] and n1["collection"] != n2["collection"]
    assert k.node_id_for(l1) == n1["id"]                                       # stable: label -> id
    # case / punctuation variants are still one node
    a, _ = k.ensure_node("Investing", kind="abstraction")
    assert k.ensure_node("investing")[0]["id"] == a["id"] == k.ensure_node("INVESTING.")[0]["id"]


def test_old_ids_keep_working_and_new_ids_avoid_them(stores):
    from memory.upgrade import run_upgrades
    vector, graph = stores
    # an old graph: 'C++' was stored as ent_c / k_c with the symbol-stripped alias 'c'
    graph.upsert_node("ent_c", kind="entity", label="C++")
    graph.set_node_fields("ent_c", collection="k_c", status="active")
    graph.register_collection("k_c", "ent_c", "entity")
    vector.create_collection("k_c")
    with graph._get_connection() as conn:
        conn.execute("INSERT INTO node_aliases (alias_norm, node_id) VALUES ('c', 'ent_c')")
        conn.execute("DELETE FROM schema_version WHERE version >= 4")
        conn.commit()
    cog = _cog(stores)                                     # start-up runs the upgrade
    k = cog.knowledge
    assert k.find_node("C++")["id"] == "ent_c"             # old node found by its label
    assert k.find_node("C") is None                        # 'c' no longer means C++
    c, created = k.ensure_node("C")
    assert created and c["id"] != "ent_c" and c["collection"] != "k_c"
    assert graph.get_node("ent_c")["label"] == "C++"      # old node untouched
    run_upgrades(cog)                                      # idempotent
    assert k.find_node("C")["id"] == c["id"]


# ------------------------------------------------------------------ 6. hierarchy: no cycles, no subject under anything

def _abstracts(graph):
    return [(e["src"], e["dst"], bool((e.get("data") or {}).get("primary")))
            for e in graph.list_edges() if e["relation"] == "abstracts"]


def test_subjects_never_nest_and_abstracts_has_no_cycle(stores):
    from memory.hierarchy import is_subject
    cog = _cog(stores)
    g = cog.graph
    _ctx(cog, "Colombo Stock Exchange", "Finance", "Investing")
    _ctx(cog, "Unit trusts", "Investing", "Finance")                  # r1: the reverse nesting
    fin, inv = cog.knowledge.find_node("Finance"), cog.knowledge.find_node("Investing")
    assert is_subject(fin) and is_subject(inv)
    pairs = {(s, d) for s, d, _ in _abstracts(g)}
    assert (inv["id"], fin["id"]) not in pairs and (fin["id"], inv["id"]) not in pairs
    for s, d, _ in _abstracts(g):
        assert not is_subject(g.get_node(d))
    cse = cog.knowledge.find_node("Colombo Stock Exchange")
    assert cog.hierarchy.route_to(cse["id"])[0] == [fin["id"], cse["id"]]
    # a cycle is refused at link time
    a, _ = cog.knowledge.ensure_node("Hub A", kind="abstraction")
    b, _ = cog.knowledge.ensure_node("Hub B", kind="abstraction")
    assert cog.knowledge.link(a["id"], b["id"], "abstracts")[1]
    eid, created, why = cog.knowledge.link_checked(b["id"], a["id"], "abstracts")
    assert eid is None and "cycle" in why


def test_multiple_membership_has_one_primary_parent(stores):
    cog = _cog(stores)
    g = cog.graph
    _ctx(cog, "Bitcoin", "Cryptocurrency", "Technology")
    _ctx(cog, "Bitcoin", "Cryptocurrency", "Investing")
    crypto = cog.knowledge.find_node("Cryptocurrency")
    ups = [(s, p) for s, d, p in _abstracts(g) if d == crypto["id"]]
    assert len(ups) == 2 and sum(p for _, p in ups) == 1                # graph membership, one primary
    tech = cog.knowledge.find_node("Technology")
    assert dict(ups)[tech["id"]] is True                                 # the first membership
    btc = cog.knowledge.find_node("Bitcoin")
    assert cog.hierarchy.route_to(btc["id"])[0] == [tech["id"], crypto["id"], btc["id"]]
    from memory.tree_layout import build_forest
    nodes = g.all_nodes(["entity", "abstraction"])
    _, children = build_forest(nodes, g.list_edges([n["id"] for n in nodes]))
    assert crypto["id"] in children[tech["id"]]                          # layout follows the primary too
    # removing the primary hub promotes the other membership
    cog.knowledge.remove_node(tech["id"])
    ups = [(s, p) for s, d, p in _abstracts(g) if d == crypto["id"]]
    assert ups == [(cog.knowledge.find_node("Investing")["id"], True)]


def test_repair_fixes_an_old_inconsistent_graph(stores, clock):
    cog = _cog(stores)
    g = cog.graph
    for i, lab in enumerate(["X", "Y", "Z"]):
        g.upsert_node(f"abs_{lab.lower()}", kind="abstraction", label=lab,
                      data={"role": "subject"} if lab == "Z" else {})
    g.link("abs_x", "abs_y", "abstracts")
    clock.advance(1)
    g.link("abs_y", "abs_x", "abstracts")      # cycle (the newer edge closes it)
    clock.advance(1)
    g.link("abs_x", "abs_z", "abstracts")      # subject under something
    assert cog.hierarchy.repair() >= 2
    pairs = [(s, d) for s, d, _ in _abstracts(g)]
    assert pairs == [("abs_x", "abs_y")] and _abstracts(g)[0][2] is True


def test_repair_removes_old_hub_member_symmetric_edges(stores):
    """D4: an old graph where a hub both abstracts a member AND has a related_to/bridges to it
    (which the current link_checked would refuse) is cleaned up by repair (startup + sleep)."""
    cog = _cog(stores)
    g = cog.graph
    hub, _ = cog.knowledge.ensure_node("Body art", kind="abstraction")
    hen, _ = cog.knowledge.ensure_node("Henna")
    g.link(hub["id"], hen["id"], "abstracts")
    g.link(hub["id"], hen["id"], "related_to")      # hand-made, both directions of the rule
    g.link(hen["id"], hub["id"], "bridges")
    assert any(e["relation"] == "related_to" for e in g.edges_of(hen["id"]))
    cog.hierarchy.repair()
    rels = {e["relation"] for e in g.edges_of(hen["id"])}
    assert "related_to" not in rels and "bridges" not in rels
    assert g.find_edge(hub["id"], hen["id"], "abstracts")                # the part-of link stays


# ------------------------------------------------------------------ 17. no related_to between a hub and its member

def test_no_related_to_between_an_abstraction_and_its_member(stores):
    cog = _cog(stores)
    k = cog.knowledge
    hen, _ = k.ensure_node("Henna")
    k.add_record(hen, "Henna paste is applied at wedding ceremonies.")
    hub, _ = k.ensure_node("Henna traditions", kind="abstraction")
    k.add_record(hub, "Henna traditions include wedding paste ceremonies.")
    k.link(hub["id"], hen["id"], "abstracts")
    cog.recall("Henna and Henna traditions")
    rel = [e for e in cog.graph.list_edges() if e["relation"] == "related_to"]
    assert rel == []
    a, created = k.link(hen["id"], hub["id"], "related_to")
    assert a is None and not created
    # symmetric links are stored once per pair
    sar, _ = k.ensure_node("Saree")
    e1, c1 = k.link(hen["id"], sar["id"], "related_to")
    e2, c2 = k.link(sar["id"], hen["id"], "related_to")
    assert c1 and not c2 and e1 == e2


# ------------------------------------------------------------------ 20. graph_link tool validates

def test_graph_link_tool_validates_dedupes_and_emits(stores, events):
    from memory.hierarchy import SELF_ID
    from tools.memory_tools import init_memory_tools
    cog = _cog(stores)
    tools = init_memory_tools(cog.vector, cog.graph, knowledge=cog.knowledge)
    a, _ = cog.knowledge.ensure_node("Hub A", kind="abstraction")
    b, _ = cog.knowledge.ensure_node("Topic B")
    assert not tools.graph_link(a["id"], b["id"], "bogus_relation").success
    assert not tools.graph_link(SELF_ID, b["id"], "has_subject").success          # structure is not hand-made
    assert not tools.graph_link(a["id"], "missing", "abstracts").success
    r = tools.graph_link(a["id"], b["id"], "abstracts")
    assert r.success and of_type(events, "edge_drawn")[-1]["src"] == a["id"]
    n_events = len(of_type(events, "edge_drawn"))
    assert tools.graph_link(a["id"], b["id"], "abstracts").success                # dedupe: no second edge
    assert len(of_type(events, "edge_drawn")) == n_events
    assert sum(e["relation"] == "abstracts" for e in cog.graph.list_edges()) == 1
    r = tools.graph_link(b["id"], a["id"], "abstracts")                           # cycle
    assert not r.success and "cycle" in r.error


# ------------------------------------------------------------------ 7. curiosity nodes follow their topic

def test_removing_a_topic_removes_its_question_nodes_and_orphans_stay_out_of_subjects(stores):
    from engine.curiosity import CuriosityEngine
    from tests.fakes.fake_llm import FakeLLMRouter
    router = FakeLLMRouter(loop=True, script=[{"content": json.dumps({"subject": "Culture"})}])
    cog = _cog(stores, router=router)
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver)
    g = cog.graph
    saree, _ = cog.knowledge.ensure_node("Saree")
    cog.knowledge.add_record(saree, "A saree is a draped garment worn in South Asia.")
    item = cog.curiosity.add("How is a saree draped?", saree["id"], 0.5, topic="Saree")
    gap = cog.curiosity.add("Who is Basil Hallward?", None, 0.5, topic="Basil Hallward", gap=True)
    cog.knowledge.remove_node(saree["id"])
    assert g.get_node(f"cur_{item['id']}") is None and cog.curiosity.get(item["id"]) is None
    assert g.get_node(f"cur_{gap['id']}") is not None                 # a pending gap question stays
    cog.attach_orphans()
    for e in g.list_edges():
        if e["relation"] == "abstracts":
            assert g.get_node(e["dst"])["kind"] in ("entity", "abstraction")
    assert not g.edges_of(f"cur_{gap['id']}")


def test_sleep_prune_removes_question_nodes_of_finished_items(stores):
    from engine.curiosity import CuriosityEngine
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver)
    it = cog.curiosity.add("What is ikat weaving?", None, 0.5, topic="Ikat")
    cog.curiosity._set(it["id"], status="done")                        # an old graph left its node behind
    SleepCycle(cog).prune()
    assert cog.graph.get_node(f"cur_{it['id']}") is None


# ------------------------------------------------------------------ 5. migrated copies; 12. registry counts

def _henna_mehndi(cog):
    from engine.sleep_cycle import SleepCycle
    k = cog.knowledge
    a, _ = k.ensure_node("Henna")
    b, _ = k.ensure_node("Mehndi")
    k.add_record(a, "Henna paste is applied at wedding ceremonies to decorate hands.")
    k.add_record(b, "Mehndi paste is applied at wedding ceremonies on the bride's hands.")
    k.add_record(b, "Mehndi night is a pre-wedding celebration with music.")
    for n in (a, b):
        k.refresh_stats(n)
    s = SleepCycle(cog)
    s.abstract()
    absn = cog.graph.get_node(s.report.abstractions[0]["node_id"])
    return a, b, absn


def test_copies_are_not_double_counted_in_recall(stores):
    cog = _cog(stores)
    a, b, absn = _henna_mehndi(cog)
    assert cog.vector.count(absn["collection"]) == 3
    res, verdict = cog.recall("henna paste wedding")
    docs = [h["document"] for h in res.hits]
    assert len(docs) == len(set(docs))                                 # each fact once
    origins = [(h.get("metadata") or {}).get("origin_id") or h["id"] for h in res.hits]
    assert len(origins) == len(set(origins))
    second = __import__("engine.sleep_cycle", fromlist=["SleepCycle"]).SleepCycle(cog)
    second.abstract()
    assert cog.vector.count(absn["collection"]) == 3                   # never re-copied


def test_copies_follow_their_originals(stores):
    cog = _cog(stores)
    a, b, absn = _henna_mehndi(cog)
    v, g = cog.vector, cog.graph
    rid = v.get_all_records(a["collection"])[0]["id"]
    v.prune(a["collection"], [rid])                                    # any delete path (tools, eviction…)
    assert v.count(a["collection"]) == 0
    left = [r["metadata"]["origin_node"] for r in v.get_all_records(absn["collection"])]
    assert a["id"] not in left and len(left) == 2
    assert g.get_collection(a["collection"])["record_count"] == 0     # 12: registry follows
    cog.knowledge.remove_node(b["id"])
    assert v.count(absn["collection"]) == 0                            # node gone -> its copies gone
    assert g.get_collection(absn["collection"])["record_count"] == 0


def test_sleep_refreshes_stale_copies_and_counts(stores):
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    a, b, absn = _henna_mehndi(cog)
    v, g = cog.vector, cog.graph
    # an original edited behind the copy's back, and a hub that no longer abstracts a member
    rec = v.get_all_records(a["collection"])[0]
    v._get_or_create_collection(a["collection"]).update(ids=[rec["id"]], documents=["Henna paste is used at weddings."])
    e = g.find_edge(absn["id"], b["id"], "abstracts")
    g.delete_edge(e)
    g.update_collection_stats(a["collection"], record_count=42)      # drifted registry
    s = SleepCycle(cog)
    s.maintain()
    docs = sorted(r["document"] for r in v.get_all_records(absn["collection"]))
    assert docs == ["Henna paste is used at weddings."]
    assert s.report.copies_updated == 1 and s.report.copies_removed == 2
    assert g.get_collection(a["collection"])["record_count"] == 1 and s.report.counts_fixed >= 1


# ------------------------------------------------------------------ 9. subjects made in sleep count against the budget

def test_sleep_subjects_respect_the_collection_budget(stores):
    from engine.curiosity import CuriosityEngine
    from engine.research_pipeline import ResearchPipeline
    from engine.sleep_cycle import SleepCycle
    from tests.fakes.fake_llm import FakeLLMRouter
    router = FakeLLMRouter(loop=True, script=[{"content": json.dumps({"subject": "Culture"})}])
    cog = _cog(stores, router=router, sleep={"max_new_collections": 0}, resolver={"use_llm": True})
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver)
    cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=lambda q, n: [], fetch_fn=lambda u: "")
    assert cog.resolver.subject_for("Saree", []) == "Culture"
    cog.knowledge.ensure_node("Saree")
    s = SleepCycle(cog)
    s.research()
    assert cog.knowledge.find_node("Culture") is None                 # budget 0: no new subject collection
    assert not s.report.errors
    from core.cognition_config import cognition_config
    s2 = SleepCycle(cog, cfg={"sleep": {**cognition_config()["sleep"], "max_new_collections": 1}})
    s2.research()
    assert cog.knowledge.find_node("Culture") is not None
    assert s2.report.collections_created == [cog.knowledge.find_node("Culture")["collection"]]


# ------------------------------------------------------------------ 10. re-research does not duplicate facts

def test_researching_a_topic_twice_stores_each_fact_once(stores):
    from engine.research_pipeline import ResearchPipeline
    cog = _cog(stores)
    rp = ResearchPipeline(cog.knowledge, router=None, search_fn=_oscar_search, fetch_fn=lambda u: OSCAR[u])
    o1 = rp.research("Oscar Wilde")
    o2 = rp.research("Oscar Wilde")
    n = cog.graph.get_node(o1.node_id)
    docs = [r["document"] for r in cog.vector.get_all_records(n["collection"])]
    assert len(docs) == len(set(docs)) == o1.records_stored and o2.records_stored == 0
    assert cog.graph.get_collection(n["collection"])["record_count"] == len(docs)


# ------------------------------------------------------------------ 13. random-hash embedding fallback is loud

def test_hash_fallback_is_flagged_and_reported(stores, tmp_path, clock, caplog):
    import logging
    from memory import embeddings
    from memory.graph_store import GraphMemory
    from memory.mind_view import graph_snapshot
    from memory.vector_store import VectorMemory
    cog = _cog(stores)                                         # WordHashEmbedder: a real (test) embedder
    assert cog.semantic_ok and graph_snapshot(cog)["cognition"]["semantic_ok"] is True
    saved = embeddings._model_instance
    embeddings._model_instance = None
    try:
        with caplog.at_level(logging.ERROR, logger="peter"):
            fn = embeddings.LocalEmbeddingFunction("definitely-not-a-model-xyz")
        if getattr(fn.model, "semantic", True) is not False:
            pytest.skip("a real sentence-transformers model is installed")
        assert any(r.levelno >= logging.ERROR and "HASH embedder" in r.getMessage() for r in caplog.records)
        v = VectorMemory(str(tmp_path / "fb"), clock=clock, user_id="fb", embedding_function=fn)
        g = GraphMemory(str(tmp_path / "fb.db"), clock=clock, user_id="fb")
        cog2 = CognitiveMemory(v, g, router=None, async_research=False)
        assert v.semantic_ok is False and cog2.semantic_ok is False
        snap = graph_snapshot(cog2)
        assert snap["cognition"]["semantic_ok"] is False and snap["stats"]["semantic_ok"] is False
    finally:
        embeddings._model_instance = saved


# ------------------------------------------------------------------ 14. aliases are indexed; near-duplicates merge

def test_aliases_are_found_by_meaning(stores):
    cog = _cog(stores)
    n, _ = cog.knowledge.ensure_node("Colombo Stock Exchange", aliases=["CSE equities market"])
    hits = cog.vector.search_entities("equities market", n_results=3)
    assert hits and hits[0]["node_id"] == n["id"] and hits[0]["via_alias"]
    assert len({h["node_id"] for h in hits}) == len(hits)              # one result per node
    m = cog.resolver.resolve_term("equities market")
    assert m is not None and m.node["id"] == n["id"] and m.via == "index"
    cog.knowledge.remove_node(n["id"])
    assert cog.vector.search_entities("equities market", n_results=3) == []   # alias vectors go too


def test_sleep_merges_near_duplicate_topics(stores, events):
    from engine.curiosity import CuriosityEngine
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver)
    k, g = cog.knowledge, cog.graph
    a, _ = k.ensure_node("Colombo Stock Exchange")
    b, _ = k.ensure_node("Stock Exchange of Colombo")
    sar, _ = k.ensure_node("Saree")
    for t in ("The Colombo Stock Exchange lists about 290 companies.",
              "The Colombo Stock Exchange indices are ASPI and S&P SL20."):
        k.add_record(a, t)
    k.add_record(b, "The Colombo Stock Exchange lists about 290 companies.")     # same fact
    k.add_record(b, "The Colombo Stock Exchange indices include ASPI.")
    k.add_record(sar, "A saree is a draped garment.")
    k.link(b["id"], sar["id"], "related_to")
    it = cog.curiosity.add("Who regulates the Colombo stock exchange?", b["id"], 0.5)
    s = SleepCycle(cog)
    s.merge_duplicates()
    assert s.report.merged and s.report.merged[0]["keeper"] == a["id"] and s.report.merged[0]["merged"] == b["id"]
    assert g.get_node(b["id"]) is None
    docs = sorted(r["document"] for r in cog.vector.get_all_records(a["collection"]))
    assert "The Colombo Stock Exchange indices include ASPI." in docs    # moved
    assert len(docs) == len(set(docs)) == 3                              # the shared fact once
    assert g.find_edge(a["id"], sar["id"], "related_to")                # edges re-pointed
    assert k.find_node("Stock Exchange of Colombo")["id"] == a["id"]    # old name is an alias now
    assert cog.curiosity.get(it["id"])["topic_node_id"] == a["id"]
    assert g.get_node(f"cur_{it['id']}") and g.find_edge(f"cur_{it['id']}", a["id"], "curious_about")
    assert of_type(events, "nodes_merged")
    assert g.get_node(sar["id"])                                         # unrelated topics untouched


def test_merge_never_folds_symbol_or_empty_topics(stores):
    """D1/D8: C / C++ / C# / 'C ++' are distinct languages, and an empty topic is never
    merged on label alone — the merge needs real records (centroids) on both sides."""
    from engine.sleep_cycle import SleepCycle, _symbol_compatible
    cog = _cog(stores)
    k, g = cog.knowledge, cog.graph
    c, _ = k.ensure_node("C")
    for t in ("C is a procedural language created by Dennis Ritchie at Bell Labs.",
              "C uses manual memory management with malloc and free."):
        k.add_record(c, t)
    cpp, _ = k.ensure_node("C++")
    for t in ("C++ adds classes and templates, designed by Bjarne Stroustrup.",
              "C++ supports RAII and the standard template library."):
        k.add_record(cpp, t)
    cs, _ = k.ensure_node("C#")           # empty on purpose
    cspace, _ = k.ensure_node("C ++")     # detached symbols: still distinct from C (D8)
    for n in (c, cpp):
        k.refresh_stats(n)
    assert len({c["id"], cpp["id"], cs["id"], cspace["id"]}) == 4       # four distinct nodes/ids
    assert not _symbol_compatible("C", "C++") and not _symbol_compatible("C", "C ++")
    assert not _symbol_compatible("C++", "C#") and not _symbol_compatible("Python 2", "Python 3")
    assert _symbol_compatible("Colombo Stock Exchange", "Colombo Stock Exchanges")
    s = SleepCycle(cog)
    s.merge_duplicates()
    assert s.report.merged == []                                        # nothing folded
    assert all(g.get_node(x) is not None for x in (c["id"], cpp["id"], cs["id"], cspace["id"]))


# ------------------------------------------------------------------ 15. evals never fake a pass

def test_eval_runner_reports_not_run_instead_of_faking(tmp_path, monkeypatch):
    import evals.runner as runner
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    s = runner.run_evals("fake")
    by_id = {r["task_id"]: r for r in s["task_results"]}
    assert all(by_id[i]["status"] == "success" and by_id[i]["passed"] for i in (4, 5, 8, 10))   # really ran
    not_run = [r for r in s["task_results"] if r["status"] == "not_run"]
    assert not_run and all(r["passed"] is False and r["reason"].startswith("NOT RUN") for r in not_run)
    assert s["all_passed"] is False and s["not_run_tasks"] == len(not_run)
    live = runner.run_evals("live")
    assert live["passed_tasks"] == 0 and live["not_run_tasks"] == 10
    assert json.loads((tmp_path / "evals_fake.json").read_text())["not_run_tasks"] == len(not_run)
