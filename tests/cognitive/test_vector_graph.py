"""Parts B-D: every node is a filled vector store, edges are grounded in the vectors,
and the mind view draws the stored vector graph. Deterministic and offline."""
from __future__ import annotations

import json
import math

import pytest

from core.cognition_config import set_cognition_config
from memory.cognition import CognitiveMemory
from memory.hierarchy import SELF_ID
from tests.cognitive.conftest import DAY, of_type


def _cog(stores, router=None, **cfg):
    set_cognition_config({"resolver": {"use_llm": False}, **cfg})
    vector, graph = stores
    return CognitiveMemory(vector, graph, router=router, async_research=False)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _facts(cog, label, facts, kind="entity"):
    n, _ = cog.knowledge.ensure_node(label, kind=kind)
    for f in facts:
        cog.knowledge.add_record(n, f)
    cog.knowledge.refresh_stats(n)
    return cog.graph.get_node(n["id"])


HENNA = ["Henna paste is applied at wedding ceremonies to decorate hands.",
         "Henna dye comes from the leaves of the henna plant."]
MEHNDI = ["Mehndi designs are drawn with henna paste on the bride's hands at weddings.",
          "Mehndi night is a pre-wedding celebration with music."]
SAREE = ["A saree is a long draped garment worn in South Asia.",
         "Kandyan saree draping is a Sri Lankan bridal style."]


# ------------------------------------------------------------------ C1. edge weights = centroid cosine

def test_edge_weight_is_the_cosine_of_the_two_centroids(stores):
    cog = _cog(stores)
    a = _facts(cog, "Henna", HENNA)
    b = _facts(cog, "Mehndi", MEHNDI)
    eid, _ = cog.knowledge.link(a["id"], b["id"], "related_to")
    e = cog.graph.get_edge(eid)
    expect = _cos(cog.vector.centroid(a["collection"]), cog.vector.centroid(b["collection"]))
    assert e["data"]["weight"] == pytest.approx(expect, abs=1e-3) and e["data"]["basis"] == "centroid"
    # more facts -> the weight follows (refresh_stats after a batch of writes)
    cog.knowledge.add_record(b, "Mehndi is a Persian word for henna.")
    cog.knowledge.refresh_stats(b)
    expect2 = _cos(cog.vector.centroid(a["collection"]), cog.vector.centroid(b["collection"]))
    assert cog.graph.get_edge(eid)["data"]["weight"] == pytest.approx(expect2, abs=1e-3)


def test_empty_collections_use_the_label_until_they_have_records(stores):
    cog = _cog(stores)
    a, _ = cog.knowledge.ensure_node("Colombo Stock Exchange")
    b, _ = cog.knowledge.ensure_node("Colombo Stock Exchange listing rules")
    pair = sorted((a["id"], b["id"]))
    eid = cog.graph.find_edge(pair[0], pair[1], "similar_to")
    assert eid and cog.graph.get_edge(eid)["data"]["basis"] == "label"


def test_peter_has_a_vector_and_has_subject_edges_are_weighted(stores):
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    inv = _facts(cog, "Investing", ["Investing means buying shares and unit trusts for returns."], kind="abstraction")
    cog.hierarchy.mark_subject(inv)
    cul = _facts(cog, "Culture", ["Culture includes weddings, clothing and festivals."], kind="abstraction")
    cog.hierarchy.mark_subject(cul)
    SleepCycle(cog).maintain()
    me = cog.graph.get_node_vector(SELF_ID)
    assert me and me["basis"] == "subjects" and me["n"] == 2
    for e in cog.graph.edges_of(SELF_ID):
        if e["relation"] == "has_subject":
            other = cog.graph.get_node(e["dst"])
            expect = _cos(me["vector"], cog.vector.centroid(other["collection"]))
            assert e["data"]["weight"] == pytest.approx(expect, abs=1e-3)


# ------------------------------------------------------------------ C2. record-level vector links

def test_record_links_are_stored_used_by_recall_and_cleaned_up(stores, events):
    cog = _cog(stores)
    g, v = cog.graph, cog.vector
    a = _facts(cog, "Henna", HENNA)
    b = _facts(cog, "Mehndi", MEHNDI)
    eid, _ = cog.knowledge.link(a["id"], b["id"], "related_to")
    links = g.vector_links_of_edges([eid])
    assert 0 < len(links) <= 3
    best = max(links, key=lambda l: l["similarity"])
    ra = v.get_records(a["collection"], [best["src_record"]])[0]
    rb = v.get_records(b["collection"], [best["dst_record"]])[0]
    assert best["similarity"] == pytest.approx(_cos(ra["embedding"], rb["embedding"]), abs=1e-3)
    # recall on Henna follows the links record-to-record into Mehndi's collection
    res = cog.graph_recall.recall("henna paste at wedding ceremonies")
    tr = of_type(events, "traverse")[-1]
    assert tr["vector_links"] and tr["vector_links"][0]["to_node"] == b["id"]
    assert set(l["to_record"] for l in tr["vector_links"]) <= set(tr["memory_ids"])
    assert eid in tr["edges"] and b["id"] in tr["path"]
    # ...and that is the ONLY way in when the neighbour collection is not searched
    set_cognition_config({"resolver": {"use_llm": False}, "recall": {"max_neighbours": 0}})
    res = cog.graph_recall.recall("henna paste at wedding ceremonies")
    pulled = [h for h in res.hits if h.get("via") == "vector_link"]
    assert pulled and pulled[0]["node_id"] == b["id"] and pulled[0]["edge_id"] == eid
    top = max((h for h in res.hits if h["node_id"] == a["id"]), key=lambda h: h["score"])
    assert pulled[0]["score"] <= top["score"]                        # weighted below the direct hit
    # deleting a linked record removes its links; removing a node removes all of them
    v.prune(b["collection"], [best["dst_record"]])
    assert not [l for l in g.vector_links_of_edges([eid]) if l["dst_record"] == best["dst_record"]]
    cog.knowledge.remove_node(b["id"])
    assert g.vector_links_of_edges([eid]) == [] and g.count_vector_links() == 0


def test_a_copy_is_never_linked_to_its_own_original(stores):
    cog = _cog(stores)
    a = _facts(cog, "Henna", HENNA)
    hub, _ = cog.knowledge.ensure_node("Body art", kind="abstraction")
    eid, _ = cog.knowledge.link(hub["id"], a["id"], "abstracts")
    cog.knowledge.copy_records(hub, a, cog.vector.get_all_records(a["collection"]))
    cog.knowledge.refresh_stats(hub)
    for l in cog.graph.vector_links_of_edges([eid]):
        src = cog.vector.get_records(hub["collection"], [l["src_record"]])[0]
        assert src["metadata"].get("origin_id") != l["dst_record"]


# ------------------------------------------------------------------ C3. similar_to from centroid kNN, refreshed

def test_similar_to_is_refreshed_from_centroids_with_hysteresis(stores):
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores, vectors={"similar_add_min": 0.4, "similar_remove_below": 0.2})
    g = cog.graph
    a = _facts(cog, "Henna", HENNA)
    b = _facts(cog, "Mehndi", MEHNDI)
    c = _facts(cog, "Saree", SAREE)
    SleepCycle(cog).maintain()
    sim_ab = _cos(cog.vector.centroid(a["collection"]), cog.vector.centroid(b["collection"]))
    sim_ac = _cos(cog.vector.centroid(a["collection"]), cog.vector.centroid(c["collection"]))
    assert sim_ab >= 0.4 > sim_ac
    pair = lambda x, y: g.find_edge(*sorted((x["id"], y["id"])), "similar_to")
    assert pair(a, b) and not pair(a, c)
    assert g.get_edge(pair(a, b))["data"]["weight"] == pytest.approx(sim_ab, abs=1e-3)
    # a hand-made similar_to between unrelated topics is dropped once the vectors disagree
    g.link(*sorted((a["id"], c["id"])), "similar_to")
    s = SleepCycle(cog)
    s.maintain()
    assert not pair(a, c) and s.report.vectors["similar_removed"] == 1
    # hysteresis: between the two thresholds an existing link stays, a new one is not made
    set_cognition_config({"resolver": {"use_llm": False},
                          "vectors": {"similar_add_min": sim_ab + 0.01, "similar_remove_below": sim_ab - 0.01}})
    SleepCycle(cog).maintain()
    assert pair(a, b)


# ------------------------------------------------------------------ B. every node is a filled vector store

class _OverviewWeb:
    """Fake web: any query gets one page per known label it mentions."""
    TEXT = {
        "investing": "Investing is putting money into shares, bonds and unit trusts for future returns. "
                     "Investing main areas include equities, fixed income and property.",
        "sri lanka financial markets": "Sri Lanka financial markets include the stock exchange and the bond market. "
                                       "Sri Lanka financial markets are regulated by the SEC.",
        "culture": "Culture covers the customs, arts, clothing and festivals of a people. "
                   "Culture main areas include weddings, music and dress.",
    }

    def __init__(self):
        self.queries = []

    def search(self, q, n):
        self.queries.append(q)
        out = []
        for lab in self.TEXT:
            if lab in q.lower():
                slug = lab.replace(" ", "-")
                out.append({"url": f"https://{slug}.example/overview", "title": f"{lab} overview", "snippet": lab})
        return out

    def fetch(self, url):
        lab = url.split("//")[1].split(".")[0].replace("-", " ")
        return self.TEXT.get(lab, "")


def _hub_chain(cog):
    from memory.entity_resolver import Match, Resolution
    from memory.graph_recall import GraphRecallResult
    cse = _facts(cog, "Colombo Stock Exchange", [
        "The Colombo Stock Exchange is the main stock exchange in Sri Lanka.",
        "The Colombo Stock Exchange has about 290 listed companies.",
        "The Colombo Stock Exchange indices are ASPI and S&P SL20."])
    res = Resolution(terms=["CSE"], matched=[Match("CSE", cse, "alias", 1.0)],
                     category="Sri Lanka financial markets", subject="Investing")
    cog.link_context(GraphRecallResult(query="CSE", resolution=res), cse["id"])
    return cse, cog.knowledge.find_node("Sri Lanka financial markets"), cog.knowledge.find_node("Investing")


def test_new_hubs_get_their_members_knowledge_at_once(stores):
    cog = _cog(stores)
    cse, hub, subj = _hub_chain(cog)
    v = cog.vector
    for n in (hub, subj):
        recs = v.get_all_records(n["collection"])
        assert len(recs) == 3, n["label"]                               # never an empty collection
        assert {r["metadata"]["origin_node"] for r in recs} == {cse["id"]}
        originals = {r["id"] for r in v.get_all_records(cse["collection"])}
        assert {r["metadata"]["origin_id"] for r in recs} == originals    # provenance to the ORIGINAL record
        assert cog.graph.get_collection(n["collection"])["record_count"] == 3
    # and its edges are weighted by real centroids now
    e = cog.graph.get_edge(cog.graph.find_edge(hub["id"], cse["id"], "abstracts"))
    assert e["data"]["basis"] == "centroid" and e["data"]["weight"] > 0.9


def test_sleep_fills_empty_hubs_with_an_overview_and_queues_the_rest(stores):
    from engine.curiosity import CuriosityEngine
    from engine.research_pipeline import ResearchPipeline
    from engine.sleep_cycle import SleepCycle
    cog = _cog(stores)
    web = _OverviewWeb()
    cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=web.search, fetch_fn=web.fetch)
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver)
    cul, _ = cog.knowledge.ensure_node("Culture", kind="abstraction", data={"role": "subject"})
    cog.hierarchy.mark_subject(cul)
    assert cog.record_count(cul) == 0
    s = SleepCycle(cog)
    s.fill()
    assert cog.record_count(cul) >= 2 and any("overview" in q for q in web.queries)
    assert s.report.filled[0]["node_id"] == cul["id"] and s.report.filled[0]["researched"] >= 2
    assert not s.budget.created                                         # filled in place: no new collection
    # the gap scan then queues whatever is still thin (here: a topic with no facts yet)
    lone, _ = cog.knowledge.ensure_node("Ikat weaving")
    s2 = SleepCycle(cog)
    s2.fill()
    s2.scan_gaps()
    assert lone["id"] in s2.report.gaps_flagged
    assert any(it["topic_node_id"] == lone["id"] for it in cog.curiosity.list())


def test_research_in_chat_fills_hubs_immediately(stores):
    from engine.research_pipeline import ResearchPipeline
    cog = _cog(stores, fill={"research_in_chat": True})
    web = _OverviewWeb()
    cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=web.search, fetch_fn=web.fetch)
    cse, hub, subj = _hub_chain(cog)
    assert any(q.startswith("Investing: overview") for q in web.queries)
    own = [r for r in cog.vector.get_all_records(subj["collection"]) if not r["metadata"].get("origin_id")]
    assert own and all(r["metadata"]["topic"] == "Investing" for r in own)   # researched into its own collection


def test_snapshot_marks_empty_collections(stores):
    from memory.mind_view import graph_snapshot
    cog = _cog(stores)
    a, _ = cog.knowledge.ensure_node("Ikat weaving")
    _facts(cog, "Henna", HENNA)
    snap = graph_snapshot(cog)
    by = {n["id"]: n for n in snap["nodes"]}
    assert by[a["id"]]["empty"] is True and by[a["id"]]["record_count"] == 0
    assert by[cog.knowledge.find_node("Henna")["id"]]["empty"] is False
    assert snap["stats"]["empty_collections"] == 1


# ------------------------------------------------------------------ D. the picture is the stored vector graph

def _d(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def test_force_layout_is_deterministic_weighted_and_peter_is_the_fixed_origin():
    from memory.graph_layout import graph_positions
    ids = ["a", "b", "c", "s"]
    edges = [{"src": "a", "dst": "b", "relation": "similar_to", "data": {"weight": 0.95}},
             {"src": "a", "dst": "c", "relation": "related_to", "data": {"weight": 0.05}},
             {"src": "s", "dst": "a", "relation": "abstracts", "data": {"weight": 0.6}},
             {"src": "self_peter", "dst": "s", "relation": "has_subject", "data": {"weight": 0.7}}]
    p1 = graph_positions(ids + ["self_peter"], edges, {})
    assert "self_peter" not in p1 and set(p1) == set(ids)
    assert p1 == graph_positions(list(reversed(ids)) + ["self_peter"], list(reversed(edges)), {})
    assert _d(p1["a"], p1["b"]) < _d(p1["a"], p1["c"])                 # similar -> shorter spring
    # seeded from the previous layout, a re-layout barely moves anything
    p2 = graph_positions(ids + ["self_peter"], edges, {}, previous=p1)
    assert max(_d(p1[i], p2[i]) for i in ids) < 1.0


def test_snapshot_draws_only_stored_edges_with_weights_and_peters_edges(stores):
    from memory.mind_view import graph_snapshot
    cog = _cog(stores)
    cse, hub, subj = _hub_chain(cog)
    hen = _facts(cog, "Henna", HENNA)
    snap = graph_snapshot(cog)
    assert snap["layout"] == "graph" and snap["self"]["id"] == SELF_ID
    drawn = {n["id"] for n in snap["nodes"]}
    assert SELF_ID not in drawn
    stored = {e["id"]: e for e in cog.graph.list_edges()}
    for e in snap["edges"]:
        assert e["id"] in stored                                        # nothing drawn that is not stored
        assert e["src"] in drawn | {SELF_ID} and e["dst"] in drawn
    rels = {(e["src"], e["dst"], e["relation"]): e for e in snap["edges"]}
    peter = rels[(SELF_ID, subj["id"], "has_subject")]                  # Peter's edge is in the snapshot
    assert snap["core_links"] == [subj["id"]]
    w = rels[(hub["id"], cse["id"], "abstracts")]
    assert w["weight"] is not None and w["primary"] is True
    assert peter["weight"] is not None
    # an unconnected topic is not pulled in by any drawn-only line
    assert not [e for e in snap["edges"] if hen["id"] in (e["src"], e["dst"]) and e["relation"] == "has_subject"]


def test_graph_layout_is_stable_and_relaxes_only_new_nodes(stores):
    """D2: an unchanged 'graph' graph gives IDENTICAL positions on the next snapshot (no drift),
    and adding one node moves that node and its neighbours while the rest of the picture holds."""
    from memory.mind_view import graph_snapshot
    cog = _cog(stores)
    cse, hub, subj = _hub_chain(cog)
    _facts(cog, "Henna", HENNA)
    _facts(cog, "Saree", SAREE)
    first = {n["id"]: tuple(n["position"]) for n in graph_snapshot(cog, layout_mode="graph")["nodes"]}
    again = {n["id"]: tuple(n["position"]) for n in graph_snapshot(cog, layout_mode="graph")["nodes"]}
    assert first == again                                               # byte-for-byte: no drift at all
    third = {n["id"]: tuple(n["position"]) for n in graph_snapshot(cog, layout_mode="graph")["nodes"]}
    assert third == first
    # add a new topic linked to Henna: it and Henna may move, the far-away subject chain must not
    hen = cog.knowledge.find_node("Henna")
    new, _ = cog.knowledge.ensure_node("Mehndi")
    for t in MEHNDI:
        cog.knowledge.add_record(new, t)
    cog.knowledge.refresh_stats(new)
    cog.knowledge.link(hen["id"], new["id"], "related_to")
    after = {n["id"]: tuple(n["position"]) for n in graph_snapshot(cog, layout_mode="graph")["nodes"]}
    assert new["id"] in after                                          # the new node was placed
    moved = {i for i in first if i in after and _d(first[i], after[i]) > 0.01}
    assert cse["id"] not in moved and subj["id"] not in moved          # the distant chain held still
    # and it is stable again afterwards
    stable = {n["id"]: tuple(n["position"]) for n in graph_snapshot(cog, layout_mode="graph")["nodes"]}
    assert stable == after


def test_research_in_chat_runs_in_the_background_when_async(stores):
    from engine.research_pipeline import ResearchPipeline
    set_cognition_config({"resolver": {"use_llm": False}, "fill": {"research_in_chat": True}})
    vector, graph = stores
    cog = CognitiveMemory(vector, graph, router=None, async_research=True)
    web = _OverviewWeb()
    cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=web.search, fetch_fn=web.fetch)
    cse, hub, subj = _hub_chain(cog)
    assert cog.record_count(subj) >= 3                     # members' knowledge right away
    cog.last_research_thread.join(timeout=20)
    assert any(q.startswith("Investing: overview") for q in web.queries)
    assert cog.own_record_count(cog.graph.get_node(subj["id"])) > 0
