"""Peter is a real node in his graph: subjects hang off him by stored edges, recall can
walk the tree top-down, and similar topics get stored similar_to links."""
from __future__ import annotations

from core.cognition_config import set_cognition_config
from memory.cognition import CognitiveMemory
from memory.hierarchy import SELF_ID, ensure_self, is_subject, link_helper
from memory.mind_view import graph_snapshot
from tests.cognitive.conftest import of_type


def _cog(stores):
    set_cognition_config({"resolver": {"use_llm": False}})
    vector, graph = stores
    return CognitiveMemory(vector, graph, router=None, async_research=False)


def _investing_tree(cog):
    k = cog.knowledge
    inv, _ = k.ensure_node("Investing", kind="abstraction", data={"role": "subject"})
    cog.hierarchy.mark_subject(inv)
    mkt, _ = k.ensure_node("Sri Lanka financial markets", kind="abstraction")
    cse, _ = k.ensure_node("Colombo Stock Exchange")
    k.link(inv["id"], mkt["id"], "abstracts")
    k.link(mkt["id"], cse["id"], "abstracts")
    for t in ("Shares are bought and sold through licensed stockbrokers on the exchange.",
              "Trading in shares happens on weekdays through stockbrokers.",
              "Brokers charge a commission on every share trade."):
        k.add_record(cse, t)
    return inv, mkt, cse


def test_peter_is_a_node_and_subjects_hang_off_him(stores):
    cog = _cog(stores)
    graph = cog.graph
    me = graph.get_node(SELF_ID)
    assert me and me["kind"] == "self" and me["label"] == "Peter"
    inv, mkt, _ = _investing_tree(cog)
    edges = [e for e in graph.edges_of(SELF_ID) if e["relation"] == "has_subject"]
    assert [e["dst"] for e in edges] == [inv["id"]]                  # stored, not just drawn
    assert cog.hierarchy.subjects()[0]["id"] == inv["id"]
    snap = graph_snapshot(cog)
    assert snap["core_links"] == [inv["id"]]
    assert SELF_ID not in {n["id"] for n in snap["nodes"]}           # the core is drawn by the view


def test_existing_subjects_are_backfilled(stores):
    vector, graph = stores
    graph.upsert_node("abs_writing", kind="abstraction", label="Writing", data={"role": "subject"})
    ensure_self(graph)
    assert graph.find_edge(SELF_ID, "abs_writing", "has_subject")
    ensure_self(graph)                                                # idempotent
    assert sum(e["relation"] == "has_subject" for e in graph.edges_of(SELF_ID)) == 1


def test_recall_walks_the_tree_when_no_topic_is_named(stores, events):
    cog = _cog(stores)
    inv, mkt, cse = _investing_tree(cog)
    result, _ = cog.recall("how do stockbrokers handle trading in shares?")
    assert [m.via for m in result.resolution.matched] == ["tree"]
    assert result.matched_nodes[0]["id"] == cse["id"]
    assert result.route == [inv["id"], mkt["id"], cse["id"]]
    assert result.hits and result.hits[0]["node_id"] == cse["id"]
    tr = of_type(events, "traverse")[-1]
    assert tr["path"][:3] == [inv["id"], mkt["id"], cse["id"]]       # pulse: Peter -> subject -> … -> topic
    assert len(tr["edges"]) >= 2


def test_direct_match_reports_its_route_from_the_subject(stores, events):
    cog = _cog(stores)
    inv, mkt, cse = _investing_tree(cog)
    result, _ = cog.recall("what is the Colombo Stock Exchange?")
    assert result.matched_nodes[0]["id"] == cse["id"]
    assert result.route == [inv["id"], mkt["id"], cse["id"]]
    assert of_type(events, "traverse")[-1]["path"][0] == inv["id"]


def test_unrelated_question_does_not_walk_into_a_wrong_topic(stores):
    cog = _cog(stores)
    _investing_tree(cog)
    assert cog.hierarchy.descend("recipe for chocolate cake") is None


def test_similar_topics_get_a_stored_link_that_is_not_a_tree_parent(stores):
    from memory.tree_layout import build_forest
    cog = _cog(stores)
    inv, mkt, cse = _investing_tree(cog)
    rules, _ = cog.knowledge.ensure_node("Colombo Stock Exchange listing rules")
    graph = cog.graph
    a, b = sorted((cse["id"], rules["id"]))
    assert graph.find_edge(a, b, "similar_to")
    assert not any(e["relation"] == "similar_to" for e in graph.edges_of(inv["id"]))  # subjects never
    nodes = graph.all_nodes(["entity", "abstraction"])
    roots, children = build_forest(nodes, graph.list_edges([n["id"] for n in nodes]))
    assert rules["id"] in roots                                        # still needs its own subject
    cog.attach_orphans()                                               # ...which it gets by similarity
    assert graph.find_edge(inv["id"], rules["id"], "abstracts")


def test_helpers_hang_under_peter_and_are_never_drawn(stores):
    cog = _cog(stores)
    graph = cog.graph
    graph.upsert_node("researcher", kind="agent", label="researcher")
    link_helper(graph, "researcher")
    assert graph.find_edge("researcher", SELF_ID, "helper_of")
    labels = {n["label"] for n in graph_snapshot(cog)["nodes"]}
    assert "researcher" not in labels and "Peter" not in labels
    assert not is_subject(graph.get_node("researcher"))
