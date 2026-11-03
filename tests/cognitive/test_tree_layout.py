"""Tree layout for the mind view: hubs above topics, questions under topics, no overlaps."""
from memory.tree_layout import build_forest, tree_positions, LEVEL_GAP


def _n(i, kind="entity", u=0.0):
    return {"id": i, "kind": kind, "label": i, "usage_score": u}


def _e(s, d, rel):
    return {"src": s, "dst": d, "relation": rel}


NODES = [_n("hub", "abstraction"), _n("dorian", u=2.0), _n("wilde", u=1.0),
         _n("q1", "curiosity"), _n("q2", "curiosity"), _n("lonely")]
EDGES = [_e("hub", "dorian", "abstracts"), _e("dorian", "wilde", "related_to"),
         _e("q1", "dorian", "curious_about"), _e("q2", "wilde", "curious_about")]


def test_hub_above_topic_and_questions_under_topics():
    p = tree_positions(NODES, EDGES)
    assert p["hub"][1] > p["dorian"][1] > p["wilde"][1] > p["q2"][1]
    assert abs(p["dorian"][1] - p["q1"][1] - LEVEL_GAP) < 1e-6
    assert all(v[2] == 0.0 for v in p.values())


def test_no_overlap_on_same_level():
    p = tree_positions(NODES, EDGES)
    rows = {}
    for i, (x, y, _) in p.items():
        rows.setdefault(round(y, 3), []).append(x)
    for xs in rows.values():
        xs.sort()
        assert all(b - a >= 1.7 for a, b in zip(xs, xs[1:]))


def test_cycles_broken_and_deterministic():
    nodes = [_n("a", u=1), _n("b", u=2), _n("c", u=3)]
    edges = [_e("a", "b", "abstracts"), _e("b", "c", "abstracts"), _e("c", "a", "abstracts")]
    roots, children = build_forest(nodes, edges)
    assert len(roots) == 1
    assert sum(len(v) for v in children.values()) == 2
    assert tree_positions(nodes, edges) == tree_positions(list(reversed(nodes)), edges)


def test_empty():
    assert tree_positions([], []) == {}


def test_core_links_only_main_subjects():
    from memory.tree_layout import core_subjects
    nodes = [{"id": "w", "kind": "abstraction", "label": "Writing", "data": {"role": "subject"}},
             _n("vic", "abstraction"), _n("dorian"), _n("loose")]
    edges = [_e("w", "vic", "abstracts"), _e("vic", "dorian", "abstracts")]
    assert core_subjects(nodes, edges) == ["w"]
    p = tree_positions(nodes, edges)
    assert p["w"][1] > p["vic"][1] > p["dorian"][1]


def test_parse_subject_normalised():
    from memory.entity_resolver import _parse_extraction
    out = _parse_extraction('{"entities": ["Colombo Stock Exchange"], "aspect": "", '
                            '"category": "Sri Lanka financial markets", "subject": "investing."}')
    assert out["subject"] == "Investing"


def test_every_topic_ends_up_under_a_main_subject(stores):
    """Loose topics and questions are put under a subject; the subject is linked to Peter."""
    from memory.cognition import CognitiveMemory, _is_subject
    from memory.mind_view import graph_snapshot
    from memory.tree_layout import build_forest
    from tests.fakes.fake_llm import FakeLLMRouter
    vector, graph = stores
    router = FakeLLMRouter([{"content": '{"subject": "investing"}'}] * 10)
    cog = CognitiveMemory(vector, graph, router=router, async_research=False)
    cse, _ = cog.knowledge.ensure_node("Colombo Stock Exchange")
    graph.upsert_node("cur_x", kind="curiosity", label="How do equity trust funds work?")
    graph.upsert_node("agent_r", kind="entity", label="researcher")
    assert cog.attach_orphans() == 2
    nodes = [n for n in graph.all_nodes(["entity", "abstraction", "curiosity"]) if n["label"] != "researcher"]
    roots, _ = build_forest(nodes, graph.list_edges([n["id"] for n in nodes]))
    assert len(roots) == 1 and _is_subject(graph.get_node(roots[0]))
    assert graph.get_node(roots[0])["label"] == "Investing"
    snap = graph_snapshot(cog, layout_mode="tree")
    assert snap["core_links"] == roots
    assert "researcher" not in {n["label"] for n in snap["nodes"]}
    assert cog.attach_orphans() == 0          # nothing left to place


# ---- 3-D cluster layout: hierarchy kept, similarity decides closeness ----

def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def test_cluster_similar_subjects_and_topics_sit_together():
    from memory.tree_layout import SUBJECT_RADIUS, cluster_positions
    subj = {"role": "subject"}
    nodes = [{"id": "fin", "kind": "abstraction", "label": "Finance", "data": subj},
             {"id": "inv", "kind": "abstraction", "label": "Investing", "data": subj},
             {"id": "wri", "kind": "abstraction", "label": "Writing", "data": subj},
             _n("cse"), _n("ut"), _n("poem")]
    edges = [_e("inv", "cse", "abstracts"), _e("inv", "ut", "abstracts"), _e("wri", "poem", "abstracts")]
    sem = {"fin": (10, 1, 0), "inv": (9, 2, 1), "wri": (-10, 0, 3),
           "cse": (9, 3, 2), "ut": (9.5, 2.5, 1.5), "poem": (-9, 1, 4)}
    p = cluster_positions(nodes, edges, sem)
    for s in ("fin", "inv", "wri"):                                    # subjects ring Peter
        assert abs(_dist(p[s], (0, 0, 0)) - SUBJECT_RADIUS) < 1e-3
    assert _dist(p["fin"], p["inv"]) < _dist(p["fin"], p["wri"])       # similar subjects adjacent
    assert _dist(p["cse"], p["ut"]) < _dist(p["cse"], p["poem"])       # similar topics together
    assert _dist(p["cse"], (0, 0, 0)) > SUBJECT_RADIUS                  # topics grow outward
    assert _dist(p["cse"], p["ut"]) >= 2.5                              # but never overlap
    assert len({c for xyz in p.values() for c in xyz[2:]}) > 1          # really 3-D (not flat)
    assert p == cluster_positions(nodes, edges, sem)                    # deterministic


def test_cluster_is_default_snapshot_layout(stores):
    from memory.cognition import CognitiveMemory
    from memory.mind_view import graph_snapshot
    vector, graph = stores
    cog = CognitiveMemory(vector, graph, router=None, async_research=False)
    cog.knowledge.ensure_node("henna")
    assert graph_snapshot(cog)["layout"] == "cluster"
