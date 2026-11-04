"""Peter's place in his own memory graph.

Before this module Peter was not in the graph at all: the "Peter" ball in the mind
view was a sphere drawn at the origin, recall jumped straight to a topic by name,
and similarity never produced a stored link outside the sleep cycle. This module
makes the hierarchy real, stored as ordinary graph edges:

    self_peter --has_subject--> Investing --abstracts--> Sri Lanka financial markets
                                                         --abstracts--> Colombo Stock Exchange
    self_peter <--helper_of-- researcher <--spawned_by-- task_…   (plumbing, never drawn)
    Colombo Stock Exchange --similar_to-- Unit trusts            (meaning-based cross link)

  * ensure_self()   - the Peter node exists (one per user graph) and every subject hangs off it
  * mark_subject()  - promote a node to a main subject: role=subject + has_subject edge
  * route_to()      - the stored path Peter -> subject -> ... -> topic (for recall pulses)
  * descend()       - recall by walking the tree top-down when no topic matched by name
  * link_similar()  - store similar_to links to the closest existing topics (by meaning)
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from core.cognition_config import cognition_config
from core.logger import logger

SELF_ID = "self_peter"
SELF_KIND = "self"
TOPIC_KINDS = ["entity", "abstraction"]
HELPER_NAMES = {"researcher", "librarian", "qa", "voice", "filesystem", "devops"}


def node_data(node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = (node or {}).get("data") or {}
    if isinstance(d, str):
        try:
            d = json.loads(d or "{}")
        except Exception:
            d = {}
    return d if isinstance(d, dict) else {}


def is_subject(node: Optional[Dict[str, Any]]) -> bool:
    return node_data(node).get("role") == "subject"


def ensure_self(graph: Any) -> str:
    """Create the Peter node if missing and hang every main subject off it. Idempotent."""
    if graph.get_node(SELF_ID) is None:
        graph.upsert_node(SELF_ID, kind=SELF_KIND, label="Peter", data={"role": "self"}, importance=1.0,
                          pinned=True)
    for n in graph.all_nodes(TOPIC_KINDS):
        if is_subject(n) and n.get("status") != "pruned":
            graph.link_once(SELF_ID, n["id"], "has_subject")
    return SELF_ID


def ensure_primary(graph: Any, node_id: str) -> None:
    """Exactly one incoming 'abstracts' edge of a node is primary (the oldest, if none is)."""
    ups = [e for e in graph.edges_of(node_id) if e["relation"] == "abstracts" and e["dst"] == node_id]
    if not ups:
        return
    ups.sort(key=lambda e: (e.get("created_at") or 0.0, e["id"]))
    prim = [e for e in ups if (e.get("data") or {}).get("primary")]
    keep = prim[0] if prim else ups[0]
    if not prim:
        graph.update_edge_data(keep["id"], primary=True)
    for e in prim[1:]:
        graph.update_edge_data(e["id"], primary=None)


def link_helper(graph: Any, agent_node_id: str) -> None:
    """Helper agents (researcher, librarian…) hang under Peter as plumbing; the view never draws them."""
    try:
        ensure_self(graph)
        graph.link_once(agent_node_id, SELF_ID, "helper_of")
    except Exception as e:
        logger.debug(f"hierarchy: helper link skipped: {e}")


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Hierarchy:
    def __init__(self, vector: Any, graph: Any, knowledge: Any = None) -> None:
        self.vector = vector
        self.graph = graph
        self.knowledge = knowledge
        self._centroids: Dict[Tuple[str, int], Optional[List[float]]] = {}
        self._label_vecs: Dict[str, List[float]] = {}
        self.vgraph: Any = None        # memory.vector_graph.VectorGraph (set by CognitiveMemory)
        try:
            ensure_self(graph)
        except Exception as e:
            logger.warning(f"hierarchy: could not create the Peter node: {e}")

    @property
    def cfg(self) -> Dict[str, Any]:
        return cognition_config()["recall"]

    # ------------------------------------------------------------------ structure

    def mark_subject(self, node: Dict[str, Any], turn_id: str = "") -> None:
        """Make node a main subject: role=subject (layout) + has_subject edge from Peter (graph).

        A main subject sits directly under Peter and never under another node, so any
        'abstracts' membership it had (it was a field hub under another subject) is removed;
        its own members keep their place."""
        if not is_subject(node):
            self.graph.upsert_node(node["id"], kind=node.get("kind", "abstraction"), label=node["label"],
                                   data={**node_data(node), "role": "subject"})
        for e in self.graph.edges_of(node["id"]):
            if e["relation"] == "abstracts" and e["dst"] == node["id"]:
                self.graph.delete_edge(e["id"])
        ensure_self(self.graph)
        if self.knowledge is not None:
            self.knowledge.link(SELF_ID, node["id"], "has_subject", turn_id=turn_id)
        else:
            self.graph.link_once(SELF_ID, node["id"], "has_subject")

    def ensure_primary(self, node_id: str) -> None:
        ensure_primary(self.graph, node_id)

    def repair(self) -> int:
        """Make a stored hierarchy sound (old graphs, hand-made links). Returns changes made:
          * no 'abstracts' edge into a main subject (or Peter);
          * no cycle in 'abstracts' (edges replayed oldest first; one that would close a
            cycle is removed);
          * every node with hubs above it has exactly one primary parent edge."""
        changes = 0
        edges = sorted((e for e in self.graph.list_edges() if e["relation"] == "abstracts"),
                       key=lambda e: (e.get("created_at") or 0.0, e["id"]))
        subj = {n["id"] for n in self.graph.all_nodes(TOPIC_KINDS) if is_subject(n)} | {SELF_ID}
        kids: Dict[str, List[str]] = {}

        def reaches(a: str, b: str) -> bool:
            seen, stack = {a}, [a]
            while stack:
                cur = stack.pop()
                if cur == b:
                    return True
                for k in kids.get(cur, []):
                    if k not in seen:
                        seen.add(k); stack.append(k)
            return False

        kept: List[Dict[str, Any]] = []
        for e in edges:
            if e["dst"] in subj or e["src"] == e["dst"] or reaches(e["dst"], e["src"]):
                self.graph.delete_edge(e["id"])
                changes += 1
                continue
            kids.setdefault(e["src"], []).append(e["dst"])
            kept.append(e)
        by_dst: Dict[str, List[Dict[str, Any]]] = {}
        for e in kept:
            by_dst.setdefault(e["dst"], []).append(e)
        for dst, ups in by_dst.items():
            prim = [e for e in ups if (e.get("data") or {}).get("primary")]
            if len(prim) != 1:
                self.ensure_primary(dst)
                changes += 1
        # D4: a hub and a node it abstracts must not also carry a symmetric link (related_to /
        # bridges / similar_to) — that is the audit item 17 rule, applied to old/hand-made graphs.
        abstract_pairs = {(e["src"], e["dst"]) for e in kept}
        for e in self.graph.list_edges():
            if e["relation"] not in ("related_to", "bridges", "similar_to"):
                continue
            if (e["src"], e["dst"]) in abstract_pairs or (e["dst"], e["src"]) in abstract_pairs:
                self.graph.delete_edge(e["id"])
                changes += 1
        return changes

    def subjects(self) -> List[Dict[str, Any]]:
        out = []
        for e in self.graph.edges_of(SELF_ID):
            if e["relation"] == "has_subject" and e["src"] == SELF_ID:
                n = self.graph.get_node(e["dst"])
                if n and n.get("status") != "pruned":
                    out.append(n)
        return sorted(out, key=lambda n: n["id"])

    def children(self, node_id: str) -> List[Dict[str, Any]]:
        out = []
        for e in self.graph.edges_of(node_id):
            if e["relation"] == "abstracts" and e["src"] == node_id:
                n = self.graph.get_node(e["dst"])
                if n and n.get("status") != "pruned" and n.get("kind") in TOPIC_KINDS:
                    out.append((e, n))
        return [n for _, n in sorted(out, key=lambda t: t[1]["id"])]

    def route_to(self, node_id: str, max_depth: int = 6) -> Tuple[List[str], List[str]]:
        """(node ids, edge ids) of the stored path subject -> ... -> node, following
        'abstracts' edges upward. Empty when the node is not under a subject."""
        nodes, edges, cur, seen = [node_id], [], node_id, {node_id}
        for _ in range(max_depth):
            cur_node = self.graph.get_node(cur)
            if is_subject(cur_node):
                nodes.reverse(); edges.reverse()
                return nodes, edges
            ups = [e for e in self.graph.edges_of(cur)
                   if e["relation"] == "abstracts" and e["dst"] == cur and e["src"] not in seen]
            if not ups:
                break
            e = self._pick_parent(ups)     # the primary parent, consistently
            edges.append(e["id"]); nodes.append(e["src"]); seen.add(e["src"]); cur = e["src"]
        return [], []

    def _pick_parent(self, ups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Primary parent edge (edge data primary=true) first, then a subject, then by id."""
        def data(e):
            d = e.get("data") or {}
            return d if isinstance(d, dict) else {}
        prim = sorted((e for e in ups if data(e).get("primary")), key=lambda e: e["src"])
        if prim:
            return prim[0]
        ups = sorted(ups, key=lambda e: e["src"])
        subj = [e for e in ups if is_subject(self.graph.get_node(e["src"]))]
        return (subj or ups)[0]

    # ------------------------------------------------------------------ meaning

    def _vec(self, node: Dict[str, Any], label_vec: Optional[List[float]]) -> List[List[float]]:
        """Vectors that stand for a node: its label, and its facts' centroid when it has any."""
        vecs = [label_vec] if label_vec else []
        coll = node.get("collection")
        if coll and self.vector.has_collection(coll):
            try:
                count = self.vector.count(coll) if hasattr(self.vector, "count") else -1
            except Exception:
                count = -1
            key = (coll, count)
            if key not in self._centroids:
                self._centroids[key] = self.vector.centroid(coll) if count != 0 else None
            if self._centroids[key]:
                vecs.append(self._centroids[key])
        return vecs

    def _scores(self, qvec: List[float], nodes: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        if not nodes:
            return []
        labels = [n.get("label") or "" for n in nodes]
        missing = sorted({l for l in labels if l not in self._label_vecs})
        if missing:   # label embeddings are cached: a walk over a big tree stays cheap
            for l, v in zip(missing, self.vector.embed(missing)):
                self._label_vecs[l] = v
        label_vecs = [self._label_vecs[l] for l in labels]
        out = []
        for n, lv in zip(nodes, label_vecs):
            out.append((max((_cos(qvec, v) for v in self._vec(n, lv)), default=0.0), n))
        return sorted(out, key=lambda t: (-t[0], t[1]["id"]))

    def descend(self, query: str, max_depth: int = 6, max_nodes: int = 400
                ) -> Optional[Tuple[Dict[str, Any], float, List[str], List[str]]]:
        """Recall by searching Peter's tree top-down: from Peter through each main subject
        down its 'abstracts' branches, score every topic on the way against the question
        (label + facts), and take the best one if it reaches tree_walk_min.
        Returns (node, score, route node ids subject -> … -> node, route edge ids) or None.
        Hub nodes (e.g. "Sri Lanka financial markets") often have no facts of their own,
        so the whole branch is searched rather than stopping at a weak middle level."""
        floor = float(self.cfg.get("tree_walk_min", 0.4))
        subjects = self.subjects()
        if not subjects or not query.strip():
            return None
        parent: Dict[str, Tuple[Optional[str], Optional[str]]] = {}   # id -> (parent id, edge id)
        order: List[Dict[str, Any]] = []
        frontier = []
        for subj in subjects:
            if subj["id"] not in parent:
                parent[subj["id"]] = (None, None)
                order.append(subj)
                frontier.append(subj)
        for _ in range(max_depth):
            nxt = []
            for cur in frontier:
                for kid in self.children(cur["id"]):
                    if kid["id"] in parent or len(order) >= max_nodes:
                        continue
                    parent[kid["id"]] = (cur["id"], self.graph.find_edge(cur["id"], kid["id"], "abstracts"))
                    order.append(kid)
                    nxt.append(kid)
            frontier = nxt
            if not frontier:
                break
        qvec = self.vector.embed([query])[0]
        scored = self._scores(qvec, order)
        score, node = scored[0]
        if score < floor:
            return None
        nodes, edges, cur = [node["id"]], [], node["id"]
        while parent[cur][0] is not None:
            p_id, e_id = parent[cur]
            if e_id:
                edges.append(e_id)
            nodes.append(p_id)
            cur = p_id
        nodes.reverse(); edges.reverse()
        return node, score, nodes, edges

    def link_similar(self, node: Dict[str, Any], turn_id: str = "") -> int:
        """Store similar_to links from node to the closest existing topics by meaning: k nearest
        by node vector (memory/vector_graph.py: centroid, or label while a collection is empty).
        Main subjects and Peter are never linked this way (they are structure)."""
        if self.vgraph is None:
            from memory.vector_graph import VectorGraph
            self.vgraph = VectorGraph(self.vector, self.graph, knowledge=self.knowledge)
        return self.vgraph.link_similar(node, turn_id=turn_id)
