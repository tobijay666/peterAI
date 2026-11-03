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
        try:
            ensure_self(graph)
        except Exception as e:
            logger.warning(f"hierarchy: could not create the Peter node: {e}")

    @property
    def cfg(self) -> Dict[str, Any]:
        return cognition_config()["recall"]

    # ------------------------------------------------------------------ structure

    def mark_subject(self, node: Dict[str, Any], turn_id: str = "") -> None:
        """Make node a main subject: role=subject (layout) + has_subject edge from Peter (graph)."""
        if not is_subject(node):
            self.graph.upsert_node(node["id"], kind=node.get("kind", "abstraction"), label=node["label"],
                                   data={**node_data(node), "role": "subject"},
                                   importance=float(node.get("importance") or 0.5),
                                   pinned=bool(node.get("pinned")))
        ensure_self(self.graph)
        self.graph.link_once(SELF_ID, node["id"], "has_subject")

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
            ups = sorted((e for e in self.graph.edges_of(cur)
                          if e["relation"] == "abstracts" and e["dst"] == cur and e["src"] not in seen),
                         key=lambda e: e["src"])
            if not ups:
                break
            subj_ups = [e for e in ups if is_subject(self.graph.get_node(e["src"]))]
            e = (subj_ups or ups)[0]
            edges.append(e["id"]); nodes.append(e["src"]); seen.add(e["src"]); cur = e["src"]
        return [], []

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
        """Store similar_to links from node to the closest existing topics by meaning.
        Main subjects and Peter are never linked this way (they are structure)."""
        cfg = self.cfg
        floor = float(cfg.get("similar_link_min", 0.6))
        max_links = int(cfg.get("similar_link_max", 3))
        if max_links <= 0 or node.get("kind") not in TOPIC_KINDS or is_subject(node):
            return 0
        label = node.get("label") or ""
        try:
            hits = self.vector.search_entities(label, n_results=max_links + 4)
        except Exception:
            return 0
        made = 0
        for h in hits:
            if made >= max_links or h["score"] < floor:
                break
            other = self.graph.get_node(h["node_id"])
            if (not other or other["id"] == node["id"] or other.get("status") == "pruned"
                    or is_subject(other) or other.get("kind") not in TOPIC_KINDS):
                continue
            if (other.get("label") or "").strip().lower() == label.strip().lower():
                continue
            a, b = sorted((node["id"], other["id"]))   # one undirected link per pair
            if self.graph.find_edge(a, b, "similar_to") or self.graph.find_edge(b, a, "similar_to"):
                continue
            if self.knowledge is not None:
                self.knowledge.link(a, b, "similar_to", turn_id=turn_id)
            else:
                self.graph.link_once(a, b, "similar_to")
            made += 1
        return made
