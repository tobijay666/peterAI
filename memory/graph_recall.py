"""GraphRecall — graph first, then the matched node's own collection (spec Phase 1.3).

For every node the resolver matched:
  * touch it (activation counter + last-accessed timestamp)
  * search its collection for the query
  * also search 1-hop abstraction / related / bridged neighbours, at a lower weight
and record the traversal path (nodes and edges in visit order) so the mind view
can send a light pulse along exactly those edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.cognition_config import cognition_config
from core.events import emit
from core.logger import logger
from memory.entity_resolver import EntityResolver, Resolution
from memory.graph_store import GraphMemory
from memory.usage import UsageTracker
from memory.vector_store import VectorMemory

NEIGHBOUR_RELATIONS = ("abstracts", "related_to", "bridges", "similar_to")


def dedupe_hits(hits: Any) -> List[Dict[str, Any]]:
    """One hit per fact: a hub's copy of a record (metadata origin_id) and the original, or two
    records with the same text, count once (the best-scoring one; the original on a tie).
    Sufficiency counts what is left, so copies can no longer make thin evidence look strong."""
    best: Dict[str, Dict[str, Any]] = {}
    by_text: Dict[str, str] = {}

    def rank(h: Dict[str, Any]):
        is_copy = bool((h.get("metadata") or {}).get("origin_id"))
        return (h["score"], not is_copy)

    for h in sorted(hits, key=lambda h: (-h["score"], h["id"])):
        meta = h.get("metadata") or {}
        key = str(meta.get("origin_id") or h["id"])
        text = " ".join((h.get("document") or "").lower().split())
        key = by_text.get(text, key)
        if key not in best or rank(h) > rank(best[key]):
            best[key] = h
        by_text.setdefault(text, key)
    return sorted(best.values(), key=lambda h: (-h["score"], h["id"]))


@dataclass
class GraphRecallResult:
    query: str
    resolution: Resolution
    hits: List[Dict[str, Any]] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    route: List[str] = field(default_factory=list)   # stored path subject -> ... -> topic
    # record-level vector links followed: {edge_id, from_node, to_node, from_record, to_record, similarity}
    vector_links: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def matched_nodes(self) -> List[Dict[str, Any]]:
        return self.resolution.nodes

    @property
    def unmatched_terms(self) -> List[str]:
        return self.resolution.unmatched

    @property
    def best_score(self) -> float:
        return max((h["score"] for h in self.hits), default=0.0)


class GraphRecall:
    def __init__(self, vector: VectorMemory, graph: GraphMemory, resolver: EntityResolver,
                 usage: UsageTracker, cfg: Optional[Dict[str, Any]] = None, hierarchy: Any = None) -> None:
        self.vector = vector
        self.graph = graph
        self.resolver = resolver
        self.usage = usage
        self._cfg = cfg
        self.hierarchy = hierarchy   # memory.hierarchy.Hierarchy: Peter -> subject -> topic

    @property
    def cfg(self) -> Dict[str, Any]:
        return (self._cfg or cognition_config())["recall"]

    def _search(self, node: Dict[str, Any], query: str, weight: float, top_k: int) -> List[Dict[str, Any]]:
        ns = node.get("collection")
        if not ns or not self.vector.has_collection(ns):
            return []
        try:
            hits = self.vector.recall(ns, query, n_results=top_k, silent=True)
        except Exception as e:
            logger.warning(f"GraphRecall: search in {ns} failed: {e}")
            return []
        self.graph.update_collection_stats(ns, accessed=True)
        out = []
        for h in hits:
            out.append({**h, "score": float(h["score"]) * weight, "raw_score": float(h["score"]),
                        "node_id": node["id"], "node_label": node.get("label", ""), "namespace": ns})
        return out

    def _follow_vector_links(self, by_id: Dict[str, Dict[str, Any]], result: GraphRecallResult) -> None:
        """A hit record with record-level vector links pulls the linked records out of the
        neighbouring collection (score = hit x link similarity x vectors.follow_weight), and
        the links are reported so the mind view can pulse record-to-record along the edge."""
        vcfg = (self._cfg or cognition_config()).get("vectors", {})
        fw, cap = float(vcfg.get("follow_weight", 0.5)), int(vcfg.get("max_follow", 6))
        if cap <= 0 or not by_id:
            return
        matched = {n["id"] for n in result.resolution.nodes}
        # links start at the records found in the matched topics themselves
        hits = sorted((h for h in by_id.values() if h.get("node_id") in matched),
                      key=lambda h: (-h["score"], h["id"]))[: max(4, int(self.cfg["top_k"]) * 2)]
        score_of = {h["id"]: h["score"] for h in hits}
        cand = []
        for ln in self.graph.vector_links_for_records(list(score_of)):
            for frm, to, fn, tn in ((ln["src_record"], ln["dst_record"], ln["src_node"], ln["dst_node"]),
                                    (ln["dst_record"], ln["src_record"], ln["dst_node"], ln["src_node"])):
                if frm in score_of and tn not in matched:
                    cand.append((score_of[frm] * float(ln["similarity"]), ln, frm, to, fn, tn))
        cand.sort(key=lambda t: (-t[0], t[3]))
        seen, chosen = set(), []
        for c in cand:
            if c[3] in seen:
                continue
            seen.add(c[3]); chosen.append(c)
            if len(chosen) >= cap:
                break
        by_node: Dict[str, List[Any]] = {}
        for c in chosen:
            by_node.setdefault(c[5], []).append(c)
        for nid, items in sorted(by_node.items()):
            node = self.graph.get_node(nid)
            if not node or not node.get("collection") or not self.vector.has_collection(node["collection"]):
                continue
            recs = {r["id"]: r for r in self.vector.get_records(node["collection"], [c[3] for c in items])}
            for score, ln, frm, to, fn, tn in items:
                r = recs.get(to)
                if r is None:
                    continue
                h = {"id": r["id"], "document": r["document"], "metadata": r["metadata"],
                     "score": score * fw, "raw_score": float(ln["similarity"]), "node_id": nid,
                     "node_label": node.get("label", ""), "namespace": node["collection"],
                     "via": "vector_link", "edge_id": ln["edge_id"], "from_record": frm}
                if r["id"] not in by_id or by_id[r["id"]]["score"] < h["score"]:
                    by_id[r["id"]] = h
                result.vector_links.append({"edge_id": ln["edge_id"], "from_node": fn, "to_node": tn,
                                            "from_record": frm, "to_record": to,
                                            "similarity": round(float(ln["similarity"]), 4)})
                if tn not in result.path:
                    result.path.append(tn)
                if ln["edge_id"] not in result.edges:
                    result.edges.append(ln["edge_id"])

    def _record_fallback(self, query: str, resolution: Resolution) -> None:
        """No entity node matched: look inside knowledge collections for a closely matching
        record (knowledge about a sub-topic is often stored under its parent topic, e.g.
        'nekath' facts inside the Poruwa ceremony collection). Capped, most-used first."""
        floor = float(self.cfg.get("record_fallback_min", 0.45))
        cap = int(self.cfg.get("record_fallback_max_collections", 64))
        nodes = [n for n in self.graph.all_nodes(["entity", "abstraction"])
                 if n.get("status") != "pruned" and n.get("collection")]
        nodes.sort(key=lambda n: -float(n.get("usage_score") or 0.0))
        found = []
        for n in nodes[:cap]:
            if not self.vector.has_collection(n["collection"]):
                continue
            hits = self.vector.recall(n["collection"], query, n_results=1, touch=False, silent=True)
            if hits and hits[0]["score"] >= floor:
                found.append((hits[0]["score"], n))
        found.sort(key=lambda t: -t[0])
        from memory.entity_resolver import Match
        for score, n in found[:2]:
            resolution.matched.append(Match(term=query, node=n, via="records", score=score))

    def recall(self, query: str, turn_id: str = "", resolution: Optional[Resolution] = None,
               emit_events: bool = True, touch: bool = True) -> GraphRecallResult:
        resolution = resolution or self.resolver.resolve(query)
        route_nodes: List[str] = []
        route_edges: List[str] = []
        if not resolution.matched and resolution.terms and self.hierarchy is not None:
            # No topic by name: walk Peter's tree top-down (subject -> field -> topic) by meaning
            try:
                walked = self.hierarchy.descend(query)
            except Exception as e:
                logger.debug(f"GraphRecall: tree walk failed: {e}")
                walked = None
            if walked is not None:
                from memory.entity_resolver import Match
                node, score, route_nodes, route_edges = walked
                resolution.matched.append(Match(term=query, node=node, via="tree", score=score))
                try:   # the pulse follows the primary route, like every other recall
                    r_nodes, r_edges = self.hierarchy.route_to(node["id"])
                    if r_nodes:
                        route_nodes, route_edges = r_nodes, r_edges
                except Exception:
                    pass
        if not resolution.matched and resolution.terms:
            self._record_fallback(query, resolution)
        result = GraphRecallResult(query=query, resolution=resolution)
        top_k = int(self.cfg["top_k"])
        nweight = float(self.cfg["neighbour_weight"])
        max_nb = int(self.cfg["max_neighbours"])

        by_id: Dict[str, Dict[str, Any]] = {}
        for node in resolution.nodes:
            result.path.append(node["id"])
            if touch:
                self.usage.touch(node["id"], turn_id=turn_id, emit_event=emit_events)
            for h in self._search(node, query, 1.0, top_k):
                if h["id"] not in by_id or by_id[h["id"]]["score"] < h["score"]:
                    by_id[h["id"]] = h
            # 1-hop neighbours (abstractions above, related / bridged peers)
            taken = 0
            for e in self.graph.edges_of(node["id"]):
                if taken >= max_nb or e["relation"] not in NEIGHBOUR_RELATIONS:
                    continue
                other_id = e["dst"] if e["src"] == node["id"] else e["src"]
                other = self.graph.get_node(other_id)
                if not other or other.get("status") == "pruned" or not other.get("collection"):
                    continue
                taken += 1
                result.edges.append(e["id"])
                if other_id not in result.path:
                    result.path.append(other_id)
                for h in self._search(other, query, nweight, max(2, top_k // 2)):
                    if h["id"] not in by_id or by_id[h["id"]]["score"] < h["score"]:
                        by_id[h["id"]] = h

        try:
            self._follow_vector_links(by_id, result)
        except Exception as e:
            logger.debug(f"GraphRecall: following vector links failed: {e}")
        result.hits = dedupe_hits(by_id.values())
        # The stored route Peter took: subject -> ... -> first matched topic. The pulse in
        # the mind view starts at Peter's core, enters at the subject and follows these edges.
        if not route_nodes and result.path and self.hierarchy is not None:
            try:
                route_nodes, route_edges = self.hierarchy.route_to(result.path[0])
            except Exception:
                route_nodes, route_edges = [], []
        result.route = route_nodes
        if emit_events:
            path = route_nodes + [n for n in result.path if n not in route_nodes]
            emit("traverse", user_id=self.graph.user_id, turn_id=turn_id, query=query[:120],
                 path=path, edges=route_edges + result.edges, memory_ids=[h["id"] for h in result.hits],
                 unmatched=resolution.unmatched, vector_links=result.vector_links)
        return result
