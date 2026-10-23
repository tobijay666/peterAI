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

NEIGHBOUR_RELATIONS = ("abstracts", "related_to", "bridges")


@dataclass
class GraphRecallResult:
    query: str
    resolution: Resolution
    hits: List[Dict[str, Any]] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)

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
                 usage: UsageTracker, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.vector = vector
        self.graph = graph
        self.resolver = resolver
        self.usage = usage
        self._cfg = cfg

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

        result.hits = sorted(by_id.values(), key=lambda h: (-h["score"], h["id"]))
        if emit_events:
            emit("traverse", user_id=self.graph.user_id, turn_id=turn_id, query=query[:120],
                 path=result.path, edges=result.edges, memory_ids=[h["id"] for h in result.hits],
                 unmatched=resolution.unmatched)
        return result
