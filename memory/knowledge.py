"""KnowledgeStore — entity/abstraction nodes bound to their own vector collection.

One place that keeps the graph node, its Chroma collection, its aliases, its
entity-index vector and the collection registry consistent:

    ensure_node("Henna")  ->  node 'ent_henna'  +  collection 'peter_k_henna'
                              + alias 'henna'    +  index vector 'Henna'
    remove_node(...)      ->  the inverse, used by pruning

Research (Stage 3) and the sleep cycle (Stage 5) both go through here.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.clock import Clock
from core.cognition_config import cognition_config
from core.events import emit
from core.logger import logger
from memory.graph_store import GraphMemory
from memory.usage import UsageTracker
from memory.vector_store import CollectionBudget, VectorMemory, slugify

_ID_PREFIX = {"entity": "ent_", "abstraction": "abs_"}


class KnowledgeStore:
    def __init__(self, vector: VectorMemory, graph: GraphMemory,
                 usage: Optional[UsageTracker] = None, clock: Optional[Clock] = None) -> None:
        self.vector = vector
        self.graph = graph
        self._clock = clock or graph._clock
        self.usage = usage or UsageTracker(graph, clock=self._clock)
        self.layout = None  # LayoutEngine, attached by CognitiveMemory

    def position_of(self, node: Dict[str, Any]) -> Optional[List[float]]:
        if self.layout is None:
            return None
        try:
            return [round(x, 3) for x in self.layout.place(node)]
        except Exception:
            return None

    @property
    def user_id(self) -> str:
        return self.graph.user_id

    # ------------------------------------------------------------------ ids

    @staticmethod
    def node_id_for(label: str, kind: str = "entity") -> str:
        return _ID_PREFIX.get(kind, "ent_") + slugify(label)

    def find_node(self, label: str) -> Optional[Dict[str, Any]]:
        """Exact lookup: alias table first, then the canonical id."""
        for nid in self.graph.resolve_alias(label):
            node = self.graph.get_node(nid)
            if node and node.get("status") != "pruned":
                return node
        for kind in ("entity", "abstraction"):
            node = self.graph.get_node(self.node_id_for(label, kind))
            if node and node.get("status") != "pruned":
                return node
        return None

    # ------------------------------------------------------------------ create

    def ensure_node(
        self,
        label: str,
        kind: str = "entity",
        aliases: Iterable[str] = (),
        budget: Optional[CollectionBudget] = None,
        cycle_id: Optional[str] = None,
        turn_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
    ) -> Tuple[Dict[str, Any], bool]:
        """Return (node, created). Creating a node also creates its collection.

        The collection is created first so a BudgetExceeded leaves nothing
        half-made. A new node starts with a baseline activation (access_count
        = 1) so it is not instantly a prune candidate.
        """
        existing = self.find_node(label)
        if existing:
            for a in aliases:
                self.graph.add_alias(existing["id"], a)
            return existing, False

        node_id = self.node_id_for(label, kind)
        namespace = VectorMemory.knowledge_namespace(label)
        if cycle_id is None and budget is not None:
            cycle_id = budget.cycle_id or None
        self.vector.create_collection(namespace, budget=budget)  # may raise BudgetExceeded

        cfg = cognition_config()
        now = self._clock.now()
        self.graph.upsert_node(node_id, kind=kind, label=label, data=data or {},
                               importance=float(cfg["usage"]["baseline_importance"]))
        self.graph.set_node_fields(node_id, collection=namespace, access_count=1, last_accessed=now,
                                   status="active", confidence=float(confidence))
        self.graph.register_collection(namespace, node_id, kind, cycle_id=cycle_id)
        for a in [label, *aliases]:
            self.graph.add_alias(node_id, a)
        self.vector.index_entity(node_id, label, kind)

        node = self.graph.get_node(node_id) or {"id": node_id, "kind": kind, "label": label}
        score = self.usage.score_node(node)
        self.graph.set_node_fields(node_id, usage_score=score)
        style = self.usage.style_of(node)
        emit("node_spawned", user_id=self.user_id, turn_id=turn_id, node_id=node_id, kind=kind,
             label=label, collection=namespace, style=style, usage_score=round(score, 4),
             position=self.position_of(node))
        logger.info(f"KnowledgeStore: new {kind} node '{label}' -> {namespace}")
        return node, True

    # ------------------------------------------------------------------ records

    def add_record(self, node: Dict[str, Any], text: str, metadata: Optional[Dict[str, Any]] = None,
                   embedding: Optional[List[float]] = None, importance: float = 0.5) -> str:
        namespace = node.get("collection") or VectorMemory.knowledge_namespace(node["label"])
        meta = {"node_id": node["id"], "trust": "untrusted_web", "confidence": 0.5}
        meta.update({k: v for k, v in (metadata or {}).items() if isinstance(v, (str, int, float, bool))})
        kind = str(meta.pop("kind", "claim"))
        rid = self.vector.store(namespace=namespace, text=text, importance=importance, kind=kind,
                                extra_metadata=meta, embedding=embedding, silent=True)
        return rid

    def refresh_stats(self, node: Dict[str, Any]) -> int:
        namespace = node.get("collection")
        if not namespace:
            return 0
        n = self.vector.count(namespace)
        self.graph.update_collection_stats(namespace, record_count=n)
        # node confidence = mean record confidence
        try:
            recs = self.vector.get_all_records(namespace)
            confs = [float(r["metadata"].get("confidence", 0.5)) for r in recs]
            if confs:
                self.graph.set_node_fields(node["id"], confidence=sum(confs) / len(confs))
        except Exception:
            pass
        return n

    # ------------------------------------------------------------------ edges

    def link(self, src: str, dst: str, relation: str, turn_id: str = "") -> Tuple[str, bool]:
        eid, created = self.graph.link_once(src, dst, relation)
        if created:
            emit("edge_drawn", user_id=self.user_id, turn_id=turn_id, edge_id=eid, src=src, dst=dst,
                 relation=relation)
        return eid, created

    # ------------------------------------------------------------------ remove

    def remove_node(self, node_id: str, turn_id: str = "") -> bool:
        """Prune a node and everything bound to it. Pinned nodes are refused."""
        node = self.graph.get_node(node_id)
        if not node or node.get("pinned"):
            return False
        namespace = node.get("collection")
        emit("node_pruned", user_id=self.user_id, turn_id=turn_id, node_id=node_id,
             label=node.get("label"), collection=namespace)
        if namespace:
            still_used = [c for c in self.graph.list_collections() if c["name"] == namespace
                          and c.get("node_id") not in (None, node_id)]
            if not still_used:
                self.vector.drop_collection(namespace)
                self.graph.unregister_collection(namespace)
        self.vector.unindex_entity(node_id)
        # curiosity items about this node go with it
        try:
            with self.graph._get_connection() as conn:
                conn.execute("DELETE FROM curiosity_queue WHERE topic_node_id = ?;", (node_id,))
                conn.commit()
        except Exception:
            pass
        return self.graph.delete_node(node_id)
