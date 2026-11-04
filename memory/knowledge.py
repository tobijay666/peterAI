"""KnowledgeStore — entity/abstraction nodes bound to their own vector collection.

One place that keeps the graph node, its Chroma collection, its aliases, its
entity-index vector and the collection registry consistent:

    ensure_node("Henna")  ->  node 'ent_henna'  +  collection 'peter_k_henna'
                              + alias 'henna'    +  index vector 'Henna'
    remove_node(...)      ->  the inverse, used by pruning

Research (Stage 3) and the sleep cycle (Stage 5) both go through here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.clock import Clock
from core.cognition_config import cognition_config
from core.events import emit
from core.logger import logger
from memory.graph_store import GraphMemory
from memory.usage import UsageTracker
from memory.vector_store import CollectionBudget, VectorMemory, label_key, slugify, topic_slug

_ID_PREFIX = {"entity": "ent_", "abstraction": "abs_"}
SYMMETRIC_RELATIONS = ("related_to", "similar_to", "bridges")
# Topic-topic relations that carry a vector weight (centroid cosine) and record-level links
WEIGHTED_RELATIONS = ("abstracts", "related_to", "bridges", "similar_to", "has_subject")


class KnowledgeStore:
    def __init__(self, vector: VectorMemory, graph: GraphMemory,
                 usage: Optional[UsageTracker] = None, clock: Optional[Clock] = None) -> None:
        self.vector = vector
        self.graph = graph
        self._clock = clock or graph._clock
        self.usage = usage or UsageTracker(graph, clock=self._clock)
        self.layout = None  # LayoutEngine, attached by CognitiveMemory
        self.on_created = None  # callback(node, turn_id) after a node is created (similarity links)
        self.on_linked = None   # callback(src, dst, relation) -> extra edge data (vector weight)
        self.after_linked = None  # callback(edge_id, src, dst, relation) after a weighted edge is stored
        self.on_records_changed = None  # callback(event, node_id, record_ids) (vector links, Part C)
        self.on_refreshed = None  # callback(node) after a batch of writes (refresh_stats)
        # During a batch (sleep bulk phases) the per-edge weight / record-link work and the
        # per-node refresh are skipped — the end-of-sleep VectorGraph.refresh_all does it all
        # once (D3). Chat paths never set these, so weights/links still appear immediately there.
        self._defer_links = False
        self._defer_refresh = False
        # Every write/delete in a knowledge collection (tools, eviction, sleep, research) comes back
        # here: registry counts stay right and copies of deleted records go too (audit items 5, 12).
        vector.add_listener(str(graph.db_path), self._on_vector_change)

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
        """Unique, stable id: 'Henna' -> 'ent_henna'; 'C++' -> 'ent_c_<hash>' (see topic_slug)."""
        return _ID_PREFIX.get(kind, "ent_") + topic_slug(label)

    @staticmethod
    def legacy_node_id_for(label: str, kind: str = "entity") -> str:
        """The id scheme before unique slugs (truncated, symbols dropped). Old nodes keep it."""
        return _ID_PREFIX.get(kind, "ent_") + slugify(label)

    def _same_label(self, node: Dict[str, Any], label: str) -> bool:
        key = label_key(label)
        if label_key(node.get("label") or "") == key:
            return True
        return GraphMemory.normalize_alias(label) in set(self.graph.aliases_of(node["id"]))

    def find_node(self, label: str) -> Optional[Dict[str, Any]]:
        """Exact lookup: alias table first, then the canonical id, then the legacy id.

        A node found by id must really carry this label (old truncated or symbol-stripped
        ids are shared by different labels: 'C' / 'C++' / 'C#')."""
        hits = []
        for nid in self.graph.resolve_alias(label):
            node = self.graph.get_node(nid)
            if node and node.get("status") != "pruned":
                hits.append(node)
        if hits:
            key = label_key(label)
            hits.sort(key=lambda n: (label_key(n.get("label") or "") != key, n["id"]))
            return hits[0]
        for kind in ("entity", "abstraction"):
            for nid in dict.fromkeys((self.node_id_for(label, kind), self.legacy_node_id_for(label, kind))):
                node = self.graph.get_node(nid)
                if node and node.get("status") != "pruned" and self._same_label(node, label):
                    return node
        return None

    def _free_id(self, label: str, kind: str) -> Tuple[str, str]:
        """(node id, namespace) for a new node that collide with nothing already stored."""
        base_id, base_ns = self.node_id_for(label, kind), VectorMemory.knowledge_namespace(label)
        node_id, namespace, i = base_id, base_ns, 1
        while True:
            taken_id = self.graph.get_node(node_id) is not None
            reg = self.graph.get_collection(namespace)
            taken_ns = bool(reg and reg.get("node_id") and reg.get("node_id") != node_id
                            and self.graph.get_node(reg["node_id"]) is not None)
            if not taken_id and not taken_ns:
                return node_id, namespace
            i += 1
            node_id, namespace = f"{base_id}_{i}", f"{base_ns}_{i}"

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
            new_aliases = [a for a in aliases if a and GraphMemory.normalize_alias(a)
                           not in set(self.graph.aliases_of(existing["id"]))]
            for a in new_aliases:
                self.graph.add_alias(existing["id"], a)
            if new_aliases:
                self.index_aliases(existing)
            return existing, False

        node_id, namespace = self._free_id(label, kind)
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
        self.vector.index_entity(node_id, label, kind, aliases=list(aliases))

        node = self.graph.get_node(node_id) or {"id": node_id, "kind": kind, "label": label}
        # NB: "Investing", "investing" and "INVESTING." are one node (same alias key); "C", "C++" and
        # "C#" are three (symbols are part of the key and of the id hash).
        score = self.usage.score_node(node)
        self.graph.set_node_fields(node_id, usage_score=score)
        style = self.usage.style_of(node)
        emit("node_spawned", user_id=self.user_id, turn_id=turn_id, node_id=node_id, kind=kind,
             label=label, collection=namespace, style=style, usage_score=round(score, 4),
             position=self.position_of(node))
        logger.info(f"KnowledgeStore: new {kind} node '{label}' -> {namespace}")
        if self.on_created is not None:
            try:
                self.on_created(node, turn_id)
            except Exception as e:
                logger.debug(f"KnowledgeStore: on_created hook failed for {node_id}: {e}")
        return node, True

    def index_aliases(self, node: Dict[str, Any]) -> None:
        """(Re)index a node's label and every alias in the entity index."""
        self.vector.index_entity(node["id"], node.get("label") or "", node.get("kind", "entity"),
                                 aliases=self.graph.aliases_of(node["id"]))

    def index_all_aliases(self) -> int:
        """Upgrade: index the aliases of every topic (older graphs indexed labels only)."""
        n = 0
        for node in self.graph.all_nodes(["entity", "abstraction"]):
            if node.get("status") == "pruned":
                continue
            own = GraphMemory.normalize_alias(node.get("label") or "")
            if any(a != own for a in self.graph.aliases_of(node["id"])):
                self.index_aliases(node)
                n += 1
        return n

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
        if self.on_refreshed is not None and not self._defer_refresh:
            try:      # edge weights + record-level vector links of this node (Part C)
                self.on_refreshed(self.graph.get_node(node["id"]) or node)
            except Exception as e:
                logger.debug(f"KnowledgeStore: on_refreshed failed for {node.get('id')}: {e}")
        # node confidence = mean record confidence
        try:
            recs = self.vector.get_all_records(namespace)
            confs = [float(r["metadata"].get("confidence", 0.5)) for r in recs]
            if confs:
                self.graph.set_node_fields(node["id"], confidence=sum(confs) / len(confs))
        except Exception:
            pass
        return n

    # ------------------------------------------------------------------ change tracking

    def _on_vector_change(self, event: str, namespace: str, ids: List[str]) -> None:
        if event == "added":
            # Fast path (D3): just bump the count. No registry lookup and no cache pop — the
            # vector-graph caches validate themselves by record count when not frozen, and
            # refresh_stats (always called after an ingest) refreshes this node's edges/links.
            self.graph.bump_collection_count(namespace, len(ids))
            return
        reg = self.graph.get_collection(namespace)
        if reg is None:
            return
        self.graph.update_collection_stats(namespace, record_count=self.vector.count(namespace))
        if event == "deleted":
            self.delete_copies(origin_ids=ids)
        if self.on_records_changed is not None and reg.get("node_id"):
            try:
                self.on_records_changed(event, reg["node_id"], ids)
            except Exception as e:
                logger.debug(f"KnowledgeStore: on_records_changed failed: {e}")

    # ------------------------------------------------------------------ copies (aggregated knowledge)

    @staticmethod
    def origin_of(record: Dict[str, Any]) -> str:
        """The original record a (possibly copied) record stands for."""
        return str((record.get("metadata") or {}).get("origin_id") or record["id"])

    def delete_copies(self, origin_node: Optional[str] = None, origin_ids: Optional[List[str]] = None) -> int:
        """Delete copies (in hub/abstraction collections) of a removed node or of deleted records."""
        if not origin_node and not origin_ids:
            return 0
        if origin_node:
            where: Dict[str, Any] = {"origin_node": origin_node}
        else:
            where = {"origin_id": {"$in": list(origin_ids)}} if len(origin_ids) > 1 else {"origin_id": origin_ids[0]}
        total = 0
        for c in self.graph.list_collections(kind="abstraction"):
            if not self.vector.has_collection(c["name"]):
                continue
            ids = [r["id"] for r in self.vector.get_where(c["name"], where)]
            if ids:
                total += self.vector.delete_ids(c["name"], ids)
        return total

    def copy_records(self, target: Dict[str, Any], source: Dict[str, Any], records: List[Dict[str, Any]],
                     limit: Optional[int] = None) -> int:
        """Copy records of `source` into the hub `target` with provenance (origin_id / origin_node /
        origin_collection), reusing the stored embeddings. A copy of a copy points at the original.
        Records already present (same origin or same text) are skipped. Returns how many were copied."""
        ns = target.get("collection")
        if not ns:
            return 0
        existing = self.vector.get_all_records(ns)
        have = {self.origin_of(r) for r in existing}
        texts = {" ".join((r["document"] or "").lower().split()) for r in existing}
        copied = 0
        for r in records:
            if limit is not None and copied >= limit:
                break
            meta0 = r.get("metadata") or {}
            origin_id = self.origin_of(r)
            text_key = " ".join((r.get("document") or "").lower().split())
            if origin_id in have or text_key in texts or not r.get("embedding"):
                continue
            if meta0.get("origin_node") == target["id"]:
                continue
            meta = {k: v for k, v in meta0.items()
                    if k not in ("access_count", "last_accessed", "created_at", "agent", "node_id")}
            meta.update({"origin_collection": meta0.get("origin_collection") or source.get("collection") or "",
                         "origin_id": origin_id, "origin_node": meta0.get("origin_node") or source["id"],
                         "kind": "migrated"})
            self.add_record(target, r["document"], meta, embedding=r["embedding"])
            have.add(origin_id)
            texts.add(text_key)
            copied += 1
        return copied

    def refresh_copies(self, hub: Dict[str, Any]) -> Dict[str, int]:
        """Sleep: drop copies whose original is gone or no longer below this hub; refresh the
        text of copies whose original changed. Returns {"removed", "updated"}."""
        out = {"removed": 0, "updated": 0}
        ns = hub.get("collection")
        if not ns or not self.vector.has_collection(ns):
            return out
        copies = [r for r in self.vector.get_all_records(ns) if (r.get("metadata") or {}).get("origin_id")]
        if not copies:
            return out
        stale: List[str] = []
        by_coll: Dict[str, List[Dict[str, Any]]] = {}
        below: Dict[str, bool] = {}
        for r in copies:
            m = r["metadata"]
            onode = str(m.get("origin_node") or "")
            node = self.graph.get_node(onode) if onode else None
            if node is None or node.get("status") == "pruned":
                stale.append(r["id"]); continue
            if onode not in below:
                below[onode] = self.is_descendant(onode, of=hub["id"])
            if not below[onode]:
                stale.append(r["id"]); continue
            by_coll.setdefault(str(m.get("origin_collection") or node.get("collection") or ""), []).append(r)
        for coll, recs in by_coll.items():
            if not coll or not self.vector.has_collection(coll):
                stale.extend(r["id"] for r in recs); continue
            originals = {o["id"]: o for o in self.vector.get_records(coll, [r["metadata"]["origin_id"] for r in recs])}
            for r in recs:
                o = originals.get(r["metadata"]["origin_id"])
                if o is None:
                    stale.append(r["id"])
                elif (o["document"] or "") != (r["document"] or ""):
                    stale.append(r["id"])
                    meta = {k: v for k, v in (r["metadata"] or {}).items()
                            if k not in ("access_count", "last_accessed", "created_at", "agent", "node_id")}
                    self.add_record(hub, o["document"], meta, embedding=o.get("embedding") or None)
                    out["updated"] += 1
        if stale:
            out["removed"] = self.vector.delete_ids(ns, stale) - out["updated"]
        return out

    # ------------------------------------------------------------------ edges

    def link(self, src: str, dst: str, relation: str, turn_id: str = "") -> Tuple[str, bool]:
        eid, created, _ = self.link_checked(src, dst, relation, turn_id=turn_id)
        return eid, created

    def link_checked(self, src: str, dst: str, relation: str, turn_id: str = ""
                     ) -> Tuple[Optional[str], bool, str]:
        """(edge_id, created, reason). edge_id is None when the link is refused.

        The memory is a graph, but its part-of relation must stay sound:
          * 'abstracts' (hub -> member) never forms a cycle and never puts a main subject
            (or Peter) under anything; a node may belong to several hubs, and the first
            membership is marked primary (edge data primary=true) - route_to, the layout
            and the recall pulse all follow the primary parent.
          * 'has_subject' only from Peter to a main subject (hierarchy.mark_subject).
          * symmetric links (related_to, similar_to, bridges) are stored once per pair and
            never between a hub and its own member (that is already an 'abstracts' edge).
        """
        from memory.hierarchy import SELF_ID, is_subject
        if src == dst:
            return None, False, "self loop"
        if relation == "abstracts":
            dst_node = self.graph.get_node(dst)
            if dst == SELF_ID or is_subject(dst_node):
                return None, False, "a main subject is never placed under another node"
            existing = self.graph.find_edge(src, dst, "abstracts")
            if existing:
                return existing, False, ""
            if self.is_descendant(src, of=dst):
                return None, False, "would create a cycle in 'abstracts'"
            primary = not any(e["relation"] == "abstracts" and e["dst"] == dst for e in self.graph.edges_of(dst))
            data = {"primary": True} if primary else None
            return self._new_edge(src, dst, relation, data, turn_id)
        if relation == "has_subject" and (src != SELF_ID or not is_subject(self.graph.get_node(dst))):
            return None, False, "has_subject links Peter to a main subject only"
        if relation in SYMMETRIC_RELATIONS:
            for a, b in ((src, dst), (dst, src)):
                existing = self.graph.find_edge(a, b, relation)
                if existing:
                    return existing, False, ""
            if self.graph.find_edge(src, dst, "abstracts") or self.graph.find_edge(dst, src, "abstracts"):
                return None, False, "already linked as hub and member"
        existing = self.graph.find_edge(src, dst, relation)
        if existing:
            return existing, False, ""
        return self._new_edge(src, dst, relation, None, turn_id)

    def _new_edge(self, src: str, dst: str, relation: str, data: Optional[Dict[str, Any]],
                  turn_id: str) -> Tuple[Optional[str], bool, str]:
        data = dict(data or {})
        weighted = relation in WEIGHTED_RELATIONS and not self._defer_links
        if self.on_linked is not None and weighted:
            try:   # vector weight of the new edge (Part C); computed before the event so the view gets it
                data.update(self.on_linked(src, dst, relation) or {})
            except Exception as e:
                logger.debug(f"KnowledgeStore: edge weight for {src}->{dst} failed: {e}")
        eid = self.graph.link(src, dst, relation, data or None)
        emit("edge_drawn", user_id=self.user_id, turn_id=turn_id, edge_id=eid, src=src, dst=dst,
             relation=relation, weight=data.get("weight"), primary=bool(data.get("primary")))
        if self.after_linked is not None and weighted:
            try:   # record-level vector links of the new edge
                self.after_linked(eid, src, dst, relation)
            except Exception as e:
                logger.debug(f"KnowledgeStore: vector links for {eid} failed: {e}")
        return eid, True, ""

    @contextmanager
    def batch(self):
        """Bulk ingest (sleep abstraction, aggregation, merge, background research): buffer the
        vector-store recounts and defer per-edge weight/link work and per-node refresh, so a
        sleep phase does its vector bookkeeping once at the end rather than per record/edge (D3)."""
        with self.vector.suspend_notifications():
            prev_l, prev_r = self._defer_links, self._defer_refresh
            self._defer_links = True
            self._defer_refresh = True
            try:
                yield
            finally:
                self._defer_links, self._defer_refresh = prev_l, prev_r

    def is_descendant(self, node_id: str, of: str, max_nodes: int = 5000) -> bool:
        """True when node_id is reachable from `of` along 'abstracts' edges (of -> ... -> node_id)."""
        if node_id == of:
            return True
        seen, stack = {of}, [of]
        while stack and len(seen) < max_nodes:
            cur = stack.pop()
            for e in self.graph.edges_of(cur):
                if e["relation"] == "abstracts" and e["src"] == cur and e["dst"] not in seen:
                    if e["dst"] == node_id:
                        return True
                    seen.add(e["dst"])
                    stack.append(e["dst"])
        return False

    # ------------------------------------------------------------------ merge

    def merge_nodes(self, keeper: Dict[str, Any], loser: Dict[str, Any], turn_id: str = "",
                    dedupe_similarity: float = 0.9) -> Dict[str, int]:
        """Fold a near-duplicate node into `keeper`: its records (stored embeddings, near-identical
        ones skipped), its edges (re-pointed through the validated link rules), its label and
        aliases (indexed), its curiosity items. Then the duplicate is removed."""
        out = {"records": 0, "edges": 0, "aliases": 0}
        kid, lid = keeper["id"], loser["id"]
        # 1. records
        if loser.get("collection") and keeper.get("collection"):
            import numpy as np
            have = [r for r in self.vector.get_all_records(keeper["collection"]) if r.get("embedding")]
            H = np.asarray([r["embedding"] for r in have], dtype=float) if have else None
            if H is not None:
                H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
            for r in self.vector.get_all_records(loser["collection"]):
                if not r.get("embedding"):
                    continue
                v = np.asarray(r["embedding"], dtype=float)
                v /= max(float(np.linalg.norm(v)), 1e-12)
                if H is not None and len(H) and float(np.max(H @ v)) >= dedupe_similarity:
                    continue
                meta = {k: v2 for k, v2 in (r.get("metadata") or {}).items()
                        if k not in ("access_count", "last_accessed", "created_at", "agent", "node_id")}
                meta["merged_from"] = lid
                self.add_record(keeper, r["document"], meta, embedding=r["embedding"])
                H = v[None, :] if H is None or not len(H) else np.vstack([H, v[None, :]])
                out["records"] += 1
        # 2. curiosity items follow (before remove_node would delete them)
        with self.graph._get_connection() as conn:
            conn.execute("UPDATE curiosity_queue SET topic_node_id = ? WHERE topic_node_id = ?;", (kid, lid))
            conn.commit()
        # 3. edges
        from memory.hierarchy import ensure_primary
        loser_edges = self.graph.edges_of(lid)
        for e in loser_edges:
            src = kid if e["src"] == lid else e["src"]
            dst = kid if e["dst"] == lid else e["dst"]
            if src == dst:
                continue
            eid, created, _ = self.link_checked(src, dst, e["relation"], turn_id=turn_id)
            if created:
                out["edges"] += 1
            self.graph.delete_edge(e["id"])     # re-pointed: the duplicate keeps nothing to cascade
        for e in loser_edges:
            if e["relation"] == "abstracts":
                ensure_primary(self.graph, e["dst"] if e["src"] == lid else kid)
        # 4. names
        for a in [loser.get("label") or ""] + self.graph.aliases_of(lid):
            if a and GraphMemory.normalize_alias(a) not in set(self.graph.aliases_of(kid)):
                self.graph.add_alias(kid, a)
                out["aliases"] += 1
        self.graph.set_node_fields(kid, access_count=int(keeper.get("access_count") or 0)
                                   + int(loser.get("access_count") or 0))
        self.remove_node(lid, turn_id=turn_id)
        fresh = self.graph.get_node(kid) or keeper
        self.index_aliases(fresh)
        self.refresh_stats(fresh)
        emit("nodes_merged", user_id=self.user_id, turn_id=turn_id, keeper=kid, merged=lid,
             label=keeper.get("label"), merged_label=loser.get("label"), **out)
        return out

    # ------------------------------------------------------------------ remove

    def remove_node(self, node_id: str, turn_id: str = "") -> bool:
        """Prune a node and everything bound to it. Pinned nodes are refused."""
        node = self.graph.get_node(node_id)
        if not node or node.get("pinned"):
            return False
        namespace = node.get("collection")
        emit("node_pruned", user_id=self.user_id, turn_id=turn_id, node_id=node_id,
             label=node.get("label"), collection=namespace)
        # copies of its records in the hubs above it are stale now (audit item 5)
        try:
            self.delete_copies(origin_node=node_id)
        except Exception as e:
            logger.debug(f"KnowledgeStore: copy cleanup for {node_id} failed: {e}")
        if namespace:
            still_used = [c for c in self.graph.list_collections() if c["name"] == namespace
                          and c.get("node_id") not in (None, node_id)]
            if not still_used:
                self.vector.drop_collection(namespace)
                self.graph.unregister_collection(namespace)
        self.vector.unindex_entity(node_id)
        children = [e["dst"] for e in self.graph.edges_of(node_id)
                    if e["relation"] == "abstracts" and e["src"] == node_id]
        # curiosity items about this node go with it: the queue rows AND their question nodes
        # (otherwise they stay as orphan nodes that attach_orphans would hang under a subject)
        try:
            cur_nodes = {e["src"] for e in self.graph.edges_of(node_id)
                         if e["relation"] == "curious_about" and e["dst"] == node_id}
            with self.graph._get_connection() as conn:
                rows = conn.execute("SELECT id FROM curiosity_queue WHERE topic_node_id = ?;", (node_id,)).fetchall()
                cur_nodes |= {f"cur_{r['id']}" for r in rows}
                conn.execute("DELETE FROM curiosity_queue WHERE topic_node_id = ?;", (node_id,))
                conn.commit()
            for cid in sorted(cur_nodes):
                cn = self.graph.get_node(cid)
                if cn and cn.get("kind") == "curiosity" and self.graph.delete_node(cid):
                    emit("node_pruned", user_id=self.user_id, turn_id=turn_id, node_id=cid,
                         label=cn.get("label"), collection=None)
        except Exception as e:
            logger.debug(f"KnowledgeStore: curiosity cleanup for {node_id} failed: {e}")
        ok = self.graph.delete_node(node_id)
        if ok and children:
            from memory.hierarchy import ensure_primary
            for c in children:          # a member that lost its primary hub gets another one
                ensure_primary(self.graph, c)
        return ok
