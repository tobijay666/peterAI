"""VectorGraph — edges grounded in the vectors (Part C).

Every topic node stands for a vector:
  * its collection's centroid (mean of its L2-normalised record embeddings), or
  * its label embedding while the collection is still empty,
  * Peter himself: the normalised mean of his main subjects' vectors.

and every topic-topic edge carries what the vectors say about it:
  * edge data ``weight`` = cosine of the two node vectors (``basis`` "centroid" when both
    sides have records, else "label"), set when the edge is drawn and recomputed in sleep;
  * record-level vector links (SQLite ``vector_links``): the top-k most similar record pairs
    across the two collections, used by recall to follow an edge record-to-record and by
    the mind view to pulse along it;
  * ``similar_to`` edges are the k nearest neighbours by node vector, refreshed every sleep
    with hysteresis (added at >= ``vectors.similar_add_min``, removed below
    ``vectors.similar_remove_below``).

Pure numpy over stored embeddings: deterministic with the test embedders.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.cognition_config import cognition_config

TOPIC_KINDS = ["entity", "abstraction"]
WEIGHTED = ("abstracts", "related_to", "bridges", "similar_to", "has_subject")


def _unit(v: Any) -> Optional[np.ndarray]:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else None


class VectorGraph:
    def __init__(self, vector: Any, graph: Any, knowledge: Any = None) -> None:
        self.vector = vector
        self.graph = graph
        self.knowledge = knowledge
        self._label_vecs: Dict[str, np.ndarray] = {}
        self._cent: Dict[str, Tuple[int, Optional[np.ndarray]]] = {}   # node id -> (record count, centroid)
        # "frozen" (trust the centroid cache without re-counting) is per THREAD: a sleep refresh
        # freezes only its own thread, so a concurrent research thread still sees fresh counts (D7).
        self._tl = threading.local()
        self._self_cache: Optional[np.ndarray] = None
        self._rec_cache: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
        # serialises the write batch of a sleep refresh with concurrent record-deletion cleanup,
        # so a link is never re-inserted for a record a deleter removed in the same instant (D7).
        self._lock = threading.RLock()

    @property
    def _frozen(self) -> bool:
        return bool(getattr(self._tl, "frozen", False))

    @_frozen.setter
    def _frozen(self, value: bool) -> None:
        self._tl.frozen = bool(value)

    @property
    def cfg(self) -> Dict[str, Any]:
        return cognition_config()["vectors"]

    # ------------------------------------------------------------------ node vectors

    def label_vec(self, label: str) -> Optional[np.ndarray]:
        key = label or ""
        if key not in self._label_vecs:
            try:
                self._label_vecs[key] = _unit(self.vector.embed([key])[0])
            except Exception:
                self._label_vecs[key] = None
        return self._label_vecs[key]

    def label_vecs(self, labels: List[str]) -> List[Optional[np.ndarray]]:
        missing = sorted({l or "" for l in labels if (l or "") not in self._label_vecs})
        if missing:
            try:
                for l, v in zip(missing, self.vector.embed(missing)):
                    self._label_vecs[l] = _unit(v)
            except Exception:
                for l in missing:
                    self._label_vecs[l] = None
        return [self._label_vecs.get(l or "") for l in labels]

    def centroid(self, node: Dict[str, Any], records: Optional[List[Dict[str, Any]]] = None) -> Optional[np.ndarray]:
        """Collection centroid, cached by record count (cheap to ask again)."""
        ns = node.get("collection")
        if not ns or not self.vector.has_collection(ns):
            return None
        hit = self._cent.get(node["id"])
        if self._frozen and hit is not None and records is None:
            return hit[1]
        n = self.vector.count(ns) if records is None else len(records)
        if hit is not None and hit[0] == n:
            return hit[1]
        c = None
        if n and records is None and hit is None:
            # warm cache: a centroid stored last sleep (node_vectors) saves reading every record
            # on the first access after a restart, when the record count still matches (D5)
            stored = self.graph.get_node_vector(node["id"])
            if stored and stored.get("basis") == "centroid" and int(stored.get("n") or -1) == n:
                v = _unit(np.asarray(stored["vector"], dtype=float))
                if v is not None:
                    self._cent[node["id"]] = (n, v)
                    return v
        if n:
            recs = records if records is not None else self.vector.get_all_records(ns)
            vecs = [r["embedding"] for r in recs if r.get("embedding")]
            if vecs:
                arr = np.asarray(vecs, dtype=float)
                arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
                c = _unit(arr.mean(axis=0))
        self._cent[node["id"]] = (n, c)
        if c is not None:
            try:
                self.graph.set_node_vector(node["id"], c.tolist(), "centroid", n)
            except Exception:
                pass
        return c

    def node_vector(self, node: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
        """(vector, basis): centroid when the node has records, else its label embedding."""
        from memory.hierarchy import SELF_ID
        if node["id"] == SELF_ID:
            v = self.self_vector()
            return v, "subjects"
        c = self.centroid(node)
        if c is not None:
            return c, "centroid"
        return self.label_vec(node.get("label") or ""), "label"

    def self_vector(self, store: bool = False) -> Optional[np.ndarray]:
        """Peter's vector: the normalised mean of his main subjects' vectors."""
        from memory.hierarchy import SELF_ID
        if self._frozen and getattr(self, "_self_cache", None) is not None and not store:
            return self._self_cache
        subs = [self.graph.get_node(e["dst"]) for e in self.graph.edges_of(SELF_ID)
                if e["relation"] == "has_subject" and e["src"] == SELF_ID]
        vecs = [self.node_vector(s)[0] for s in subs if s]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return None
        v = _unit(np.mean(np.asarray(vecs), axis=0))
        self._self_cache = v
        if store and v is not None:
            self.graph.set_node_vector(SELF_ID, v.tolist(), "subjects", len(vecs))
        return v

    def pair_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, str]:
        """Cosine of two nodes. Centroids when both have records; otherwise their labels
        (the label is the only vector an empty collection has)."""
        from memory.hierarchy import SELF_ID
        if SELF_ID in (a["id"], b["id"]):
            other = b if a["id"] == SELF_ID else a
            sv, ov = self.self_vector(), self.node_vector(other)[0]
            if sv is None or ov is None:
                return 0.0, "subjects"
            return float(sv @ ov), "subjects"
        ca, cb = self.centroid(a), self.centroid(b)
        if ca is not None and cb is not None:
            return float(ca @ cb), "centroid"
        la, lb = self.label_vec(a.get("label") or ""), self.label_vec(b.get("label") or "")
        if la is None or lb is None:
            return 0.0, "label"
        return float(la @ lb), "label"

    # ------------------------------------------------------------------ edge weights

    def weight_data(self, src: str, dst: str, relation: str) -> Dict[str, Any]:
        """KnowledgeStore.on_linked hook: data for a new weighted edge."""
        a, b = self.graph.get_node(src), self.graph.get_node(dst)
        if not a or not b:
            return {}
        w, basis = self.pair_similarity(a, b)
        return {"weight": round(w, 4), "basis": basis}

    def refresh_edge(self, edge: Dict[str, Any], nodes: Optional[Dict[str, Dict[str, Any]]] = None,
                     records: Optional[Dict[str, List[Dict[str, Any]]]] = None
                     ) -> Tuple[Dict[str, Any], Optional[List[Tuple[str, str, str, str, float]]]]:
        """One edge's new data (weight) and its record-level link rows (None: no links for
        this relation). Nothing is written; callers batch the writes."""
        get = (lambda i: nodes.get(i)) if nodes is not None else self.graph.get_node
        a, b = get(edge["src"]), get(edge["dst"])
        data = dict(edge.get("data") or {})
        if not a or not b:
            return data, None
        w, basis = self.pair_similarity(a, b)
        data.update({"weight": round(w, 4), "basis": basis})
        rows = self.record_pairs(a, b, records=records) if edge["relation"] != "has_subject" else None
        return data, rows

    def _write(self, results: List[Tuple[str, Dict[str, Any], Optional[List[Any]]]]) -> None:
        # the lock serialises this with on_records_changed's link cleanup (D7)
        with self._lock:
            self.graph.set_edge_data_many([(eid, data) for eid, data, _ in results])
            self.graph.replace_vector_links_many({eid: rows for eid, _, rows in results if rows is not None})

    # ------------------------------------------------------------------ record-level links

    def _records(self, node: Dict[str, Any], cache: Optional[Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """A node's records with embeddings. Cached across calls by record count, so refreshing
        the edges of a well-connected node does not re-read every neighbour collection."""
        if cache is not None and node["id"] in cache:
            return cache[node["id"]]
        ns = node.get("collection")
        if not ns or not self.vector.has_collection(ns):
            recs: List[Dict[str, Any]] = []
        else:
            hit = self._rec_cache.get(node["id"])
            n = hit[0] if (self._frozen and hit is not None) else self.vector.count(ns)
            if hit is not None and hit[0] == n:
                recs = hit[1]
            else:
                recs = [r for r in self.vector.get_all_records(ns) if r.get("embedding")] if n else []
                if len(self._rec_cache) > 512:
                    self._rec_cache.clear()
                self._rec_cache[node["id"]] = (n, recs)
        if cache is not None:
            cache[node["id"]] = recs
        return recs

    def record_pairs(self, a: Dict[str, Any], b: Dict[str, Any],
                     records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> List[Tuple[str, str, str, str, float]]:
        """The top-k most similar record pairs across a's and b's collections (a copy is
        never paired with its own original): rows (a, b, record_a, record_b, similarity)."""
        from memory.knowledge import KnowledgeStore
        k = int(self.cfg.get("record_links_k", 3))
        floor = float(self.cfg.get("record_link_min", 0.3))
        ra, rb = self._records(a, records), self._records(b, records)
        rows: List[Tuple[str, str, str, str, float]] = []
        if k > 0 and ra and rb:
            A = np.asarray([r["embedding"] for r in ra], dtype=float)
            B = np.asarray([r["embedding"] for r in rb], dtype=float)
            A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
            B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
            S = A @ B.T
            oa = [KnowledgeStore.origin_of(r) for r in ra]
            ob = [KnowledgeStore.origin_of(r) for r in rb]
            ta = [" ".join((r["document"] or "").lower().split()) for r in ra]
            tb = [" ".join((r["document"] or "").lower().split()) for r in rb]
            for flat in np.argsort(-S, axis=None, kind="stable"):
                i, j = divmod(int(flat), S.shape[1])
                sim = float(S[i, j])
                if sim < floor or len(rows) >= k:
                    break
                if oa[i] == ob[j] or ta[i] == tb[j]:
                    continue        # a hub's copy and its original are the same fact, not a link
                rows.append((a["id"], b["id"], ra[i]["id"], rb[j]["id"], round(sim, 4)))
        return rows

    def link_records(self, edge_id: str, a: Dict[str, Any], b: Dict[str, Any],
                     records: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> int:
        """Store the record-level links of one (new) edge. Returns how many were stored."""
        rows = self.record_pairs(a, b, records=records)
        self.graph.replace_vector_links(edge_id, rows)
        return len(rows)

    def on_records_changed(self, event: str, node_id: str, ids: List[str]) -> None:
        """KnowledgeStore hook: deleted records take their vector links with them."""
        self._rec_cache.pop(node_id, None)
        self._cent.pop(node_id, None)
        if event == "deleted":
            with self._lock:
                self.graph.delete_vector_links_for_records(ids)

    def refresh_node(self, node: Dict[str, Any]) -> int:
        """After a batch of writes to one node (research, aggregation): its edges' weights and
        record links. Returns the number of edges refreshed."""
        self._cent.pop(node["id"], None)
        edges = [e for e in self.graph.edges_of(node["id"]) if e["relation"] in WEIGHTED]
        records: Dict[str, List[Dict[str, Any]]] = {}
        results = []
        for e in edges:
            data, rows = self.refresh_edge(e, records=records)
            results.append((e["id"], data, rows))
        self._write(results)
        return len(results)

    # ------------------------------------------------------------------ similar_to (kNN by vector)

    def link_similar(self, node: Dict[str, Any], turn_id: str = "") -> int:
        """Store similar_to links from node to its nearest topics (>= similar_add_min, at most
        similar_k). Candidates come from the entity index (one query, like the rest of recall)
        rather than an O(N) scan of every node, so a new topic is cheap to create (D3); the
        sleep cycle's refresh_all then does the authoritative centroid kNN with hysteresis.
        Main subjects and Peter are structure, never linked this way."""
        from memory.hierarchy import is_subject
        cfg = self.cfg
        add_min = float(cfg.get("similar_add_min", 0.6))
        k = int(cfg.get("similar_k", 3))
        if k <= 0 or node.get("kind") not in TOPIC_KINDS or is_subject(node) or self.knowledge is None:
            return 0
        label = (node.get("label") or "").strip()
        try:
            hits = self.vector.search_entities(label, n_results=k + 6)
        except Exception:
            return 0
        made = 0
        for h in hits:
            if made >= k or float(h.get("score", 0.0)) < add_min:
                break
            other = self.graph.get_node(h["node_id"])
            if (not other or other["id"] == node["id"] or other.get("status") == "pruned"
                    or is_subject(other) or other.get("kind") not in TOPIC_KINDS):
                continue
            if (other.get("label") or "").strip().lower() == label.lower():
                continue
            a, b = sorted((node["id"], other["id"]))     # one undirected link per pair
            eid, created, _ = self.knowledge.link_checked(a, b, "similar_to", turn_id=turn_id)
            if created:
                made += 1
        return made

    # ------------------------------------------------------------------ sleep: everything

    def refresh_all(self) -> Dict[str, int]:
        """Sleep: recompute every node vector, every weighted edge's weight and record links,
        Peter's vector, and the similar_to kNN with hysteresis."""
        from memory.hierarchy import SELF_ID, is_subject
        self._cent.clear()
        self._rec_cache.clear()
        cfg = self.cfg
        out = {"edges": 0, "links": 0, "similar_added": 0, "similar_removed": 0}
        nodes = {n["id"]: n for n in self.graph.all_nodes(TOPIC_KINDS) if n.get("status") != "pruned"}
        me = self.graph.get_node(SELF_ID)
        if me:
            nodes[SELF_ID] = me
        records: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes.values():
            if n["id"] != SELF_ID:
                self.centroid(n, records=self._records(n, records))
        self.self_vector(store=True)

        # similar_to: kNN over node vectors, hysteresis
        add_min = float(cfg.get("similar_add_min", 0.6))
        drop_below = float(cfg.get("similar_remove_below", 0.45))
        k = int(cfg.get("similar_k", 3))
        topics = sorted((n for nid, n in nodes.items() if nid != SELF_ID and not is_subject(n)), key=lambda n: n["id"])
        abstracts = set(self.graph.edge_pairs("abstracts"))
        existing = {}
        for e in self.graph.list_edges():
            if e["relation"] == "similar_to":
                existing[tuple(sorted((e["src"], e["dst"])))] = e
        if topics:
            ids = [n["id"] for n in topics]
            C = [self._cent.get(i, (0, None))[1] for i in ids]
            L = self.label_vecs([n.get("label") or "" for n in topics])
            dim = next((len(v) for v in list(C) + list(L) if v is not None), 0)
            if dim:
                Z = np.zeros(dim)
                Cm = np.asarray([c if c is not None else Z for c in C])
                Lm = np.asarray([l if l is not None else Z for l in L])
                has_c = np.asarray([c is not None for c in C])
                S = np.where(np.outer(has_c, has_c), Cm @ Cm.T, Lm @ Lm.T)
                np.fill_diagonal(S, -1.0)
                labels = [(n.get("label") or "").strip().lower() for n in topics]
                wanted = set()
                for i in range(len(ids)):
                    order = np.argsort(-S[i], kind="stable")[: max(k, 0)]
                    for j in order:
                        j = int(j)
                        if S[i, j] < add_min or labels[i] == labels[j]:
                            continue
                        if (ids[i], ids[j]) in abstracts or (ids[j], ids[i]) in abstracts:
                            continue
                        wanted.add(tuple(sorted((ids[i], ids[j]))))
                index = {nid: i for i, nid in enumerate(ids)}
                for pair, e in list(existing.items()):
                    i, j = index.get(pair[0]), index.get(pair[1])
                    if i is None or j is None:
                        continue
                    if S[i, j] < drop_below:
                        self.graph.delete_edge(e["id"])
                        existing.pop(pair)
                        out["similar_removed"] += 1
                for pair in sorted(wanted):
                    if pair not in existing and self.knowledge is not None:
                        eid, created, _ = self.knowledge.link_checked(pair[0], pair[1], "similar_to")
                        if created:
                            out["similar_added"] += 1

        # weights + record links of every weighted edge between live nodes (one write batch)
        results = []
        self._frozen = True
        try:
            for e in self.graph.list_edges():
                if e["relation"] not in WEIGHTED or e["src"] not in nodes or e["dst"] not in nodes:
                    continue
                data, rows = self.refresh_edge(e, nodes=nodes, records=records)
                results.append((e["id"], data, rows))
        finally:
            self._frozen = False
        self._write(results)
        out["edges"] = len(results)
        out["links"] = self.graph.count_vector_links()
        return out
