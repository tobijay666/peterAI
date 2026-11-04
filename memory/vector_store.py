"""ChromaDB namespaced vector store implementation."""

import hashlib
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from core.events import emit
from core.logger import logger
from core.clock import Clock, SystemClock
from memory.embeddings import LocalEmbeddingFunction, is_semantic


class BudgetExceeded(Exception):
    """Raised when a CollectionBudget has no new collections left."""


class CollectionBudget:
    """Caps how many NEW vector collections one sleep cycle may create (spec: max 5).

    Passed into VectorMemory.create_collection(); the store enforces the cap so
    callers cannot forget it. Re-opening an existing collection costs nothing.
    """

    def __init__(self, max_new: int = 5, cycle_id: str = "") -> None:
        self.max_new = int(max_new)
        self.cycle_id = cycle_id
        self.created: List[str] = []

    @property
    def remaining(self) -> int:
        return max(0, self.max_new - len(self.created))

    def can_create(self) -> bool:
        return self.remaining > 0

    def consume(self, namespace: str) -> None:
        if not self.can_create():
            raise BudgetExceeded(
                f"Collection budget exhausted ({self.max_new} new collections this cycle); "
                f"refused '{namespace}'."
            )
        self.created.append(namespace)


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SLUG_MAX = 40
# Characters that only separate words: dropping them from a slug loses nothing.
# Anything else ("+", "#", "%", "@", non-ASCII letters...) carries meaning ("C" / "C++" / "C#").
_SEPARATORS = set(" \t\r\n.,!?;:'\"-_()[]{}/&\u2019\u2018\u201c\u201d")


def slugify(label: str, max_len: int = _SLUG_MAX) -> str:
    """'Wedding Cultures!' -> 'wedding_cultures' (Chroma-safe, lower-case). Lossy: see topic_slug."""
    slug = _SLUG_RE.sub("_", (label or "").lower()).strip("_")
    slug = slug[:max_len].strip("_")
    return slug or "topic"


def label_key(label: str) -> str:
    """Canonical text of a label that keeps meaningful symbols: 'C++' -> 'c++', 'Investing.' -> 'investing'."""
    return " ".join(re.findall(r"[a-z0-9]+[+#]*|[+#]+", (label or "").lower()))


def slug_is_lossy(label: str) -> bool:
    """True when two different labels could share the plain slug: it was truncated, or the
    label has characters the slug drops that are not mere separators."""
    full = _SLUG_RE.sub("_", (label or "").lower()).strip("_")
    if len(full) > _SLUG_MAX:
        return True
    return any(not (ch.isascii() and ch.isalnum()) and ch not in _SEPARATORS for ch in (label or ""))


def topic_slug(label: str) -> str:
    """Unique, stable slug for a topic label, used for node ids and collection names.

    Plain labels keep the readable slug ('Henna' -> 'henna', unchanged from earlier
    versions). Lossy ones get a short hash of the canonical label appended
    ('C++' -> 'c_3a1f09b2', 'International Monetary Fund lending ... Asia' ->
    'international_monetary_fund_len_5c0e7d21'), so they never collide.
    """
    slug = slugify(label)
    if not slug_is_lossy(label):
        return slug
    digest = hashlib.sha1((label_key(label) or (label or "").strip().lower()).encode("utf-8")).hexdigest()[:8]
    return f"{slug[:_SLUG_MAX - 9].strip('_') or 'topic'}_{digest}"


class VectorMemory:
    """Manages persistent namespaced vector memory collections using ChromaDB.

    Two families of collections live side by side:
      * agent namespaces (DEFAULT_NAMESPACES) — unchanged from v1
      * knowledge collections, one per entity/abstraction node, named
        ``peter_k_<slug>`` (namespace ``k_<slug>``), created on demand.
    Plus one entity index (``peter_index_entities``) holding one vector per
    entity/abstraction label, so the resolver finds a node by meaning.
    """

    KNOWLEDGE_PREFIX = "k_"
    INDEX_NAMESPACE = "index_entities"

    DEFAULT_NAMESPACES = [
        "peter",
        "researcher",
        "filesystem",
        "librarian",
        "devops",
        "qa",
        "voice",
    ]

    def __init__(self, persist_directory: str = "data/chroma", embedding_model: str = "all-MiniLM-L6-v2",
                 clock: Optional[Clock] = None, user_id: str = "local", embedding_function: Any = None):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embed_fn = embedding_function if embedding_function is not None else LocalEmbeddingFunction(embedding_model)
        self._clock = clock or SystemClock()
        self.user_id = user_id

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collections: Dict[str, Any] = {}
        # Change listeners (KnowledgeStore): fn(event, namespace, ids) with event "added" | "deleted".
        # Every write/delete path notifies them, so registry counts, migrated copies and
        # record-level vector links stay consistent whoever deletes (tools, eviction, sleep).
        self._listeners: Dict[str, Any] = {}
        self._suspended: Optional[set] = None   # set of namespaces with buffered "added" events

        for ns in self.DEFAULT_NAMESPACES:
            self._get_or_create_collection(ns)

        # Migrate legacy Emma knowledge database if present
        self.migrate_legacy_db()

    def migrate_legacy_db(self, legacy_db_path: str = "db/emma_knowledge.db") -> int:
        """Migrates findings from legacy SQLite Emma DB into librarian namespace (Section 9.4)."""
        legacy_path = Path(legacy_db_path)
        if not legacy_path.exists():
            return 0

        logger.info(f"VectorMemory: Migrating legacy knowledge database from {legacy_path}...")
        migrated_count = 0
        try:
            conn = sqlite3.connect(str(legacy_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='findings';")
            if not cursor.fetchone():
                conn.close()
                return 0

            cursor.execute("SELECT * FROM findings;")
            rows = cursor.fetchall()
            for r in rows:
                content = r["content"] if "content" in r.keys() else str(dict(r))
                file_path = r["file_path"] if "file_path" in r.keys() else ""
                self.store(
                    namespace="librarian",
                    text=content,
                    importance=0.6,
                    pinned=False,
                    kind="legacy_finding",
                    tags=f"migrated,legacy,{file_path}",
                )
                migrated_count += 1
            conn.close()
            logger.info(f"VectorMemory: Successfully migrated {migrated_count} legacy findings to 'librarian' namespace.")
        except Exception as e:
            logger.error(f"VectorMemory: Legacy migration error: {e}")
        return migrated_count

    def _get_or_create_collection(self, namespace: str):
        if namespace not in self.collections:
            coll = self.client.get_or_create_collection(
                name=f"peter_{namespace}",
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.collections[namespace] = coll
        return self.collections[namespace]

    def store(
        self,
        namespace: str,
        text: str,
        importance: float = 0.5,
        pinned: bool = False,
        kind: str = "finding",
        tags: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        silent: bool = False,
    ) -> str:
        """Stores a text record in the specified namespace.

        ``embedding`` stores a precomputed vector (used when migrating records
        between collections, so nothing is re-embedded). ``silent`` skips the
        mind-view event (bulk writes emit their own summary event).
        """
        coll = self._get_or_create_collection(namespace)
        doc_id = str(uuid.uuid4())
        now = self._clock.now()

        meta = {
            "agent": namespace,
            "created_at": float(now),
            "last_accessed": float(now),
            "access_count": int(1),
            "importance": float(importance),
            "pinned": bool(pinned),
            "kind": str(kind),
            "tags": str(tags),
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        if embedding is not None:
            coll.add(documents=[text], metadatas=[meta], ids=[doc_id], embeddings=[list(map(float, embedding))])
        else:
            coll.add(documents=[text], metadatas=[meta], ids=[doc_id])
        logger.debug(f"VectorMemory: Stored record [{doc_id}] in namespace '{namespace}'.")
        self._notify("added", namespace, [doc_id])
        if not silent:
            emit("store", user_id=self.user_id, namespace=namespace, memory_id=doc_id, kind=kind)
        return doc_id

    def recall(
        self,
        namespace: str,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
        touch: bool = True,
        silent: bool = False,
    ) -> List[Dict[str, Any]]:
        """Recalls top-k matching documents from a namespace, updating hit metrics.

        ``query_embedding`` searches by a precomputed vector (e.g. a centroid);
        ``touch=False`` reads without counting it as an access (audits, sleep).
        """
        coll = self._get_or_create_collection(namespace)
        if query_embedding is not None:
            kwargs = {"query_embeddings": [list(map(float, query_embedding))],
                      "n_results": min(n_results, max(1, coll.count()))}
        else:
            kwargs = {"query_texts": [query], "n_results": min(n_results, max(1, coll.count()))}
        if where_filter:
            kwargs["where"] = where_filter

        if coll.count() == 0:
            return []

        try:
            res = coll.query(**kwargs)
        except Exception as e:
            logger.error(f"VectorMemory: Recall error in '{namespace}': {e}")
            return []

        results = []
        if not res or not res.get("documents"):
            return results

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        ids = res["ids"][0]
        distances = res.get("distances", [[]])[0]

        now = self._clock.now()
        for i in range(len(docs)):
            doc_id = ids[i]
            meta = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else 0.0

            # Update access statistics
            if touch:
                new_access_count = int(meta.get("access_count", 0)) + 1
                meta["access_count"] = new_access_count
                meta["last_accessed"] = float(now)

                try:
                    coll.update(ids=[doc_id], metadatas=[meta])
                except Exception as e:
                    logger.warning(f"Failed updating access stats for {doc_id}: {e}")

            results.append(
                {
                    "id": doc_id,
                    "document": docs[i],
                    "metadata": meta,
                    "score": 1.0 - float(dist),
                }
            )

        if results and not silent:
            emit("recall", user_id=self.user_id, namespace=namespace,
                 memory_ids=[r["id"] for r in results], count=len(results))
        return results

    def pin(self, namespace: str, doc_id: str, pinned: bool = True) -> bool:
        """Sets the pinned flag for a document to prevent eviction."""
        coll = self._get_or_create_collection(namespace)
        try:
            res = coll.get(ids=[doc_id])
            if res and res.get("metadatas"):
                meta = res["metadatas"][0]
                meta["pinned"] = pinned
                coll.update(ids=[doc_id], metadatas=[meta])
                return True
        except Exception as e:
            logger.error(f"VectorMemory: Pin error for {doc_id}: {e}")
        return False

    def prune(self, namespace: str, doc_ids: List[str]) -> int:
        """Deletes specific documents by ID from a namespace unless pinned."""
        coll = self._get_or_create_collection(namespace)
        if not doc_ids:
            return 0

        to_delete = []
        try:
            records = coll.get(ids=doc_ids)
            if records and records.get("metadatas"):
                for i, meta in enumerate(records["metadatas"]):
                    if not meta.get("pinned", False):
                        to_delete.append(records["ids"][i])
                    else:
                        logger.info(f"VectorMemory: Skipped pruning pinned record {records['ids'][i]}")

            if to_delete:
                coll.delete(ids=to_delete)
                logger.info(f"VectorMemory: Pruned {len(to_delete)} records from '{namespace}'.")
                self._notify("deleted", namespace, to_delete)
            return len(to_delete)
        except Exception as e:
            logger.error(f"VectorMemory: Prune failed in '{namespace}': {e}")
            return 0

    def get_all_records(self, namespace: str) -> List[Dict[str, Any]]:
        coll = self._get_or_create_collection(namespace)
        try:
            if coll.count() == 0:
                return []
            res = coll.get(include=["documents", "metadatas", "embeddings"])
        except Exception as e:
            logger.warning(f"VectorMemory: get_all_records on '{namespace}' failed: {e}")
            return []

        records = []
        # ChromaDB returns embeddings as a numpy array or None — never use
        # "or []" on it (numpy raises ValueError for multi-element truth test)
        raw_embeddings = res.get("embeddings")
        embeddings = raw_embeddings if raw_embeddings is not None else []
        for i in range(len(res["ids"])):
            try:
                emb = list(embeddings[i]) if i < len(embeddings) else []
            except (TypeError, IndexError):
                emb = []
            records.append(
                {
                    "id": res["ids"][i],
                    "document": res["documents"][i],
                    "metadata": res["metadatas"][i],
                    "embedding": emb,
                }
            )
        return records

    def adjust_importance(
        self,
        namespace: str,
        doc_id: str,
        delta: float,
    ) -> Optional[float]:
        """Adjust importance of a document by delta, clamped to [0, 1].
        Pinned records are never modified (B4, FR-12.3).
        Returns the new importance value, or None if the record is not found or is pinned.
        """
        coll = self._get_or_create_collection(namespace)
        try:
            res = coll.get(ids=[doc_id])
            if not res or not res.get("metadatas"):
                return None
            meta = res["metadatas"][0]
            if meta.get("pinned", False):
                logger.debug(f"VectorMemory: Skipping importance adjustment for pinned record {doc_id}")
                return None
            current = float(meta.get("importance", 0.5))
            new_val = max(0.0, min(1.0, current + delta))
            meta["importance"] = new_val
            coll.update(ids=[doc_id], metadatas=[meta])
            return new_val
        except Exception as e:
            logger.error(f"VectorMemory.adjust_importance failed for {doc_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Change notification + bulk helpers
    # ------------------------------------------------------------------

    def add_listener(self, key: str, fn: Any) -> None:
        """Register fn(event, namespace, ids); one listener per key (re-registering replaces it)."""
        self._listeners[str(key)] = fn

    def _notify(self, event: str, namespace: str, ids: List[str]) -> None:
        if not ids or not self.is_knowledge_namespace(namespace):
            return
        # Bulk insert (sleep abstraction, aggregation, research): buffer the per-record "added"
        # notifications and flush one recount per namespace at the end (D3). Deletes stay
        # immediate — they must drop copies / vector links at once for correctness.
        if self._suspended is not None and event == "added":
            self._suspended.add(namespace)
            return
        if not self._listeners:
            return
        for fn in list(self._listeners.values()):
            try:
                fn(event, namespace, list(ids))
            except Exception as e:
                logger.warning(f"VectorMemory: listener failed on {event} in '{namespace}': {e}")

    @contextmanager
    def suspend_notifications(self):
        """Within the block, "added" notifications are buffered; on exit each touched namespace
        gets one "recount" so the registry and caches catch up once instead of per record."""
        if self._suspended is not None:
            yield                                 # already suspended by an outer block
            return
        self._suspended = set()
        try:
            yield
        finally:
            touched, self._suspended = self._suspended, None
            for ns in sorted(touched):
                self._notify("recount", ns, ["*"])

    def delete_ids(self, namespace: str, ids: List[str], notify: bool = True) -> int:
        """Delete records by id regardless of pinning (used for derived copies)."""
        if not ids or not self.has_collection(namespace):
            return 0
        coll = self._get_or_create_collection(namespace)
        try:
            present = coll.get(ids=list(ids), include=[])["ids"]
            if present:
                coll.delete(ids=present)
        except Exception as e:
            logger.warning(f"VectorMemory.delete_ids failed in '{namespace}': {e}")
            return 0
        if present and notify:
            self._notify("deleted", namespace, present)
        return len(present)

    def get_where(self, namespace: str, where: Dict[str, Any], with_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Records matching a metadata filter (no access-stat update)."""
        if not self.has_collection(namespace):
            return []
        coll = self._get_or_create_collection(namespace)
        try:
            if coll.count() == 0:
                return []
            inc = ["documents", "metadatas"] + (["embeddings"] if with_embeddings else [])
            res = coll.get(where=where, include=inc)
        except Exception as e:
            logger.debug(f"VectorMemory.get_where failed in '{namespace}': {e}")
            return []
        raw = res.get("embeddings") if with_embeddings else None
        embs = raw if raw is not None else []
        out = []
        for i, rid in enumerate(res["ids"]):
            out.append({"id": rid, "document": res["documents"][i], "metadata": res["metadatas"][i] or {},
                        "embedding": list(embs[i]) if i < len(embs) else []})
        return out

    # ------------------------------------------------------------------
    # Dynamic knowledge collections (cognitive memory, Stage 1)
    # ------------------------------------------------------------------

    @classmethod
    def knowledge_namespace(cls, label_or_slug: str) -> str:
        """'Henna' -> 'k_henna' (Chroma collection 'peter_k_henna'); unique per label (topic_slug)."""
        return cls.KNOWLEDGE_PREFIX + topic_slug(label_or_slug)

    @classmethod
    def is_knowledge_namespace(cls, namespace: str) -> bool:
        return namespace.startswith(cls.KNOWLEDGE_PREFIX)

    def _chroma_names(self) -> List[str]:
        out = []
        for c in self.client.list_collections():
            out.append(c.name if hasattr(c, "name") else str(c))
        return out

    def list_knowledge_namespaces(self) -> List[str]:
        prefix = "peter_" + self.KNOWLEDGE_PREFIX
        return sorted(n[len("peter_"):] for n in self._chroma_names() if n.startswith(prefix))

    def list_namespaces(self) -> List[str]:
        """Agent namespaces first, then every knowledge collection that exists."""
        return list(self.DEFAULT_NAMESPACES) + self.list_knowledge_namespaces()

    @property
    def semantic_ok(self) -> bool:
        """False when the embedder is the random-hash fallback (no meaning in the vectors)."""
        return is_semantic(self.embed_fn)

    def has_collection(self, namespace: str) -> bool:
        if namespace in self.collections:
            return True
        return f"peter_{namespace}" in self._chroma_names()

    def create_collection(self, namespace: str, budget: Optional[CollectionBudget] = None) -> bool:
        """Create a knowledge collection. Returns True if it is new.

        An existing collection is reopened for free. A new one consumes one unit
        of ``budget`` (when given) and raises BudgetExceeded when none is left.
        """
        if self.has_collection(namespace):
            self._get_or_create_collection(namespace)
            return False
        if budget is not None:
            budget.consume(namespace)
        self._get_or_create_collection(namespace)
        logger.info(f"VectorMemory: created collection 'peter_{namespace}'")
        return True

    def drop_collection(self, namespace: str) -> bool:
        """Delete a whole knowledge collection (agent namespaces are refused)."""
        if namespace in self.DEFAULT_NAMESPACES:
            logger.warning(f"VectorMemory: refusing to drop agent namespace '{namespace}'")
            return False
        self.collections.pop(namespace, None)
        try:
            self.client.delete_collection(f"peter_{namespace}")
            return True
        except Exception as e:
            logger.warning(f"VectorMemory: drop_collection('{namespace}') failed: {e}")
            return False

    def count(self, namespace: str) -> int:
        if not self.has_collection(namespace):
            return 0
        return self._get_or_create_collection(namespace).count()

    def get_records(self, namespace: str, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        coll = self._get_or_create_collection(namespace)
        res = coll.get(ids=list(ids), include=["documents", "metadatas", "embeddings"])
        raw = res.get("embeddings")
        embs = raw if raw is not None else []
        out = []
        for i, rid in enumerate(res["ids"]):
            out.append({
                "id": rid,
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
                "embedding": list(embs[i]) if i < len(embs) else [],
            })
        return out

    def update_metadata(self, namespace: str, doc_id: str, **fields: Any) -> bool:
        coll = self._get_or_create_collection(namespace)
        try:
            res = coll.get(ids=[doc_id])
            if not res or not res.get("metadatas"):
                return False
            meta = dict(res["metadatas"][0] or {})
            for k, v in fields.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            coll.update(ids=[doc_id], metadatas=[meta])
            return True
        except Exception as e:
            logger.warning(f"VectorMemory.update_metadata failed for {doc_id}: {e}")
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.embed_fn(list(texts))
        return [list(map(float, v)) for v in vecs]

    def centroid(self, namespace: str) -> Optional[List[float]]:
        """Mean of the (L2-normalised) embeddings in a collection, or None if empty."""
        recs = self.get_all_records(namespace)
        vecs = [r["embedding"] for r in recs if r.get("embedding")]
        if not vecs:
            return None
        import numpy as np
        arr = np.asarray(vecs, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mean = (arr / norms).mean(axis=0)
        n = float(np.linalg.norm(mean)) or 1.0
        return (mean / n).tolist()

    # ------------------------------------------------------------------
    # Entity index: one vector per entity / abstraction label
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_vec_id(node_id: str, alias: str) -> str:
        return f"{node_id}::alias::{hashlib.sha1(alias.strip().lower().encode('utf-8')).hexdigest()[:10]}"

    def index_entity(self, node_id: str, label: str, kind: str = "entity", aliases: Any = ()) -> None:
        """One vector for the label plus one per alias (all pointing at node_id), so a topic is
        found by meaning under any of its names ('CSE' -> Colombo Stock Exchange)."""
        coll = self._get_or_create_collection(self.INDEX_NAMESPACE)
        meta = {"node_id": node_id, "kind": kind, "label": label}
        ids, docs, metas = [node_id], [label], [meta]
        seen = {(label or "").strip().lower()}
        for a in aliases or ():
            a = (a or "").strip()
            if not a or a.lower() in seen:
                continue
            seen.add(a.lower())
            ids.append(self._alias_vec_id(node_id, a))
            docs.append(a)
            metas.append({**meta, "alias": a})
        try:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception as e:
            logger.warning(f"VectorMemory.index_entity failed for {node_id}: {e}")

    def unindex_entity(self, node_id: str) -> None:
        try:
            coll = self._get_or_create_collection(self.INDEX_NAMESPACE)
            coll.delete(ids=[node_id])
            coll.delete(where={"node_id": node_id})       # its alias vectors
        except Exception:
            pass

    def search_entities(self, text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Semantic lookup of entity/abstraction nodes by label or alias. One result per node
        (its best-matching name). Never touches access stats."""
        coll = self._get_or_create_collection(self.INDEX_NAMESPACE)
        total = coll.count()
        if total == 0:
            return []
        try:
            res = coll.query(query_texts=[text], n_results=min(max(n_results * 3, n_results + 4), total))
        except Exception as e:
            logger.warning(f"VectorMemory.search_entities failed: {e}")
            return []
        best: Dict[str, Dict[str, Any]] = {}
        for i, vid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i] or {}
            dist = res.get("distances", [[0.0]])[0][i]
            nid = str(meta.get("node_id") or vid)
            score = 1.0 - float(dist)
            if nid not in best or score > best[nid]["score"]:
                best[nid] = {"node_id": nid, "label": meta.get("label", ""), "kind": meta.get("kind", "entity"),
                             "score": score, "via_alias": meta.get("alias") or None}
        return sorted(best.values(), key=lambda h: (-h["score"], h["node_id"]))[:n_results]
