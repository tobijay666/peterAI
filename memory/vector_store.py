"""ChromaDB namespaced vector store implementation."""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from core.logger import logger
from core.clock import Clock, SystemClock
from memory.embeddings import LocalEmbeddingFunction


class VectorMemory:
    """Manages persistent namespaced vector memory collections using ChromaDB."""

    DEFAULT_NAMESPACES = [
        "peter",
        "researcher",
        "filesystem",
        "librarian",
        "devops",
        "qa",
        "voice",
    ]

    def __init__(self, persist_directory: str = "data/chroma", embedding_model: str = "all-MiniLM-L6-v2", clock: Optional[Clock] = None):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embed_fn = LocalEmbeddingFunction(embedding_model)
        self._clock = clock or SystemClock()

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collections: Dict[str, Any] = {}

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
    ) -> str:
        """Stores a text record in the specified agent namespace."""
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

        coll.add(documents=[text], metadatas=[meta], ids=[doc_id])
        logger.debug(f"VectorMemory: Stored record [{doc_id}] in namespace '{namespace}'.")
        return doc_id

    def recall(
        self,
        namespace: str,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Recalls top-k matching documents from a namespace, updating hit metrics."""
        coll = self._get_or_create_collection(namespace)
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
            return len(to_delete)
        except Exception as e:
            logger.error(f"VectorMemory: Prune failed in '{namespace}': {e}")
            return 0

    def get_all_records(self, namespace: str) -> List[Dict[str, Any]]:
        coll = self._get_or_create_collection(namespace)
        if coll.count() == 0:
            return []
        res = coll.get(include=["documents", "metadatas"])
        records = []
        for i in range(len(res["ids"])):
            records.append(
                {
                    "id": res["ids"][i],
                    "document": res["documents"][i],
                    "metadata": res["metadatas"][i],
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
