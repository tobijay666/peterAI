"""Background decay-based eviction scheduler for vector and graph memory."""

import threading
import time
from typing import Optional
from core.logger import logger
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory


class EvictionScheduler:
    """Applies decay-based memory eviction policy every 24 hours."""

    def __init__(
        self,
        vector_store: VectorMemory,
        graph_store: GraphMemory,
        interval_hours: float = 24.0,
        max_age_days: float = 30.0,
        min_access_count: int = 2,
        importance_threshold: float = 0.3,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.interval_sec = interval_hours * 3600.0
        self.max_age_days = max_age_days
        self.min_access_count = min_access_count
        self.importance_threshold = importance_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EvictionThread")
        self._thread.start()
        logger.info("EvictionScheduler started.")

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self.run_eviction()
            time.sleep(self.interval_sec)

    def run_eviction(self) -> None:
        """Evaluates decay criteria across vector namespaces and graph database."""
        logger.info("EvictionScheduler: Running decay eviction cycle...")
        now = time.time()
        max_age_sec = self.max_age_days * 86400.0

        total_evicted_vectors = 0
        for ns in self.vector_store.DEFAULT_NAMESPACES:
            records = self.vector_store.get_all_records(ns)
            evict_ids = []
            for r in records:
                meta = r.get("metadata", {})
                pinned = meta.get("pinned", False)
                if pinned:
                    continue

                last_accessed = meta.get("last_accessed", now)
                access_count = meta.get("access_count", 0)
                importance = meta.get("importance", 0.5)

                age_sec = now - last_accessed
                if (
                    age_sec > max_age_sec
                    and access_count < self.min_access_count
                    and importance < self.importance_threshold
                ):
                    evict_ids.append(r["id"])

            if evict_ids:
                pruned_cnt = self.vector_store.prune(ns, evict_ids)
                total_evicted_vectors += pruned_cnt

        evicted_graph_nodes = self.graph_store.prune(
            max_age_days=self.max_age_days, min_importance=self.importance_threshold
        )
        logger.info(
            f"EvictionScheduler: Eviction completed. Removed {total_evicted_vectors} vector documents and {evicted_graph_nodes} graph nodes."
        )
