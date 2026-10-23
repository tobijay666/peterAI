"""
Per-user PeterAgent cache.

Keeps one bootstrapped Peter per active user so the heavy bootstrap
(Chroma, SQLite, embedder load) only runs once per session.

Idle entries are evicted after IDLE_TIMEOUT_S seconds (default 30 min).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from core.logger import logger
from core.user_context import UserContext

IDLE_TIMEOUT_S: int = 1800  # 30 minutes


@dataclass
class _Entry:
    peter: Any
    tts: Any
    guardrails: Any
    eviction: Any
    worker_pool: Any
    router: Any
    learning: Any  # LearningService
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()


class PeterPool:
    """Thread-safe cache of one Peter instance per user_id."""

    def __init__(self, idle_timeout_s: int = IDLE_TIMEOUT_S) -> None:
        self._pool: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout_s
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(self, ctx: UserContext) -> Tuple[Any, Any, Any, Any]:
        """Returns (peter, guardrails, router, learning) for the given user context.

        Creates a new Peter instance on the first call for each user_id.
        Subsequent calls return the cached instance and reset the idle timer.
        """
        uid = ctx.user_id
        with self._lock:
            entry = self._pool.get(uid)
            if entry is not None:
                entry.touch()
                logger.debug(f"PeterPool: Cache hit for user '{uid}'.")
                return entry.peter, entry.guardrails, entry.router, entry.learning

        # Bootstrap outside the lock (it can take a few seconds)
        logger.info(f"PeterPool: Creating Peter for user '{uid}'.")
        from peter import bootstrap_for_user  # import here to avoid circular
        peter, tts, guardrails, eviction, worker_pool, router = bootstrap_for_user(ctx)
        from server.learning import LearningService
        learning = LearningService(ctx, peter)

        entry = _Entry(
            peter=peter, tts=tts, guardrails=guardrails,
            eviction=eviction, worker_pool=worker_pool, router=router,
            learning=learning,
        )
        with self._lock:
            # Another thread might have created an entry while we were outside
            existing = self._pool.get(uid)
            if existing:
                # Shut down the one we just created and use the existing one
                try:
                    worker_pool.stop()
                    eviction.stop()
                except Exception:
                    pass
                existing.touch()
                return existing.peter, existing.guardrails, existing.router, existing.learning
            self._pool[uid] = entry

        return peter, guardrails, router, learning

    def evict(self, user_id: str) -> None:
        """Immediately evicts and shuts down a user's Peter instance."""
        with self._lock:
            entry = self._pool.pop(user_id, None)
        if entry:
            self._shutdown(user_id, entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reap_loop(self) -> None:
        while True:
            time.sleep(60)
            now = time.monotonic()
            evict_ids = []
            with self._lock:
                for uid, entry in list(self._pool.items()):
                    if now - entry.last_used > self._idle_timeout:
                        evict_ids.append(uid)
                for uid in evict_ids:
                    del self._pool[uid]
            for uid in evict_ids:
                # entry already removed — just log (workers already stopped in _shutdown)
                logger.info(f"PeterPool: Evicted idle session for user '{uid}'.")

    def _shutdown(self, uid: str, entry: _Entry) -> None:
        try:
            entry.worker_pool.stop()
        except Exception as e:
            logger.warning(f"PeterPool: worker_pool.stop() error for '{uid}': {e}")
        try:
            entry.eviction.stop()
        except Exception as e:
            logger.warning(f"PeterPool: eviction.stop() error for '{uid}': {e}")


# ---------------------------------------------------------------------------
# Module-level singleton (imported by server.app and server.routes.*)
# ---------------------------------------------------------------------------
pool = PeterPool()
