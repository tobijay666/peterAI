"""
LearningService — wires the chat loop to Peter's memory system.

After every chat turn this service:
  1. Stores the conversation pair (user msg + Peter reply) in the vector
     store so future turns can recall it.
  2. Emits a "learned" event to the 3-D view (green growth pulse).
  3. Runs a lightweight keyword-based skill-trigger extraction pass:
     if the same topic cluster has been stored ≥ CLUSTER_MIN times for
     this user, it creates a candidate skill node in the graph.
  4. Updates per-user memory stats (counts by namespace/kind) that are
     served by GET /api/memory/stats.
  5. Queues curiosity questions (cognitive memory, Stage 4) and marks the
     user active for the idle-triggered sleep cycle (Stage 5).

All operations are fire-and-forget: failures are logged, never raised.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

from core.events import emit
from core.logger import logger
from core.user_context import UserContext

# Minimum number of similar conversation turns before a skill-trigger is born.
CLUSTER_MIN = 3
# Max importance decay factor applied to old conversation episodes each day.
EPISODE_IMPORTANCE = 0.55


class LearningService:
    """Per-user learning wiring.  One instance lives inside the Peter pool entry."""

    def __init__(self, user_ctx: UserContext, peter: Any) -> None:
        self._ctx = user_ctx
        self._peter = peter
        self._lock = threading.Lock()
        # Simple in-process keyword → count map for micro-clustering
        self._topic_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_turn(
        self,
        user_msg: str,
        peter_reply: str,
        task_id: str,
    ) -> None:
        """Called after every successful chat turn.  Never raises."""
        threading.Thread(
            target=self._process,
            args=(user_msg, peter_reply, task_id),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process(
        self,
        user_msg: str,
        peter_reply: str,
        task_id: str,
    ) -> None:
        """Background: store, emit, maybe extract skill trigger."""
        uid = self._ctx.user_id

        # 1. Store the turn as an episode in "peter" namespace
        try:
            episode_text = (
                f"User asked: {user_msg.strip()}\n"
                f"Peter replied: {peter_reply.strip()}"
            )
            vector = self._peter.vector
            mem_id = vector.store(
                namespace="peter",
                text=episode_text,
                importance=EPISODE_IMPORTANCE,
                pinned=False,
                kind="episode",
                tags=f"chat,task:{task_id}",
                extra_metadata={"task_id": task_id, "trust": "user"},
            )
            logger.debug(f"[learning] stored episode {mem_id} for user {uid}")
        except Exception as e:
            logger.error(f"[learning] store episode failed: {e}")
            return

        # 2. Emit "learned" event → 3-D view shows growth glow on Peter sphere
        try:
            emit(
                "learned",
                user_id=uid,
                turn_id=task_id,
                memory_id=mem_id,
                namespace="peter",
            )
        except Exception as e:
            logger.warning(f"[learning] emit learned failed: {e}")

        # 3. Micro skill-trigger extraction
        try:
            self._maybe_extract_skill(user_msg, peter_reply, task_id)
        except Exception as e:
            logger.warning(f"[learning] skill extraction failed: {e}")

        # 4. Curiosity: queue adjacent questions Peter wants to learn (Stage 4)
        try:
            cognition = getattr(self._peter, "cognition", None)
            if cognition is not None:
                cognition.after_turn(user_msg, peter_reply, turn_id=task_id)
        except Exception as e:
            logger.warning(f"[learning] curiosity generation failed: {e}")

        # 5. Idle tracking for the sleep cycle (Stage 5)
        try:
            from server.idle import idle_tracker
            idle_tracker.mark_active(uid)
        except Exception:
            pass

    def _maybe_extract_skill(
        self,
        user_msg: str,
        peter_reply: str,
        task_id: str,
    ) -> None:
        """Create a candidate skill node when the same topic repeats ≥ CLUSTER_MIN times."""
        topic = _extract_topic(user_msg)
        if not topic:
            return

        with self._lock:
            self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1
            count = self._topic_counts[topic]

        # Only on exact threshold crossing (not every turn after)
        if count != CLUSTER_MIN:
            return

        uid = self._ctx.user_id
        logger.info(f"[learning] topic '{topic}' crossed threshold → creating candidate skill")

        try:
            from memory.skills import SkillStore
            graph = self._peter.graph
            skill_store = SkillStore(graph)
            trigger = f"help with {topic}"
            plan_template = [
                {"step": "recall", "query": topic},
                {"step": "answer", "topic": topic},
            ]
            skill_id = skill_store.create_candidate(
                trigger=trigger,
                plan_template=plan_template,
                source_task_ids=[task_id],
                created_by="learning_service",
            )

            # Emit a skill_created event so the 3-D knowledge graph reacts
            emit(
                "skill_created",
                user_id=uid,
                turn_id=task_id,
                skill_id=skill_id,
                trigger=trigger,
            )
            logger.info(f"[learning] candidate skill {skill_id} created: {trigger}")
        except Exception as e:
            logger.error(f"[learning] skill creation failed: {e}")


# ------------------------------------------------------------------
# Memory stats helper (called by /api/memory/stats)
# ------------------------------------------------------------------

def get_memory_stats(peter: Any, user_ctx: UserContext) -> Dict[str, Any]:
    """Return counts of records per namespace and kind, plus total."""
    vector = peter.vector
    stats: Dict[str, Any] = {
        "user_id": user_ctx.user_id,
        "namespaces": {},
        "total": 0,
        "by_kind": {},
        "limit": int(__import__("os").getenv("MEMORY_LIMIT_PER_USER", "5000")),
    }
    for ns in vector.list_namespaces():
        try:
            recs = vector.get_all_records(ns)
            ns_count = len(recs)
            stats["namespaces"][ns] = ns_count
            stats["total"] += ns_count
            for r in recs:
                kind = r.get("metadata", {}).get("kind", "unknown")
                stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        except Exception:
            stats["namespaces"][ns] = 0
    return stats


# ------------------------------------------------------------------
# Simple keyword topic extraction
# ------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "how", "what", "when", "where", "why", "who", "which",
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "and", "or", "but", "so",
    "for", "of", "to", "in", "on", "at", "by", "from", "with",
    "about", "please", "help", "tell", "explain", "give", "make",
    "get", "let", "want", "need", "like", "just", "also", "very",
    "much", "more", "most", "some", "any", "all", "no", "not",
    "peter", "ai",
}


def _extract_topic(text: str) -> str:
    """Return a normalised 1-3 word topic key, or '' if extraction fails."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    keywords = [w for w in words if w not in _STOP and len(w) > 3]
    if not keywords:
        return ""
    # Use up to 2 most frequent keywords as the topic key
    freq: Dict[str, int] = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=lambda w: -freq[w])[:2]
    return " ".join(sorted(top))
