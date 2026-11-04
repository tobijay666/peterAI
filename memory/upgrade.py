"""One-time upgrades of an existing user's memory, run when CognitiveMemory starts.

Existing graphs must keep working: old node ids and collection names stay as they
are (only new nodes use the unique-slug scheme), and everything else that newer
code expects is filled in here, once per graph (schema_version rows record it):

  v4  alias keys with symbols ('c++' next to the old 'c'), drop old symbol-stripped
      keys that now mean a different label; index aliases in the entity index;
      repair the hierarchy (no cycles in 'abstracts', nothing above a subject,
      one primary parent per node).
  v5  edge weights and record-level vector links (Part C) are computed by the next
      sleep cycle (SleepCycle.maintain -> VectorGraph.refresh_all); empty hub collections
      are filled by the sleep cycle's fill step. Nothing heavy runs at start-up.
"""
from __future__ import annotations

from typing import Any

from core.logger import logger

TOPIC_KINDS = ["entity", "abstraction"]


def schema_version(graph: Any) -> int:
    with graph._get_connection() as conn:
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return int(row["v"] or 0) if row else 0


def _mark(graph: Any, version: int) -> None:
    with graph._get_connection() as conn:
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", (version, graph._clock.now()))
        conn.commit()


def upgrade_aliases(graph: Any) -> int:
    """New alias keys keep '+'/'#'. Add the new key of every topic label; remove the old
    key when it differs (it was 'c' for 'C++', which now means the topic 'C')."""
    changed = 0
    for n in graph.all_nodes(TOPIC_KINDS):
        label = n.get("label") or ""
        new, old = graph.normalize_alias(label), graph.legacy_normalize_alias(label)
        if new and new not in graph.aliases_of(n["id"]):
            graph.add_alias(n["id"], label)
            changed += 1
        if old and old != new:
            graph.remove_alias(n["id"], old)
            changed += 1
    return changed


def run_upgrades(cognition: Any) -> None:
    """Idempotent; cheap on an up-to-date graph (one SELECT)."""
    graph = cognition.graph
    try:
        v = schema_version(graph)
    except Exception as e:
        logger.warning(f"upgrade: could not read schema version: {e}")
        return
    if v < 4:
        try:
            n = upgrade_aliases(graph)
            hier = getattr(cognition, "hierarchy", None)
            if hier is not None and hasattr(hier, "repair"):
                n += hier.repair()
            knowledge = getattr(cognition, "knowledge", None)
            if knowledge is not None and hasattr(knowledge, "index_all_aliases"):
                n += knowledge.index_all_aliases()
            _mark(graph, 4)
            if n:
                logger.info(f"upgrade: memory of '{graph.user_id}' upgraded to v4 ({n} changes)")
        except Exception as e:
            logger.error(f"upgrade: v4 failed for '{graph.user_id}': {e}")
