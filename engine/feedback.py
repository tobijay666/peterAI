"""Feedback engine — good/bad commands bound to the last completed task (FR-17).

feedback.record_good(task_id) and record_bad(task_id, note) persist feedback
and trigger downstream effects:
- Rate the task in the OutcomeLedger (FR-17.1)
- Lower importance of cited memory ids (FR-17.2)
- Count against the skill used (FR-17.2)
"""
from __future__ import annotations
from typing import Optional
from core.clock import Clock, SystemClock
from core.logger import logger
from memory.outcomes import OutcomeLedger
from memory.vector_store import VectorMemory

IMPORTANCE_BAD_FEEDBACK_DELTA = -0.20  # overridden by evolution.yaml if loaded


class FeedbackEngine:
    def __init__(
        self,
        ledger: OutcomeLedger,
        vector: VectorMemory,
        bad_feedback_delta: float = IMPORTANCE_BAD_FEEDBACK_DELTA,
        clock: Optional[Clock] = None,
    ) -> None:
        self.ledger = ledger
        self.vector = vector
        self.bad_feedback_delta = bad_feedback_delta
        self._clock = clock or SystemClock()

    def record_good(self, task_id: str) -> None:
        """Record positive feedback for a task (FR-17.1)."""
        self.ledger.rate_task(task_id, rating=+1)
        logger.info(f"FeedbackEngine: +1 for task {task_id}")

    def record_bad(self, task_id: str, note: Optional[str] = None) -> None:
        """Record negative feedback, lower cited memory importance, count against skill (FR-17.2)."""
        self.ledger.rate_task(task_id, rating=-1, note=note)
        logger.info(f"FeedbackEngine: -1 for task {task_id}")

        # Lower importance of cited memories
        memory_ids = self.ledger.get_cited_memory_ids(task_id)
        for mid in memory_ids:
            for ns in self.vector.DEFAULT_NAMESPACES:
                result = self.vector.adjust_importance(ns, mid, self.bad_feedback_delta)
                if result is not None:
                    logger.debug(f"FeedbackEngine: adjusted importance of {mid} in {ns}: {result}")
                    break
