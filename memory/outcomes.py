"""OutcomeLedger — records every tool dispatch, step and task outcome (FR-12)."""
from __future__ import annotations
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.clock import Clock, SystemClock
from core.logger import logger


class OutcomeLedger:
    """Persists outcomes to the graph DB (outcomes + feedback tables from schema v2).

    IMPORTANT (FR-12.2): every public method catches all exceptions and logs
    them. A ledger failure must never propagate to the caller.
    """

    def __init__(self, db_path: str = "data/peter_graph.db", clock: Optional[Clock] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or SystemClock()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def record(
        self,
        task_id: str,
        level: str,  # "tool" | "step" | "task"
        agent: str,
        status: str,  # "success" | "failure" | "timeout" | "blocked"
        step_id: Optional[str] = None,
        tool: Optional[str] = None,
        skill_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        latency_ms: Optional[int] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Write one outcome row. Never raises (FR-12.2)."""
        row_id = str(uuid.uuid4())
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO outcomes
                      (id, task_id, step_id, level, agent, tool, skill_id,
                       provider, model, status, latency_ms, tokens_in,
                       tokens_out, memory_ids, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row_id, task_id, step_id, level, agent, tool,
                        skill_id, provider, model, status, latency_ms,
                        tokens_in, tokens_out,
                        json.dumps(memory_ids) if memory_ids else None,
                        self._clock.now(),
                    ),
                )
                conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"OutcomeLedger.record failed (non-fatal): {e}")
            return None

    def rate_task(
        self,
        task_id: str,
        rating: int,  # +1 or -1
        note: Optional[str] = None,
    ) -> Optional[str]:
        """Persist user feedback for a task. Never raises (FR-12.2)."""
        row_id = str(uuid.uuid4())
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO feedback (id, task_id, rating, note, created_at) VALUES (?,?,?,?,?)",
                    (row_id, task_id, rating, note, self._clock.now()),
                )
                conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"OutcomeLedger.rate_task failed (non-fatal): {e}")
            return None

    def stats(
        self,
        group_by: str = "agent",
        window_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return aggregated stats grouped by agent/tool/model/skill. Never raises."""
        try:
            cutoff = self._clock.now() - window_seconds if window_seconds else 0.0
            valid_groups = {"agent", "tool", "model", "skill_id"}
            col = group_by if group_by in valid_groups else "agent"
            with self._conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {col} as key,
                           COUNT(*) as total,
                           SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successes,
                           AVG(latency_ms) as avg_latency_ms,
                           MAX(latency_ms) as p_max_latency_ms
                    FROM outcomes
                    WHERE created_at >= ?
                    GROUP BY {col}
                    ORDER BY total DESC
                    """,
                    (cutoff,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"OutcomeLedger.stats failed (non-fatal): {e}")
            return []

    def get_task_outcomes(self, task_id: str) -> List[Dict[str, Any]]:
        """Return all outcome rows for a given task_id. Never raises."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM outcomes WHERE task_id = ? ORDER BY created_at",
                    (task_id,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"OutcomeLedger.get_task_outcomes failed (non-fatal): {e}")
            return []

    def get_cited_memory_ids(self, task_id: str) -> List[str]:
        """Return all memory_ids cited across all outcome rows for a task."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT memory_ids FROM outcomes WHERE task_id = ? AND memory_ids IS NOT NULL",
                    (task_id,),
                ).fetchall()
            ids: List[str] = []
            for r in rows:
                try:
                    ids.extend(json.loads(r["memory_ids"]))
                except Exception:
                    pass
            return list(set(ids))
        except Exception as e:
            logger.error(f"OutcomeLedger.get_cited_memory_ids failed (non-fatal): {e}")
            return []
