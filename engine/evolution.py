"""EvolutionCoordinator — orchestrates the daily learning cycle (FR-20).

Cycle order (fixed, tested):
  snapshot → consolidate → promote/retire skills → decay eviction → journal

Honours EVOLUTION_MODE (from .env):
  shadow: reports what would happen, performs nothing (FR-20.1)
  active: executes all mutations
  off: does nothing

Every automatic mutation is written to the evolution journal (FR-20.2).
"""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.clock import Clock, SystemClock
from core.events import event_bus
from core.logger import logger

JOURNAL_PATH = Path("data/evolution/journal.jsonl")
EVOLUTION_MODE_ENV = os.environ.get("EVOLUTION_MODE", "shadow")


class EvolutionCoordinator:
    """Runs the daily cycle and journals all mutations.

    Args:
        vector: VectorMemory
        graph: GraphMemory
        router: LLMRouter
        guardrails: GuardrailManager
        ledger: OutcomeLedger
        skill_store: SkillStore
        consolidation_config: dict from evolution.yaml['consolidation']
        mode: 'shadow' | 'active' | 'off' (default from EVOLUTION_MODE env)
        clock: injectable clock
    """

    def __init__(
        self,
        vector: Any,
        graph: Any,
        router: Any,
        guardrails: Any,
        ledger: Any,
        skill_store: Any,
        consolidation_config: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        clock: Optional[Clock] = None,
        interval_hours: float = 24.0,
    ) -> None:
        self.vector = vector
        self.graph = graph
        self.router = router
        self.guardrails = guardrails
        self.ledger = ledger
        self.skill_store = skill_store
        self.consolidation_config = consolidation_config or {}
        self.mode = mode or EVOLUTION_MODE_ENV
        self._clock = clock or SystemClock()
        self.interval_hours = interval_hours
        self._running = False
        self._thread: Optional[threading.Thread] = None
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Subscribe to task completion events for outcome tracking
        event_bus.subscribe("task.compiled", self._on_task_compiled)

    def start(self) -> None:
        """Start the background cycle thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EvolutionThread")
        self._thread.start()
        logger.info(f"EvolutionCoordinator started (mode={self.mode})")

    def stop(self) -> None:
        self._running = False

    def run_cycle(self) -> Dict[str, Any]:
        """Run one full evolution cycle. Returns a report dict."""
        if self.mode == "off":
            logger.info("EvolutionCoordinator: mode=off, skipping cycle.")
            return {"mode": "off", "actions": []}

        is_shadow = (self.mode == "shadow")
        report: Dict[str, Any] = {"mode": self.mode, "actions": [], "errors": []}

        logger.info(f"EvolutionCoordinator: starting cycle (mode={self.mode})")

        # Step 1: Snapshot
        try:
            from memory.snapshot import SnapshotManager
            snap_mgr = SnapshotManager(clock=self._clock)
            if not is_shadow:
                snap_path = snap_mgr.take_snapshot()
                self._journal({"step": "snapshot", "path": str(snap_path)})
                report["actions"].append({"step": "snapshot", "path": str(snap_path)})
            else:
                report["actions"].append({"step": "snapshot", "note": "shadow: would take snapshot"})
        except Exception as e:
            report["errors"].append(f"snapshot: {e}")
            logger.error(f"EvolutionCoordinator: snapshot error: {e}")

        # Step 2: Consolidation
        try:
            from memory.consolidation import ConsolidationPass
            cp = ConsolidationPass(
                vector=self.vector,
                graph=self.graph,
                router=self.router,
                guardrails=self.guardrails,
                config=self.consolidation_config,
                clock=self._clock,
            )
            con_report = cp.run(dry_run=is_shadow)
            action = {
                "step": "consolidation",
                "semantic_written": con_report.semantic_records_written,
                "skills_created": con_report.skill_candidates_created,
                "sources_decayed": con_report.sources_decayed,
                "dry_run": is_shadow,
            }
            report["actions"].append(action)
            if not is_shadow:
                self._journal({**action, "step": "consolidation"})
        except Exception as e:
            report["errors"].append(f"consolidation: {e}")
            logger.error(f"EvolutionCoordinator: consolidation error: {e}")

        # Step 3: Promote / retire skills
        try:
            action = self._run_skill_maintenance(is_shadow)
            report["actions"].append(action)
        except Exception as e:
            report["errors"].append(f"skill_maintenance: {e}")
            logger.error(f"EvolutionCoordinator: skill maintenance error: {e}")

        # Step 4: Decay eviction (after consolidation — FR-14.5)
        try:
            from memory.eviction import EvictionScheduler
            evict = EvictionScheduler(
                vector_store=self.vector,
                graph_store=self.graph,
                clock=self._clock,
            )
            if not is_shadow:
                evict.run_eviction()
                self._journal({"step": "eviction", "note": "ran"})
                report["actions"].append({"step": "eviction", "note": "ran"})
            else:
                report["actions"].append({"step": "eviction", "note": "shadow: would run eviction"})
        except Exception as e:
            report["errors"].append(f"eviction: {e}")
            logger.error(f"EvolutionCoordinator: eviction error: {e}")

        logger.info(f"EvolutionCoordinator: cycle complete. report={report}")
        return report

    def _run_skill_maintenance(self, is_shadow: bool) -> Dict[str, Any]:
        """Evaluate all candidate/active skills for promotion/deprecation."""
        promoted = []
        deprecated = []

        all_skills = self.skill_store.list_skills()
        for skill in all_skills:
            sid = skill["skill_id"]
            status = skill.get("status", "candidate")
            
            # Get stats from ledger
            try:
                with self.ledger._conn() as conn:
                    row = conn.execute(
                        "SELECT uses, successes FROM v_skill_stats WHERE skill_id = ?",
                        (sid,),
                    ).fetchone()
                    stats = dict(row) if row else {"uses": 0, "successes": 0}
            except Exception:
                stats = {"uses": 0, "successes": 0}

            if status == "candidate" and not is_shadow:
                if self.skill_store.evaluate_promotion(sid, stats):
                    promoted.append(sid)
                    self._journal({"step": "skill_promoted", "skill_id": sid, "stats": stats})
                    logger.info(f"EvolutionCoordinator: promoted skill {sid} to active")

            elif status == "active" and not is_shadow:
                if self.skill_store.evaluate_deprecation(sid, stats):
                    deprecated.append(sid)
                    self._journal({"step": "skill_deprecated", "skill_id": sid, "stats": stats})
                    logger.info(f"EvolutionCoordinator: deprecated skill {sid}")

        return {"step": "skill_maintenance", "promoted": promoted, "deprecated": deprecated, "dry_run": is_shadow}

    def _on_task_compiled(self, topic: str, payload: Dict[str, Any]) -> None:
        """Handle task.compiled events for real-time updates."""
        task_id = payload.get("task_id", "")
        logger.debug(f"EvolutionCoordinator: task compiled: {task_id}")

    def _journal(self, entry: Dict[str, Any]) -> None:
        """Append a structured entry to the evolution journal (FR-20.2)."""
        try:
            entry["timestamp"] = self._clock.now()
            entry["mode"] = self.mode
            with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"EvolutionCoordinator: journal write failed: {e}")

    def _loop(self) -> None:
        interval_sec = self.interval_hours * 3600.0
        while self._running:
            self.run_cycle()
            # Use real time.sleep for the actual wait (not clock)
            time.sleep(interval_sec)

    def get_journal_entries(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Read last N journal entries."""
        try:
            lines = JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
            return [json.loads(l) for l in lines[-last_n:] if l.strip()]
        except Exception:
            return []
