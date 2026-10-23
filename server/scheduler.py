"""
Per-user nightly evolution scheduler.

Runs one EvolutionCoordinator cycle for every user that has an active
Peter session in the pool.  Scheduled daily via APScheduler (if installed)
or a simple background thread.

The scheduler is started by server.app on startup.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from core.logger import logger


class NightlyScheduler:
    """Runs the evolution cycle for all active users once a day."""

    def __init__(self, interval_hours: float = 24.0) -> None:
        self._interval  = interval_hours * 3600.0
        self._running   = False
        self._thread: Any = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="NightlyEvolution"
        )
        self._thread.start()
        logger.info(f"NightlyScheduler started (interval={self._interval / 3600:.1f}h)")

    def stop(self) -> None:
        self._running = False

    def run_now(self) -> None:
        """Run one cycle immediately (useful for testing / manual trigger)."""
        self._run_cycle()

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            if self._running:
                self._run_cycle()

    def _run_cycle(self) -> None:
        """Run the evolution cycle for every user currently in the pool."""
        try:
            from server.peter_pool import pool  # import late to avoid circular
            with pool._lock:
                uids = list(pool._pool.keys())
            for uid in uids:
                try:
                    from server.idle import idle_tracker
                    report = idle_tracker.run_sleep(uid, run_cycle_for_user, reason="timer")
                    logger.info(f"[scheduler] Evolution cycle for user '{uid}': {report}")
                except Exception as e:
                    logger.error(f"[scheduler] Evolution cycle failed for user '{uid}': {e}")
        except Exception as e:
            logger.error(f"[scheduler] _run_cycle outer error: {e}")


def build_coordinator(peter: Any, router: Any, guardrails: Any, mode: Optional[str] = None,
                      should_pause: Any = None, interval_hours: float = 24.0) -> Any:
    """EvolutionCoordinator for one user's Peter, with cognitive memory + per-user snapshots."""
    import yaml
    from engine.evolution import EvolutionCoordinator
    from memory.outcomes import OutcomeLedger
    from memory.skills import SkillStore
    from memory.snapshot import snapshot_manager_for

    try:
        with open("config/evolution.yaml", "r") as f:
            evo_cfg = yaml.safe_load(f) or {}
    except Exception:
        evo_cfg = {}
    cognition = getattr(peter, "cognition", None)
    return EvolutionCoordinator(
        vector=peter.vector,
        graph=peter.graph,
        router=router,
        guardrails=guardrails,
        ledger=OutcomeLedger(db_path=str(peter.graph.db_path)),
        skill_store=SkillStore(graph=peter.graph),
        consolidation_config=evo_cfg.get("consolidation", {}),
        mode=mode,
        interval_hours=interval_hours,
        cognition=cognition,
        snapshot_manager=snapshot_manager_for(peter.vector, peter.graph),
        should_pause=should_pause,
    )


def run_cycle_for_user(user_id: str, reason: str = "timer", mode: Optional[str] = None,
                       should_pause: Any = None) -> Dict[str, Any]:
    """One sleep/evolution cycle for a user that has a live Peter in the pool."""
    from server.peter_pool import pool
    with pool._lock:
        entry = pool._pool.get(user_id)
    if entry is None:
        return {"skipped": "user not active"}
    coord = build_coordinator(entry.peter, entry.router, entry.guardrails, mode=mode, should_pause=should_pause)
    return coord.run_cycle(reason=reason)


# Module-level singleton
nightly = NightlyScheduler()
