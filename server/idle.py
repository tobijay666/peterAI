"""Idle-triggered sleep (cognitive memory, Stage 5).

Peter sleeps when a user has been idle for ``sleep.idle_minutes`` and no cycle
ran for that user in the last ``sleep.min_hours_between`` hours. The nightly
timer stays as a fallback. A new message during sleep makes the cycle finish
its current step and skip the rest (``should_pause``).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional, Set

from core.cognition_config import cognition_config
from core.logger import logger


class IdleTracker:
    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._now = monotonic
        self._last_active: Dict[str, float] = {}
        self._last_sleep: Dict[str, float] = {}
        self._sleeping: Set[str] = set()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------ bookkeeping

    def mark_active(self, user_id: str) -> None:
        with self._lock:
            self._last_active[user_id] = self._now()

    def active_since(self, user_id: str, t: float) -> bool:
        with self._lock:
            return self._last_active.get(user_id, -1.0) > t

    def is_sleeping(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._sleeping

    def due(self, user_id: str) -> bool:
        cfg = cognition_config()["sleep"]
        now = self._now()
        with self._lock:
            if user_id in self._sleeping:
                return False
            last_active = self._last_active.get(user_id)
            if last_active is None:
                return False  # never active in this process: nothing new to consolidate
            if now - last_active < float(cfg["idle_minutes"]) * 60.0:
                return False
            last_sleep = self._last_sleep.get(user_id)
            if last_sleep is not None and now - last_sleep < float(cfg["min_hours_between"]) * 3600.0:
                return False
            if last_sleep is not None and last_sleep > last_active:
                return False  # already slept since the last activity
            return True

    # ------------------------------------------------------------------ running a cycle

    def run_sleep(self, user_id: str, runner: Callable[..., object], reason: str = "idle",
                  mode: Optional[str] = None) -> Optional[object]:
        """Run one cycle for a user via ``runner(user_id, reason=, mode=, should_pause=)``."""
        with self._lock:
            if user_id in self._sleeping:
                return None
            self._sleeping.add(user_id)
        started = self._now()
        try:
            return runner(user_id, reason=reason, mode=mode,
                          should_pause=lambda: self.active_since(user_id, started))
        finally:
            with self._lock:
                self._sleeping.discard(user_id)
                self._last_sleep[user_id] = self._now()

    def check_once(self, user_ids, runner: Callable[..., object]) -> list:
        """Start a background sleep for every due user. Returns the user ids started."""
        started = []
        for uid in list(user_ids):
            if self.due(uid):
                threading.Thread(target=self.run_sleep, args=(uid, runner, "idle"), daemon=True,
                                 name=f"Sleep-{uid}").start()
                started.append(uid)
        return started

    def start(self, interval_s: float = 60.0) -> None:
        if self._running:
            return
        self._running = True

        def loop() -> None:
            from server.peter_pool import pool
            from server.scheduler import run_cycle_for_user
            while self._running:
                time.sleep(interval_s)
                try:
                    with pool._lock:
                        uids = list(pool._pool.keys())
                    for uid in self.check_once(uids, run_cycle_for_user):
                        logger.info(f"[idle] user '{uid}' idle — starting sleep cycle")
                except Exception as e:
                    logger.error(f"[idle] check failed: {e}")

        self._thread = threading.Thread(target=loop, daemon=True, name="IdleSleepTrigger")
        self._thread.start()

    def stop(self) -> None:
        self._running = False


idle_tracker = IdleTracker()
