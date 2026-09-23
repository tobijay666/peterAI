"""Background thread probing provider health with circuit breaker logic."""

import threading
import time
from typing import Dict, Tuple
from core.logger import logger
from llm.router import LLMRouter


class HealthProber:
    """Probes provider model endpoints and manages cooldown for failing endpoints."""

    def __init__(self, router: LLMRouter, probe_interval_sec: int = 600, cooldown_sec: int = 300):
        self.router = router
        self.probe_interval_sec = probe_interval_sec
        self.cooldown_sec = cooldown_sec
        # Map (provider, model) -> (is_alive, last_fail_timestamp, failure_count)
        self.health_state: Dict[Tuple[str, str], Tuple[bool, float, int]] = {}
        self._running = False
        self._thread: threading.Thread = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._probe_loop, daemon=True, name="HealthProberThread")
        self._thread.start()
        logger.info("HealthProber background thread started.")

    def stop(self) -> None:
        self._running = False

    def _probe_loop(self) -> None:
        while self._running:
            self.run_full_probe()
            time.sleep(self.probe_interval_sec)

    def run_full_probe(self) -> None:
        """Pings models across providers to evaluate endpoint responsiveness."""
        logger.debug("HealthProber: Initiating provider health probes...")
        for prov_key, provider in self.router.providers.items():
            if not provider.is_available():
                continue

            for model in provider.models:
                alive = provider.probe(model, timeout=5.0)
                key = (prov_key, model)
                prev_alive, last_fail, fail_cnt = self.health_state.get(key, (True, 0.0, 0))

                if alive:
                    self.health_state[key] = (True, 0.0, 0)
                else:
                    new_fail_cnt = fail_cnt + 1
                    self.health_state[key] = (False, time.time(), new_fail_cnt)
                    logger.warning(
                        f"HealthProber: Model {prov_key}:{model} failed probe (fail count: {new_fail_cnt})"
                    )

    def is_model_alive(self, provider_key: str, model: str) -> bool:
        key = (provider_key.lower(), model)
        if key not in self.health_state:
            return True

        alive, last_fail, fail_cnt = self.health_state[key]
        if alive:
            return True

        # Check circuit breaker cooldown
        if fail_cnt >= 3:
            if time.time() - last_fail > self.cooldown_sec:
                logger.info(f"HealthProber: Cooldown expired for {provider_key}:{model}, allowing retry.")
                return True
            return False
        return True
