"""OpenRouter provider implementation with dynamic model discovery and 15-day caching."""

import json
from pathlib import Path
from typing import List, Optional
import requests
from core.clock import SystemClock
from core.logger import logger
from llm.providers.base import BaseProvider

CACHE_FILE = Path("data/openrouter_models_cache.json")
CACHE_TTL_SECONDS = 15 * 86400  # 15 days


class OpenRouterProvider(BaseProvider):

    def __init__(
        self,
        name: str = "OpenRouter",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        models: Optional[List[str]] = None,
        priority: int = 30,
        cache_path: Optional[Path] = None,
    ):
        self.cache_path = cache_path or CACHE_FILE
        if models is None:
            models = self.get_cached_or_remote_models()

        super().__init__(
            name=name, base_url=base_url, api_key_env=api_key_env, models=models, priority=priority
        )

    def get_cached_or_remote_models(self, force_refresh: bool = False) -> List[str]:
        """Loads models from local cache if within 15 days TTL, otherwise fetches from API."""
        default_fallback = [
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-r1",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-flash",
            "qwen/qwen-2.5-72b-instruct",
        ]

        now = SystemClock().now()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not force_refresh and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                ts = cache_data.get("timestamp", 0)
                cached_models = cache_data.get("models", [])
                if (now - ts) < CACHE_TTL_SECONDS and cached_models:
                    logger.info(f"OpenRouter: Loaded {len(cached_models)} models from local cache (age: int({(now - ts) / 86400}) days).")
                    return self._select_priority_models(cached_models, default_fallback)
            except Exception as e:
                logger.warning(f"OpenRouter: Failed reading models cache ({e}). Re-fetching...")

        # Fetch remote models list
        fetched = self.fetch_remote_models()
        if fetched:
            try:
                cache_payload = {"timestamp": now, "models": fetched}
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, indent=2)
                logger.info(f"OpenRouter: Cached {len(fetched)} models to {self.cache_path} for 15 days.")
            except Exception as e:
                logger.warning(f"OpenRouter: Failed writing models cache ({e}).")
            return self._select_priority_models(fetched, default_fallback)

        # Fallback if fetch failed and no cache
        return default_fallback

    def fetch_remote_models(self) -> List[str]:
        """Queries OpenRouter /api/v1/models endpoint."""
        url = "https://openrouter.ai/api/v1/models"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                data = res.json()
                items = data.get("data", [])
                models = [item["id"] for item in items if "id" in item]
                return models
            else:
                logger.warning(f"OpenRouter models API returned status {res.status_code}")
        except Exception as e:
            logger.warning(f"OpenRouter models API request failed: {e}")
        return []

    def _select_priority_models(self, full_list: List[str], preferred_defaults: List[str]) -> List[str]:
        """Combines preferred models with top models from full list."""
        selected = []
        full_set = set(full_list)

        # Add preferred models first if present
        for pref in preferred_defaults:
            if pref in full_set:
                selected.append(pref)

        # Add other popular models from full list up to top 15
        for m in full_list:
            if m not in selected:
                selected.append(m)
            if len(selected) >= 15:
                break

        return selected or preferred_defaults
