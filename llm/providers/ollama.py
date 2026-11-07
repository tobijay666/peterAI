"""Ollama local LLM provider — zero-network-cost first-priority fallback.

Connects to a locally running Ollama server (http://localhost:11434).
Ollama exposes an OpenAI-compatible /v1 endpoint, so it fits naturally
into the existing BaseProvider without any special handling.

Configuration:
  OLLAMA_API_KEY  : Not required by Ollama, but the openai client needs a
                    non-empty string. Defaults to the sentinel "ollama".
  OLLAMA_MODELS   : Comma-separated list of model names to try in order.
                    Defaults to common small models.
"""

import os
from typing import List, Optional

from llm.providers.base import BaseProvider


class OllamaProvider(BaseProvider):
    """Connects to a locally running Ollama instance at http://localhost:11434."""

    def __init__(
        self,
        name: str = "Ollama",
        base_url: str = "http://localhost:11434/v1",
        api_key_env: str = "OLLAMA_API_KEY",
        models: Optional[List[str]] = None,
        priority: int = 5,        # Highest priority — local, free, no rate limits
        timeout: float = 60.0,    # Ollama can be slow on first token (model load)
    ):
        # Ollama doesn't use real auth; inject a sentinel so the openai client
        # doesn't refuse to initialise.
        if not os.getenv("OLLAMA_API_KEY"):
            os.environ["OLLAMA_API_KEY"] = "ollama"

        if models is None:
            env_models = os.getenv("OLLAMA_MODELS", "").strip()
            if env_models:
                models = [m.strip() for m in env_models.split(",") if m.strip()]
            else:
                models = [
                    "llama3.2:3b",
                    "llama3.1:8b",
                    "qwen2.5:7b",
                    "mistral:7b",
                    "phi3:mini",
                ]

        super().__init__(
            name=name,
            base_url=base_url,
            api_key_env=api_key_env,
            models=models,
            priority=priority,
            timeout=timeout,
        )

    def is_available(self) -> bool:
        """Returns True only if the Ollama server is reachable AND has models installed.

        Also trims self.models to only those actually present on the server,
        so the model ladder never tries a model that isn't pulled.
        """
        if not self.client:
            return False
        try:
            import requests as _requests
            r = _requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code != 200:
                return False
            server_models = r.json().get("models", [])
            if not server_models:
                return False

            # Build a set of installed model names (with and without tags)
            installed_full = {m.get("name", "") for m in server_models}
            installed_base = {n.split(":")[0] for n in installed_full}

            # Keep only self.models that are installed
            available = [
                m for m in self.models
                if m in installed_full or m.split(":")[0] in installed_base
            ]

            if available:
                self.models = available

            return bool(available)
        except Exception:
            return False
