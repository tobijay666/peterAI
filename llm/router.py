"""LLM Router managing multi-provider fail-soft execution policies."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from core.interfaces import Message
from core.logger import logger
from llm.providers.base import BaseProvider, ProviderError
from llm.providers.groq import GroqProvider
from llm.providers.huggingface import HuggingFaceProvider
from llm.providers.nvidia import NvidiaProvider
from llm.providers.openrouter import OpenRouterProvider

PROVIDER_CLASSES = {
    "nvidia": NvidiaProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "huggingface": HuggingFaceProvider,
}


class LLMRouter:
    """Orchestrates provider execution policies, fallback sequences, and token accounting."""

    def __init__(
        self,
        config_path: str = "config/providers.yaml",
        providers: Optional[Dict[str, BaseProvider]] = None,
    ):
        self.config_path = Path(config_path)
        self.providers: Dict[str, BaseProvider] = {}
        self.token_stats: Dict[str, int] = {}
        if providers is not None:
            # Test/injection path: skip config loading entirely
            self.providers = providers
            self.token_stats = {k: 0 for k in providers}
        else:
            self.load_config()

    def load_config(self) -> None:
        if not self.config_path.exists():
            logger.warning(f"Router config {self.config_path} missing. Initializing default providers.")
            self._init_defaults()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                prov_data = data.get("providers", {})
                for key, p_info in prov_data.items():
                    cls = PROVIDER_CLASSES.get(key.lower(), BaseProvider)
                    provider = cls(
                        name=p_info.get("name", key),
                        base_url=p_info.get("base_url", ""),
                        api_key_env=p_info.get("api_key_env", ""),
                        models=p_info.get("models", []),
                        priority=p_info.get("priority", 50),
                    )
                    self.providers[key.lower()] = provider
                    self.token_stats[key.lower()] = 0
            logger.info(f"LLMRouter initialized with {len(self.providers)} providers.")
        except Exception as e:
            logger.error(f"Failed to load LLM Router config: {e}. Falling back to defaults.")
            self._init_defaults()

    def _init_defaults(self) -> None:
        self.providers = {
            "nvidia": NvidiaProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
            "huggingface": HuggingFaceProvider(),
        }
        for k in self.providers:
            self.token_stats[k] = 0

    def complete(
        self,
        messages: List[Message],
        policy: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Executes LLM completion using policy fallback ladder."""
        if not policy:
            # Sort all configured providers by priority
            sorted_provs = sorted(self.providers.keys(), key=lambda k: self.providers[k].priority)
            policy = sorted_provs

        errors = []
        for prov_key in policy:
            prov_key_clean = prov_key.lower()
            provider = self.providers.get(prov_key_clean)

            if not provider or not provider.is_available():
                logger.debug(f"Skipping provider {prov_key} (not available or missing API key).")
                continue

            try:
                res = provider.complete(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Track usage
                usage = res.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                self.token_stats[prov_key_clean] += total_tokens

                return res
            except Exception as e:
                msg = f"Provider {prov_key} failed: {e}"
                logger.warning(msg)
                errors.append(msg)
                continue

        raise ProviderError(f"All providers in policy {policy} failed. Details: {errors}")

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_stats.copy()
