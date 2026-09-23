"""OpenRouter provider implementation."""

from typing import List
from llm.providers.base import BaseProvider


class OpenRouterProvider(BaseProvider):

    def __init__(
        self,
        name: str = "OpenRouter",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        models: List[str] = None,
        priority: int = 30,
    ):
        if models is None:
            models = [
                "meta-llama/llama-3.3-70b-instruct",
                "deepseek/deepseek-r1",
                "anthropic/claude-3.5-sonnet",
            ]
        super().__init__(
            name=name, base_url=base_url, api_key_env=api_key_env, models=models, priority=priority
        )
