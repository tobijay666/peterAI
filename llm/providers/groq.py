"""Groq provider implementation."""

from typing import List
from llm.providers.base import BaseProvider


class GroqProvider(BaseProvider):

    def __init__(
        self,
        name: str = "Groq",
        base_url: str = "https://api.groq.com/openai/v1",
        api_key_env: str = "GROQ_API_KEY",
        models: List[str] = None,
        priority: int = 20,
    ):
        if models is None:
            models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        super().__init__(
            name=name, base_url=base_url, api_key_env=api_key_env, models=models, priority=priority
        )
