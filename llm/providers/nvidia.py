"""NVIDIA NIM provider implementation."""

from typing import List
from llm.providers.base import BaseProvider


class NvidiaProvider(BaseProvider):

    def __init__(
        self,
        name: str = "NVIDIA",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key_env: str = "NVIDIA_API_KEY",
        models: List[str] = None,
        priority: int = 10,
    ):
        if models is None:
            models = [
                "deepseek-ai/deepseek-v4.1-flash",
                "z-ai/glm-5-3",
                "z-ai/glm-5-3-flash",
                "moonshotai/kimi-k3",
                "nvidia/nemotron-3.5-lightning-30b-a3b",
                "meta/llama-3.3-70b-instruct",
            ]
        super().__init__(
            name=name, base_url=base_url, api_key_env=api_key_env, models=models, priority=priority
        )
