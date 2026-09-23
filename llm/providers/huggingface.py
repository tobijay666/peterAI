"""HuggingFace Inference API provider implementation."""

from typing import List
from llm.providers.base import BaseProvider


class HuggingFaceProvider(BaseProvider):

    def __init__(
        self,
        name: str = "HuggingFace",
        base_url: str = "https://api-inference.huggingface.co/v1",
        api_key_env: str = "HF_API_KEY",
        models: List[str] = None,
        priority: int = 40,
    ):
        if models is None:
            models = [
                "meta-llama/Llama-3.3-70B-Instruct",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-72B-Instruct",
            ]
        super().__init__(
            name=name, base_url=base_url, api_key_env=api_key_env, models=models, priority=priority
        )
