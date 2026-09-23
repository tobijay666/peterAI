"""Base LLM Provider interface using OpenAI-compatible protocol."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import openai
from core.interfaces import Message
from core.logger import logger


class ProviderError(Exception):
    """Raised when provider call fails."""

    pass


class BaseProvider(ABC):
    """Abstract base class for OpenAI-compatible API providers."""

    def __init__(self, name: str, base_url: str, api_key_env: str, models: List[str], priority: int = 10):
        self.name = name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.models = models
        self.priority = priority
        self.api_key = os.getenv(api_key_env, "")
        self.client: Optional[openai.OpenAI] = None
        self._init_client()

    def _init_client(self) -> None:
        if self.api_key:
            try:
                self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for {self.name}: {e}")
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        """Returns True if the API key is configured."""
        return bool(self.api_key and self.client)

    def probe(self, model: str, timeout: float = 8.0) -> bool:
        """Health check probe using a 1-token ping."""
        if not self.is_available():
            return False
        try:
            res = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=timeout,
            )
            return bool(res.choices and len(res.choices) > 0)
        except Exception as e:
            logger.debug(f"Health probe failed for {self.name}:{model}: {e}")
            return False

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Iterates through provider's model ladder to perform completion."""
        if not self.is_available():
            raise ProviderError(f"Provider {self.name} is missing API key ({self.api_key_env}).")

        formatted_msgs = [m.to_dict() for m in messages]
        last_exception = None

        for model in self.models:
            try:
                logger.debug(f"Attempting completion via {self.name} with model {model}...")
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": formatted_msgs,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools

                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]

                # Convert response to dictionary structure
                msg_obj = choice.message
                tool_calls_dict = None
                if msg_obj.tool_calls:
                    tool_calls_dict = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg_obj.tool_calls
                    ]

                result = {
                    "provider": self.name,
                    "model": model,
                    "content": msg_obj.content or "",
                    "role": msg_obj.role,
                    "tool_calls": tool_calls_dict,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                    if response.usage
                    else {},
                }
                logger.info(f"Completion successful from {self.name}:{model}")
                return result
            except Exception as e:
                logger.warning(f"Model {model} failed on provider {self.name}: {e}")
                last_exception = e
                continue

        raise ProviderError(f"All models exhausted on provider {self.name}. Last error: {last_exception}")
