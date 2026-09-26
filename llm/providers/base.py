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
        self.last_rate_limits: Dict[str, Any] = {}
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

    def _extract_rate_limits(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Extracts standard rate limit headers (Groq, OpenAI, etc.)."""
        rl = {}
        header_map = {
            "x-ratelimit-limit-requests": "limit_requests",
            "x-ratelimit-limit-tokens": "limit_tokens",
            "x-ratelimit-remaining-requests": "remaining_requests",
            "x-ratelimit-remaining-tokens": "remaining_tokens",
            "x-ratelimit-reset-requests": "reset_requests",
            "x-ratelimit-reset-tokens": "reset_tokens",
            "retry-after": "retry_after",
        }
        for h_key, r_key in header_map.items():
            val = headers.get(h_key) or headers.get(h_key.lower())
            if val is not None:
                rl[r_key] = val

        if rl:
            self.last_rate_limits = rl
            rem_req = rl.get("remaining_requests")
            rem_tok = rl.get("remaining_tokens")
            if rem_req and str(rem_req).isdigit() and int(rem_req) < 10:
                logger.warning(f"[{self.name}] Low remaining requests: {rem_req}")
            if rem_tok and str(rem_tok).isdigit() and int(rem_tok) < 1000:
                logger.warning(f"[{self.name}] Low remaining tokens: {rem_tok}")

        return rl

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

                headers = {}
                if hasattr(self.client.chat.completions, "with_raw_response"):
                    try:
                        raw_res = self.client.chat.completions.with_raw_response.create(**kwargs)
                        response = raw_res.parse()
                        headers = dict(raw_res.headers)
                    except Exception:
                        response = self.client.chat.completions.create(**kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)

                rate_limits = self._extract_rate_limits(headers)

                choice = response.choices[0]
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
                    "rate_limits": rate_limits,
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
