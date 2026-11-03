"""Base LLM Provider interface using OpenAI-compatible protocol."""

import re
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import openai
from core.interfaces import Message
from core.logger import logger



_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (and an unterminated leading one)."""
    text = _THINK_RE.sub("", text or "")
    if text.lstrip().startswith("<think>"):
        return ""
    return text


def _is_request_specific_error(e: Exception) -> bool:
    """A 400 caused by this request's content (e.g. the model emitted a tool call that failed
    schema validation) says nothing about the model's health."""
    msg = str(e).lower()
    status = getattr(e, "status_code", None)
    return (status == 400 or "error code: 400" in msg) and any(
        k in msg for k in ("tool_use_failed", "tool call validation", "invalid_request_error", "failed_generation"))

class ProviderError(Exception):
    """Raised when provider call fails."""

    pass


class BaseProvider(ABC):
    """Abstract base class for OpenAI-compatible API providers.

    Key improvements over v1:
    - Per-session model blacklist: failed models are skipped immediately on
      subsequent steps, rather than waiting for the 10-minute health prober.
    - Hard per-request timeout (COMPLETION_TIMEOUT_SEC, default 90s) to
      prevent indefinite hangs on stalled provider connections.
    """

    def __init__(self, name: str, base_url: str, api_key_env: str, models: List[str], priority: int = 10):
        self.name = name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.models = models
        self.priority = priority
        self.api_key = os.getenv(api_key_env, "")
        self.client: Optional[openai.OpenAI] = None
        self.last_rate_limits: Dict[str, Any] = {}
        # Per-session model blacklist: models that failed during the current run.
        # Updated immediately on failure so the NEXT step skips the dead model,
        # instead of waiting for the 10-minute background health probe cycle.
        self._failed_models: set = set()
        self._completion_timeout: float = float(os.getenv("COMPLETION_TIMEOUT_SEC", "15"))
        self._init_client()

    def _init_client(self) -> None:
        if self.api_key:
            try:
                self.client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self._completion_timeout,
                    max_retries=0,
                )
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for {self.name}: {e}")
                self.client = None
        else:
            self.client = None

    def is_available(self) -> bool:
        """Returns True if the API key is configured and client is initialized."""
        return bool(self.api_key and self.client)

    def mark_model_failed(self, model: str) -> None:
        """Immediately blacklists a model for the duration of this session.

        Called internally after any completion failure. The health prober
        calls clear_failed_models() once a re-probe succeeds, allowing
        re-admission.
        """
        if model not in self._failed_models:
            self._failed_models.add(model)
            logger.warning(f"[{self.name}] Model '{model}' blacklisted for this session after failure.")

    def clear_failed_models(self) -> None:
        """Clears the session blacklist (called by health prober on successful re-probe)."""
        self._failed_models.clear()

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
            if res.choices and len(res.choices) > 0:
                # Successful probe — remove from blacklist so it can be used again
                self._failed_models.discard(model)
                return True
            return False
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
        """Iterates through provider's model ladder to perform completion.

        Blacklisted models are skipped immediately.
        Each request enforces COMPLETION_TIMEOUT_SEC (default 90s) to prevent
        indefinite hangs on stalled provider connections.
        """
        if not self.is_available():
            raise ProviderError(f"Provider {self.name} is missing API key ({self.api_key_env}).")

        formatted_msgs = [m.to_dict() for m in messages]
        last_exception = None

        for model in self.models:
            # Skip models that already failed this session — no waiting
            if model in self._failed_models:
                logger.debug(f"[{self.name}] Skipping blacklisted model '{model}'.")
                continue

            try:
                logger.debug(f"Attempting completion via {self.name} with model {model}...")
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": formatted_msgs,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": self._completion_timeout,  # Hard timeout — prevents 27-min hangs
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
                    # Some reasoning models (e.g. Qwen3) put their thinking inside the reply
                    "content": _strip_thinking(msg_obj.content or ""),
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
                if _is_request_specific_error(e):
                    # The model produced one bad tool call / the request was rejected:
                    # try the next model for THIS call, but keep the model for later calls.
                    logger.info(f"[{self.name}] '{model}' rejected this request only; not blacklisting.")
                else:
                    # Timeouts, outages, missing models, auth: skip it for the rest of the session
                    self.mark_model_failed(model)
                last_exception = e
                continue

        raise ProviderError(f"All models exhausted on provider {self.name}. Last error: {last_exception}")
