"""FakeLLM — scripted replies for deterministic tests (A3)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from core.interfaces import Message


class ScriptExhausted(Exception):
    pass


class FakeLLMRouter:
    """A drop-in replacement for LLMRouter that returns scripted replies.

    Args:
        script: List of reply dicts in order. Each dict is returned as-is.
                  If exhausted, raises ScriptExhausted unless loop=True.
        loop: If True, cycle through the script indefinitely.
        default_reply: Reply to use when script is exhausted (if loop=False).

    Reply format (same as LLMRouter.complete return):
        {
            "content": "text reply",
            "tool_calls": [],   # optional
            "usage": {"total_tokens": 10},
            "provider": "fake",
            "model": "fake-model",
        }
    """

    def __init__(
        self,
        script: Optional[List[Dict[str, Any]]] = None,
        loop: bool = False,
        default_reply: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._script = list(script or [])
        self._loop = loop
        self._default = default_reply or {
            "content": "[FakeLLM default reply]",
            "tool_calls": [],
            "usage": {"total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2},
            "provider": "fake",
            "model": "fake-model",
        }
        self._index = 0
        self.calls: List[Dict[str, Any]] = []  # history for assertions

    def complete(
        self,
        messages: List[Message],
        policy: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        self.calls.append({"messages": messages, "policy": policy, "tools": tools})
        if self._index < len(self._script):
            reply = self._script[self._index]
            self._index = (self._index + 1) % len(self._script) if self._loop else self._index + 1
            return reply
        if self._loop and self._script:
            reply = self._script[self._index % len(self._script)]
            self._index += 1
            return reply
        return self._default

    def get_token_usage(self) -> Dict[str, int]:
        return {"fake": sum(c.get("usage", {}).get("total_tokens", 0) for c in self.calls)}


FakeLLM = FakeLLMRouter
