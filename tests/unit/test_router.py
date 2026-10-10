"""Test LLMRouter: provider failure fallback, circuit breaking (FR-9.4)."""

import pytest
from typing import Any, Dict, List, Optional
from core.interfaces import Message
from llm.router import LLMRouter
from llm.providers.base import BaseProvider, ProviderError

class FakeProvider(BaseProvider):
    def __init__(self, name: str, should_fail: bool = False, is_down: bool = False):
        super().__init__(name=name, base_url="", api_key_env="FAKE_KEY", models=[])
        self.should_fail = should_fail
        self.is_down = is_down
        self.call_count = 0

    def is_available(self) -> bool:
        return not self.is_down

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        self.call_count += 1
        if self.should_fail:
            raise ProviderError(f"{self.name} failed")
        return {"content": f"{self.name} success", "usage": {"total_tokens": 10}}


@pytest.fixture
def router():
    providers = {
        "prov1": FakeProvider("prov1", should_fail=True),
        "prov2": FakeProvider("prov2", should_fail=False),
        "prov3": FakeProvider("prov3", is_down=True),
    }
    r = LLMRouter(providers=providers)
    return r

def test_router_fallback(router):
    """Provider failure falls through to next provider (FR-9.4)."""
    messages = [Message(role="user", content="hi")]
    policy = ["prov1", "prov2"]
    
    res = router.complete(messages=messages, policy=policy)
    
    assert res["content"] == "prov2 success"
    assert router.providers["prov1"].call_count == 1
    assert router.providers["prov2"].call_count == 1
    assert router.token_stats["prov2"] == 10

def test_router_all_fail(router):
    """All providers down raises ProviderError."""
    messages = [Message(role="user", content="hi")]
    policy = ["prov1", "prov3"]
    
    with pytest.raises(ProviderError, match="All providers in policy .* failed"):
        router.complete(messages=messages, policy=policy)

    assert router.providers["prov1"].call_count == 1
    assert router.providers["prov3"].call_count == 0  # Is down, so skipped
