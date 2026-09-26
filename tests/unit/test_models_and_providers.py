"""Unit tests for updated LLM providers, model ladders, rate limit parsing, and OpenRouter caching."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from core.interfaces import Message
from llm.providers.base import BaseProvider, ProviderError
from llm.providers.groq import GroqProvider
from llm.providers.nvidia import NvidiaProvider
from llm.providers.openrouter import OpenRouterProvider
from llm.router import LLMRouter


def test_groq_provider_models_and_rate_limits():
    """Validates Groq provider model ladder and rate limit header extraction."""
    provider = GroqProvider()
    assert "openai/gpt-oss-120b" in provider.models
    assert "openai/gpt-oss-20b" in provider.models
    assert "qwen/qwen3.8-27b" in provider.models

    headers = {
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-limit-tokens": "18000",
        "x-ratelimit-remaining-requests": "14370",
        "x-ratelimit-remaining-tokens": "17997",
        "x-ratelimit-reset-requests": "2m59s",
        "x-ratelimit-reset-tokens": "7.66s",
        "retry-after": "2",
    }
    parsed = provider._extract_rate_limits(headers)
    assert parsed["remaining_requests"] == "14370"
    assert parsed["remaining_tokens"] == "17997"
    assert parsed["retry_after"] == "2"
    assert provider.last_rate_limits == parsed


def test_nvidia_provider_models():
    """Validates NVIDIA provider updated model ladder."""
    provider = NvidiaProvider()
    expected_models = [
        "deepseek-ai/deepseek-v4.1-flash",
        "z-ai/glm-5-3",
        "z-ai/glm-5-3-flash",
        "moonshotai/kimi-k3",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "meta/llama-3.3-70b-instruct",
    ]
    for model in expected_models:
        assert model in provider.models


def test_openrouter_caching_and_ttl(temp_dir):
    """Validates OpenRouter model caching and 15-day TTL handling."""
    cache_file = temp_dir / "openrouter_models_cache.json"

    # Pre-populate valid cache (<15 days old)
    now = time.time()
    valid_cache = {
        "timestamp": now - (5 * 86400),  # 5 days old
        "models": ["meta-llama/llama-3.3-70b-instruct", "custom/test-model-1"],
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(valid_cache, f)

    provider = OpenRouterProvider(cache_path=cache_file)
    assert "meta-llama/llama-3.3-70b-instruct" in provider.models
    assert "custom/test-model-1" in provider.models

    # Test expired cache (>15 days old) triggers re-fetch
    expired_cache = {
        "timestamp": now - (16 * 86400),  # 16 days old
        "models": ["old/expired-model"],
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(expired_cache, f)

    with patch.object(OpenRouterProvider, "fetch_remote_models", return_value=["fetched/new-model"]):
        provider_refreshed = OpenRouterProvider(cache_path=cache_file)
        assert "fetched/new-model" in provider_refreshed.models

        # Verify updated cache file was saved
        with open(cache_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert "fetched/new-model" in saved_data["models"]


def test_router_integration_and_model_completion():
    """Validates LLMRouter integration with updated providers using mock completion."""
    router = LLMRouter("config/providers.yaml")
    assert "groq" in router.providers
    assert "nvidia" in router.providers
    assert "openrouter" in router.providers

    mock_provider = GroqProvider()
    mock_provider.api_key = "mock_key"

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello, Sir Anton."
    mock_choice.message.role = "assistant"
    mock_choice.message.tool_calls = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_raw_response = MagicMock()
    mock_raw_response.parse.return_value = mock_response
    mock_raw_response.headers = {
        "x-ratelimit-remaining-requests": "14000",
        "x-ratelimit-remaining-tokens": "17500",
    }

    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response
    mock_client.chat.completions.create.return_value = mock_response

    mock_provider.client = mock_client
    router.providers["groq"] = mock_provider

    res = router.complete(messages=[Message(role="user", content="Hello")], policy=["groq"])
    assert res["provider"] == "Groq"
    assert res["content"] == "Hello, Sir Anton."
    assert res["rate_limits"]["remaining_requests"] == "14000"
    assert router.get_token_usage()["groq"] == 15
