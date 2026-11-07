"""Fixtures for the cognitive-memory test suite (Stages 0–8).

Everything is deterministic and offline: WordHashEmbedder for similarity,
FakeClock for time, FakeLLMRouter / scripted functions for the LLM, and fake
search/fetch functions for the web.
"""
from __future__ import annotations

import pytest

from core.clock import FakeClock
from core.cognition_config import set_cognition_config
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tests.fakes.word_embedder import WordHashEmbedder

DAY = 86400.0


@pytest.fixture(autouse=True)
def _reset_cognition_config():
    set_cognition_config(None)
    yield
    set_cognition_config(None)


@pytest.fixture
def clock():
    # Start well after epoch so "days since" arithmetic never goes negative
    return FakeClock(start=1_000 * DAY)


@pytest.fixture
def stores(tmp_path, clock):
    vector = VectorMemory(persist_directory=str(tmp_path / "chroma"), clock=clock,
                          user_id="u-test", embedding_function=WordHashEmbedder())
    graph = GraphMemory(database_path=str(tmp_path / "graph.db"), clock=clock, user_id="u-test")
    yield vector, graph
    graph.close()


@pytest.fixture
def events(monkeypatch):
    """Capture every emit() call made during the test: list of payload dicts."""
    captured = []
    import core.events as ev

    def fake_log(user_id, payload):
        captured.append(payload)

    monkeypatch.setattr(ev, "_write_event_log", fake_log)
    return captured


def of_type(events, t):
    return [e for e in events if e.get("type") == t]
