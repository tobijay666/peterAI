"""Pytest fixtures for unit and integration testing."""

import sys
import tempfile
import pytest
from pathlib import Path
from core.guardrails import GuardrailManager
from memory.vector_store import VectorMemory
from memory.graph_store import GraphMemory

from core.clock import FakeClock
from tests.fakes.fake_llm import FakeLLMRouter
from tests.fakes.fake_embedder import FakeEmbeddingFunction


@pytest.fixture
def temp_dir():
    # ignore_cleanup_errors prevents false PermissionError teardown failures on Windows
    # caused by SQLite WAL / ChromaDB sqlite3 files being briefly locked after tests.
    kwargs = {"ignore_cleanup_errors": True} if sys.platform == "win32" else {}
    with tempfile.TemporaryDirectory(**kwargs) as tmp:
        yield Path(tmp)


@pytest.fixture
def guardrails(temp_dir):
    config_file = temp_dir / "guardrails.json"
    config_file.write_text(
        """{
  "rules": [
    {
      "id": "block_path_traversal",
      "severity": "block",
      "applies_to": ["tool_call"],
      "pattern": "(\\\\.\\\\.[\\\\/\\\\]|%2e%2e)",
      "description": "Path traversal check"
    },
    {
      "id": "block_secret_extraction",
      "severity": "block",
      "applies_to": ["tool_call", "output"],
      "pattern": "(?i)(\\\\.(env|pem|key)|id_rsa|\\\\.ssh)",
      "description": "Secret check"
    }
  ]
}""",
        encoding="utf-8",
    )
    return GuardrailManager(str(config_file))


@pytest.fixture
def vector_memory(temp_dir):
    chroma_dir = temp_dir / "chroma"
    return VectorMemory(persist_directory=str(chroma_dir))


@pytest.fixture
def graph_memory(temp_dir):
    db_file = temp_dir / "test_graph.db"
    gm = GraphMemory(database_path=str(db_file))
    yield gm
    gm.close()  # checkpoint WAL and release file lock before temp dir cleanup

@pytest.fixture
def fake_clock():
    """A FakeClock starting at epoch 0. Advance with clock.advance(days=N)."""
    return FakeClock(start=0.0)

@pytest.fixture
def fake_llm():
    """A FakeLLMRouter with default empty script (returns default reply)."""
    return FakeLLMRouter()

@pytest.fixture
def fake_embedder():
    """A FakeEmbeddingFunction — deterministic, no network required."""
    return FakeEmbeddingFunction()
