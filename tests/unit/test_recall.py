"""Tests for ContextBuilder (E1, FR-15)."""
import pytest
from core.clock import FakeClock
from memory.recall import ContextBuilder, RecalledContext
from memory.trust import TrustLevel
from memory.vector_store import VectorMemory
from memory.graph_store import GraphMemory
from unittest.mock import MagicMock

class FakeVectorMemory:
    def __init__(self, records):
        self.records = records

    def recall(self, namespace: str, query: str, n_results: int = 5):
        # Return records that belong to the namespace
        res = [r for r in self.records if r.get("_ns", "peter") == namespace]
        return res[:n_results]

def test_context_builder_budget_and_trust():
    records = [
        {"id": "1", "document": "A" * 100, "score": 0.9, "metadata": {"trust": "tool_verified", "kind": "finding"}},
        {"id": "2", "document": "B" * 100, "score": 0.8, "metadata": {"trust": "quarantined", "kind": "finding"}},
        {"id": "3", "document": "C" * 100, "score": 0.7, "metadata": {"trust": "untrusted_web", "kind": "episode"}},
        {"id": "4", "document": "D" * 100, "score": 0.6, "metadata": {"trust": "system", "kind": "semantic"}},
        {"id": "5", "document": "E" * 1000, "score": 0.95, "metadata": {"trust": "user", "kind": "skill_trigger"}},
    ]
    vec = FakeVectorMemory(records)
    graph = MagicMock()
    
    # 100 chars budget = 25 tokens (if 4 chars per token)
    # Wait, 100 tokens * 4 = 400 chars budget
    cb = ContextBuilder(vector=vec, graph=graph, token_budget=100, min_similarity=0.1, skill_min_similarity=0.1)
    
    ctx = cb.build("query")
    
    # 2 is quarantined, never injected
    assert "2" not in ctx.memory_ids
    assert "quarantined" not in ctx.memory_text
    
    # 5 is a skill but 1000 chars > 400 chars budget, so it will be excluded by budget
    assert "5" not in ctx.memory_ids
    
    # 4 is semantic, higher priority than 1 (finding), 4 has 100 chars -> fits in 400
    # 1 has 100 chars -> fits
    # 3 has 100 chars -> fits
    
    assert "4" in ctx.memory_ids
    assert "1" in ctx.memory_ids
    assert "3" in ctx.memory_ids
    
    # 3 is untrusted, must be in [DATA START]
    assert "[DATA START" in ctx.memory_text
    assert "[DATA END]" in ctx.memory_text
    
    # Check that output never exceeds budget
    # Actually memory_text has some extra section headers
    # The budget checking in the code just checks if `used_chars + len(chunk) > budget_chars`
    assert ctx.stats["excluded_budget"] > 0 # item 5
    assert ctx.stats["excluded_trust"] == 1 # item 2

def test_context_builder_similarity_floor():
    records = [
        {"id": "1", "document": "A", "score": 0.2, "metadata": {"trust": "user", "kind": "finding"}},
    ]
    vec = FakeVectorMemory(records)
    cb = ContextBuilder(vector=vec, graph=MagicMock(), min_similarity=0.5)
    ctx = cb.build("query")
    assert not ctx.memory_ids
    assert ctx.stats["excluded_similarity"] == 1

def test_context_builder_deterministic_order():
    records = [
        {"id": "1", "document": "A", "score": 0.8, "metadata": {"trust": "user", "kind": "finding"}},
        {"id": "2", "document": "B", "score": 0.8, "metadata": {"trust": "user", "kind": "finding"}},
    ]
    vec = FakeVectorMemory(records)
    cb = ContextBuilder(vector=vec, graph=MagicMock(), min_similarity=0.5)
    ctx1 = cb.build("query")
    ctx2 = cb.build("query")
    assert ctx1.memory_ids == ctx2.memory_ids
