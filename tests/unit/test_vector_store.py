"""Unit tests for VectorMemory operations."""

from memory.vector_store import VectorMemory


def test_vector_store_roundtrip(vector_memory: VectorMemory):
    doc_id = vector_memory.store(
        namespace="peter", text="Sir Anton requested python multi-agent system architecture.", importance=0.8
    )
    assert doc_id is not None

    recalled = vector_memory.recall(namespace="peter", query="multi-agent python system")
    assert len(recalled) > 0
    assert "Sir Anton" in recalled[0]["document"]


def test_vector_store_pinning(vector_memory: VectorMemory):
    doc_id = vector_memory.store(namespace="researcher", text="Pinned critical finding", pinned=True)
    vector_memory.pin(namespace="researcher", doc_id=doc_id, pinned=True)

    # Prune should skip pinned record
    pruned = vector_memory.prune(namespace="researcher", doc_ids=[doc_id])
    assert pruned == 0

    recalled = vector_memory.recall(namespace="researcher", query="critical finding")
    assert len(recalled) == 1
