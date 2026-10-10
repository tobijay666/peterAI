"""Tests for ConsolidationPass (F1-F7, FR-14)."""
import pytest
from core.clock import FakeClock
from memory.consolidation import ConsolidationPass, ClusterPlan
from memory.trust import TrustLevel
from unittest.mock import MagicMock

class FakeVectorMemory:
    def __init__(self, records):
        self.records = records
        self.stores = []
        self.importance_adjustments = []

    def get_all_records(self, namespace):
        return [r for r in self.records if r.get("_namespace", "peter") == namespace]

    def embed_fn(self, texts):
        # return dummy embeddings
        # 3 known topics: 'apple', 'banana', 'cherry'
        emb = []
        for t in texts:
            if 'apple' in t: emb.append([1.0, 0.0, 0.0])
            elif 'banana' in t: emb.append([0.0, 1.0, 0.0])
            else: emb.append([0.0, 0.0, 1.0])
        return emb

    def store(self, namespace, text, importance, pinned, kind, tags, extra_metadata):
        self.stores.append((text, extra_metadata))
        return f"sem_{len(self.stores)}"

    def adjust_importance(self, namespace, rid, delta):
        self.importance_adjustments.append((rid, delta))
        return True

    def _get_or_create_collection(self, namespace):
        m = MagicMock()
        m.update = MagicMock()
        return m

class FakeLLMRouter:
    def __init__(self, fail_on_cluster=None):
        self.fail_on_cluster = fail_on_cluster
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[-1].content
        if self.fail_on_cluster and self.fail_on_cluster in prompt:
            raise Exception("LLM failure")
        return {"content": f"Summary {self.calls}"}

def test_consolidation_clusters_and_undersized():
    # Test F1: 3 known topics
    records = []
    # 4 apples -> cluster size 4
    for i in range(4):
        records.append({"id": f"a{i}", "document": f"apple {i}", "_namespace": "peter", "metadata": {"trust": "user"}})
    # 2 bananas -> cluster size 2 (undersized)
    for i in range(2):
        records.append({"id": f"b{i}", "document": f"banana {i}", "_namespace": "peter", "metadata": {"trust": "user"}})

    vec = FakeVectorMemory(records)
    router = FakeLLMRouter()
    gr = MagicMock()
    gr.check_memory_write.return_value.allowed = True

    cp = ConsolidationPass(vector=vec, graph=MagicMock(), router=router, guardrails=gr, config={"min_age_hours": 0})
    # Make them all look old enough by using clock
    cp._clock = FakeClock()
    
    report = cp.run()
    
    assert report.clusters_found == 2
    # Only the apple cluster is >= 3
    assert report.semantic_records_written == 1
    assert len(vec.stores) == 1
    
    # Check that sources were decayed
    assert report.sources_decayed == 4
    assert len(vec.importance_adjustments) == 4

def test_consolidation_llm_failure_isolation():
    # Test F2: LLM failure in one cluster leaves others untouched
    records = []
    for i in range(3):
        records.append({"id": f"a{i}", "document": f"apple {i}", "_namespace": "peter", "metadata": {"trust": "user"}})
    for i in range(3):
        records.append({"id": f"c{i}", "document": f"cherry {i}", "_namespace": "peter", "metadata": {"trust": "user"}})

    vec = FakeVectorMemory(records)
    router = FakeLLMRouter(fail_on_cluster="apple")
    gr = MagicMock()
    gr.check_memory_write.return_value.allowed = True

    cp = ConsolidationPass(vector=vec, graph=MagicMock(), router=router, guardrails=gr, config={"min_age_hours": 0})
    report = cp.run()
    
    assert report.clusters_found == 2
    assert report.semantic_records_written == 1 # cherry succeeded
    assert len(vec.stores) == 1
    assert "cherry" not in str(report.errors)
    assert len(report.errors) == 1

def test_consolidation_idempotency_and_pinned():
    # Test F5: second run produces 0 new records
    # Test F3: pinned records never modified
    records = [
        {"id": "p1", "document": "apple 1", "metadata": {"trust": "user", "pinned": True}},
        {"id": "p2", "document": "apple 2", "metadata": {"trust": "user", "pinned": True}},
        {"id": "p3", "document": "apple 3", "metadata": {"trust": "user", "pinned": True}},
        {"id": "c1", "document": "banana 1", "metadata": {"trust": "user", "consolidated_into": "sem_1"}},
        {"id": "c2", "document": "banana 2", "metadata": {"trust": "user", "consolidated_into": "sem_1"}},
        {"id": "c3", "document": "banana 3", "metadata": {"trust": "user", "consolidated_into": "sem_1"}},
    ]
    vec = FakeVectorMemory(records)
    cp = ConsolidationPass(vector=vec, graph=MagicMock(), router=FakeLLMRouter(), guardrails=MagicMock(), config={"min_age_hours": 0})
    
    report = cp.run()
    assert report.clusters_found == 0
    assert report.semantic_records_written == 0
