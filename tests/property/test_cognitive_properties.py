"""Property tests for cognitive memory invariants (Stage 8, S8.2)."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from memory.usage import percentile_ranks, usage_score
from memory.vector_store import BudgetExceeded, CollectionBudget, VectorMemory, slugify
from memory.visual_style import visual_style

DAY = 86400.0
CHROMA_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")


@settings(max_examples=300, deadline=None)
@given(n=st.integers(0, 10_000), extra=st.integers(1, 1000), days=st.floats(0, 3650), imp=st.floats(0, 1),
       half=st.floats(0.5, 365))
def test_usage_rises_with_access_count(n, extra, days, imp, half):
    now = days * DAY
    assert usage_score(n + extra, 0.0, imp, now, half) >= usage_score(n, 0.0, imp, now, half)


@settings(max_examples=300, deadline=None)
@given(n=st.integers(1, 10_000), d1=st.floats(0, 3650), d2=st.floats(0, 3650), imp=st.floats(0, 1),
       half=st.floats(0.5, 365))
def test_usage_falls_with_time(n, d1, d2, imp, half):
    early, late = sorted((d1, d2))
    assert usage_score(n, 0.0, imp, late * DAY, half) <= usage_score(n, 0.0, imp, early * DAY, half) + 1e-12


@settings(max_examples=200, deadline=None)
@given(st.dictionaries(st.text(min_size=1, max_size=6), st.floats(0, 100, allow_nan=False), min_size=1, max_size=60))
def test_percentiles_are_bounded_and_order_preserving(scores):
    p = percentile_ranks(scores)
    assert set(p) == set(scores)
    assert all(0.0 <= v <= 1.0 for v in p.values())
    for a in scores:
        for b in scores:
            if scores[a] < scores[b]:
                assert p[a] < p[b]
            if scores[a] == scores[b]:
                assert p[a] == p[b]


@settings(max_examples=200, deadline=None)
@given(st.floats(0, 1), st.floats(0, 1))
def test_more_usage_never_looks_smaller_or_dimmer(a, b):
    lo, hi = sorted((a, b))
    s_lo, s_hi = visual_style(lo), visual_style(hi)
    assert s_hi["scale"] >= s_lo["scale"] and s_hi["opacity"] >= s_lo["opacity"]
    assert s_hi["emissive"] >= s_lo["emissive"] and s_hi["saturation"] >= s_lo["saturation"]


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=200))
def test_collection_names_are_always_chroma_safe(label):
    name = "peter_" + VectorMemory.knowledge_namespace(label)
    assert CHROMA_NAME.match(name), name
    assert ".." not in name
    assert slugify(label) == slugify(slugify(label))   # idempotent


_TMP = Path(tempfile.mkdtemp(prefix="peter-prop-"))
_VECTOR = None


def _vector() -> VectorMemory:
    global _VECTOR
    if _VECTOR is None:
        from tests.fakes.word_embedder import WordHashEmbedder
        _VECTOR = VectorMemory(str(_TMP / "chroma"), embedding_function=WordHashEmbedder())
    return _VECTOR


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cap=st.integers(0, 6), names=st.lists(st.sampled_from([f"t{i}" for i in range(10)]), max_size=14),
       run=st.uuids())
def test_cycle_budget_is_never_exceeded(cap, names, run):
    """Any sequence of create_collection calls makes at most `cap` new collections;
    re-opening an existing one is always free; the refusal leaves nothing behind."""
    vector = _vector()
    prefix = f"k_{run.hex[:8]}_"
    budget = CollectionBudget(max_new=cap)
    created = set()
    for n in names:
        ns = prefix + n
        try:
            if vector.create_collection(ns, budget=budget):
                created.add(ns)
        except BudgetExceeded:
            assert len(created) == cap
            assert not vector.has_collection(ns)
    assert len(created) <= cap
    assert set(budget.created) == created
    for ns in created:
        vector.drop_collection(ns)
