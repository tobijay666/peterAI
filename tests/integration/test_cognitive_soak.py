"""60-day cognitive soak as a test (marked slow; ~1 min). Full 90 days: scripts/soak_cognitive.py."""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_memory_reaches_steady_state_over_60_days():
    from scripts.soak_cognitive import run
    res = run(days=60, seed=11)
    assert res["passed"], res
