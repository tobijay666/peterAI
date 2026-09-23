"""Hypothesis property-based tests for GuardrailManager."""

from hypothesis import given, strategies as st
from core.guardrails import GuardrailManager


@given(st.text())
def test_guardrails_never_crash(input_text: str):
    gm = GuardrailManager("config/guardrails.json")
    res = gm.check_input(input_text)
    assert res is not None
    assert isinstance(res.allowed, bool)
