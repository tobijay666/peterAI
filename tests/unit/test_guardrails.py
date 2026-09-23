"""Unit tests for GuardrailManager rule evaluation."""

import pytest
from core.guardrails import GuardrailManager


@pytest.fixture
def guardrails_manager():
    return GuardrailManager("config/guardrails.json")


def test_guardrail_path_traversal_blocked(guardrails_manager: GuardrailManager):
    res = guardrails_manager.check_tool_call("read_file", {"file_path": "../etc/passwd"})
    assert not res.allowed
    assert res.rule_id == "block_path_traversal"
    assert res.action == "block"


def test_guardrail_secret_file_blocked(guardrails_manager: GuardrailManager):
    res = guardrails_manager.check_tool_call("read_file", {"file_path": "d:/project/.env"})
    assert not res.allowed
    assert res.rule_id == "block_secret_extraction"


def test_guardrail_clean_input_allowed(guardrails_manager: GuardrailManager):
    res = guardrails_manager.check_tool_call("read_file", {"file_path": "d:/project/readme.txt"})
    assert res.allowed
    assert res.action == "allow"
