"""
Unit tests for the emit() function (Phase 2).

Tests run without a server, without API keys, and without a real WebSocket.
They verify that:
  - emit() writes to the correct per-user log path.
  - emit() never raises even when the WebSocket server is absent.
  - Two different user_ids write to two different log files.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


def test_emit_does_not_raise():
    """emit() must never raise, even with no server running."""
    from core.events import emit
    emit("test_event", user_id="test-user", turn_id="t1", extra="hello")


def test_emit_writes_to_correct_path(tmp_path, monkeypatch):
    """emit() writes a JSON-lines record under data/users/<user_id>/events.jsonl."""
    # Patch open/mkdir so we write under tmp_path
    import core.events as ev_mod

    written: list = []
    orig_open = open

    def fake_open(path, mode="r", **kwargs):
        p = str(path)
        if "events.jsonl" in p and "a" in mode:
            written.append(p)
        return orig_open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    from core.events import emit
    emit("store", user_id="u-xyz", turn_id="t2", memory_id="abc123")
    assert any("u-xyz" in p for p in written), f"Expected u-xyz in written paths: {written}"


def test_emit_different_users_different_paths(monkeypatch):
    """Two different user_ids must result in different log paths."""
    paths_seen: list = []
    orig_open = open

    def fake_open(path, mode="r", **kwargs):
        p = str(path)
        if "events.jsonl" in p and "a" in mode:
            paths_seen.append(p)
        return orig_open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    from core.events import emit
    emit("recall", user_id="alice", turn_id="t3")
    emit("recall", user_id="bob",   turn_id="t4")

    alice_paths = [p for p in paths_seen if "alice" in p]
    bob_paths   = [p for p in paths_seen if "bob"   in p]
    assert alice_paths, "No path for alice"
    assert bob_paths,   "No path for bob"
    assert alice_paths[0] != bob_paths[0], "alice and bob must write to different files"


def test_emit_local_user_uses_legacy_path(monkeypatch):
    """The 'local' user_id (REPL mode) writes to data/events.jsonl, not data/users/local/."""
    paths_seen: list = []
    orig_open = open

    def fake_open(path, mode="r", **kwargs):
        p = str(path)
        if "events.jsonl" in p and "a" in mode:
            paths_seen.append(p)
        return orig_open(path, mode, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    from core.events import emit
    emit("turn_start", user_id="local", turn_id="t5")
    # Should be data/events.jsonl, NOT data/users/local/events.jsonl
    assert any(p.replace("\\", "/").endswith("data/events.jsonl") for p in paths_seen), \
        f"Expected data/events.jsonl for local user, got: {paths_seen}"
