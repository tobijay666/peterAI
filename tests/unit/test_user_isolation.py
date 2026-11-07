"""
Unit tests for user-data isolation (Phase 1).

These tests are self-contained and do not require API keys, a running
server, ChromaDB or SQLite — they only verify that the helper functions
compute the correct paths and that two different users never share them.
"""

import pytest
from pathlib import Path
from core.user_context import UserContext, anonymous_context, web_context


class TestUserContext:
    def test_anonymous_uses_legacy_paths(self):
        """The anonymous (REPL) context must resolve to the original data paths."""
        ctx = anonymous_context()
        assert ctx.chroma_dir   == Path("data/chroma"),          ctx.chroma_dir
        assert ctx.graph_db_path == Path("data/peter_graph.db"), ctx.graph_db_path

    def test_web_users_are_scoped_under_users(self):
        ctx = web_context("user-abc", "Alice")
        assert "users" in ctx.chroma_dir.parts
        assert "user-abc" in ctx.chroma_dir.parts
        assert "users" in ctx.graph_db_path.parts
        assert "user-abc" in ctx.graph_db_path.parts

    def test_two_users_never_share_paths(self):
        a = web_context("alice", "Alice")
        b = web_context("bob",   "Bob")
        assert a.chroma_dir    != b.chroma_dir
        assert a.graph_db_path != b.graph_db_path
        assert a.events_log_path != b.events_log_path

    def test_anonymous_is_not_web(self):
        anon = anonymous_context()
        web  = web_context("u1")
        assert anon.is_anonymous is True
        assert web.is_anonymous  is False

    def test_anonymous_path_never_contains_users(self):
        ctx = anonymous_context()
        assert "users" not in ctx.chroma_dir.parts
        assert "users" not in ctx.graph_db_path.parts

    def test_web_user_id_in_all_paths(self):
        uid = "test-user-99"
        ctx = web_context(uid)
        assert uid in str(ctx.chroma_dir)
        assert uid in str(ctx.graph_db_path)
        assert uid in str(ctx.events_log_path)
