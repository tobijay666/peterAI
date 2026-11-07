"""
UserContext — carries identity and resource limits for one signed-in user.

This is the single place that defines how a user_id flows through the system.
Every call that touches memory, tools, or the agent loop receives a
UserContext so user A can never touch user B's data.

For the terminal REPL the default anonymous context is used, which behaves
exactly as before (no change to existing behaviour).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------

@dataclass
class UserContext:
    """Carries the identity and per-user resource paths for one session."""

    user_id: str
    """Opaque user identifier (JWT sub claim, or 'local' for the REPL)."""

    display_name: str = "User"
    """Human-readable name shown in Peter's responses."""

    data_root: Path = field(default_factory=lambda: Path("data"))
    """Root directory under which this user's data lives."""

    max_messages_per_day: int = field(
        default_factory=lambda: int(os.getenv("USER_MAX_MESSAGES_PER_DAY", "200"))
    )
    """Daily message cap — enforced by the server, ignored in REPL mode."""

    is_anonymous: bool = False
    """True for the local REPL user — skips quota and web-only restrictions."""

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @property
    def chroma_dir(self) -> Path:
        """ChromaDB persist directory scoped to this user."""
        if self.is_anonymous:
            return self.data_root / "chroma"
        return self.data_root / "users" / self.user_id / "chroma"

    @property
    def graph_db_path(self) -> Path:
        """SQLite graph-store path scoped to this user."""
        if self.is_anonymous:
            return self.data_root / "peter_graph.db"
        return self.data_root / "users" / self.user_id / "peter_graph.db"

    @property
    def events_log_path(self) -> Path:
        """JSON-lines event log for this user's session (used for replay)."""
        if self.is_anonymous:
            return self.data_root / "events.jsonl"
        return self.data_root / "users" / self.user_id / "events.jsonl"

    def ensure_dirs(self) -> None:
        """Create all required data directories for this user."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.graph_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_log_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def anonymous_context() -> UserContext:
    """
    Returns the default context used by the terminal REPL.
    Paths resolve to the same locations as before multi-user support was added,
    so the REPL continues to work without any migration.
    """
    return UserContext(
        user_id="local",
        display_name="Sir Anton",
        is_anonymous=True,
    )


def web_context(user_id: str, display_name: str = "User") -> UserContext:
    """
    Returns a context for a web-authenticated user.
    Data is stored under data/users/<user_id>/.
    """
    ctx = UserContext(
        user_id=user_id,
        display_name=display_name,
        is_anonymous=False,
    )
    ctx.ensure_dirs()
    return ctx
