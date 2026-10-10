"""Injectable clock abstraction for Peter AI.

Replaces all direct time.time() calls in the codebase so that tests can
use a FakeClock to advance time deterministically without sleeping.

Usage:
    # Production code (SystemClock is the default)
    from core.clock import SystemClock
    clock = SystemClock()
    now = clock.now()

    # Tests
    from core.clock import FakeClock
    clock = FakeClock(start=0.0)
    clock.advance(days=31)
    now = clock.now()   # returns 31 * 86400

Design rule: time.time() MUST NOT appear outside this module.
A test (test_no_direct_time.py) enforces this by grepping the source tree.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for an injectable clock that returns the current epoch time."""

    def now(self) -> float:
        """Return the current time as a float (seconds since epoch)."""
        ...


class SystemClock:
    """Production clock — delegates to time.time()."""

    def now(self) -> float:  # noqa: D102
        return time.time()


class FakeClock:
    """Test clock — starts at a fixed epoch and advances on demand.

    Args:
        start: Initial epoch time (default 0.0 — Jan 1 1970).

    Example::

        clock = FakeClock(start=1_000_000.0)
        clock.advance(days=31)
        assert clock.now() == 1_000_000.0 + 31 * 86_400
    """

    def __init__(self, start: float = 0.0) -> None:
        self._current = start

    def now(self) -> float:  # noqa: D102
        return self._current

    def advance(
        self,
        seconds: float = 0.0,
        minutes: float = 0.0,
        hours: float = 0.0,
        days: float = 0.0,
    ) -> None:
        """Advance the clock by the given duration.

        All arguments are additive::

            clock.advance(days=1, hours=2)  # advances by 26 hours
        """
        self._current += seconds + minutes * 60 + hours * 3600 + days * 86_400

    def set(self, epoch: float) -> None:
        """Directly set the clock to an absolute epoch value."""
        self._current = epoch
