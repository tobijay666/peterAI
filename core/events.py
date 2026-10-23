"""In-process pub/sub event bus for decoupled component communication."""

from typing import Any, Callable, Dict, List
from core.logger import logger

EventHandler = Callable[[str, Dict[str, Any]], None]


class EventBus:
    """Pub/Sub event dispatcher for agent interactions and system signals."""

    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribes a callback handler to a specific topic pattern."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)
        logger.debug(f"EventBus: Subscribed handler to [{topic}]")

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publishes a event payload to all matching topic subscribers."""
        logger.debug(f"EventBus: Published to [{topic}] -> {payload.get('event_type', topic)}")
        handlers = self._subscribers.get(topic, [])
        for handler in handlers:
            try:
                handler(topic, payload)
            except Exception as e:
                logger.error(f"EventBus error in handler for topic [{topic}]: {e}")


# Global EventBus Singleton
event_bus = EventBus()


# ---------------------------------------------------------------------------
# Structured event emitter (Phase 2)
# ---------------------------------------------------------------------------
# emit() is the single call-site for all mind-view events.  It:
#   1. Appends the event to a per-user JSON-lines log (for replay).
#   2. Pushes the event to the user's WebSocket queue (non-blocking).
#   3. Publishes it on the EventBus so any existing subscriber still works.
#
# It is designed to never raise: a broken WebSocket or full log never
# propagates back to the agent loop.

import json as _json
import threading as _threading
import time as _time
from pathlib import Path as _Path
from typing import Optional as _Optional

_log_lock = _threading.Lock()

# ---------------------------------------------------------------------------
# Mind-view event schema (v1)
# ---------------------------------------------------------------------------
# Every payload carries: type, v, user_id, turn_id, ts. The extra fields each
# event type carries are listed here; scripts/gen_docs.py renders this table
# into docs/generated/event_topics.md.
EVENT_SCHEMA_VERSION = 1

MIND_EVENTS: Dict[str, str] = {
    # Agent loop
    "turn_start":       "agent",
    "turn_end":         "agent",
    "llm_call":         "agent, step",
    "tool_call":        "agent, tool",
    "delegate":         "agent, query",
    "context_injected": "agent, memory_ids, skill_hint",
    # Vector memory
    "store":            "namespace, memory_id, kind",
    "recall":           "namespace, memory_ids, count",
    "learned":          "memory_id, namespace",
    "skill_created":    "skill_id, trigger",
    # Graph-indexed recall (Stage 2)
    "traverse":         "query, path (node ids in visit order), edges (edge ids)",
    "node_touched":     "node_id, usage_score, tier, style",
    "gap_detected":     "term, reason, missing_aspects",
    # Research (Stage 3)
    "research_start":   "topic, queries",
    "research_step":    "topic, url, status",
    "research_done":    "topic, node_id, records, position, style",
    "research_answer":  "question, answer (cited follow-up after background research)",
    "node_spawned":     "node_id, kind, label, collection, style, position",
    "edge_drawn":       "edge_id, src, dst, relation",
    # Curiosity (Stage 4)
    "curiosity_added":  "id, question, topic_node_id, priority",
    # Sleep cycle (Stage 5)
    "sleep_start":      "reason, mode",
    "sleep_phase":      "name",
    "embedding_migrated": "from_node, to_node, count",
    "node_fading":      "node_id, usage_score, style",
    "node_pruned":      "node_id, collection",
    "layout_updated":   "moves [{node_id, position}] after the sleep-time refit",
    "sleep_end":        "report",
}

EVENT_TOPICS = [
    "task.compiled", "task.all_steps_completed", "task.failed",
    "outcome.recorded", "evolution.cycle_started", "evolution.cycle_completed",
    "skill.promoted", "skill.deprecated",
] + [f"mind.{name}  ({fields})" for name, fields in MIND_EVENTS.items()]


def emit(
    event_type: str,
    user_id: str = "local",
    turn_id: str = "",
    **kwargs: Any,
) -> None:
    """Emit a structured mind-view event.

    Args:
        event_type: Short string identifying the event, e.g. "recall", "store",
                    "llm_call", "tool_call", "step_start", "step_end", "delegate".
        user_id:    The user this event belongs to (matches UserContext.user_id).
        turn_id:    Opaque string tying all events in one conversation turn together.
        **kwargs:   Arbitrary extra fields included in the JSON payload.

    Examples:
        emit("recall", user_id=ctx.user_id, turn_id=tid, memory_ids=["abc"])
        emit("store",  user_id=ctx.user_id, turn_id=tid, memory_id="xyz", namespace="peter")
        emit("delegate", user_id=ctx.user_id, turn_id=tid, agent="researcher")
    """
    payload: Dict[str, Any] = {
        "type":    event_type,
        "v":       EVENT_SCHEMA_VERSION,
        "user_id": user_id,
        "turn_id": turn_id,
        "ts":      _time.time(),
        **kwargs,
    }

    # 1. Write to the per-user JSON-lines log
    _write_event_log(user_id, payload)

    # 2. Push to the WebSocket queue (imported lazily to avoid circular imports)
    try:
        from server.ws import push_event  # type: ignore
        push_event(user_id, payload)
    except Exception:
        pass  # server not running (REPL mode) — silently skip

    # 3. Publish on the internal event bus (backwards compat)
    try:
        event_bus.publish(f"mind.{event_type}", payload)
    except Exception:
        pass


def _write_event_log(user_id: str, payload: Dict[str, Any]) -> None:
    """Append one JSON-lines record to the user's event log. Never raises."""
    try:
        # Derive the log path from UserContext if possible, otherwise use a fallback.
        try:
            from core.user_context import web_context, anonymous_context  # type: ignore
            if user_id == "local":
                log_path = _Path("data/events.jsonl")
            else:
                log_path = _Path("data") / "users" / user_id / "events.jsonl"
        except Exception:
            log_path = _Path("data/events.jsonl")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = _json.dumps(payload)
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass  # log write errors must never surface to the caller
