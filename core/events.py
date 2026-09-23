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
