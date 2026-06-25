import asyncio
from typing import Awaitable, Callable

from orchestration.events import Event


EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Untyped fan-out bus: every subscriber receives every Event."""

    def __init__(self) -> None:
        """Create an empty subscriber list."""
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Register a sync or async handler to receive future events."""
        self._subscribers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a previously-subscribed handler; no-op if unknown."""
        try:
            self._subscribers.remove(handler)
        except ValueError:
            pass

    async def publish(self, event: Event) -> None:
        """Deliver `event` to every subscriber, awaiting coroutine returns (backpressure)."""
        for h in list(self._subscribers):
            res = h(event)
            if asyncio.iscoroutine(res):
                await res
