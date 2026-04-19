"""
Async, in-process event bus.

Every event the executor produces flows through this bus. The audit writer is
always subscribed. Strategies and the risk engine subscribe to the slices
relevant to them (Decision 2: "Feedback: async event bus, per strategy").

Design:
- publish() is fire-and-forget; the bus queues and a background pump fans out
  to subscribers. This keeps the hot path (order placement, fill receipt)
  free of slow subscribers.
- subscribers pull from their own asyncio.Queue; slow subscribers apply
  backpressure only to themselves, not to the publisher.
- start() / stop() lifecycle. stop() drains pending events to already-queued
  subscribers before returning.

Not a cross-process bus; executor is one asyncio process by design
(spec: "Fully async throughout (asyncio). No threading.").
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .events import Event, EventType
from .logging import get_logger


log = get_logger("executor.event_bus")


# A filter function takes an Event and returns True if the subscriber wants it.
EventFilter = Callable[[Event], bool]
# A subscriber callback. Called from the bus pump task.
AsyncSubscriber = Callable[[Event], Awaitable[None]]


@dataclass
class _Subscription:
    subscriber_id: str
    queue: asyncio.Queue[Event]
    filter_fn: EventFilter | None = None
    on_event: AsyncSubscriber | None = None
    drop_count: int = 0
    closed: bool = False


class EventBus:
    def __init__(self, *, maxsize: int = 10_000) -> None:
        self._inbox: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subs: dict[str, _Subscription] = {}
        self._pump_task: asyncio.Task[None] | None = None
        self._running = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._pump_task = asyncio.create_task(self._pump(), name="event-bus-pump")
        log.info("event_bus.start")

    async def stop(self, *, drain_timeout_sec: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        # Let pump drain whatever is in the inbox.
        try:
            await asyncio.wait_for(self._inbox.join(), timeout=drain_timeout_sec)
        except asyncio.TimeoutError:
            log.warning("event_bus.stop.drain_timeout", pending=self._inbox.qsize())
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except (asyncio.CancelledError, Exception):
                pass
        # Close subscriber queues so any iterators exit cleanly.
        for sub in list(self._subs.values()):
            sub.closed = True
            try:
                sub.queue.put_nowait(_SENTINEL)  # wake any awaiter
            except asyncio.QueueFull:
                pass
        log.info("event_bus.stop")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Enqueue event for fan-out. Await point: structured log before and after."""
        log.debug(
            "event_bus.publish.enter",
            event_type=event.event_type.value,
            event_id=event.event_id,
            intent_id=event.intent_id,
            strategy_id=event.strategy_id,
            venue=event.venue,
        )
        await self._inbox.put(event)
        log.debug(
            "event_bus.publish.done",
            event_type=event.event_type.value,
            event_id=event.event_id,
        )

    def publish_nowait(self, event: Event) -> None:
        """Sync publish. Raises asyncio.QueueFull if overloaded."""
        self._inbox.put_nowait(event)

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        subscriber_id: str,
        *,
        filter_fn: EventFilter | None = None,
        on_event: AsyncSubscriber | None = None,
        queue_maxsize: int = 1000,
    ) -> _Subscription:
        """
        Register a subscriber. Two modes:
          - Pass on_event for push-style (called from pump task).
          - Omit on_event and call stream(subscription) for pull-style AsyncIterator.
        """
        async with self._lock:
            if subscriber_id in self._subs:
                raise ValueError(f"subscriber {subscriber_id!r} already registered")
            sub = _Subscription(
                subscriber_id=subscriber_id,
                queue=asyncio.Queue(maxsize=queue_maxsize),
                filter_fn=filter_fn,
                on_event=on_event,
            )
            self._subs[subscriber_id] = sub
        log.info("event_bus.subscribe", subscriber_id=subscriber_id, push=on_event is not None)
        return sub

    async def unsubscribe(self, subscriber_id: str) -> None:
        async with self._lock:
            sub = self._subs.pop(subscriber_id, None)
        if sub is not None:
            sub.closed = True
            try:
                sub.queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                pass
        log.info("event_bus.unsubscribe", subscriber_id=subscriber_id)

    async def stream(self, subscription: _Subscription) -> AsyncIterator[Event]:
        """Pull-style iterator for a subscription created without on_event."""
        while not subscription.closed:
            item = await subscription.queue.get()
            if item is _SENTINEL:
                return
            try:
                yield item
            finally:
                subscription.queue.task_done()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _pump(self) -> None:
        log.info("event_bus.pump.start")
        try:
            while True:
                event = await self._inbox.get()
                try:
                    await self._fanout(event)
                finally:
                    self._inbox.task_done()
        except asyncio.CancelledError:
            log.info("event_bus.pump.cancelled")
            raise
        except Exception as exc:  # pragma: no cover — guard against silent death
            log.error("event_bus.pump.crash", error=str(exc))
            raise

    async def _fanout(self, event: Event) -> None:
        # Copy subs to a list so unsubscribes during iteration don't break us.
        subs = list(self._subs.values())
        for sub in subs:
            if sub.closed:
                continue
            if sub.filter_fn is not None:
                try:
                    if not sub.filter_fn(event):
                        continue
                except Exception as exc:
                    log.warning(
                        "event_bus.filter.crash",
                        subscriber_id=sub.subscriber_id,
                        error=str(exc),
                    )
                    continue
            if sub.on_event is not None:
                try:
                    await sub.on_event(event)
                except Exception as exc:
                    log.error(
                        "event_bus.subscriber.crash",
                        subscriber_id=sub.subscriber_id,
                        event_type=event.event_type.value,
                        error=str(exc),
                    )
            else:
                try:
                    sub.queue.put_nowait(event)
                except asyncio.QueueFull:
                    sub.drop_count += 1
                    log.warning(
                        "event_bus.subscriber.drop",
                        subscriber_id=sub.subscriber_id,
                        drop_count=sub.drop_count,
                        event_type=event.event_type.value,
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "inbox_size": self._inbox.qsize(),
            "subscribers": [
                {
                    "id": s.subscriber_id,
                    "queue_size": s.queue.qsize(),
                    "drops": s.drop_count,
                    "closed": s.closed,
                }
                for s in self._subs.values()
            ],
        }


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def filter_types(*types: EventType) -> EventFilter:
    wanted = set(types)
    def _f(e: Event) -> bool:
        return e.event_type in wanted
    return _f


def filter_strategy(strategy_id: str) -> EventFilter:
    def _f(e: Event) -> bool:
        return e.strategy_id == strategy_id
    return _f


def filter_any(filters: Iterable[EventFilter]) -> EventFilter:
    fs = tuple(filters)
    def _f(e: Event) -> bool:
        return any(f(e) for f in fs)
    return _f
