"""Event bus fan-out, filters, lifecycle."""
from __future__ import annotations

import asyncio

import pytest

from executor.core.event_bus import EventBus, filter_types
from executor.core.events import Event, EventType, Source


pytestmark = pytest.mark.asyncio


async def test_publish_fanout_push_subscriber():
    bus = EventBus()
    await bus.start()
    received: list[Event] = []

    async def sink(e: Event) -> None:
        received.append(e)

    await bus.subscribe("sink", on_event=sink)

    await bus.publish(Event.make(EventType.WARN, Source.EXECUTOR, {"i": 1}))
    await bus.publish(Event.make(EventType.ERROR, Source.EXECUTOR, {"i": 2}))

    await asyncio.sleep(0.05)
    await bus.stop()

    assert [e.payload["i"] for e in received] == [1, 2]


async def test_filter_only_matching():
    bus = EventBus()
    await bus.start()
    received: list[Event] = []

    async def sink(e: Event) -> None:
        received.append(e)

    await bus.subscribe("warns", filter_fn=filter_types(EventType.WARN), on_event=sink)

    await bus.publish(Event.make(EventType.WARN, Source.EXECUTOR, {"i": 1}))
    await bus.publish(Event.make(EventType.ERROR, Source.EXECUTOR, {"i": 2}))
    await bus.publish(Event.make(EventType.WARN, Source.EXECUTOR, {"i": 3}))

    await asyncio.sleep(0.05)
    await bus.stop()

    assert [e.payload["i"] for e in received] == [1, 3]


async def test_pull_style_stream():
    bus = EventBus()
    await bus.start()
    sub = await bus.subscribe("puller")
    got: list[int] = []

    async def reader():
        async for e in bus.stream(sub):
            got.append(e.payload["i"])
            if len(got) == 2:
                return

    task = asyncio.create_task(reader())
    await bus.publish(Event.make(EventType.WARN, Source.EXECUTOR, {"i": 1}))
    await bus.publish(Event.make(EventType.WARN, Source.EXECUTOR, {"i": 2}))
    await asyncio.wait_for(task, timeout=2.0)
    await bus.stop()
    assert got == [1, 2]


async def test_subscriber_crash_does_not_kill_bus():
    bus = EventBus()
    await bus.start()
    delivered: list[int] = []

    async def bad(e: Event) -> None:
        raise RuntimeError("boom")

    async def good(e: Event) -> None:
        delivered.append(e.payload["i"])

    await bus.subscribe("bad", on_event=bad)
    await bus.subscribe("good", on_event=good)
    await bus.publish(Event.make(EventType.WARN, Source.EXECUTOR, {"i": 1}))
    await asyncio.sleep(0.05)
    await bus.stop()
    assert delivered == [1]
