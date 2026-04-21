"""Telemetry HTTP endpoint — /health and /pipeline_stats."""
from __future__ import annotations

import asyncio
import time

import aiohttp
import pytest

from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType, Source
from executor.core.telemetry import TelemetryServer


@pytest.mark.asyncio
async def test_health_endpoint_returns_status_and_uptime():
    srv = TelemetryServer(host="127.0.0.1", port=0, daemon_mode=True)
    await srv.start()
    # TCPSite with port=0 binds ephemeral; aiohttp exposes it via runner sites.
    actual_port = srv._site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{actual_port}/health") as r:
                body = await r.json()
        assert body["status"] == "ok"
        assert body["daemon_mode"] is True
        assert body["uptime_sec"] >= 0
        assert "started_at" in body
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_pipeline_stats_counts_events():
    srv = TelemetryServer(host="127.0.0.1", port=0, daemon_mode=True)
    await srv.start()
    actual_port = srv._site._server.sockets[0].getsockname()[1]
    try:
        # Feed three events directly (mimicking the bus subscription).
        for _ in range(3):
            await srv.on_event(Event.make(EventType.INTENT_EMITTED, source="strategy:u"))
        await srv.on_event(Event.make(EventType.FILL, source=Source.venue("v1")))

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{actual_port}/pipeline_stats") as r:
                body = await r.json()
        assert body["lifetime"]["INTENT_EMITTED"] == 3
        assert body["lifetime"]["FILL"] == 1
        assert body["lifetime_total"] == 4
        assert body["last_hour_total"] == 4
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_pipeline_stats_includes_orchestrator_block():
    srv = TelemetryServer(host="127.0.0.1", port=0, daemon_mode=True)
    await srv.start()
    actual_port = srv._site._server.sockets[0].getsockname()[1]

    class _FakeOrch:
        def stats(self):
            return {"intents_received": 7, "admitted": 5, "rejected": 2, "filled_legs": 10}

    srv.set_orchestrator(_FakeOrch())
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{actual_port}/pipeline_stats") as r:
                body = await r.json()
        assert body["orchestrator"]["intents_received"] == 7
        assert body["orchestrator"]["filled_legs"] == 10
    finally:
        await srv.stop()
