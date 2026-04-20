"""AttributionServer — HTTP endpoints."""
from __future__ import annotations

import asyncio
import socket
from decimal import Decimal
from pathlib import Path

import aiohttp
import pytest

from executor.attribution.server import AttributionServer
from executor.attribution.tracker import AttributionTracker
from executor.core.types import Side


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _start(tmp_path: Path, port: int) -> tuple[AttributionServer, AttributionTracker]:
    t = AttributionTracker(db_path=tmp_path / "attr.sqlite", exit_horizon_sec=1)
    s = AttributionServer(t, host="127.0.0.1", port=port)
    await s.start()
    return s, t


def test_health_endpoint(tmp_path: Path) -> None:
    port = _free_port()

    async def go() -> None:
        s, _ = await _start(tmp_path, port)
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"http://127.0.0.1:{port}/attribution/health") as r:
                    assert r.status == 200
                    body = await r.json()
                    assert body["ok"] is True
        finally:
            await s.stop()

    asyncio.run(go())


def test_summary_endpoint_returns_aggregated(tmp_path: Path) -> None:
    port = _free_port()

    async def go() -> None:
        s, t = await _start(tmp_path, port)
        try:
            t.note_decision("i1", Decimal("0.50"))
            t.note_arrival("i1", "l1", Decimal("0.50"))
            t.on_fill(
                fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
                strategy_id="alpha", venue="kalshi", market_id="m",
                side=Side.BUY, size=Decimal("1"), fill_price=Decimal("0.50"),
                fill_ts_ns=1, intent_price=Decimal("0.50"),
            )
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"http://127.0.0.1:{port}/attribution/summary") as r:
                    body = await r.json()
                    assert body["total_fills"] == 1
                    assert "alpha" in body["strategies"]
        finally:
            await s.stop()
            t.close()

    asyncio.run(go())


def test_summary_endpoint_bad_since_ns(tmp_path: Path) -> None:
    port = _free_port()

    async def go() -> None:
        s, t = await _start(tmp_path, port)
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"http://127.0.0.1:{port}/attribution/summary?since_ns=abc"
                ) as r:
                    assert r.status == 400
        finally:
            await s.stop()
            t.close()

    asyncio.run(go())
