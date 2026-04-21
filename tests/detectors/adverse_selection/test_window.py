"""WindowAdverseSelectionDetector — venue-pause + gate flag."""
from __future__ import annotations

import asyncio
from decimal import Decimal

from executor.core.events import Event, EventType
from executor.core.types import Side
from executor.detectors.adverse_selection.window import (
    WindowAdverseSelectionDetector,
)


def test_initial_state_unflagged() -> None:
    d = WindowAdverseSelectionDetector(window=5, adverse_threshold=0.6, move_threshold_sigma=2.0)
    assert d.is_flagged(strategy_id="x", market_id="m1") is False
    paused, _ = d.venue_pause_detail("kalshi")
    assert paused is False
    assert d.is_venue_paused("kalshi") is False


def test_threshold_trips_after_window_full() -> None:
    d = WindowAdverseSelectionDetector(
        window=4, adverse_threshold=0.5, move_threshold_sigma=1.0, pause_sec=60
    )
    # Seed mid history so sigma is non-trivial.
    for v in (Decimal("0.5"), Decimal("0.51"), Decimal("0.49"), Decimal("0.50"),
              Decimal("0.48"), Decimal("0.52")):
        d.observe_mid("kalshi", v)

    base_ns = 10**12
    # 4 BUY fills at 0.50, then mid drops sharply at +60s — every fill adverse.
    for i in range(4):
        asyncio.run(d.observe_fill(
            venue="kalshi", market_id="m1", side=Side.BUY,
            fill_price=Decimal("0.50"), fill_ts_ns=base_ns + i,
        ))
    # Push the post-fill horizon to >= 60s with a large adverse drop.
    drop_ns = base_ns + 70 * 1_000_000_000
    last_flag = None
    for i in range(4):
        flag = asyncio.run(d.update_mark(
            venue="kalshi", market_id="m1",
            mid=Decimal("0.30"), now_ns=drop_ns + i,
        ))
        last_flag = last_flag or flag
    paused, _ = d.venue_pause_detail("kalshi", now_ns=drop_ns + 1000)
    assert paused is True
    assert last_flag is not None
    assert d.is_flagged(strategy_id="x", market_id="m1") is True


def test_no_trip_when_movement_below_sigma() -> None:
    d = WindowAdverseSelectionDetector(
        window=4, adverse_threshold=0.5, move_threshold_sigma=20.0,
    )
    for v in (Decimal("0.5"), Decimal("0.51"), Decimal("0.49"), Decimal("0.50")):
        d.observe_mid("kalshi", v)
    base_ns = 10**12
    for i in range(4):
        asyncio.run(d.observe_fill(
            venue="kalshi", market_id="m1", side=Side.BUY,
            fill_price=Decimal("0.50"), fill_ts_ns=base_ns + i,
        ))
    drop_ns = base_ns + 70 * 1_000_000_000
    for i in range(4):
        asyncio.run(d.update_mark(
            venue="kalshi", market_id="m1",
            mid=Decimal("0.495"), now_ns=drop_ns + i,
        ))
    paused, _ = d.venue_pause_detail("kalshi", now_ns=drop_ns + 100)
    assert paused is False


def test_clear_venue_releases_pause() -> None:
    d = WindowAdverseSelectionDetector(window=2, adverse_threshold=0.5, move_threshold_sigma=0.5)
    for v in (Decimal("0.5"), Decimal("0.51")):
        d.observe_mid("kalshi", v)
    base_ns = 10**12
    for i in range(2):
        asyncio.run(d.observe_fill(
            venue="kalshi", market_id="m1", side=Side.BUY,
            fill_price=Decimal("0.50"), fill_ts_ns=base_ns + i,
        ))
    drop_ns = base_ns + 70 * 1_000_000_000
    for i in range(2):
        asyncio.run(d.update_mark(
            venue="kalshi", market_id="m1",
            mid=Decimal("0.30"), now_ns=drop_ns + i,
        ))
    paused, _ = d.venue_pause_detail("kalshi", now_ns=drop_ns + 1)
    if paused:
        d.clear_venue("kalshi")
    paused2, _ = d.venue_pause_detail("kalshi", now_ns=drop_ns + 1)
    assert paused2 is False


def test_emit_flag_event_publishes_anomaly() -> None:
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    d = WindowAdverseSelectionDetector(
        window=2, adverse_threshold=0.5, move_threshold_sigma=0.5, publish=pub
    )
    for v in (Decimal("0.50"), Decimal("0.51"), Decimal("0.49"),
              Decimal("0.50"), Decimal("0.51"), Decimal("0.50")):
        d.observe_mid("kalshi", v)
    base_ns = 10**12
    flag = None
    for i in range(2):
        asyncio.run(d.observe_fill(
            venue="kalshi", market_id="m1", side=Side.BUY,
            fill_price=Decimal("0.50"), fill_ts_ns=base_ns + i,
        ))
    for i in range(2):
        f = asyncio.run(d.update_mark(
            venue="kalshi", market_id="m1",
            mid=Decimal("0.30"), now_ns=base_ns + 70_000_000_000 + i,
        ))
        flag = flag or f
    assert flag is not None
    asyncio.run(d.emit_flag_event(flag, venue="kalshi"))
    [ev] = captured
    assert ev.event_type is EventType.ANOMALY_DETECTED
    assert ev.payload["kind"] == "ADVERSE_SELECTION_DETECTED"


def test_snapshot_reports_counts() -> None:
    d = WindowAdverseSelectionDetector(window=3, adverse_threshold=0.6, move_threshold_sigma=2.0)
    for v in (Decimal("0.5"), Decimal("0.51"), Decimal("0.50")):
        d.observe_mid("kalshi", v)
    base_ns = 10**12
    asyncio.run(d.observe_fill(
        venue="kalshi", market_id="m1", side=Side.BUY,
        fill_price=Decimal("0.50"), fill_ts_ns=base_ns,
    ))
    snap = d.snapshot()
    assert snap["kalshi"]["fills_in_window"] == 1
    assert snap["kalshi"]["decided_fills"] == 0


def test_sell_side_adverse_direction_inverted() -> None:
    """For SELL, mid going UP is adverse."""
    d = WindowAdverseSelectionDetector(
        window=3, adverse_threshold=0.5, move_threshold_sigma=0.5,
    )
    for v in (Decimal("0.50"), Decimal("0.51"), Decimal("0.49"),
              Decimal("0.50"), Decimal("0.51"), Decimal("0.50")):
        d.observe_mid("kalshi", v)
    base_ns = 10**12
    for i in range(3):
        asyncio.run(d.observe_fill(
            venue="kalshi", market_id="m1", side=Side.SELL,
            fill_price=Decimal("0.50"), fill_ts_ns=base_ns + i,
        ))
    drop_ns = base_ns + 70_000_000_000
    flag = None
    for i in range(3):
        f = asyncio.run(d.update_mark(
            venue="kalshi", market_id="m1",
            mid=Decimal("0.70"),  # UP move == adverse for SELL
            now_ns=drop_ns + i,
        ))
        flag = flag or f
    assert flag is not None
