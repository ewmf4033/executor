"""WindowAdverseSelectionDetector — venue-pause + gate flag."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.events import Event, EventType
from executor.core.types import Side
from executor.detectors.adverse_selection.window import (
    WindowAdverseSelectionDetector,
)
from executor.risk.state import RiskState


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


# ---------------------------------------------------------------------------
# Phase 4.9 Item 2: durable pause persistence.
# ---------------------------------------------------------------------------


def _trip_pause(d: WindowAdverseSelectionDetector, venue: str = "kalshi") -> None:
    """Helper — trip an adverse-selection pause for `venue`."""
    for v in (Decimal("0.50"), Decimal("0.51"), Decimal("0.49"),
              Decimal("0.50"), Decimal("0.51"), Decimal("0.50")):
        d.observe_mid(venue, v)
    base_ns = 10**12
    for i in range(d.window):
        asyncio.run(d.observe_fill(
            venue=venue, market_id="m1", side=Side.BUY,
            fill_price=Decimal("0.50"), fill_ts_ns=base_ns + i,
        ))
    drop_ns = base_ns + 70_000_000_000
    for i in range(d.window):
        asyncio.run(d.update_mark(
            venue=venue, market_id="m1",
            mid=Decimal("0.30"), now_ns=drop_ns + i,
        ))


async def _make_state(tmp_path: Path) -> RiskState:
    state = RiskState(db_path=tmp_path / "risk.sqlite")
    await state.load()
    return state


def test_pause_persists_across_restart(tmp_path: Path) -> None:
    state_a = asyncio.run(_make_state(tmp_path))
    d_a = WindowAdverseSelectionDetector(
        window=2, adverse_threshold=0.5, move_threshold_sigma=0.5, state=state_a,
    )
    _trip_pause(d_a)
    assert d_a.is_venue_paused("kalshi") is True
    state_a.close()

    # Detector B opens the same state DB fresh — no in-memory carry-over.
    state_b = asyncio.run(_make_state(tmp_path))
    d_b = WindowAdverseSelectionDetector(
        window=2, adverse_threshold=0.5, move_threshold_sigma=0.5,
    )
    assert d_b.is_venue_paused("kalshi") is False  # before rehydrate
    d_b.load_from_state(state_b)
    assert d_b.is_venue_paused("kalshi") is True
    state_b.close()


def test_resume_clears_persisted_state(tmp_path: Path) -> None:
    state_a = asyncio.run(_make_state(tmp_path))
    d_a = WindowAdverseSelectionDetector(
        window=2, adverse_threshold=0.5, move_threshold_sigma=0.5, state=state_a,
    )
    _trip_pause(d_a)
    assert d_a.is_venue_paused("kalshi") is True
    d_a.clear_venue("kalshi")
    state_a.close()

    state_b = asyncio.run(_make_state(tmp_path))
    d_b = WindowAdverseSelectionDetector(
        window=2, adverse_threshold=0.5, move_threshold_sigma=0.5,
    )
    d_b.load_from_state(state_b)
    assert d_b.is_venue_paused("kalshi") is False
    state_b.close()


def test_load_from_empty_state_is_noop(tmp_path: Path) -> None:
    state = asyncio.run(_make_state(tmp_path))
    d = WindowAdverseSelectionDetector(
        window=2, adverse_threshold=0.5, move_threshold_sigma=0.5,
    )
    # Should not crash on a fresh DB with no rows.
    d.load_from_state(state)
    assert d.is_venue_paused("kalshi") is False
    assert state.list_adverse_pauses() == []
    state.close()
