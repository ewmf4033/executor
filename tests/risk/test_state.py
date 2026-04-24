"""Risk state — cache load, rebuild from venues, corruption fallback."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.types import Position
from executor.risk.state import (
    RiskState,
    RiskStateCorruptInCapitalMode,
    utc_date_str,
)


pytestmark = pytest.mark.asyncio


class FakeAdapter:
    def __init__(self, positions):
        self._positions = positions

    async def get_account(self):
        class A:
            pass
        return A()

    async def get_positions(self):
        return self._positions


async def test_fresh_load_empty(tmp_path: Path):
    s = RiskState(db_path=tmp_path / "rs.sqlite")
    outcome = await s.load()
    assert outcome == "loaded"
    assert s.total_exposure() == Decimal("0")


async def test_rebuild_from_venue_get_positions(tmp_path: Path):
    # Pre-delete any cache; no file exists → rebuild from venues.
    path = tmp_path / "rs.sqlite"
    # force a "corrupt" scenario for rebuild pathway: write garbage.
    path.write_bytes(b"this is not a sqlite database")
    positions = [
        Position(
            market_id="MKT-1", venue="kalshi", outcome_id="YES",
            size=Decimal("50"), avg_price_prob=Decimal("0.60"),
            unrealized_pnl=Decimal("0"), as_of_ts=time.time_ns(),
        ),
        Position(
            market_id="MKT-2", venue="kalshi", outcome_id="NO",
            size=Decimal("20"), avg_price_prob=Decimal("0.40"),
            unrealized_pnl=Decimal("0"), as_of_ts=time.time_ns(),
        ),
    ]
    s = RiskState(db_path=path)
    outcome = await s.load(venues={"kalshi": FakeAdapter(positions)})
    assert outcome == "rebuilt_venues"
    # 50*0.60 + 20*0.40 = 38
    assert s.total_exposure() == Decimal("38.00")
    assert s.exposure_by_venue("kalshi") == Decimal("38.00")


async def test_corrupted_sqlite_falls_back_to_audit_replay(tmp_path: Path):
    # Write a garbage SQLite file.
    bad_path = tmp_path / "rs.sqlite"
    bad_path.write_bytes(b"\x00\x01corrupt\x00")
    # Build a minimal audit db with a FILL event from today.
    audit = tmp_path / "audit.sqlite"
    conn = sqlite3.connect(str(audit))
    conn.executescript(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, ts_ns INTEGER, event_type TEXT,
            source TEXT, intent_id TEXT, leg_id TEXT, venue TEXT,
            market_id TEXT, strategy_id TEXT, payload_json TEXT, schema_version INTEGER
        );
        """
    )
    now = time.time_ns()
    conn.execute(
        "INSERT INTO events VALUES (?, ?, 'FILL', 'venue:kalshi', NULL, NULL, 'kalshi', 'MKT-1', 's1', ?, 1)",
        ("e1", now, json.dumps({"size": "10", "price_prob": "0.60", "fee": "0.05"})),
    )
    conn.commit()
    conn.close()

    s = RiskState(db_path=bad_path)
    outcome = await s.load(venues={}, audit_db_path=audit)
    assert outcome == "rebuilt_audit_fallback"
    # Daily pnl should be -0.05 (audit replay records fees as pnl delta in Phase 3).
    assert s.daily_pnl("") == Decimal("-0.05")


async def test_record_and_read_exposures(tmp_path: Path):
    s = RiskState(db_path=tmp_path / "rs.sqlite")
    await s.load()
    s.add_exposure(venue="kalshi", market_id="M", outcome_id="YES",
                   dollars=Decimal("10"), event_id="E")
    s.add_exposure(venue="kalshi", market_id="M", outcome_id="YES",
                   dollars=Decimal("5"))
    assert s.exposure("kalshi", "M", "YES") == Decimal("15")
    assert s.exposure_by_event("E") == Decimal("15")


async def test_daily_pnl_isolation_per_strategy(tmp_path: Path):
    s = RiskState(db_path=tmp_path / "rs.sqlite")
    await s.load()
    s.record_pnl("A", Decimal("-50"))
    s.record_pnl("B", Decimal("20"))
    assert s.daily_pnl("A") == Decimal("-50")
    assert s.daily_pnl("B") == Decimal("20")


async def test_config_hash_roundtrip(tmp_path: Path):
    s = RiskState(db_path=tmp_path / "rs.sqlite")
    await s.load()
    s.record_config_hash("abc123")
    h, ts = s.current_config_hash()
    assert h == "abc123"
    assert ts is not None


# ---------------------------------------------------------------------------
# Phase 4.13.1 Fix #C — corrupt cache halts startup under capital_mode
# ---------------------------------------------------------------------------


async def test_riskstate_corrupt_capital_mode_halts(tmp_path: Path):
    """With capital_mode=True, a corrupt cache file causes load() to raise
    RiskStateCorruptInCapitalMode instead of silently rebuilding from
    venues or replaying audit."""
    path = tmp_path / "rs.sqlite"
    path.write_bytes(b"this is not a sqlite database")
    positions = [
        Position(
            market_id="MKT-1", venue="kalshi", outcome_id="YES",
            size=Decimal("50"), avg_price_prob=Decimal("0.60"),
            unrealized_pnl=Decimal("0"), as_of_ts=time.time_ns(),
        ),
    ]
    s = RiskState(db_path=path)
    # Even with venues available, load must refuse.
    with pytest.raises(RiskStateCorruptInCapitalMode):
        await s.load(
            venues={"kalshi": FakeAdapter(positions)}, capital_mode=True
        )


async def test_riskstate_corrupt_paper_mode_rebuilds(tmp_path: Path):
    """With capital_mode=False (default), the existing best-effort
    rebuild-from-venues behavior is preserved for paper observation."""
    path = tmp_path / "rs.sqlite"
    path.write_bytes(b"this is not a sqlite database")
    positions = [
        Position(
            market_id="MKT-1", venue="kalshi", outcome_id="YES",
            size=Decimal("50"), avg_price_prob=Decimal("0.60"),
            unrealized_pnl=Decimal("0"), as_of_ts=time.time_ns(),
        ),
    ]
    s = RiskState(db_path=path)
    outcome = await s.load(venues={"kalshi": FakeAdapter(positions)})
    assert outcome == "rebuilt_venues"
    # Explicit capital_mode=False works the same way.
    path2 = tmp_path / "rs2.sqlite"
    path2.write_bytes(b"corrupt again")
    s2 = RiskState(db_path=path2)
    outcome2 = await s2.load(
        venues={"kalshi": FakeAdapter(positions)}, capital_mode=False
    )
    assert outcome2 == "rebuilt_venues"
