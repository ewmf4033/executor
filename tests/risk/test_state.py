"""Risk state — cache load, rebuild from venues, corruption fallback."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.types import Position
from executor.risk.state import (
    OperatorLivenessStore,
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


# ---------------------------------------------------------------------------
# Phase 4.14b — OperatorLivenessStore tests.
# ---------------------------------------------------------------------------


async def _fresh_live_store(tmp_path: Path) -> tuple[RiskState, OperatorLivenessStore]:
    s = RiskState(db_path=tmp_path / "rs.sqlite")
    await s.load()
    return s, OperatorLivenessStore(s.connection)


async def test_operator_liveness_arm_persists(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    now = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=now)
    snap = store.load()
    assert snap.armed is True
    assert snap.timeout_sec == 600
    assert snap.armed_ts_ns == now
    assert snap.last_heartbeat_ts_ns == now
    assert snap.armed_by_source == "executorctl"
    assert snap.kill_mode_at_arm == "NONE"
    assert snap.disarmed_reason is None
    s.close()


async def test_operator_liveness_disarm_clears_state(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    now = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=now)
    store.disarm(reason="session end", now_ns=now + 1)
    snap = store.load()
    assert snap.armed is False
    assert snap.disarmed_reason == "session end"
    s.close()


async def test_operator_liveness_heartbeat_updates_ts(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    t0 = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=t0)
    t1 = t0 + 5_000_000_000  # +5s
    ok = store.heartbeat(now_ns=t1)
    assert ok is True
    snap = store.load()
    assert snap.last_heartbeat_ts_ns == t1
    s.close()


async def test_operator_liveness_heartbeat_noop_when_disarmed(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    # No arm at all — default row is disarmed.
    ok = store.heartbeat(now_ns=time.time_ns())
    assert ok is False
    snap = store.load()
    assert snap.armed is False
    assert snap.last_heartbeat_ts_ns == 0
    s.close()


async def test_operator_liveness_survives_restart(tmp_path: Path):
    # Open, arm, close — then reopen the same DB and verify the row
    # reflects armed state with its original armed_ts_ns preserved.
    path = tmp_path / "rs.sqlite"
    s1 = RiskState(db_path=path)
    await s1.load()
    store1 = OperatorLivenessStore(s1.connection)
    t0 = time.time_ns()
    store1.arm(timeout_sec=1200, source="executorctl", kill_mode="SOFT", now_ns=t0)
    s1.close()

    s2 = RiskState(db_path=path)
    await s2.load()
    store2 = OperatorLivenessStore(s2.connection)
    snap = store2.load()
    assert snap.armed is True
    assert snap.armed_ts_ns == t0
    assert snap.timeout_sec == 1200
    assert snap.last_heartbeat_ts_ns == t0
    assert snap.kill_mode_at_arm == "SOFT"
    s2.close()


# ---------------------------------------------------------------------------
# Phase 4.14e — heartbeat/disarm atomicity + writer serialization.
#
# Before 4.14e, ``heartbeat`` performed a ``load()`` → branch-on-armed
# → ``UPDATE`` against a shared ``check_same_thread=False`` sqlite
# connection. A ``disarm()`` landing between the load and the update
# could leave the row disarmed while still accepting the heartbeat
# write (and the caller would still see ``True``). The ultrareview
# Run 2 flagged this as B1/B2; the fix replaces heartbeat with a
# single conditional ``UPDATE ... WHERE armed = 1`` and adds a
# writer-serializing ``threading.RLock`` around every mutation.
# ---------------------------------------------------------------------------


async def test_heartbeat_after_disarm_does_not_rearm(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    t0 = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=t0)
    store.disarm(reason="ops", now_ns=t0 + 1)
    # Late heartbeat arriving after disarm must not flip armed back on.
    ok = store.heartbeat(now_ns=t0 + 10_000_000_000)
    assert ok is False
    snap = store.load()
    assert snap.armed is False
    s.close()


async def test_heartbeat_after_disarm_does_not_advance_last_heartbeat(
    tmp_path: Path,
):
    s, store = await _fresh_live_store(tmp_path)
    t0 = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=t0)
    pre = store.load().last_heartbeat_ts_ns
    assert pre == t0
    store.disarm(reason="ops", now_ns=t0 + 1)
    # Heartbeat on disarmed row must be a no-op — last_heartbeat_ts_ns
    # frozen at its pre-disarm value.
    store.heartbeat(now_ns=t0 + 5_000_000_000)
    assert store.load().last_heartbeat_ts_ns == pre
    s.close()


async def test_heartbeat_returns_not_accepted_when_disarmed(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    # Default row is disarmed — caller must see False, not True-on-stale-read.
    assert store.heartbeat(now_ns=time.time_ns()) is False
    s.close()


async def test_disarm_after_heartbeat_wins_when_ordered_last(tmp_path: Path):
    s, store = await _fresh_live_store(tmp_path)
    t0 = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=t0)
    assert store.heartbeat(now_ns=t0 + 1_000_000_000) is True
    store.disarm(reason="late", now_ns=t0 + 2_000_000_000)
    snap = store.load()
    assert snap.armed is False
    assert snap.disarmed_reason == "late"
    # Another heartbeat now cannot resurrect armed state.
    assert store.heartbeat(now_ns=t0 + 3_000_000_000) is False
    assert store.load().armed is False
    s.close()


async def test_liveness_store_serializes_writes(tmp_path: Path):
    """Sanity check: the writer lock lets many concurrent heartbeats
    from different threads run without corrupting state or tripping
    ``sqlite3`` recursive-cursor errors. End state must be ``armed=True``
    with ``last_heartbeat_ts_ns`` equal to one of the submitted values."""
    s, store = await _fresh_live_store(tmp_path)
    t0 = time.time_ns()
    store.arm(timeout_sec=600, source="executorctl", kill_mode="NONE", now_ns=t0)

    submitted: list[int] = []
    errors: list[BaseException] = []

    def _hammer(offset: int) -> None:
        try:
            ts = t0 + offset
            submitted.append(ts)
            store.heartbeat(now_ns=ts)
        except BaseException as exc:  # pragma: no cover — would fail test
            errors.append(exc)

    threads = [
        threading.Thread(target=_hammer, args=(i * 1_000_000,))
        for i in range(1, 21)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert errors == []
    snap = store.load()
    assert snap.armed is True
    # The final persisted heartbeat must be one of the submitted
    # timestamps — never a partial/garbage value.
    assert snap.last_heartbeat_ts_ns in submitted
    s.close()
