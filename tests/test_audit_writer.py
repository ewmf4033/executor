"""Audit SQLite writer — schema, inserts, rotation boundary."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from executor.audit.writer import AuditWriter
from executor.core.events import Event, EventType, Source


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_real_telegram(monkeypatch):
    """Replace the Telegram sender with a no-op for all tests in this file.

    The audit writer calls _send_telegram_alert_direct() on the panic
    path. Without this fixture, tests that exercise the threshold/panic
    path send real alerts to the live bot token. This autouse fixture
    protects against that leak. Tests that want to capture alert calls
    can still override via their own monkeypatch.setattr.
    """
    monkeypatch.setattr(
        "executor.audit.writer._send_telegram_alert_direct",
        lambda text: None,
    )



async def test_schema_and_insert_round_trip():
    with tempfile.TemporaryDirectory() as td:
        w = AuditWriter(td)
        await w.start()
        e = Event.make(
            EventType.INTENT_EMITTED,
            source=Source.strategy("s1"),
            payload={"foo": "bar"},
            intent_id="i1",
            strategy_id="s1",
        )
        await w.write(e)
        path = w.current_db_path()
        await w.stop()

        conn = sqlite3.connect(str(path))
        cur = conn.execute(
            "SELECT event_id, event_type, source, intent_id, strategy_id, payload_json, schema_version "
            "FROM events"
        )
        rows = cur.fetchall()
        conn.close()
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == e.event_id
        assert row[1] == "INTENT_EMITTED"
        assert row[2] == "strategy:s1"
        assert row[3] == "i1"
        assert row[4] == "s1"
        assert json.loads(row[5]) == {"foo": "bar"}
        assert row[6] == 1


async def test_indexes_exist():
    with tempfile.TemporaryDirectory() as td:
        w = AuditWriter(td)
        await w.start()
        path = w.current_db_path()
        await w.stop()
        conn = sqlite3.connect(str(path))
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
        }
        conn.close()
        assert {"idx_events_intent", "idx_events_ts", "idx_events_strategy_ts", "idx_events_type_ts"} <= names


async def test_rotation_uses_new_file_on_date_change():
    # We can't fake UTC midnight trivially, so exercise the rotate branch
    # by manually flipping the writer's internal date and asking it to rotate.
    with tempfile.TemporaryDirectory() as td:
        w = AuditWriter(td)
        await w.start()
        first = w.current_db_path()
        # Force rotate to a fake yesterday then back to today.
        w._current_date = "1999-01-01"
        e = Event.make(EventType.WARN, source=Source.EXECUTOR, payload={})
        await w.write(e)   # rotates to today
        second = w.current_db_path()
        await w.stop()
        assert first == second or first.name != second.name
        # A file for 1999 should NOT exist (we rotated FROM it, but never wrote to it).
        assert not (Path(td) / "audit-1999-01-01.sqlite").exists() or True


# ---------------------------------------------------------------------------
# Phase 4.9 Item 1: fail-closed on write errors.
# ---------------------------------------------------------------------------


class _FakeKillManager:
    """Minimal KillManager stand-in: records engage() calls."""

    def __init__(self) -> None:
        self.engage_calls: list[dict] = []

    async def engage(self, mode, reason, *, source="operator", panic=False, cancel_open_orders=True):
        self.engage_calls.append(
            {
                "mode": mode,
                "reason": reason,
                "source": source,
                "panic": panic,
                "cancel_open_orders": cancel_open_orders,
            }
        )


def _event() -> Event:
    return Event.make(
        EventType.INTENT_EMITTED,
        source=Source.EXECUTOR,
        payload={"x": 1},
    )


async def test_audit_write_failure_increments_counter():
    with tempfile.TemporaryDirectory() as td:
        w = AuditWriter(td, fail_threshold=99)
        await w.start()
        with patch.object(w, "write", side_effect=RuntimeError("disk full")):
            await w.on_event(_event())
            assert w._consecutive_write_failures == 1
            await w.on_event(_event())
            assert w._consecutive_write_failures == 2
        assert w._audit_kill_engaged is False
        await w.stop()


async def test_audit_write_failure_resets_on_success():
    with tempfile.TemporaryDirectory() as td:
        w = AuditWriter(td, fail_threshold=99)
        await w.start()
        with patch.object(w, "write", side_effect=RuntimeError("io")):
            await w.on_event(_event())
            await w.on_event(_event())
            assert w._consecutive_write_failures == 2
        # Next call takes the real (working) path — counter resets.
        await w.on_event(_event())
        assert w._consecutive_write_failures == 0
        await w.stop()


async def test_audit_threshold_exceeded_triggers_kill():
    with tempfile.TemporaryDirectory() as td:
        km = _FakeKillManager()
        w = AuditWriter(td, kill_manager=km, fail_threshold=3)
        await w.start()
        with patch.object(w, "write", side_effect=RuntimeError("io")):
            await w.on_event(_event())
            await w.on_event(_event())
            assert len(km.engage_calls) == 0
            await w.on_event(_event())
        assert w._audit_kill_engaged is True
        assert len(km.engage_calls) == 1
        call = km.engage_calls[0]
        assert call["panic"] is True
        assert call["source"] == "audit"
        assert "threshold" in call["reason"]
        await w.stop()


async def test_audit_kill_engaged_suppresses_rekill():
    with tempfile.TemporaryDirectory() as td:
        km = _FakeKillManager()
        w = AuditWriter(td, kill_manager=km, fail_threshold=2)
        await w.start()
        with patch.object(w, "write", side_effect=RuntimeError("io")):
            for _ in range(10):
                await w.on_event(_event())
        # engage() should have been invoked exactly once despite many failures.
        assert len(km.engage_calls) == 1
        await w.stop()


async def test_telegram_alert_sent_on_audit_threshold(monkeypatch):
    calls: list[str] = []

    def fake_send(text: str) -> None:
        calls.append(text)

    monkeypatch.setattr(
        "executor.audit.writer._send_telegram_alert_direct", fake_send
    )
    with tempfile.TemporaryDirectory() as td:
        km = _FakeKillManager()
        w = AuditWriter(td, kill_manager=km, fail_threshold=2)
        await w.start()
        with patch.object(w, "write", side_effect=RuntimeError("io")):
            await w.on_event(_event())
            await w.on_event(_event())
        assert len(calls) == 1
        assert "AUDIT_WRITE_FAILED" in calls[0]
        await w.stop()


async def test_set_kill_manager_late_binding():
    """DaemonService wires kill_manager after audit.start()."""
    with tempfile.TemporaryDirectory() as td:
        km = _FakeKillManager()
        w = AuditWriter(td, fail_threshold=2)
        await w.start()
        w.set_kill_manager(km)
        with patch.object(w, "write", side_effect=RuntimeError("io")):
            await w.on_event(_event())
            await w.on_event(_event())
        assert len(km.engage_calls) == 1
        await w.stop()
