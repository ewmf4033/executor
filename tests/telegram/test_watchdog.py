"""Telegram polling watchdog (Phase 4.14c)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType
from executor.kill.manager import KillManager
from executor.kill.state import KillMode, KillStateStore
from executor.telegram.bot import TelegramBot
from executor.telegram.watchdog import TelegramWatchdog


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _CapturingBus:
    """Minimal EventBus stand-in that records published events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)


class _FailingBus:
    async def publish(self, event: Event) -> None:
        raise RuntimeError("bus down")


class _FakeBot:
    """Drop-in bot replacement with a controllable last_activity_ts."""

    def __init__(self, *, last_activity: float | None = None) -> None:
        self._last_activity = (
            last_activity if last_activity is not None else time.monotonic()
        )
        self.start_calls = 0
        self.stop_calls = 0
        self.start_raises = False
        self.stop_raises = False
        # After a (re)start, advance the liveness timestamp so the next
        # watchdog poll sees a healthy bot.
        self.refresh_on_start = True

    def last_activity_ts(self) -> float:
        return self._last_activity

    def set_last_activity(self, ts: float) -> None:
        self._last_activity = ts

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_raises:
            raise RuntimeError("boom")
        if self.refresh_on_start:
            self._last_activity = time.monotonic()

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_raises:
            raise RuntimeError("boom")


def _kill_mgr(tmp_path: Path, captured: list[Event]) -> KillManager:
    store = KillStateStore(tmp_path / "kill.sqlite")

    async def pub(ev: Event) -> None:
        captured.append(ev)

    return KillManager(store=store, publish=pub)


def _watchdog(
    *,
    bot: _FakeBot,
    kill_mgr: KillManager,
    bus: Any,
    stall_threshold_sec: int = 120,
    poll_interval_sec: int = 10,
    max_restarts: int = 3,
    restart_window_sec: int = 300,
    escalate_on_max: bool = True,
) -> TelegramWatchdog:
    return TelegramWatchdog(
        bot=bot,  # type: ignore[arg-type]
        kill_mgr=kill_mgr,
        bus=bus,  # type: ignore[arg-type]
        stall_threshold_sec=stall_threshold_sec,
        poll_interval_sec=poll_interval_sec,
        max_restarts=max_restarts,
        restart_window_sec=restart_window_sec,
        escalate_on_max=escalate_on_max,
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_watchdog_healthy_no_action(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot()  # fresh last_activity_ts
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus)

    asyncio.run(wd._check_once())

    assert bot.start_calls == 0
    assert bot.stop_calls == 0
    assert bus.events == []
    assert km.mode is KillMode.NONE


def test_watchdog_detects_stall(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 150.0)
    bot.refresh_on_start = False  # leave stale so we can inspect state
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus)

    asyncio.run(wd._check_once())

    # Stall emitted + restart triggered.
    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_STALL_DETECTED in kinds
    assert bot.stop_calls == 1
    assert bot.start_calls == 1
    assert km.mode is KillMode.NONE  # not yet escalated
    # restart_ts timestamp was appended.
    assert len(wd._restart_ts) == 1


def test_watchdog_restart_ok_clears_next_check(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 150.0)
    bot.refresh_on_start = True  # start() refreshes last_activity
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus)

    asyncio.run(wd._check_once())
    assert bot.start_calls == 1
    # Second check sees a fresh last_activity_ts -> no additional action.
    prior_events = len(bus.events)
    asyncio.run(wd._check_once())
    assert bot.start_calls == 1
    assert len(bus.events) == prior_events


def test_watchdog_max_restarts_triggers_escalation(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot, kill_mgr=km, bus=bus, max_restarts=3, restart_window_sec=300
    )
    # Pre-populate restart_ts to simulate 3 prior restarts within window.
    now = time.monotonic()
    for i in range(3):
        wd._restart_ts.append(now - 10.0)

    asyncio.run(wd._check_once())

    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_STALL_DETECTED in kinds
    assert EventType.TELEGRAM_WATCHDOG_ESCALATED in kinds
    # Escalation engaged SOFT kill.
    assert km.mode is KillMode.SOFT
    # Bot was NOT restarted on the escalation path.
    assert bot.start_calls == 0
    assert bot.stop_calls == 0
    assert wd._escalated is True


def test_watchdog_escalation_only_fires_once(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus, max_restarts=3)
    now = time.monotonic()
    for _ in range(3):
        wd._restart_ts.append(now - 10.0)

    asyncio.run(wd._check_once())
    n_escalations_1 = sum(
        1 for e in bus.events if e.event_type == EventType.TELEGRAM_WATCHDOG_ESCALATED
    )
    assert n_escalations_1 == 1

    # Repeat — another stall but escalation already latched.
    asyncio.run(wd._check_once())
    n_escalations_2 = sum(
        1 for e in bus.events if e.event_type == EventType.TELEGRAM_WATCHDOG_ESCALATED
    )
    assert n_escalations_2 == 1


def test_watchdog_restart_window_prunes_old_timestamps(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot, kill_mgr=km, bus=bus, max_restarts=3, restart_window_sec=300
    )
    # Seed with 3 stale timestamps from > 300s ago.
    old = time.monotonic() - 400.0
    for _ in range(3):
        wd._restart_ts.append(old)

    asyncio.run(wd._check_once())

    # Old timestamps pruned → restart was allowed, not escalation.
    assert bot.start_calls == 1
    assert bot.stop_calls == 1
    assert km.mode is KillMode.NONE
    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_WATCHDOG_ESCALATED not in kinds
    # One fresh timestamp appended.
    assert len(wd._restart_ts) == 1


def test_watchdog_escalate_on_max_false_skips_kill(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot,
        kill_mgr=km,
        bus=bus,
        max_restarts=3,
        restart_window_sec=300,
        escalate_on_max=False,
    )
    now = time.monotonic()
    for _ in range(3):
        wd._restart_ts.append(now - 10.0)

    asyncio.run(wd._check_once())

    # Stall event still emitted, but no escalation event and no kill.
    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_STALL_DETECTED in kinds
    assert EventType.TELEGRAM_WATCHDOG_ESCALATED not in kinds
    assert km.mode is KillMode.NONE


def test_watchdog_bus_publish_failure_does_not_break_loop(tmp_path: Path) -> None:
    captured: list[Event] = []
    km = _kill_mgr(tmp_path, captured)
    bus = _FailingBus()
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus)

    # Must not raise despite bus.publish always failing.
    asyncio.run(wd._check_once())

    # Restart still attempted.
    assert bot.start_calls == 1
    assert bot.stop_calls == 1


def test_bot_last_activity_ts_initialized(tmp_path: Path) -> None:
    """last_activity_ts() returns a monotonic timestamp immediately after
    construction so the watchdog doesn't mis-fire on startup."""
    store = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=store)
    bot = TelegramBot(
        kill_manager=km, token="tok", chat_id="42", rate_limit_sec=0.0
    )
    ts = bot.last_activity_ts()
    assert isinstance(ts, float)
    # Must be close to now (within 5s).
    assert abs(time.monotonic() - ts) < 5.0


def test_watchdog_run_exits_on_stop(tmp_path: Path) -> None:
    """run() loop exits promptly when stop() is called."""
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot()
    wd = _watchdog(
        bot=bot, kill_mgr=km, bus=bus, poll_interval_sec=1
    )

    async def _exercise() -> None:
        task = asyncio.create_task(wd.run())
        await asyncio.sleep(0.05)
        await wd.stop()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(_exercise())
