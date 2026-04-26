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
    restart_timeout_sec: float = 1.0,
    post_stop_pause_sec: float = 0.0,
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
        restart_timeout_sec=restart_timeout_sec,
        post_stop_pause_sec=post_stop_pause_sec,
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


# --------------------------------------------------------------------------
# Phase 4.14d — Codex review fixes
# --------------------------------------------------------------------------


class _HangingStopBot(_FakeBot):
    """bot.stop() never returns — simulates a wedged poll task."""

    async def stop(self) -> None:
        self.stop_calls += 1
        await asyncio.Event().wait()  # forever


class _HangingStartBot(_FakeBot):
    """bot.start() never returns — simulates a wedged network init."""

    async def start(self) -> None:
        self.start_calls += 1
        await asyncio.Event().wait()  # forever


def test_watchdog_restart_stop_timeout_does_not_hang_loop(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _HangingStopBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot,
        kill_mgr=km,
        bus=bus,
        restart_timeout_sec=0.01,
    )

    async def _exercise() -> None:
        # _check_once must return promptly even though stop() hangs.
        await asyncio.wait_for(wd._check_once(), timeout=1.0)

    asyncio.run(_exercise())

    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_STALL_DETECTED in kinds
    # Timeout on stop() must surface as a restart-failed event.
    assert EventType.TELEGRAM_WATCHDOG_RESTART_FAILED in kinds
    # Failure counts toward the restart budget.
    assert len(wd._restart_ts) == 1
    # start() must NOT have been attempted because stop() timed out.
    assert bot.start_calls == 0
    # Watchdog can run another check immediately.
    asyncio.run(asyncio.wait_for(wd._check_once(), timeout=1.0))


def test_watchdog_restart_start_timeout_does_not_hang_loop(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _HangingStartBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot,
        kill_mgr=km,
        bus=bus,
        restart_timeout_sec=0.01,
    )

    async def _exercise() -> None:
        await asyncio.wait_for(wd._check_once(), timeout=1.0)

    asyncio.run(_exercise())

    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_STALL_DETECTED in kinds
    assert EventType.TELEGRAM_WATCHDOG_RESTART_FAILED in kinds
    # phase must be "start" since stop() succeeded.
    failed = [
        e for e in bus.events
        if e.event_type == EventType.TELEGRAM_WATCHDOG_RESTART_FAILED
    ]
    assert failed and failed[0].payload.get("phase") == "start"
    assert len(wd._restart_ts) == 1
    assert bot.stop_calls == 1


def test_watchdog_restart_timeout_counts_toward_escalation_budget(
    tmp_path: Path,
) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _HangingStopBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot,
        kill_mgr=km,
        bus=bus,
        restart_timeout_sec=0.01,
        max_restarts=2,
    )

    async def _exercise() -> None:
        # Two failed restart attempts fill the budget.
        await asyncio.wait_for(wd._check_once(), timeout=1.0)
        await asyncio.wait_for(wd._check_once(), timeout=1.0)
        # Third stall with full budget -> escalation.
        await asyncio.wait_for(wd._check_once(), timeout=1.0)

    asyncio.run(_exercise())

    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_WATCHDOG_ESCALATED in kinds
    # Budget full.
    assert len(wd._restart_ts) >= 2
    # SOFT kill engaged (monotonic severity: no prior stricter kill).
    assert km.mode is KillMode.SOFT


def test_watchdog_escalation_does_not_downgrade_existing_hard_or_panic(
    tmp_path: Path,
) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    # Operator engages HARD+PANIC before watchdog escalation.
    asyncio.run(km.engage(KillMode.HARD, "operator panic", panic=True))
    assert km.mode is KillMode.HARD
    assert km.snapshot().panic is True

    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus, max_restarts=1)
    wd._restart_ts.append(time.monotonic() - 10.0)  # budget full

    asyncio.run(wd._check_once())

    # Escalation event still emitted (forensic trail).
    kinds = [e.event_type for e in bus.events]
    assert EventType.TELEGRAM_WATCHDOG_ESCALATED in kinds
    # Final state preserves stricter HARD + panic.
    assert km.mode is KillMode.HARD
    assert km.snapshot().panic is True

    # Escalation payload reports that the SOFT engage was skipped.
    esc = [
        e for e in bus.events
        if e.event_type == EventType.TELEGRAM_WATCHDOG_ESCALATED
    ][0]
    assert esc.payload.get("action") == "soft_engage_attempted"
    assert esc.payload.get("result") == "skipped_existing_stricter_kill"
    assert esc.payload.get("current_mode") == "HARD"
    assert esc.payload.get("panic") is True


# ---------------------------------------------------------------------------
# Phase 4.14e — escalation result classification (B3).
#
# The pre-4.14e watchdog defaulted to ``result="soft_kill_engaged"`` and
# only overrode it when the final mode was not SOFT. This misreported
# SOFT → SOFT (no-op) as an engagement. The fix snapshots kill state
# before engage and classifies from the actual prior mode, and also
# reports explicit ``effective`` / ``skipped_weaker`` flags in the
# payload.
# ---------------------------------------------------------------------------


def test_watchdog_escalation_reports_noop_when_soft_already_active(
    tmp_path: Path,
) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    # Pre-engage SOFT (e.g. strategy gate earlier tripped). Watchdog's
    # subsequent SOFT engage is a no-op in KillManager.
    asyncio.run(km.engage(KillMode.SOFT, "prior soft"))
    assert km.mode is KillMode.SOFT

    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus, max_restarts=1)
    wd._restart_ts.append(time.monotonic() - 10.0)  # budget full

    asyncio.run(wd._check_once())

    esc = [
        e for e in bus.events
        if e.event_type == EventType.TELEGRAM_WATCHDOG_ESCALATED
    ]
    assert esc, "escalation event must still emit for audit"
    p = esc[0].payload
    # The watchdog must NOT claim it engaged SOFT when SOFT was already on.
    assert p["result"] == "noop_already_soft"
    assert p["effective"] is False
    assert p["skipped_weaker"] is True
    assert p["prev_mode"] == "SOFT"
    assert p["current_mode"] == "SOFT"
    assert p["panic"] is False
    # Final kill state: still SOFT (unchanged).
    assert km.mode is KillMode.SOFT


def test_watchdog_escalation_reports_skipped_when_existing_panic(
    tmp_path: Path,
) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    asyncio.run(km.engage(KillMode.HARD, "operator panic", panic=True))
    assert km.snapshot().panic is True

    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus, max_restarts=1)
    wd._restart_ts.append(time.monotonic() - 10.0)

    asyncio.run(wd._check_once())

    esc = [
        e for e in bus.events
        if e.event_type == EventType.TELEGRAM_WATCHDOG_ESCALATED
    ][0]
    assert esc.payload["result"] == "skipped_existing_stricter_kill"
    assert esc.payload["effective"] is False
    assert esc.payload["skipped_weaker"] is True
    assert esc.payload["prev_mode"] == "HARD"
    assert esc.payload["prev_panic"] is True
    assert esc.payload["current_mode"] == "HARD"
    assert esc.payload["panic"] is True


def test_watchdog_escalation_payload_matches_actual_transition(
    tmp_path: Path,
) -> None:
    """Happy path: NONE → SOFT via watchdog escalation. Payload must say
    the SOFT engage was actually effective (not inferred from the mode)."""
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(bot=bot, kill_mgr=km, bus=bus, max_restarts=1)
    wd._restart_ts.append(time.monotonic() - 10.0)

    asyncio.run(wd._check_once())

    esc = [
        e for e in bus.events
        if e.event_type == EventType.TELEGRAM_WATCHDOG_ESCALATED
    ][0]
    assert esc.payload["result"] == "soft_kill_engaged"
    assert esc.payload["effective"] is True
    assert esc.payload["skipped_weaker"] is False
    assert esc.payload["prev_mode"] == "NONE"
    assert esc.payload["current_mode"] == "SOFT"
    assert km.mode is KillMode.SOFT


# ---------------------------------------------------------------------------
# Phase 4.14e (H8) — bounded escalation latch reset.
#
# After escalating, the watchdog clears ``_escalated`` only after a full
# healthy window during which:
#   - the restart-timestamp deque is empty (no stalls inside
#     restart_window_sec), and
#   - the observed gap has been below ``stall_threshold_sec`` for at
#     least ``stall_threshold_sec`` contiguous seconds.
# Any intervening stall resets the healthy-window timer to None.
# ---------------------------------------------------------------------------


def test_watchdog_escalated_resets_only_after_healthy_window(
    tmp_path: Path,
) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    # stall_threshold_sec is 2s so the "healthy window" is 2s.
    wd = _watchdog(
        bot=bot,
        kill_mgr=km,
        bus=bus,
        stall_threshold_sec=2,
        restart_window_sec=10,
        max_restarts=1,
    )
    # Pre-fill budget so the first stall check escalates.
    wd._restart_ts.append(time.monotonic() - 1.0)

    asyncio.run(wd._check_once())
    assert wd._escalated is True
    # Bot now healthy; restart-deque still holds the recent timestamp.
    bot.set_last_activity(time.monotonic())
    asyncio.run(wd._check_once())
    # Healthy gap observed but restart_ts not yet empty → no reset yet.
    assert wd._escalated is True

    # Prune all restart timestamps by advancing past the restart
    # window and running another healthy check (which will prune).
    wd._restart_ts.clear()  # simulate window drain
    bot.set_last_activity(time.monotonic())
    asyncio.run(wd._check_once())
    # First healthy check after empty deque records the start of the
    # healthy window — latch NOT yet cleared.
    assert wd._escalated is True
    assert wd._healthy_since_ts is not None

    # Force the healthy window to appear fully elapsed by rewinding
    # the tracked start (equivalent to waiting stall_threshold_sec).
    wd._healthy_since_ts = time.monotonic() - 5.0  # > 2s
    bot.set_last_activity(time.monotonic())
    asyncio.run(wd._check_once())
    assert wd._escalated is False, "healthy window elapsed — latch must clear"
    assert wd._healthy_since_ts is None


def test_watchdog_healthy_window_resets_on_stall(tmp_path: Path) -> None:
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot,
        kill_mgr=km,
        bus=bus,
        stall_threshold_sec=2,
        restart_window_sec=10,
        max_restarts=1,
    )
    wd._restart_ts.append(time.monotonic() - 1.0)
    asyncio.run(wd._check_once())  # escalate
    assert wd._escalated is True

    # Start healthy window.
    wd._restart_ts.clear()
    bot.set_last_activity(time.monotonic())
    asyncio.run(wd._check_once())
    assert wd._healthy_since_ts is not None
    # A subsequent stall resets the healthy-window tracker.
    bot.set_last_activity(time.monotonic() - 200.0)
    asyncio.run(wd._check_once())
    assert wd._healthy_since_ts is None
    assert wd._escalated is True  # still latched


def test_watchdog_events_use_telegram_watchdog_source(tmp_path: Path) -> None:
    """All watchdog-originated events use Source.TELEGRAM_WATCHDOG."""
    captured: list[Event] = []
    bus = _CapturingBus()
    km = _kill_mgr(tmp_path, captured)
    bot = _FakeBot(last_activity=time.monotonic() - 200.0)
    bot.refresh_on_start = False
    wd = _watchdog(
        bot=bot, kill_mgr=km, bus=bus, restart_timeout_sec=0.01, max_restarts=1
    )

    async def _exercise() -> None:
        # One successful restart, then a second stall → escalation.
        bot.refresh_on_start = True
        await asyncio.wait_for(wd._check_once(), timeout=1.0)
        bot.refresh_on_start = False
        bot.set_last_activity(time.monotonic() - 200.0)
        await asyncio.wait_for(wd._check_once(), timeout=1.0)

    asyncio.run(_exercise())

    watchdog_event_types = {
        EventType.TELEGRAM_STALL_DETECTED,
        EventType.TELEGRAM_WATCHDOG_RESTARTED,
        EventType.TELEGRAM_WATCHDOG_RESTART_FAILED,
        EventType.TELEGRAM_WATCHDOG_ESCALATED,
    }
    for ev in bus.events:
        if ev.event_type in watchdog_event_types:
            assert ev.source == "telegram_watchdog", (
                f"{ev.event_type} has source={ev.source}, expected telegram_watchdog"
            )
    # Verify at least some watchdog events were emitted.
    assert any(ev.event_type in watchdog_event_types for ev in bus.events)
