"""Telegram polling watchdog (Phase 4.14c, hardened in 4.14d).

Monitors ``TelegramBot.last_activity_ts()`` at a configurable cadence
(``poll_interval_sec``). If the gap exceeds ``stall_threshold_sec`` the
watchdog attempts to restart the bot task. Restart attempts are tracked
in a rolling ``restart_window_sec`` window; if ``max_restarts`` is
exhausted and ``escalate_on_max`` is True, the watchdog engages the
kill-switch in SOFT mode.

Escalation policy (frozen by spec):
    - Never auto-HARDs on Telegram health alone. Losing operator
      awareness is a reason to stop admitting new intents, not to
      cancel open orders.
    - SOFT kill engagement is routed directly through ``KillManager``
      (not via Telegram) so the escalation path does not depend on the
      very surface that failed.
    - ``KillManager.engage`` enforces monotonic severity (4.14d): if
      the operator has already engaged HARD/PANIC, the watchdog's SOFT
      engage becomes a no-op. The escalation event is still emitted so
      the audit trail records the watchdog's intent; the event payload
      reports the effective outcome.
    - Escalation fires exactly once per watchdog lifecycle; subsequent
      stalls do not re-engage (guarded by ``_escalated``).

Phase 4.14d hardening (Codex review):
    - ``bot.stop()`` and ``bot.start()`` are wrapped in
      ``asyncio.wait_for(..., timeout=restart_timeout_sec)`` so a hung
      bot lifecycle call cannot wedge the watchdog loop.
    - Timed-out or failed restart attempts count toward the restart
      budget — a chronically-hung bot must eventually escalate to
      SOFT (subject to monotonic severity) rather than retry forever.
    - Watchdog-originated events use ``Source.TELEGRAM_WATCHDOG`` so
      audit/alerting can separate operator-awareness degradation from
      command-path Telegram activity.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

from ..core.event_bus import EventBus
from ..core.events import Event, EventType, Source
from ..core.logging import get_logger
from ..kill.manager import KillManager
from ..kill.state import KillMode
from .bot import TelegramBot


log = get_logger("executor.telegram.watchdog")


DEFAULT_RESTART_TIMEOUT_SEC = 30.0
DEFAULT_POST_STOP_PAUSE_SEC = 1.0


class TelegramWatchdog:
    """Watches TelegramBot polling liveness and escalates on stall."""

    def __init__(
        self,
        bot: TelegramBot,
        kill_mgr: KillManager,
        bus: EventBus,
        *,
        stall_threshold_sec: int = 120,
        poll_interval_sec: int = 10,
        max_restarts: int = 3,
        restart_window_sec: int = 300,
        escalate_on_max: bool = True,
        restart_timeout_sec: float = DEFAULT_RESTART_TIMEOUT_SEC,
        post_stop_pause_sec: float = DEFAULT_POST_STOP_PAUSE_SEC,
    ) -> None:
        self._bot = bot
        self._kill = kill_mgr
        self._bus = bus
        self._stall_threshold_sec = stall_threshold_sec
        self._poll_interval_sec = poll_interval_sec
        self._max_restarts = max_restarts
        self._restart_window_sec = restart_window_sec
        self._escalate_on_max = escalate_on_max
        self._restart_timeout_sec = restart_timeout_sec
        self._post_stop_pause_sec = post_stop_pause_sec
        self._restart_ts: deque[float] = deque()
        self._stop_event = asyncio.Event()
        self._escalated = False
        # Phase 4.14e (H8) — once-only escalation latch needs a bounded
        # reset. Prior versions set ``_escalated=True`` and never cleared
        # it, so a recovered bot (subsequent healthy windows) still
        # counted the watchdog as "fired"; a second genuine stall in
        # the same daemon lifetime would emit no kill signal. The reset
        # rule, spelled out: clear ``_escalated`` only after
        #   (a) the rolling restart-timestamp deque is empty
        #       (pruned of everything inside ``restart_window_sec``), AND
        #   (b) a full ``stall_threshold_sec`` has elapsed since we first
        #       observed a healthy gap following the escalation.
        # ``_healthy_since_ts`` is set the first time we see a healthy
        # gap while ``_escalated`` is True, and reset to ``None`` on any
        # subsequent stall. Gating on (a) prevents oscillation: a bot
        # that flaps between stall and brief recovery cannot drain the
        # latch.
        self._healthy_since_ts: float | None = None

    async def run(self) -> None:
        """Main watchdog loop. Runs until :meth:`stop` is called."""
        log.info(
            "telegram.watchdog.start",
            stall_threshold_sec=self._stall_threshold_sec,
            poll_interval_sec=self._poll_interval_sec,
            max_restarts=self._max_restarts,
            restart_window_sec=self._restart_window_sec,
            escalate_on_max=self._escalate_on_max,
            restart_timeout_sec=self._restart_timeout_sec,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._poll_interval_sec,
                    )
                    break  # stop was signalled
                except asyncio.TimeoutError:
                    pass  # normal poll cadence
                try:
                    await self._check_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning(
                        "telegram.watchdog.check_failed", error=str(exc)
                    )
        finally:
            log.info("telegram.watchdog.stop")

    async def _check_once(self) -> None:
        now = time.monotonic()
        last_activity = self._bot.last_activity_ts()
        gap = now - last_activity

        if gap < self._stall_threshold_sec:
            # Phase 4.14e (H8): bounded reset of the escalation latch.
            # We only clear ``_escalated`` when the bot has genuinely
            # recovered, defined as:
            #   - the rolling restart-timestamp deque is empty
            #     (no recent restarts within restart_window_sec), AND
            #   - at least one full stall_threshold_sec has elapsed
            #     since we first observed a healthy gap after the
            #     escalation.
            # Pruning the deque here (before the empty-check) mirrors
            # the prune below so the reset path observes the same
            # rolling window.
            cutoff = now - self._restart_window_sec
            while self._restart_ts and self._restart_ts[0] < cutoff:
                self._restart_ts.popleft()
            if self._escalated and not self._restart_ts:
                if self._healthy_since_ts is None:
                    self._healthy_since_ts = now
                elif now - self._healthy_since_ts >= self._stall_threshold_sec:
                    log.info(
                        "telegram.watchdog.escalated_cleared",
                        healthy_window_sec=round(
                            now - self._healthy_since_ts, 2
                        ),
                    )
                    self._escalated = False
                    self._healthy_since_ts = None
            return  # healthy

        # Stall seen — reset the healthy-window tracker.
        self._healthy_since_ts = None

        # Stall detected.
        await self._emit_stall(gap=gap)

        # Prune restart timestamps outside the rolling window.
        cutoff = now - self._restart_window_sec
        while self._restart_ts and self._restart_ts[0] < cutoff:
            self._restart_ts.popleft()

        if len(self._restart_ts) >= self._max_restarts:
            if self._escalate_on_max and not self._escalated:
                await self._escalate(gap=gap)
                self._escalated = True
            return

        # Phase 4.14d: always count the attempt toward the budget,
        # even if the underlying bot.stop()/bot.start() hangs or
        # raises. A pathological bot that never returns from stop()
        # must still escalate; otherwise the watchdog would retry
        # forever with no progress.
        self._restart_ts.append(now)
        await self._restart_bot()

    async def _emit_stall(self, *, gap: float) -> None:
        try:
            await self._bus.publish(
                Event.make(
                    EventType.TELEGRAM_STALL_DETECTED,
                    source=Source.TELEGRAM_WATCHDOG,
                    payload={
                        "gap_sec": round(gap, 2),
                        "threshold_sec": self._stall_threshold_sec,
                        "restart_count_in_window": len(self._restart_ts),
                    },
                )
            )
        except Exception as exc:
            # Never block the watchdog loop on event-bus failures.
            log.warning("telegram.watchdog.emit_failed", error=str(exc))

    async def _restart_bot(self) -> None:
        """Attempt a bot stop/start cycle with timeout containment.

        Any failure (TimeoutError or ordinary Exception from either
        lifecycle call) is logged and emitted as
        TELEGRAM_WATCHDOG_RESTART_FAILED. The attempt has already been
        counted toward the restart budget by the caller.
        """
        log.warning(
            "telegram.watchdog.restart_attempt",
            restart_count=len(self._restart_ts),
            max_restarts=self._max_restarts,
        )

        # Step 1: stop.
        stop_ok, stop_phase_err = await self._bounded_lifecycle(
            self._bot.stop, phase="stop"
        )
        if not stop_ok:
            await self._emit_restart_failed(
                phase="stop", error=stop_phase_err
            )
            return

        # Brief pause so sockets close cleanly. Use wait_for on the
        # stop_event so shutdown does not have to wait out the pause.
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self._post_stop_pause_sec,
            )
            return  # shutdown signalled during pause
        except asyncio.TimeoutError:
            pass

        # Step 2: start.
        start_ok, start_phase_err = await self._bounded_lifecycle(
            self._bot.start, phase="start"
        )
        if not start_ok:
            await self._emit_restart_failed(
                phase="start", error=start_phase_err
            )
            return

        await self._emit_restarted()

    async def _bounded_lifecycle(
        self, call, *, phase: str
    ) -> tuple[bool, str]:
        """Run a bot.start/stop call with timeout + exception containment.

        Returns ``(ok, error_string)``. ``ok=False`` covers both
        timeouts and ordinary exceptions; the caller reports a single
        failure event per phase.
        """
        try:
            await asyncio.wait_for(
                call(), timeout=self._restart_timeout_sec
            )
            return True, ""
        except asyncio.TimeoutError:
            log.error(
                "telegram.watchdog.restart_timeout",
                phase=phase,
                timeout_sec=self._restart_timeout_sec,
            )
            return False, f"timeout after {self._restart_timeout_sec}s"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error(
                "telegram.watchdog.restart_failed",
                phase=phase,
                error=str(exc),
            )
            return False, str(exc)

    async def _emit_restarted(self) -> None:
        try:
            await self._bus.publish(
                Event.make(
                    EventType.TELEGRAM_WATCHDOG_RESTARTED,
                    source=Source.TELEGRAM_WATCHDOG,
                    payload={
                        "restart_count_in_window": len(self._restart_ts),
                    },
                )
            )
        except Exception as exc:
            log.warning(
                "telegram.watchdog.restarted_emit_failed", error=str(exc)
            )

    async def _emit_restart_failed(self, *, phase: str, error: str) -> None:
        try:
            await self._bus.publish(
                Event.make(
                    EventType.TELEGRAM_WATCHDOG_RESTART_FAILED,
                    source=Source.TELEGRAM_WATCHDOG,
                    payload={
                        "phase": phase,
                        "error": error,
                        "restart_count_in_window": len(self._restart_ts),
                        "restart_timeout_sec": self._restart_timeout_sec,
                    },
                )
            )
        except Exception as exc:
            log.warning(
                "telegram.watchdog.restart_failed_emit_failed",
                error=str(exc),
            )

    async def _escalate(self, *, gap: float) -> None:
        """Engage SOFT kill via KillManager; emit escalation event with
        the effective before/after kill state.

        Phase 4.14e — truthful result classification. Prior to this
        change the payload defaulted to ``result="soft_kill_engaged"``
        and only switched to ``"skipped_existing_stricter_kill"``
        when the final mode was not SOFT. That defaulting was wrong
        for the SOFT → SOFT case: a watchdog SOFT engage on an
        already-SOFT row is a no-op (KillManager's monotonic
        severity logic drops equal-severity engages) but the
        emitted event still claimed ``soft_kill_engaged``. The fix
        snapshots the kill state before the engage call and
        classifies the result from the actual prior-mode:

            prev NONE  → SOFT     =>  "soft_kill_engaged"
            prev SOFT  → SOFT     =>  "noop_already_soft"
            prev HARD* → SOFT     =>  "skipped_existing_stricter_kill"

        The payload also carries ``effective`` / ``skipped_weaker``
        so audit consumers don't have to infer the no-op from the
        mode strings, and ``prev_mode`` / ``prev_panic`` to show the
        actual transition rather than just the post-engage state.
        """
        log.error(
            "telegram.watchdog.escalating",
            gap_sec=round(gap, 2),
            restart_count=len(self._restart_ts),
            max_restarts=self._max_restarts,
        )

        attempted_mode = KillMode.SOFT
        # Snapshot BEFORE engage so the classification reflects the
        # actual transition we drove, not just the final state.
        prev_snap = self._kill.snapshot()
        prev_mode = prev_snap.mode
        prev_panic = prev_snap.panic

        result = "kill_engage_error"  # pessimistic default — overridden on success
        effective = False
        skipped_weaker = False
        current_mode_after = prev_mode
        panic_after = prev_panic
        try:
            snap = await self._kill.engage(
                attempted_mode,
                "telegram watchdog: stall exceeded max restarts",
                source="telegram_watchdog",
            )
            current_mode_after = snap.mode
            panic_after = snap.panic

            if prev_mode == KillMode.NONE:
                # NONE → SOFT: a real engage landed.
                result = "soft_kill_engaged"
                effective = True
            elif prev_mode == KillMode.SOFT:
                # SOFT → SOFT: equal-severity engage is a no-op in
                # KillManager; the audit trail must not claim an
                # engagement happened.
                result = "noop_already_soft"
                effective = False
                skipped_weaker = True
            else:
                # HARD or HARD+PANIC → our SOFT request was dropped by
                # monotonic severity; the stricter prior mode stands.
                result = "skipped_existing_stricter_kill"
                effective = False
                skipped_weaker = True
        except Exception as exc:
            log.error(
                "telegram.watchdog.escalation_kill_failed", error=str(exc)
            )

        try:
            await self._bus.publish(
                Event.make(
                    EventType.TELEGRAM_WATCHDOG_ESCALATED,
                    source=Source.TELEGRAM_WATCHDOG,
                    payload={
                        "gap_sec": round(gap, 2),
                        "restart_count": len(self._restart_ts),
                        "action": "soft_engage_attempted",
                        "result": result,
                        "effective": bool(effective),
                        "skipped_weaker": bool(skipped_weaker),
                        "prev_mode": prev_mode.value,
                        "prev_panic": bool(prev_panic),
                        "current_mode": current_mode_after.value,
                        "panic": panic_after,
                        "requested_mode": attempted_mode.value,
                    },
                )
            )
        except Exception as exc:
            # Never block escalation on event-bus failures.
            log.warning(
                "telegram.watchdog.escalation_emit_failed", error=str(exc)
            )

    async def stop(self) -> None:
        self._stop_event.set()
