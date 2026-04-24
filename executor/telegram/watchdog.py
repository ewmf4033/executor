"""Telegram polling watchdog (Phase 4.14c).

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
    - Escalation fires exactly once per watchdog lifecycle; subsequent
      stalls do not re-engage (guarded by ``_escalated``).
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
    ) -> None:
        self._bot = bot
        self._kill = kill_mgr
        self._bus = bus
        self._stall_threshold_sec = stall_threshold_sec
        self._poll_interval_sec = poll_interval_sec
        self._max_restarts = max_restarts
        self._restart_window_sec = restart_window_sec
        self._escalate_on_max = escalate_on_max
        self._restart_ts: deque[float] = deque()
        self._stop_event = asyncio.Event()
        self._escalated = False

    async def run(self) -> None:
        """Main watchdog loop. Runs until :meth:`stop` is called."""
        log.info(
            "telegram.watchdog.start",
            stall_threshold_sec=self._stall_threshold_sec,
            poll_interval_sec=self._poll_interval_sec,
            max_restarts=self._max_restarts,
            restart_window_sec=self._restart_window_sec,
            escalate_on_max=self._escalate_on_max,
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
            return  # healthy

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

        await self._restart_bot()
        self._restart_ts.append(now)

    async def _emit_stall(self, *, gap: float) -> None:
        try:
            await self._bus.publish(
                Event.make(
                    EventType.TELEGRAM_STALL_DETECTED,
                    source=Source.TELEGRAM,
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
        try:
            log.warning(
                "telegram.watchdog.restart_attempt",
                restart_count=len(self._restart_ts),
            )
            await self._bot.stop()
            # Brief pause so sockets close cleanly.
            await asyncio.sleep(1.0)
            await self._bot.start()
            log.info("telegram.watchdog.restart_ok")
        except Exception as exc:
            log.error("telegram.watchdog.restart_failed", error=str(exc))

    async def _escalate(self, *, gap: float) -> None:
        log.error(
            "telegram.watchdog.escalating",
            gap_sec=round(gap, 2),
            restart_count=len(self._restart_ts),
            max_restarts=self._max_restarts,
        )
        try:
            await self._bus.publish(
                Event.make(
                    EventType.TELEGRAM_WATCHDOG_ESCALATED,
                    source=Source.TELEGRAM,
                    payload={
                        "gap_sec": round(gap, 2),
                        "restart_count": len(self._restart_ts),
                        "action": "SOFT_KILL_ENGAGED",
                    },
                )
            )
        except Exception as exc:
            # Never block escalation on event-bus failures.
            log.warning(
                "telegram.watchdog.escalation_emit_failed", error=str(exc)
            )

        try:
            await self._kill.engage(
                KillMode.SOFT,
                "telegram watchdog: stall exceeded max restarts",
                source="telegram_watchdog",
            )
        except Exception as exc:
            log.error(
                "telegram.watchdog.escalation_kill_failed", error=str(exc)
            )

    async def stop(self) -> None:
        self._stop_event.set()
