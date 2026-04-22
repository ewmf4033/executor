"""
Telemetry HTTP endpoint for the executor daemon.

Minimal aiohttp server on port 9879 (attribution owns 9878):
  GET /health           — process uptime + daemon mode flag
  GET /pipeline_stats   — event counts (1h + lifetime), orchestrator counters

For humans reading curl output. Not a UI.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from aiohttp import web

from .events import Event
from .logging import get_logger


log = get_logger("executor.telemetry")


DEFAULT_PORT = 9879
DEFAULT_HOST = "127.0.0.1"


class TelemetryServer:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        daemon_mode: bool = True,
    ) -> None:
        self._host = host
        self._port = port
        self._daemon_mode = daemon_mode
        self._started_at = time.time()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

        # Rolling event count buckets.
        self._lifetime_counts: Counter[str] = Counter()
        # last_hour is approximated with a fixed-size deque of (ts, type)
        # items; we trim on each read. A few MB even at 1 kHz sustained.
        self._last_hour: list[tuple[float, str]] = []

        # Orchestrator hook — set by DaemonService at wire-time.
        self._orchestrator: Any = None
        # AuditWriter hook — set by DaemonService at wire-time.
        # Phase 4.10 (4.9.1-a): surface audit write-failure state so
        # external monitoring can see the fail-closed counter without
        # tailing logs.
        self._audit_writer: Any = None

    def set_orchestrator(self, orch: Any) -> None:
        self._orchestrator = orch

    def set_audit_writer(self, audit: Any) -> None:
        self._audit_writer = audit

    async def on_event(self, event: Event) -> None:
        """Bus subscriber callback."""
        etype = event.event_type.value
        self._lifetime_counts[etype] += 1
        self._last_hour.append((time.time(), etype))
        # Cheap periodic trim.
        if len(self._last_hour) > 20_000:
            self._trim_hour()

    def _trim_hour(self) -> None:
        cutoff = time.time() - 3600
        # Binary search would be cleaner; linear is fine at this cardinality.
        i = 0
        for i in range(len(self._last_hour)):
            if self._last_hour[i][0] >= cutoff:
                break
        else:
            self._last_hour = []
            return
        self._last_hour = self._last_hour[i:]

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/pipeline_stats", self._pipeline_stats)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        log.info("telemetry.start", host=self._host, port=self._port)

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        log.info("telemetry.stop")

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "uptime_sec": round(time.time() - self._started_at, 2),
                "started_at": self._started_at,
                "daemon_mode": self._daemon_mode,
            }
        )

    async def _pipeline_stats(self, request: web.Request) -> web.Response:
        self._trim_hour()
        hour_counts: Counter[str] = Counter(t for _, t in self._last_hour)
        body: dict[str, Any] = {
            "lifetime": dict(self._lifetime_counts),
            "last_hour": dict(hour_counts),
            "lifetime_total": sum(self._lifetime_counts.values()),
            "last_hour_total": sum(hour_counts.values()),
        }
        if self._orchestrator is not None and hasattr(self._orchestrator, "stats"):
            body["orchestrator"] = self._orchestrator.stats()
        if self._audit_writer is not None:
            # Phase 4.10 (4.9.1-a): expose AuditWriter fail-closed state.
            # Field names match the private attributes set by AuditWriter;
            # stable enough for monitoring since they are covered by tests.
            body["audit_writer"] = {
                "consecutive_write_failures": getattr(
                    self._audit_writer, "_consecutive_write_failures", 0
                ),
                "kill_engaged": getattr(
                    self._audit_writer, "_audit_kill_engaged", False
                ),
            }
        return web.json_response(body)
