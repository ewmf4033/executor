"""
Tiny aiohttp HTTP server exposing GET /attribution/summary.

Single file. Listens on port 9878 by default. Started as an asyncio task
inside the executor process; gracefully shuts down on stop().
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from aiohttp import web

from ..core.logging import get_logger
from .tracker import AttributionTracker


log = get_logger("executor.attribution.server")


DEFAULT_PORT = 9878
DEFAULT_HOST = "127.0.0.1"


class AttributionServer:
    def __init__(
        self,
        tracker: AttributionTracker,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._tracker = tracker
        self._host = host
        self._port = port
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/attribution/summary", self._summary)
        app.router.add_get("/attribution/health", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        log.info("attribution.server.start", host=self._host, port=self._port)

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        log.info("attribution.server.stop")

    async def _summary(self, request: web.Request) -> web.Response:
        try:
            since_ns = int(request.query.get("since_ns", "0") or 0)
        except ValueError:
            return web.json_response({"error": "since_ns must be int"}, status=400)
        strategy_id = request.query.get("strategy_id")
        data = self._tracker.summary(since_ns=since_ns, strategy_id=strategy_id)
        return web.json_response(data)

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "ts": time.time()})

    @property
    def port(self) -> int:
        # Useful in tests when port=0 is requested for ephemeral binding.
        if self._site is None:
            return self._port
        # aiohttp doesn't expose the bound port directly on TCPSite; we keep
        # the configured port. Tests should pass an explicit port.
        return self._port
