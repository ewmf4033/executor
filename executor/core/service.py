"""
Executor service main loop.

Phase 1: start the event bus + audit writer, log EXECUTOR_LIFECYCLE {start,stop},
wait on a stop signal. No venue adapters, no risk engine, no strategies attached.
Purpose is to prove the scaffolding runs as a systemd service and the audit log
receives writes.
"""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from ..audit.writer import AuditWriter
from .event_bus import EventBus
from .events import Event, EventType, Source
from .logging import configure, get_logger


log = get_logger("executor.service")


DEFAULT_AUDIT_DIR = "/root/executor/audit-logs"


class ExecutorService:
    def __init__(self, *, audit_dir: str | os.PathLike[str] | None = None) -> None:
        self.audit_dir = Path(audit_dir or os.environ.get("EXECUTOR_AUDIT_DIR", DEFAULT_AUDIT_DIR))
        self.bus = EventBus()
        self.audit = AuditWriter(self.audit_dir)
        self._stop_event = asyncio.Event()
        self._subscription = None

    async def start(self) -> None:
        configure()
        await self.audit.start()
        self._subscription = await self.bus.subscribe("audit", on_event=self.audit.on_event)
        await self.bus.start()
        await self.bus.publish(
            Event.make(
                EventType.EXECUTOR_LIFECYCLE,
                source=Source.EXECUTOR,
                payload={"state": "STARTED", "audit_dir": str(self.audit_dir)},
            )
        )
        log.info("service.started", audit_dir=str(self.audit_dir))

    async def stop(self) -> None:
        await self.bus.publish(
            Event.make(
                EventType.EXECUTOR_LIFECYCLE,
                source=Source.EXECUTOR,
                payload={"state": "STOPPING"},
            )
        )
        await self.bus.stop()
        await self.audit.stop()
        log.info("service.stopped")

    async def run_until_signaled(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except NotImplementedError:
                pass
        await self.start()
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()


async def amain() -> None:
    svc = ExecutorService()
    await svc.run_until_signaled()


def main() -> None:
    configure()
    asyncio.run(amain())


if __name__ == "__main__":
    main()
