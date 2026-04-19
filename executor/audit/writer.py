"""
Audit log writer.

Append-only SQLite, daily rotation, gzip after 30 days, keep forever
(Decision 4 of /root/trading-wiki/specs/0d-executor.md).

- One file per UTC date: audit-YYYY-MM-DD.sqlite
- Current day is opened with WAL journaling.
- On rotation, the previous day's DB is finalized (VACUUM optional),
  left as-is for 30 days, then gzipped by maintenance (reaper runs
  once per startup + daily).
- Writer subscribes to the event bus and fans every Event into
  the current-day DB.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.events import Event
from ..core.logging import get_logger


log = get_logger("executor.audit")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    ts_ns INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    intent_id TEXT,
    leg_id TEXT,
    venue TEXT,
    market_id TEXT,
    strategy_id TEXT,
    payload_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_events_intent ON events(intent_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_ns);
CREATE INDEX IF NOT EXISTS idx_events_strategy_ts ON events(strategy_id, ts_ns);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, ts_ns);
"""


GZIP_AFTER_DAYS = 30


class AuditWriter:
    """
    One writer per executor process. Owns a single sqlite3.Connection that
    is swapped at UTC-midnight rotation.
    """

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._current_date: str | None = None     # YYYY-MM-DD
        self._lock = asyncio.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._open_for_today()
        self._started = True
        # Fire-and-forget reaper — gzip >30d files.
        asyncio.create_task(self._reap_once(), name="audit-reap")
        log.info("audit.start", path=str(self._current_path()))

    async def stop(self) -> None:
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                finally:
                    self._conn.close()
                self._conn = None
            self._started = False
        log.info("audit.stop")

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def write(self, event: Event) -> None:
        """
        Serialize + insert. Holds the lock briefly; actual SQLite I/O is
        dispatched to a worker thread so the event loop stays free.
        """
        log.debug(
            "audit.write.enter",
            event_type=event.event_type.value,
            event_id=event.event_id,
        )
        async with self._lock:
            self._rotate_if_needed()
            conn = self._conn
            assert conn is not None
            row = (
                event.event_id,
                int(event.ts_ns),
                event.event_type.value,
                event.source,
                event.intent_id,
                event.leg_id,
                event.venue,
                event.market_id,
                event.strategy_id,
                json.dumps(event.payload, default=_json_default, separators=(",", ":")),
                int(event.schema_version),
            )
            await asyncio.to_thread(_insert_row, conn, row)
        log.debug(
            "audit.write.done",
            event_type=event.event_type.value,
            event_id=event.event_id,
        )

    async def on_event(self, event: Event) -> None:
        """Adapter for EventBus.subscribe(on_event=...)."""
        try:
            await self.write(event)
        except Exception as exc:
            # Never let audit failure take down the bus — log loud and move on.
            # If audit is offline we want to know, but we do not stop trading.
            log.error(
                "audit.write.crash",
                event_type=event.event_type.value,
                event_id=event.event_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Rotation + reaping
    # ------------------------------------------------------------------

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _path_for(self, date_str: str) -> Path:
        return self._base_dir / f"audit-{date_str}.sqlite"

    def _current_path(self) -> Path:
        assert self._current_date is not None
        return self._path_for(self._current_date)

    def _open_for_today(self) -> None:
        date_str = self._today_str()
        path = self._path_for(date_str)
        conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA_SQL)
        self._conn = conn
        self._current_date = date_str

    def _rotate_if_needed(self) -> None:
        today = self._today_str()
        if self._current_date == today:
            return
        old = self._current_date
        log.info("audit.rotate", from_date=old, to_date=today)
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
            self._conn = None
        self._open_for_today()

    async def _reap_once(self) -> None:
        """Gzip audit files older than GZIP_AFTER_DAYS. Never deletes."""
        try:
            now = time.time()
            cutoff = now - (GZIP_AFTER_DAYS * 86400)
            for p in self._base_dir.glob("audit-*.sqlite"):
                try:
                    if p.stat().st_mtime >= cutoff:
                        continue
                    if p == self._current_path():
                        continue
                    gz_path = p.with_suffix(p.suffix + ".gz")
                    if gz_path.exists():
                        continue
                    await asyncio.to_thread(_gzip_file, p, gz_path)
                    # Only remove original after successful gzip.
                    p.unlink()
                    log.info("audit.reap.gzipped", path=str(gz_path))
                except Exception as exc:
                    log.warning("audit.reap.error", path=str(p), error=str(exc))
        except Exception as exc:  # pragma: no cover
            log.error("audit.reap.crash", error=str(exc))

    # ------------------------------------------------------------------
    # Read helpers (used by tests and future replay tools)
    # ------------------------------------------------------------------

    def current_db_path(self) -> Path:
        return self._current_path()

    def count(self) -> int:
        assert self._conn is not None
        cur = self._conn.execute("SELECT COUNT(*) FROM events")
        return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Helpers (module-level so asyncio.to_thread can pickle them trivially)
# ---------------------------------------------------------------------------


_INSERT_SQL = (
    "INSERT INTO events "
    "(event_id, ts_ns, event_type, source, intent_id, leg_id, venue, "
    " market_id, strategy_id, payload_json, schema_version) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _insert_row(conn: sqlite3.Connection, row: tuple[Any, ...]) -> None:
    conn.execute(_INSERT_SQL, row)


def _gzip_file(src: Path, dst: Path) -> None:
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        while True:
            chunk = f_in.read(1 << 20)
            if not chunk:
                break
            f_out.write(chunk)


def _json_default(obj: Any) -> Any:
    """Coerce Decimal, Enum, datetime, and other non-JSON types to strings."""
    # Decimal -> str (preserve precision).
    try:
        from decimal import Decimal
        if isinstance(obj, Decimal):
            return str(obj)
    except Exception:
        pass
    if hasattr(obj, "value") and hasattr(obj, "name"):
        # Enum-like.
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    return str(obj)
