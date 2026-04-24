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
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.events import Event
from ..core.logging import get_logger

if TYPE_CHECKING:
    from ..kill.manager import KillManager


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

DEFAULT_FAIL_THRESHOLD = 3


class AuditWriter:
    """
    One writer per executor process. Owns a single sqlite3.Connection that
    is swapped at UTC-midnight rotation.

    Phase 4.9 Item 1 (fail-closed): on.event tracks consecutive write
    failures. After EXECUTOR_AUDIT_FAIL_THRESHOLD failures in a row (default
    3), the kill switch is engaged (panic) so new intents stop flowing.
    Escalation is emitted via stderr + direct Telegram (bypassing audit,
    which is broken). Recovery is manual — operator restarts the daemon.
    """

    def __init__(
        self,
        base_dir: str | os.PathLike[str],
        *,
        kill_manager: "KillManager | None" = None,
        fail_threshold: int | None = None,
        capital_mode: bool = False,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._current_date: str | None = None     # YYYY-MM-DD
        self._lock = asyncio.Lock()
        self._started = False
        # Fail-closed plumbing.
        self._kill_manager: "KillManager | None" = kill_manager
        if fail_threshold is None:
            try:
                fail_threshold = int(
                    os.environ.get("EXECUTOR_AUDIT_FAIL_THRESHOLD", DEFAULT_FAIL_THRESHOLD)
                )
            except ValueError:
                fail_threshold = DEFAULT_FAIL_THRESHOLD
        self._fail_threshold = max(1, int(fail_threshold))
        self._consecutive_write_failures = 0
        self._audit_kill_engaged = False
        # Phase 4.13.1 Fix #B (GPT-5.5 review #2): in capital_mode, audit
        # write-failure escalation goes HARD + cancel_open_orders instead of
        # SOFT. Paper mode retains SOFT-only behavior so observation tests
        # are unaffected.
        self._capital_mode = bool(capital_mode)

    def set_kill_manager(self, kill_manager: "KillManager") -> None:
        """Late-binding for DaemonService: audit.start() runs before KillManager
        exists, so we plumb the reference in afterwards."""
        self._kill_manager = kill_manager

    def set_capital_mode(self, capital_mode: bool) -> None:
        """Late-binding for DaemonService: AuditWriter is constructed before
        ConfigManager has loaded YAML, so DaemonService plumbs the capital_mode
        flag in after config resolution."""
        self._capital_mode = bool(capital_mode)

    def _audit_kill_mode(self):
        """Return the KillMode to engage when fail-closed escalation trips.

        In capital_mode=True: HARD (stops new intents AND cancels open orders
        — combined with cancel_open_orders=True in the engage() call this is
        the "stop everything and clean up" path for real-capital audit
        breakage). In paper mode: SOFT (stop new intents only, no venue
        actions). The import is deferred so audit/writer.py does not create
        an import cycle with kill/state.py at module load.
        """
        from ..kill.state import KillMode

        return KillMode.HARD if self._capital_mode else KillMode.SOFT

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

    def write_direct(self, event: Event) -> None:
        """Emergency synchronous write — bypasses the event bus.

        Phase 4.7 Review 3 Issue 2: used by Orchestrator._emit_crash as
        a fallback when bus.publish() itself raises (bus stopping, queue
        full). This is best-effort: caller must wrap in try/except; a
        failure here means we've lost observability of one crash event
        but haven't lost the intent state.
        """
        if self._conn is None:
            raise RuntimeError("audit writer not started")
        self._rotate_if_needed()
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
        _insert_row(self._conn, row)

    async def on_event(self, event: Event) -> None:
        """Adapter for EventBus.subscribe(on_event=...).

        Phase 4.9 Item 1: fail-closed on persistent write errors.
        Tracks consecutive failures; at threshold, engages the kill switch
        (panic) and emits a stderr + Telegram alert bypassing audit.
        """
        try:
            await self.write(event)
        except Exception as exc:
            self._consecutive_write_failures += 1
            log.warning(
                "audit.write.crash",
                event_type=event.event_type.value,
                event_id=event.event_id,
                error=str(exc),
                consecutive_failures=self._consecutive_write_failures,
                threshold=self._fail_threshold,
            )
            if (
                self._consecutive_write_failures >= self._fail_threshold
                and not self._audit_kill_engaged
            ):
                self._audit_kill_engaged = True
                await self._trigger_audit_kill(exc)
            return
        # Success path — reset the counter.
        if self._consecutive_write_failures != 0:
            log.info(
                "audit.write.recovered",
                after_failures=self._consecutive_write_failures,
            )
        self._consecutive_write_failures = 0

    async def _trigger_audit_kill(self, last_exc: BaseException) -> None:
        """Engage kill switch + emit stderr ERROR + direct Telegram alert.
        Called exactly once per process lifetime (guarded by
        _audit_kill_engaged). Never raises."""
        reason = "audit_write_failed_threshold_exceeded"
        msg = (
            f"AUDIT_WRITE_FAILED: {self._consecutive_write_failures} consecutive "
            f"audit write failures (threshold {self._fail_threshold}); "
            f"last error: {last_exc!r}. Kill switch engaging (panic)."
        )
        # stderr first — guaranteed observable via systemd journal even if
        # logging and telegram both fail.
        try:
            print(f"ERROR [executor.audit] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass
        log.error("audit.kill.triggered", reason=reason, msg=msg)
        # Engage the kill switch (panic=True -> manual_only pinned).
        # Phase 4.13.1 Fix #B: in capital_mode, escalate to HARD and cancel
        # open orders. Paper mode retains SOFT-only (no cancels) because
        # observation windows must not disturb venue state on synthetic
        # failures. The mode decision goes through self._audit_kill_mode()
        # so KillMode import stays deferred.
        mode = self._audit_kill_mode()
        cancel_open = bool(self._capital_mode)
        if self._kill_manager is not None:
            try:
                await self._kill_manager.engage(
                    mode,
                    reason,
                    source="audit",
                    panic=True,
                    cancel_open_orders=cancel_open,
                )
            except Exception as exc:
                log.error("audit.kill.engage_failed", error=str(exc))
        else:
            log.error("audit.kill.no_kill_manager")
        # Direct Telegram alert — bypasses the bus/audit entirely.
        await asyncio.to_thread(_send_telegram_alert_direct, msg)

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
        # Phase 4.7 Q7: default to synchronous=FULL for durability. FULL
        # fsyncs after every commit (~2-3x slower) but never loses a
        # committed event under SIGKILL/OOM/power loss — important for
        # the audit log's "keep forever" contract. Tests can set
        # EXECUTOR_AUDIT_DURABILITY=NORMAL to regain speed.
        durability = os.environ.get("EXECUTOR_AUDIT_DURABILITY", "FULL").upper()
        if durability not in ("NORMAL", "FULL"):
            durability = "FULL"
        conn.execute(f"PRAGMA synchronous={durability}")
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


def _send_telegram_alert_direct(text: str) -> None:
    """Synchronous, best-effort Telegram send that does NOT touch the bus
    or audit. Used when audit itself is the failure — everything normal is
    broken so we do the most primitive thing possible: stdlib urllib POST
    with a short timeout. Env vars TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    must be set or we no-op."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        # stderr fallback — do not re-enter log.error path (structlog may
        # itself rely on channels that are unhealthy right now).
        try:
            print(
                f"ERROR [executor.audit] telegram alert send failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass


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
