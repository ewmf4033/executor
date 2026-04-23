"""
KillState — persistent kill-switch state.

Modes:
  NONE  — normal operation
  SOFT  — stop accepting new intents; in-flight orders complete
  HARD  — SOFT + cancel all open orders on every venue (no auto-flatten)

Auto-resume circuit breaker (per spec):
  - Operator may re-enable trading after a HARD via `resume()`.
  - We track sequential auto-resume attempts that did NOT see a healthy
    60s window of normal operation. After 3 failed strikes we pin to
    `manual_only=True`; only an explicit operator action with
    `force=True` clears it.

Panic mode:
  - `/kill panic` engages HARD with `panic=True`. While panic is engaged,
    `/kill resume` is rejected for `panic_cooldown_sec` seconds.

Persistence:
  Single-row table in risk_state.sqlite (kill_state table). Writer is
  blocking; the kill path is rare and must be durable.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


log = logging.getLogger("executor.kill.state")


class KillMode(str, Enum):
    NONE = "NONE"
    SOFT = "SOFT"
    HARD = "HARD"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kill_switch_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    mode TEXT NOT NULL,
    reason TEXT NOT NULL,
    engaged_ts_ns INTEGER NOT NULL,
    panic INTEGER NOT NULL DEFAULT 0,
    panic_until_ns INTEGER NOT NULL DEFAULT 0,
    resume_strikes INTEGER NOT NULL DEFAULT 0,
    last_resume_ts_ns INTEGER NOT NULL DEFAULT 0,
    manual_only INTEGER NOT NULL DEFAULT 0,
    extra_json TEXT NOT NULL DEFAULT '{}'
);
"""


@dataclass
class KillStateSnapshot:
    mode: KillMode = KillMode.NONE
    reason: str = ""
    engaged_ts_ns: int = 0
    panic: bool = False
    panic_until_ns: int = 0
    resume_strikes: int = 0
    last_resume_ts_ns: int = 0
    manual_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "engaged_ts_ns": self.engaged_ts_ns,
            "panic": self.panic,
            "panic_until_ns": self.panic_until_ns,
            "resume_strikes": self.resume_strikes,
            "last_resume_ts_ns": self.last_resume_ts_ns,
            "manual_only": self.manual_only,
            "extra": dict(self.extra),
        }


FORCE_RESET_ENV_VAR = "EXECUTOR_FORCE_RESET_KILL_STATE"


class KillStateStore:
    """SQLite-backed single-row kill state, rebuilt on process restart.

    Phase 4.11 (Review 8 finding 0c-5) introduced corruption recovery: the
    backing SQLite file is renamed aside with a nanosecond-precision suffix
    and a fresh DB is opened at the original path.

    Phase 4.13 (GPT-5.5 architectural review, Finding #1) changed the
    rebuild's *initial state*. Previously the rebuilt DB started in
    mode=NONE, which was labeled "fail-safe" but is in fact fail-OPEN for a
    capital-trading daemon: a process that cannot verify its prior kill
    state must not resume trading un-killed. The rebuilt DB now seeds a
    HARD + ``manual_only=True`` record with ``reason="KILL_DB_CORRUPT_REBUILT"``
    so trading stays stopped until an operator inspects the backup and
    explicitly resolves via KillManager.

    Operator bypass: setting env var ``EXECUTOR_FORCE_RESET_KILL_STATE=1``
    before process startup causes the corruption rebuild to seed a NONE
    record instead. The store records ``force_reset_used=True`` so the
    startup path can emit a ``KILL_STATE_FORCE_RESET`` audit event.
    Legitimate only after investigating what caused corruption.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rebuilt_from_corruption: bool = False
        self.force_reset_used: bool = False
        self._corruption_backup_path: Path | None = None
        self._corruption_ts_ns: int = 0
        self._conn = self._open_or_rebuild()
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        if self.rebuilt_from_corruption:
            self._seed_post_corruption_state()

    def _open_or_rebuild(self) -> sqlite3.Connection:
        """Try to open; on sqlite3.DatabaseError, back up the corrupt file
        with a ns-timestamp suffix and open a fresh DB at the same path."""
        try:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            # Force SQLite to validate the header/page-1 so corruption
            # surfaces at open time rather than first query.
            conn.execute("PRAGMA schema_version").fetchone()
            return conn
        except sqlite3.DatabaseError as exc:
            return self._handle_corruption(exc)

    def _handle_corruption(self, exc: Exception) -> sqlite3.Connection:
        # Use nanosecond epoch so rapid successive corruptions don't
        # collide on the backup filename.
        ns_ts = time.time_ns()
        backup_path = self._db_path.with_name(
            f"{self._db_path.name}.corrupt-{ns_ts}"
        )
        # Tight collision guard: if by chance the filename already exists
        # (same-nanosecond call on some OS timers), bump until unique.
        while backup_path.exists():
            ns_ts += 1
            backup_path = self._db_path.with_name(
                f"{self._db_path.name}.corrupt-{ns_ts}"
            )
        try:
            if self._db_path.exists():
                self._db_path.rename(backup_path)
                self._corruption_backup_path = backup_path
        except OSError as rename_exc:
            log.error(
                "kill.state.corrupt_rename_failed path=%s error=%s original_error=%s",
                str(self._db_path),
                str(rename_exc),
                str(exc),
            )
            # Best effort: drop the file so a fresh open succeeds.
            try:
                self._db_path.unlink()
            except OSError:
                pass
        self.rebuilt_from_corruption = True
        self._corruption_ts_ns = ns_ts
        log.error(
            "kill.state.corrupt_rebuilt path=%s backup=%s original_error=%s",
            str(self._db_path),
            str(backup_path),
            str(exc),
        )
        return sqlite3.connect(str(self._db_path), check_same_thread=False)

    def _seed_post_corruption_state(self) -> None:
        """Seed the freshly-rebuilt DB with a fail-closed initial state.

        Without operator bypass: HARD + manual_only so the daemon starts
        stopped until explicit resolution. With bypass env var set: NONE,
        and ``force_reset_used`` is flipped so the startup path can emit
        a ``KILL_STATE_FORCE_RESET`` audit event.
        """
        force_reset = os.environ.get(FORCE_RESET_ENV_VAR, "") == "1"
        if force_reset:
            self.force_reset_used = True
            snap = KillStateSnapshot(
                mode=KillMode.NONE,
                reason="KILL_DB_CORRUPT_REBUILT_FORCE_RESET",
                engaged_ts_ns=self._corruption_ts_ns,
                manual_only=False,
                extra={
                    "rebuilt_from_corruption": True,
                    "force_reset_used": True,
                    "backup_path": str(self._corruption_backup_path)
                    if self._corruption_backup_path
                    else None,
                },
            )
            log.error(
                "kill.state.force_reset_honored path=%s backup=%s",
                str(self._db_path),
                str(self._corruption_backup_path),
            )
        else:
            snap = KillStateSnapshot(
                mode=KillMode.HARD,
                reason="KILL_DB_CORRUPT_REBUILT",
                engaged_ts_ns=self._corruption_ts_ns,
                manual_only=True,
                extra={
                    "rebuilt_from_corruption": True,
                    "backup_path": str(self._corruption_backup_path)
                    if self._corruption_backup_path
                    else None,
                },
            )
            log.error(
                "kill.state.rebuilt_fail_closed path=%s backup=%s note=%s",
                str(self._db_path),
                str(self._corruption_backup_path),
                "HARD+manual_only; set EXECUTOR_FORCE_RESET_KILL_STATE=1 to override",
            )
        self.save(snap)

    def load(self) -> KillStateSnapshot:
        row = self._conn.execute(
            "SELECT mode, reason, engaged_ts_ns, panic, panic_until_ns, "
            "resume_strikes, last_resume_ts_ns, manual_only, extra_json "
            "FROM kill_switch_state WHERE id=1"
        ).fetchone()
        if row is None:
            return KillStateSnapshot()
        try:
            extra = json.loads(row[8] or "{}")
        except json.JSONDecodeError:
            extra = {}
        return KillStateSnapshot(
            mode=KillMode(row[0]),
            reason=row[1],
            engaged_ts_ns=int(row[2]),
            panic=bool(row[3]),
            panic_until_ns=int(row[4]),
            resume_strikes=int(row[5]),
            last_resume_ts_ns=int(row[6]),
            manual_only=bool(row[7]),
            extra=extra,
        )

    def save(self, snap: KillStateSnapshot) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kill_switch_state "
            "(id, mode, reason, engaged_ts_ns, panic, panic_until_ns, "
            " resume_strikes, last_resume_ts_ns, manual_only, extra_json) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snap.mode.value,
                snap.reason,
                int(snap.engaged_ts_ns),
                1 if snap.panic else 0,
                int(snap.panic_until_ns),
                int(snap.resume_strikes),
                int(snap.last_resume_ts_ns),
                1 if snap.manual_only else 0,
                json.dumps(snap.extra, default=str),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()
