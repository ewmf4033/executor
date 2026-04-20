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
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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


class KillStateStore:
    """SQLite-backed single-row kill state, rebuilt on process restart."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

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
