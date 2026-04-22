#!/usr/bin/env python3
"""
Paper-mode exposure daily reset.

Problem this solves (4.22 morning diagnostic):
In paper mode running against synthetic random-walk quotes, market_exposure
accumulates forever. Over 24+ hours, every intent rejects at the exposure
gate because remaining capacity is below 1 contract cost. Strategy runs at
full emission rate, 0% admit rate, useless paper data.

Scope:
- Zeros the `exposures` table in risk_state.sqlite.
- Does NOT touch `strategy_exposures` (Build Zero 0c failure budget).
- Does NOT touch `daily_pnl` (those are separate accounting).
- Gated strictly by PAPER_MODE=true env var. In any other mode this is a
  silent no-op (exit 0). When Phase 5 turns on real capital this script
  must be removed or gated by a stricter check.

Cron: /etc/cron.d/executor-paper-reset runs at 00:00 UTC daily, BEFORE
reconciliation at 00:05.

Flags:
    --dry-run      Show what would happen, don't write.
    --db PATH      Override default risk_state.sqlite location.
    --log PATH     Override /var/log/executor-paper-reset.log.

Phase 4.10 (4.9.1-b).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sqlite3
import sys
from pathlib import Path


DEFAULT_DB = Path("/root/executor/state/risk_state.sqlite")
DEFAULT_LOG = Path("/var/log/executor-paper-reset.log")


def _log(msg: str, log_path: Path) -> None:
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}\n"
    # stdout for cron/systemd journal.
    sys.stdout.write(line)
    sys.stdout.flush()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(line)
    except OSError:
        # Logging shouldn't fail the reset.
        pass


def _paper_mode_active() -> bool:
    return os.environ.get("PAPER_MODE", "").lower() in ("1", "true", "yes", "on")


def _count_exposures(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM exposures").fetchone()
    return int(row[0]) if row else 0


def reset(db_path: Path, log_path: Path, dry_run: bool) -> int:
    """Returns process exit code."""
    if not _paper_mode_active():
        # Silent no-op in live mode — guards against accidental live wipe.
        _log("paper_mode=false; skipping (exit 0)", log_path)
        return 0

    if not db_path.exists():
        _log(f"db_missing={db_path}; nothing to do", log_path)
        return 0

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.DatabaseError as exc:
        _log(f"db_open_failed={exc}", log_path)
        return 2

    try:
        pre = _count_exposures(conn)
        if dry_run:
            _log(f"dry_run exposures_rows={pre} would_delete={pre}", log_path)
            return 0
        conn.execute("DELETE FROM exposures")
        conn.commit()
        post = _count_exposures(conn)
        _log(
            f"reset exposures_rows_before={pre} exposures_rows_after={post} "
            f"strategy_exposures_untouched=true db={db_path}",
            log_path,
        )
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper-mode exposure daily reset.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args(argv)
    return reset(args.db, args.log, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
