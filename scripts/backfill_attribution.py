#!/usr/bin/env python3
"""
One-shot: backfill cost_basis_dollars + venue_fee_bps for pre-migration rows.

Problem: Phase 4.8 added cost_basis_dollars + venue_fee_bps columns. Rows
inserted before the migration (2026-04-21) have NULL in both.

For each NULL row:
    cost_basis_dollars =
        size * fill_price + fee    (BUY)
        size * fill_price - fee    (SELL)
    venue_fee_bps =
        fee / (size * fill_price) * 10000   (when denominator > 0)

Idempotent (only touches NULL rows). Safe to re-run.

Flags:
    --dry-run   Show what would change, write nothing.
    --db PATH   Override default attribution.sqlite location.

Phase 4.10 (Item 5).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_DB = Path("/root/executor/state/attribution.sqlite")
DEFAULT_LOG = Path("/var/log/attribution-backfill.log")


def _log(msg: str, log_path: Path) -> None:
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(line)
    except OSError:
        pass


def _dec(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None


def backfill(db_path: Path, log_path: Path, dry_run: bool) -> int:
    if not db_path.exists():
        _log(f"db_missing={db_path}", log_path)
        return 2

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT fill_id, side, size, fill_price, fee, "
            "cost_basis_dollars, venue_fee_bps FROM attribution "
            "WHERE cost_basis_dollars IS NULL OR venue_fee_bps IS NULL"
        ).fetchall()

        n_seen = len(rows)
        n_updated = 0
        n_skipped_bad = 0

        updates: list[tuple[str | None, str | None, str]] = []
        for fill_id, side, size, fill_price, fee, cb_existing, fee_bps_existing in rows:
            size_d = _dec(size)
            price_d = _dec(fill_price)
            fee_d = _dec(fee) if fee is not None else Decimal("0")
            if size_d is None or price_d is None or fee_d is None:
                n_skipped_bad += 1
                continue

            notional = size_d * price_d

            new_cb: str | None = cb_existing
            if cb_existing is None:
                if side == "BUY":
                    new_cb = str(notional + fee_d)
                elif side == "SELL":
                    new_cb = str(notional - fee_d)
                else:
                    # Unknown side — leave NULL, log.
                    n_skipped_bad += 1
                    continue

            new_fee_bps: str | None = fee_bps_existing
            if fee_bps_existing is None and notional > 0 and fee is not None:
                new_fee_bps = str((fee_d / notional) * Decimal("10000"))
            # If notional is 0 or fee is NULL, leave fee_bps NULL — those
            # rows genuinely can't be computed.

            updates.append((new_cb, new_fee_bps, fill_id))

        if dry_run:
            _log(
                f"dry_run null_rows={n_seen} would_update={len(updates)} "
                f"skipped_bad={n_skipped_bad}",
                log_path,
            )
            return 0

        with conn:
            for new_cb, new_fee_bps, fill_id in updates:
                conn.execute(
                    "UPDATE attribution "
                    "SET cost_basis_dollars = COALESCE(cost_basis_dollars, ?), "
                    "    venue_fee_bps      = COALESCE(venue_fee_bps, ?) "
                    "WHERE fill_id = ?",
                    (new_cb, new_fee_bps, fill_id),
                )
                n_updated += 1

        _log(
            f"backfilled null_rows={n_seen} updated={n_updated} "
            f"skipped_bad={n_skipped_bad} db={db_path}",
            log_path,
        )
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill attribution cost_basis_dollars + venue_fee_bps.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args(argv)
    return backfill(args.db, args.log, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
