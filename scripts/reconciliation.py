#!/usr/bin/env python3
"""Daily reconciliation of attribution fills.

Checks internal consistency of the attribution DB for the prior 24-hour
UTC window and optionally sends a Telegram alert on mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

DEFAULT_DB = "/root/executor/state/attribution.sqlite"
DELTA_ABS_THRESHOLD = 10.0   # $10 absolute
DELTA_PCT_THRESHOLD = 0.01   # 1 % of notional
CONSISTENCY_THRESHOLD = 0.01 # $0.01 per-fill recomputation tolerance


def _window_ns() -> tuple[int, int]:
    """Return (start_ns, end_ns) for the prior 24-hour UTC window.

    When run at 00:05 by cron this effectively covers the previous calendar
    day.  Using a rolling 24-hour lookback avoids missing fills that land
    right around midnight.
    """
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=1)
    return int(start.timestamp() * 1e9), int(now_utc.timestamp() * 1e9)


def _send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(json.dumps({"warn": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping alert"}),
              file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"telegram send failed: {exc}"}), file=sys.stderr)


def reconcile(db_path: str, dry_run: bool = False) -> dict:
    """Run reconciliation and return a result dict."""
    start_ns, end_ns = _window_ns()

    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = cur.execute(
        "SELECT * FROM attribution WHERE fill_ts_ns >= ? AND fill_ts_ns < ?",
        (start_ns, end_ns),
    ).fetchall()
    con.close()

    if not rows:
        result = {
            "status": "ok",
            "message": "no fills in window — nothing to reconcile",
            "window_start": start_ns,
            "window_end": end_ns,
            "fill_count": 0,
        }
        print(json.dumps(result))
        return result

    # ---- aggregate by venue ----
    venues: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        venues.setdefault(r["venue"], []).append(r)

    total_notional = 0.0
    total_fees = 0.0
    total_fills = len(rows)
    consistency_errors: list[str] = []
    venue_summaries: list[dict] = []

    for venue, fills in venues.items():
        venue_notional = 0.0
        venue_fees = 0.0
        venue_net_flow = 0.0

        for f in fills:
            size = float(f["size"])
            fill_price = float(f["fill_price"])
            fee = float(f["fee"]) if f["fee"] is not None else 0.0
            try:
                cost_basis = float(f["cost_basis_dollars"]) if f["cost_basis_dollars"] is not None else 0.0
            except (IndexError, KeyError):
                # Pre-migration row without cost_basis_dollars column
                cost_basis = (notional + fee) if side == "BUY" else (notional - fee)
            side = f["side"]

            notional = size * fill_price
            venue_notional += notional
            venue_fees += fee

            # Recompute expected cost_basis: positive for BUY, negative for SELL
            if side == "BUY":
                expected = notional + fee
                venue_net_flow += cost_basis
            else:
                expected = -(notional - fee)
                venue_net_flow -= abs(cost_basis)

            delta = abs(cost_basis - expected)
            if delta > CONSISTENCY_THRESHOLD:
                consistency_errors.append(
                    f"fill {f['fill_id']}: cost_basis={cost_basis:.4f} vs expected={expected:.4f} (delta={delta:.4f})"
                )

        total_notional += venue_notional
        total_fees += venue_fees
        venue_summaries.append({
            "venue": venue,
            "fills": len(fills),
            "notional": round(venue_notional, 4),
            "fees": round(venue_fees, 4),
            "net_flow": round(venue_net_flow, 4),
        })

    # ---- decide outcome ----
    has_mismatch = len(consistency_errors) > 0

    result = {
        "status": "mismatch" if has_mismatch else "ok",
        "window_start": start_ns,
        "window_end": end_ns,
        "fill_count": total_fills,
        "total_notional": round(total_notional, 4),
        "total_fees": round(total_fees, 4),
        "venues": venue_summaries,
        "consistency_errors": consistency_errors,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    if has_mismatch:
        result["message"] = "internal consistency mismatch detected"
    else:
        result["message"] = "all fills match — reconciliation ok"

    print(json.dumps(result))

    # ---- alert logic ----
    if has_mismatch and not dry_run:
        threshold = max(DELTA_ABS_THRESHOLD, DELTA_PCT_THRESHOLD * total_notional)
        # Sum of absolute deltas
        total_delta = sum(
            abs(_get_cost_basis(r) - _expected_cost(r))
            for r in rows
        )
        if total_delta > threshold:
            lines = [
                "<b>Reconciliation Alert</b>",
                f"Fills: {total_fills}  |  Notional: ${total_notional:,.2f}",
                f"Total delta: ${total_delta:,.4f}  (threshold ${threshold:,.2f})",
                "",
                *consistency_errors[:10],
            ]
            _send_telegram("\n".join(lines))

    return result


def _get_cost_basis(row: sqlite3.Row) -> float:
    try:
        val = row["cost_basis_dollars"]
        return float(val) if val is not None else 0.0
    except (IndexError, KeyError):
        # Pre-migration: recompute from raw fields
        return _expected_cost(row)


def _expected_cost(row: sqlite3.Row) -> float:
    size = float(row["size"])
    fill_price = float(row["fill_price"])
    fee = float(row["fee"]) if row["fee"] is not None else 0.0
    if row["side"] == "BUY":
        return size * fill_price + fee
    return -(size * fill_price - fee)


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily attribution reconciliation")
    parser.add_argument("--dry-run", action="store_true", help="Compute only, no Telegram alert")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to attribution.sqlite")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(json.dumps({"error": f"db not found: {db}"}), file=sys.stderr)
        return 1

    try:
        result = reconcile(str(db), dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    return 1 if result.get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
