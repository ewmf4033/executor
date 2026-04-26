#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("data/manual_executions/executions.csv")
FIELDS = [
    "timestamp_utc",
    "system",
    "market_ticker",
    "side",
    "entry_price",
    "size_dollars",
    "alert_edge",
    "reason",
    "notes",
]

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log a manually executed Tier 1 experiment trade. Does not place orders."
    )
    parser.add_argument("--system", required=True, help="executor, sharpedge, masta, manual, etc.")
    parser.add_argument("--ticker", required=True, help="Market ticker or market identifier")
    parser.add_argument("--side", required=True, choices=["buy", "sell", "yes", "no"])
    parser.add_argument("--entry-price", required=True, type=float, help="Decimal price, e.g. 0.42")
    parser.add_argument("--size-dollars", required=True, type=float)
    parser.add_argument("--alert-edge", default="", help="Edge from alert/model, e.g. 0.035 or 3.5%")
    parser.add_argument("--reason", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if not (0 < args.entry_price < 1):
        raise SystemExit("entry-price must be a decimal between 0 and 1, e.g. 0.42")
    if args.size_dollars <= 0:
        raise SystemExit("size-dollars must be > 0")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": args.system,
        "market_ticker": args.ticker,
        "side": args.side,
        "entry_price": args.entry_price,
        "size_dollars": args.size_dollars,
        "alert_edge": args.alert_edge,
        "reason": args.reason,
        "notes": args.notes,
    }

    write_header = not OUT.exists() or OUT.stat().st_size == 0
    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"logged manual execution -> {OUT}")
    for k in FIELDS:
        print(f"{k}: {row[k]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
