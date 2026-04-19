"""
Phase 1 smoke: emit a fake intent and verify the correct event sequence
lands in the audit log.

Expected sequence in a fresh run:
    EXECUTOR_LIFECYCLE (state=STARTED)
    INTENT_EMITTED
    EXECUTOR_LIFECYCLE (state=STOPPING)

Run:
    /root/executor/.venv/bin/python /root/executor/scripts/smoke_phase1.py
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path

# Ensure repo import works when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executor.core.intent import Intent
from executor.core.service import ExecutorService
from executor.core.types import Side
from executor.strategies.base import Strategy


class FakeStrategy(Strategy):
    strategy_id = "smoke-fake"
    required_capabilities = {"kalshi": frozenset({"supports_limit"})}
    kelly_cap = Decimal("0.25")

    async def run(self) -> None:  # unused in smoke
        return None


async def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        svc = ExecutorService(audit_dir=td)
        await svc.start()

        strat = FakeStrategy()
        strat.attach(svc.bus.publish)

        now = time.time_ns()
        intent = Intent.single(
            strategy_id=strat.strategy_id,
            venue="kalshi",
            market_id="SMOKE-MARKET-1",
            outcome_id="YES",
            side=Side.BUY,
            target_exposure=10,
            price_limit=0.55,
            confidence=0.6,
            edge_estimate=0.03,
            time_horizon_sec=120,
            created_ts=now,
            expires_ts=now + 60_000_000_000,
        )
        await strat.emit(intent)

        # Let the bus pump + audit writer flush.
        await asyncio.sleep(0.1)

        await svc.stop()

        # Verify sequence.
        db_path = sorted(Path(td).glob("audit-*.sqlite"))[-1]
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, strategy_id, intent_id FROM events ORDER BY ts_ns ASC"
        ).fetchall()
        conn.close()

        print(f"audit db: {db_path}")
        for r in rows:
            print(f"  {r[0]:<22} strategy={r[1]} intent={r[2]}")

        types = [r[0] for r in rows]
        expected = [
            "EXECUTOR_LIFECYCLE",   # STARTED
            "INTENT_EMITTED",
            "EXECUTOR_LIFECYCLE",   # STOPPING
        ]
        ok = types == expected
        if ok:
            print("OK: event sequence matches expected.")
            return 0
        else:
            print(f"FAIL: expected {expected}, got {types}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
