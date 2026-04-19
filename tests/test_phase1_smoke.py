"""
Phase-1 exit criterion test.

Start the service, attach a fake strategy, emit a single-leg intent, stop,
assert the audit log contains the expected event sequence:

    EXECUTOR_LIFECYCLE (STARTED)
    INTENT_EMITTED
    EXECUTOR_LIFECYCLE (STOPPING)
"""
from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.intent import Intent
from executor.core.service import ExecutorService
from executor.core.types import Side
from executor.strategies.base import Strategy


pytestmark = pytest.mark.asyncio


class FakeStrategy(Strategy):
    strategy_id = "phase1-fake"
    required_capabilities = {"kalshi": frozenset({"supports_limit"})}
    kelly_cap = Decimal("0.25")

    async def run(self) -> None:
        return None


async def test_fake_intent_produces_correct_event_sequence():
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
        await asyncio.sleep(0.1)
        await svc.stop()

        db_path = sorted(Path(td).glob("audit-*.sqlite"))[-1]
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, strategy_id, intent_id FROM events ORDER BY ts_ns ASC"
        ).fetchall()
        conn.close()

        types = [r[0] for r in rows]
        assert types == [
            "EXECUTOR_LIFECYCLE",   # STARTED
            "INTENT_EMITTED",
            "EXECUTOR_LIFECYCLE",   # STOPPING
        ]

        # INTENT_EMITTED row has strategy + intent ids populated.
        assert rows[1][1] == "phase1-fake"
        assert rows[1][2] == intent.intent_id
