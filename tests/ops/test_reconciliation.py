"""Tests for scripts/reconciliation.py daily reconciliation."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parents[2] / "scripts" / "reconciliation.py")
PYTHON = sys.executable

SCHEMA = """\
CREATE TABLE IF NOT EXISTS attribution (
    fill_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    leg_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    size TEXT NOT NULL,
    intent_price TEXT,
    decision_price TEXT,
    arrival_price TEXT,
    fill_price TEXT NOT NULL,
    exit_price TEXT,
    strategy_edge TEXT,
    execution_cost TEXT,
    short_term_alpha TEXT,
    fee TEXT,
    fill_ts_ns INTEGER NOT NULL,
    settled_ts_ns INTEGER NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}',
    cost_basis_dollars TEXT,
    venue_fee_bps TEXT
);
"""


def _recent_ts_ns() -> int:
    """Timestamp ~1 hour ago in nanoseconds."""
    return int(time.time() * 1e9) - 3_600_000_000_000


def _create_db(path: Path, rows: list[tuple] | None = None) -> Path:
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    if rows:
        con.executemany(
            "INSERT INTO attribution "
            "(fill_id, intent_id, leg_id, strategy_id, venue, market_id, "
            "side, size, fill_price, fee, cost_basis_dollars, fill_ts_ns, settled_ts_ns) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
    con.close()
    return path


def _run(db_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, SCRIPT, "--dry-run", "--db", str(db_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestReconciliation:
    def test_reconciliation_match_no_alert(self, tmp_path: Path) -> None:
        """Consistent fills should produce status ok / match."""
        ts = _recent_ts_ns()
        # BUY 10 @ 1.00, fee 0.10 -> cost_basis = 10*1.00 + 0.10 = 10.10
        # BUY  5 @ 2.00, fee 0.05 -> cost_basis =  5*2.00 + 0.05 = 10.05
        # SELL 3 @ 3.00, fee 0.03 -> cost_basis = -(3*3.00 - 0.03) = -8.97
        rows = [
            ("f1", "i1", "l1", "s1", "hyperliquid", "BTC-PERP", "BUY",
             "10", "1.00", "0.10", "10.10", ts, ts),
            ("f2", "i2", "l2", "s1", "hyperliquid", "ETH-PERP", "BUY",
             "5", "2.00", "0.05", "10.05", ts, ts),
            ("f3", "i3", "l3", "s1", "hyperliquid", "BTC-PERP", "SELL",
             "3", "3.00", "0.03", "-8.97", ts, ts),
        ]
        db = _create_db(tmp_path / "attr.sqlite", rows)
        proc = _run(db)

        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        out = proc.stdout.lower()
        assert "ok" in out or "match" in out, f"unexpected output: {proc.stdout}"

    def test_reconciliation_mismatch_alerts(self, tmp_path: Path) -> None:
        """Deliberately wrong cost_basis should be flagged."""
        ts = _recent_ts_ns()
        # BUY 1 @ 5.00, fee 0.50 -> expected cost_basis = 5.50, but we set 999.99
        rows = [
            ("f1", "i1", "l1", "s1", "hyperliquid", "BTC-PERP", "BUY",
             "1", "5.00", "0.50", "999.99", ts, ts),
        ]
        db = _create_db(tmp_path / "attr.sqlite", rows)
        proc = _run(db)

        out = proc.stdout.lower()
        assert "mismatch" in out or "inconsistency" in out or "error" in out, \
            f"expected mismatch indicator in: {proc.stdout}"

    def test_reconciliation_empty_db(self, tmp_path: Path) -> None:
        """Empty DB with schema but no fills should exit 0."""
        db = _create_db(tmp_path / "attr.sqlite")
        proc = _run(db)

        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        out = proc.stdout.lower()
        assert "ok" in out or "nothing" in out or "no fills" in out, \
            f"unexpected output: {proc.stdout}"
