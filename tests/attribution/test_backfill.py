"""Tests for scripts/backfill_attribution.py — Phase 4.10 (Item 5)."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_attribution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_attribution", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backfill_attribution"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE attribution (
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
    )
    return conn


def _insert(conn: sqlite3.Connection, *, fill_id: str, side: str, size: str,
            fill_price: str, fee: str | None, cb: str | None = None,
            fee_bps: str | None = None) -> None:
    conn.execute(
        "INSERT INTO attribution VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fill_id, "i-1", "l-1", "strat", "kalshi", "MKT1", side, size,
            None, None, None, fill_price, None, None, None, None, fee,
            0, 0, "{}", cb, fee_bps,
        ),
    )


def test_buy_backfill_adds_fee_to_notional(tmp_path: Path) -> None:
    mod = _load_module()
    db = tmp_path / "attr.sqlite"
    conn = _make_db(db)
    _insert(conn, fill_id="b1", side="BUY", size="100", fill_price="0.65", fee="0.50")
    conn.commit()
    conn.close()

    rc = mod.backfill(db, tmp_path / "log", dry_run=False)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT cost_basis_dollars, venue_fee_bps FROM attribution WHERE fill_id='b1'"
        ).fetchone()
    finally:
        conn.close()
    assert Decimal(row[0]) == Decimal("65.50")  # 100 * 0.65 + 0.50
    # 0.50 / 65 * 10000 = 76.923...
    assert Decimal(row[1]).quantize(Decimal("0.01")) == Decimal("76.92")


def test_sell_backfill_subtracts_fee(tmp_path: Path) -> None:
    mod = _load_module()
    db = tmp_path / "attr.sqlite"
    conn = _make_db(db)
    _insert(conn, fill_id="s1", side="SELL", size="100", fill_price="0.65", fee="0.50")
    conn.commit()
    conn.close()

    rc = mod.backfill(db, tmp_path / "log", dry_run=False)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT cost_basis_dollars FROM attribution WHERE fill_id='s1'"
        ).fetchone()
    finally:
        conn.close()
    assert Decimal(row[0]) == Decimal("64.50")  # 100 * 0.65 - 0.50


def test_dry_run_modifies_nothing(tmp_path: Path) -> None:
    mod = _load_module()
    db = tmp_path / "attr.sqlite"
    conn = _make_db(db)
    _insert(conn, fill_id="x1", side="BUY", size="1", fill_price="0.5", fee="0.01")
    conn.commit()
    conn.close()

    rc = mod.backfill(db, tmp_path / "log", dry_run=True)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT cost_basis_dollars, venue_fee_bps FROM attribution WHERE fill_id='x1'"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] is None
    assert row[1] is None


def test_idempotent_skips_already_populated(tmp_path: Path) -> None:
    mod = _load_module()
    db = tmp_path / "attr.sqlite"
    conn = _make_db(db)
    _insert(conn, fill_id="done", side="BUY", size="1", fill_price="0.5",
            fee="0.01", cb="0.51", fee_bps="200")
    _insert(conn, fill_id="todo", side="BUY", size="1", fill_price="0.5", fee="0.01")
    conn.commit()
    conn.close()

    rc = mod.backfill(db, tmp_path / "log", dry_run=False)
    assert rc == 0
    # Second run must be a no-op on the already-populated row.
    rc = mod.backfill(db, tmp_path / "log", dry_run=False)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        done = conn.execute(
            "SELECT cost_basis_dollars FROM attribution WHERE fill_id='done'"
        ).fetchone()
        todo = conn.execute(
            "SELECT cost_basis_dollars FROM attribution WHERE fill_id='todo'"
        ).fetchone()
    finally:
        conn.close()
    assert done[0] == "0.51", "must not overwrite pre-existing cost_basis"
    assert Decimal(todo[0]) == Decimal("0.51")


def test_null_fee_leaves_fee_bps_null_but_computes_cost_basis(tmp_path: Path) -> None:
    """Pre-migration rows without fee are still partially salvageable."""
    mod = _load_module()
    db = tmp_path / "attr.sqlite"
    conn = _make_db(db)
    _insert(conn, fill_id="nofee", side="BUY", size="10", fill_price="0.4", fee=None)
    conn.commit()
    conn.close()

    rc = mod.backfill(db, tmp_path / "log", dry_run=False)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT cost_basis_dollars, venue_fee_bps FROM attribution WHERE fill_id='nofee'"
        ).fetchone()
    finally:
        conn.close()
    assert Decimal(row[0]) == Decimal("4.0")  # 10 * 0.4 + 0
    assert row[1] is None  # fee missing — cannot compute bps
