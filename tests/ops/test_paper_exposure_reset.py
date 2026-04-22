"""Tests for scripts/paper_exposure_reset.py — Phase 4.10 (4.9.1-b)."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "paper_exposure_reset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("paper_exposure_reset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_exposure_reset"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE exposures (
            venue TEXT NOT NULL,
            market_id TEXT NOT NULL,
            outcome_id TEXT NOT NULL,
            dollars TEXT NOT NULL,
            event_id TEXT,
            updated_ts_ns INTEGER NOT NULL,
            PRIMARY KEY (venue, market_id, outcome_id)
        );
        CREATE TABLE strategy_exposures (
            strategy_id TEXT PRIMARY KEY,
            dollars TEXT NOT NULL,
            updated_ts_ns INTEGER NOT NULL
        );
        CREATE TABLE daily_pnl (
            date TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            pnl TEXT NOT NULL,
            PRIMARY KEY (date, strategy_id)
        );
        INSERT INTO exposures VALUES ('kalshi', 'MKT1', 'YES', '123.45', NULL, 0);
        INSERT INTO exposures VALUES ('kalshi', 'MKT2', 'YES', '50.00', NULL, 0);
        INSERT INTO exposures VALUES ('kalshi', 'MKT3', 'NO',  '25.00', NULL, 0);
        INSERT INTO strategy_exposures VALUES ('yes_no_cross', '100.00', 0);
        INSERT INTO daily_pnl VALUES ('2026-04-22', 'yes_no_cross', '12.34');
        """
    )
    conn.commit()
    return conn


def test_live_mode_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With PAPER_MODE unset, the reset must be a silent no-op."""
    mod = _load_module()
    db = tmp_path / "risk.sqlite"
    conn = _make_db(db)
    conn.close()
    monkeypatch.delenv("PAPER_MODE", raising=False)

    rc = mod.reset(db, tmp_path / "reset.log", dry_run=False)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]
    finally:
        conn.close()
    assert rows == 3, "live mode must not touch exposures"


def test_paper_mode_resets_exposures_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    db = tmp_path / "risk.sqlite"
    conn = _make_db(db)
    conn.close()
    monkeypatch.setenv("PAPER_MODE", "true")

    rc = mod.reset(db, tmp_path / "reset.log", dry_run=False)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        exposures = conn.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]
        strategy = conn.execute(
            "SELECT COUNT(*) FROM strategy_exposures"
        ).fetchone()[0]
        pnl = conn.execute("SELECT COUNT(*) FROM daily_pnl").fetchone()[0]
    finally:
        conn.close()
    assert exposures == 0, "paper mode must clear exposures"
    assert strategy == 1, "strategy_exposures must be untouched (Build Zero 0c)"
    assert pnl == 1, "daily_pnl must be untouched"


def test_dry_run_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    db = tmp_path / "risk.sqlite"
    conn = _make_db(db)
    conn.close()
    monkeypatch.setenv("PAPER_MODE", "true")

    rc = mod.reset(db, tmp_path / "reset.log", dry_run=True)
    assert rc == 0

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]
    finally:
        conn.close()
    assert rows == 3, "dry-run must not modify rows"


def test_missing_db_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setenv("PAPER_MODE", "true")
    rc = mod.reset(tmp_path / "does_not_exist.sqlite", tmp_path / "reset.log", dry_run=False)
    assert rc == 0
