"""Tests for executor/analysis/edge_half_life.py — Phase 4.10 (0h scaffold)."""
from __future__ import annotations

import math
import random
import sqlite3
from pathlib import Path

import pytest

from executor.analysis.edge_half_life import (
    EdgeDecayTracker,
    HalfLifeResult,
    _linear_regression,
)


def _make_attr_db(path: Path) -> sqlite3.Connection:
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


def _seed_decay(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    half_life_hours: float,
    n: int,
    now_ns: int,
    predicted_edge: float = 0.05,
    noise: float = 0.0,
    seed: int = 1234,
) -> None:
    """Seed `n` fills with ages uniformly in [0, 2*half_life] hours.
    realized = predicted * exp(-ln(2)/half_life * age) + noise.
    """
    rng = random.Random(seed)
    lamb = math.log(2) / half_life_hours
    for i in range(n):
        age_h = rng.uniform(0.1, 2.0 * half_life_hours)
        fill_ts = now_ns - int(age_h * 3600 * 1_000_000_000)
        realized = predicted_edge * math.exp(-lamb * age_h)
        if noise:
            realized += rng.gauss(0.0, noise * predicted_edge)
        conn.execute(
            "INSERT INTO attribution VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"fill-{i}", "i-1", "l-1", strategy_id, "kalshi", "MKT1", "BUY",
                "100", None, None, None, "0.50", None,
                str(predicted_edge), None, str(realized), "0",
                fill_ts, fill_ts + 1, "{}", None, None,
            ),
        )
    conn.commit()


# ----------------------------------------------------------------------
# Linear regression sanity
# ----------------------------------------------------------------------

def test_linear_regression_recovers_slope() -> None:
    xs = [float(i) for i in range(10)]
    ys = [2.0 * x + 1.0 for x in xs]
    slope, intercept, r2 = _linear_regression(xs, ys)
    assert math.isclose(slope, 2.0, abs_tol=1e-9)
    assert math.isclose(intercept, 1.0, abs_tol=1e-9)
    assert math.isclose(r2, 1.0, abs_tol=1e-9)


# ----------------------------------------------------------------------
# Synthetic decay curves — known half-lives
# ----------------------------------------------------------------------

@pytest.mark.parametrize("hl_hours", [1.0, 6.0, 24.0, 7 * 24.0])
def test_half_life_matches_synthetic(tmp_path: Path, hl_hours: float) -> None:
    now_ns = 1_700_000_000 * 1_000_000_000
    db = tmp_path / "attr.sqlite"
    conn = _make_attr_db(db)
    _seed_decay(conn, strategy_id="s1", half_life_hours=hl_hours, n=200, now_ns=now_ns)
    conn.close()

    tracker = EdgeDecayTracker(db)
    # Window wide enough to cover 2*hl.
    window_days = max(14, int(hl_hours / 24 * 4) + 1)
    result = tracker.compute_half_life("s1", window_days=window_days, now_ns=now_ns)

    assert result.half_life_hours is not None, f"got reason={result.reason}"
    # No noise -> fit should recover half-life to ~0.1% precision.
    rel_err = abs(result.half_life_hours - hl_hours) / hl_hours
    assert rel_err < 0.01, (
        f"expected ~{hl_hours}, got {result.half_life_hours}, rel_err={rel_err:.4f}"
    )
    assert result.r_squared is not None and result.r_squared > 0.99


# ----------------------------------------------------------------------
# Insufficient data
# ----------------------------------------------------------------------

def test_returns_none_for_insufficient_data(tmp_path: Path) -> None:
    now_ns = 1_700_000_000 * 1_000_000_000
    db = tmp_path / "attr.sqlite"
    conn = _make_attr_db(db)
    _seed_decay(conn, strategy_id="s1", half_life_hours=6.0, n=10, now_ns=now_ns)
    conn.close()

    tracker = EdgeDecayTracker(db)
    result = tracker.compute_half_life("s1", window_days=14, now_ns=now_ns)
    assert result.half_life_hours is None
    assert result.reason is not None
    assert "insufficient_data" in result.reason


# ----------------------------------------------------------------------
# Noisy decay: returns a fit with lower R²
# ----------------------------------------------------------------------

def test_noisy_decay_still_fits_with_lower_r2(tmp_path: Path) -> None:
    now_ns = 1_700_000_000 * 1_000_000_000
    db = tmp_path / "attr.sqlite"
    conn = _make_attr_db(db)
    _seed_decay(
        conn, strategy_id="s1", half_life_hours=6.0, n=300, now_ns=now_ns,
        noise=0.15, seed=42,
    )
    conn.close()

    tracker = EdgeDecayTracker(db)
    result = tracker.compute_half_life("s1", window_days=14, now_ns=now_ns)
    assert result.half_life_hours is not None
    # Noisier -> within ~30% of true half-life, and R² somewhere between
    # 0.5 and 0.98.
    rel_err = abs(result.half_life_hours - 6.0) / 6.0
    assert rel_err < 0.3, f"rel_err={rel_err}"
    assert result.r_squared is not None
    assert 0.4 < result.r_squared < 0.99


# ----------------------------------------------------------------------
# Decay curve output
# ----------------------------------------------------------------------

def test_decay_curve_buckets_sorted_ascending(tmp_path: Path) -> None:
    now_ns = 1_700_000_000 * 1_000_000_000
    db = tmp_path / "attr.sqlite"
    conn = _make_attr_db(db)
    _seed_decay(conn, strategy_id="s1", half_life_hours=6.0, n=100, now_ns=now_ns)
    conn.close()

    tracker = EdgeDecayTracker(db)
    curve = tracker.compute_decay_curve(
        "s1", bucket_hours=1, window_days=14, now_ns=now_ns
    )
    assert curve, "curve must not be empty"
    hours = [h for h, _ in curve]
    assert hours == sorted(hours)
    # Monotonic decay (mostly) -> earlier buckets > later buckets on average.
    mids = [r for _, r in curve]
    assert mids[0] > mids[-1], "expected decay from early to late buckets"


# ----------------------------------------------------------------------
# Missing database
# ----------------------------------------------------------------------

def test_missing_db_returns_insufficient(tmp_path: Path) -> None:
    tracker = EdgeDecayTracker(tmp_path / "does_not_exist.sqlite")
    result = tracker.compute_half_life("s1", window_days=14)
    assert result.half_life_hours is None
    assert result.n_fills == 0
