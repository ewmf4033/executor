"""
Edge half-life framework — 0h scaffold.

Measures how fast a strategy's predicted edge decays into realized return
after a fill.

The tracker reads attribution rows (strategy_edge = predicted edge at decision
time, short_term_alpha = realized edge over exit_horizon_sec post-fill) and
fits an exponential decay curve:

    realized(t) = predicted * exp(-lambda * t)
    half_life   = ln(2) / lambda

Two public methods:

    compute_half_life(strategy_id, window_days=14) -> float | None
        Best-fit half-life in hours. None if n<20 fills or fit is unreliable.

    compute_decay_curve(strategy_id, bucket_hours=1)
        -> list[tuple[hour, mean_realized_to_predicted_ratio]]
        Bucketed observed-vs-predicted ratio, for plotting.

Status: scaffold only. Pre-live, `short_term_alpha` is filled from synthetic
random-walk quotes, so values are noisy around zero and half-life fits are
not meaningful. Phase 5 + first live strategy will provide real signal.

CLI:
    python -m executor.analysis.edge_half_life \
        --strategy=yes_no_cross --window-days=14

Phase 4.10 (0h).
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_DB = Path("/root/executor/state/attribution.sqlite")
MIN_FILLS = 20


def _dec(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None


@dataclass
class HalfLifeResult:
    strategy_id: str
    window_days: int
    n_fills: int
    half_life_hours: float | None
    r_squared: float | None
    reason: str | None = None  # populated when half_life_hours is None


class EdgeDecayTracker:
    """Reads attribution rows and computes decay statistics.

    No mutation of the attribution DB. Connection opened per-call so the
    tracker is safe to run from a CLI or a future metrics endpoint without
    fighting with the live writer.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self._db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Decay curve
    # ------------------------------------------------------------------

    def compute_decay_curve(
        self,
        strategy_id: str,
        *,
        bucket_hours: int = 1,
        window_days: int = 14,
        now_ns: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return [(hour_bucket, mean ratio realized/predicted)] sorted by hour.

        Hour_bucket = hours elapsed since fill_ts_ns, floor-bucketed.
        Ratio clamps: if predicted == 0, row is dropped.
        """
        rows = self._load_rows(strategy_id, window_days=window_days, now_ns=now_ns)
        bucketed: dict[int, list[float]] = {}
        now_ns = now_ns or time.time_ns()
        for fill_ts_ns, predicted, realized in rows:
            if predicted == 0:
                continue
            age_hours = max(0, int((now_ns - fill_ts_ns) // (3600 * 1_000_000_000)))
            bucket = (age_hours // max(1, bucket_hours)) * max(1, bucket_hours)
            ratio = realized / predicted
            bucketed.setdefault(bucket, []).append(ratio)
        return sorted(
            (h, statistics.fmean(vs)) for h, vs in bucketed.items() if vs
        )

    # ------------------------------------------------------------------
    # Half-life fit
    # ------------------------------------------------------------------

    def compute_half_life(
        self,
        strategy_id: str,
        *,
        window_days: int = 14,
        now_ns: int | None = None,
    ) -> HalfLifeResult:
        """Exponential fit on (t_hours, log(ratio)). Returns HalfLifeResult.

        The fit projects onto the hypothesis that realized = predicted * e^{-lambda t}.
        Rows where ratio <= 0 (realized and predicted with opposite sign,
        or realized <= 0) are dropped from the log-fit — they represent
        "past half-life" noise and can't be log-transformed. If that
        leaves fewer than MIN_FILLS points, return None with reason.
        """
        rows = self._load_rows(strategy_id, window_days=window_days, now_ns=now_ns)
        if len(rows) < MIN_FILLS:
            return HalfLifeResult(
                strategy_id=strategy_id,
                window_days=window_days,
                n_fills=len(rows),
                half_life_hours=None,
                r_squared=None,
                reason=f"insufficient_data n={len(rows)} min={MIN_FILLS}",
            )

        now_ns = now_ns or time.time_ns()
        xs: list[float] = []
        ys: list[float] = []
        for fill_ts_ns, predicted, realized in rows:
            if predicted == 0:
                continue
            ratio = realized / predicted
            if ratio <= 0:
                continue
            # Age in hours from fill to now. Synthetic tests override now_ns
            # to make this deterministic.
            age_h = (now_ns - fill_ts_ns) / (3600 * 1_000_000_000)
            if age_h <= 0:
                continue
            xs.append(age_h)
            ys.append(math.log(ratio))

        if len(xs) < MIN_FILLS:
            return HalfLifeResult(
                strategy_id=strategy_id,
                window_days=window_days,
                n_fills=len(rows),
                half_life_hours=None,
                r_squared=None,
                reason=f"insufficient_positive_ratios n_log_valid={len(xs)}",
            )

        slope, intercept, r2 = _linear_regression(xs, ys)
        if slope >= 0:
            # Fit says edge grows over time; meaningless for half-life.
            return HalfLifeResult(
                strategy_id=strategy_id,
                window_days=window_days,
                n_fills=len(rows),
                half_life_hours=None,
                r_squared=r2,
                reason=f"non_decaying_fit slope={slope:.4f}",
            )

        half_life = math.log(2) / (-slope)
        return HalfLifeResult(
            strategy_id=strategy_id,
            window_days=window_days,
            n_fills=len(rows),
            half_life_hours=half_life,
            r_squared=r2,
            reason=None,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_rows(
        self,
        strategy_id: str,
        *,
        window_days: int,
        now_ns: int | None,
    ) -> list[tuple[int, float, float]]:
        """Returns [(fill_ts_ns, predicted_edge, realized_return)]."""
        if not self._db_path.exists():
            return []
        now_ns = now_ns or time.time_ns()
        since_ns = now_ns - window_days * 24 * 3600 * 1_000_000_000
        conn = sqlite3.connect(str(self._db_path))
        try:
            cur = conn.execute(
                "SELECT fill_ts_ns, strategy_edge, short_term_alpha "
                "FROM attribution "
                "WHERE strategy_id = ? AND fill_ts_ns >= ? "
                "AND strategy_edge IS NOT NULL AND short_term_alpha IS NOT NULL",
                (strategy_id, since_ns),
            )
            out: list[tuple[int, float, float]] = []
            for ts, predicted_raw, realized_raw in cur:
                p = _dec(predicted_raw)
                r = _dec(realized_raw)
                if p is None or r is None:
                    continue
                out.append((int(ts), float(p), float(r)))
            return out
        finally:
            conn.close()


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple OLS. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, mean_y, 0.0
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, intercept, r2


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Edge half-life scaffold.")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--bucket-hours", type=int, default=1)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)

    tracker = EdgeDecayTracker(args.db)
    result = tracker.compute_half_life(
        args.strategy, window_days=args.window_days
    )
    print(f"strategy={result.strategy_id}")
    print(f"window_days={result.window_days}")
    print(f"n_fills={result.n_fills}")
    print(f"half_life_hours={result.half_life_hours}")
    print(f"r_squared={result.r_squared}")
    if result.reason:
        print(f"reason={result.reason}")
    curve = tracker.compute_decay_curve(
        args.strategy,
        bucket_hours=args.bucket_hours,
        window_days=args.window_days,
    )
    if curve:
        print("decay_curve (hour, mean_ratio):")
        for h, r in curve:
            print(f"  {h:4d}h  {r:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
