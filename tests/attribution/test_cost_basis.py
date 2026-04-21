"""Tests for cost_basis_dollars and venue_fee_bps attribution columns."""
from decimal import Decimal

from executor.attribution.tracker import AttributionTracker
from executor.core.types import Side


def _make_tracker(tmp_path):
    return AttributionTracker(db_path=tmp_path / "attr.sqlite")


def _fill(tracker, *, side, size, fill_price, fee):
    return tracker.on_fill(
        fill_id="f1",
        order_id="o1",
        intent_id="i1",
        leg_id="l1",
        strategy_id="s1",
        venue="poly",
        market_id="m1",
        side=side,
        size=Decimal(str(size)),
        fill_price=Decimal(str(fill_price)),
        fill_ts_ns=1_000_000_000,
        intent_price=None,
        fee=Decimal(str(fee)) if fee is not None else None,
    )


def test_cost_basis_buy(tmp_path):
    tracker = _make_tracker(tmp_path)
    rec = _fill(tracker, side=Side.BUY, size=10, fill_price="0.55", fee="0.02")
    assert rec.cost_basis_dollars == Decimal("10") * Decimal("0.55") + Decimal("0.02")
    assert rec.cost_basis_dollars == Decimal("5.52")
    # Round-trip through DB
    loaded = tracker.get_record("f1")
    assert loaded is not None
    assert loaded.cost_basis_dollars == Decimal("5.52")


def test_cost_basis_sell(tmp_path):
    tracker = _make_tracker(tmp_path)
    rec = _fill(tracker, side=Side.SELL, size=10, fill_price="0.55", fee="0.02")
    assert rec.cost_basis_dollars == Decimal("10") * Decimal("0.55") - Decimal("0.02")
    assert rec.cost_basis_dollars == Decimal("5.48")
    loaded = tracker.get_record("f1")
    assert loaded is not None
    assert loaded.cost_basis_dollars == Decimal("5.48")


def test_venue_fee_bps(tmp_path):
    tracker = _make_tracker(tmp_path)
    rec = _fill(tracker, side=Side.BUY, size=10, fill_price="0.55", fee="0.02")
    expected = (Decimal("0.02") / (Decimal("10") * Decimal("0.55"))) * Decimal("10000")
    assert rec.venue_fee_bps == expected
    loaded = tracker.get_record("f1")
    assert loaded is not None
    assert loaded.venue_fee_bps == expected


def test_cost_basis_no_fee(tmp_path):
    tracker = _make_tracker(tmp_path)
    rec = _fill(tracker, side=Side.BUY, size=10, fill_price="0.55", fee=None)
    assert rec.cost_basis_dollars == Decimal("10") * Decimal("0.55")
    assert rec.cost_basis_dollars == Decimal("5.50")
    assert rec.venue_fee_bps is None
    loaded = tracker.get_record("f1")
    assert loaded is not None
    assert loaded.cost_basis_dollars == Decimal("5.50")
    assert loaded.venue_fee_bps is None
