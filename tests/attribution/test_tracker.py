"""AttributionTracker — slippage breakdown, settlement, summary."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from executor.attribution.tracker import AttributionTracker
from executor.core.types import Side
from executor.risk.state import RiskState


def _tracker(tmp_path: Path, *, exit_horizon_sec: int = 1) -> AttributionTracker:
    return AttributionTracker(
        db_path=tmp_path / "attr.sqlite",
        exit_horizon_sec=exit_horizon_sec,
    )


def test_on_fill_persists_record(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    rec = t.on_fill(
        fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
        strategy_id="s", venue="kalshi", market_id="m",
        side=Side.BUY, size=Decimal("10"), fill_price=Decimal("0.50"),
        fill_ts_ns=1, intent_price=Decimal("0.49"),
    )
    assert rec.fill_price == Decimal("0.50")
    out = t.get_record("f1")
    assert out is not None
    assert out.fill_price == Decimal("0.50")
    assert out.intent_price == Decimal("0.49")
    t.close()


def test_buy_strategy_edge_and_execution_cost(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.note_decision("i1", Decimal("0.40"))
    t.note_arrival("i1", "l1", Decimal("0.45"))
    rec = t.on_fill(
        fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
        strategy_id="s", venue="kalshi", market_id="m",
        side=Side.BUY, size=Decimal("1"), fill_price=Decimal("0.48"),
        fill_ts_ns=1, intent_price=Decimal("0.45"),
    )
    # BUY: decision-arrival = 0.40-0.45 = -0.05; signed for BUY stays = -0.05.
    assert rec.strategy_edge == Decimal("-0.05")
    # arrival-fill = 0.45-0.48 = -0.03 (we paid 3c worse than arrival mid).
    assert rec.execution_cost == Decimal("-0.03")
    t.close()


def test_sell_signs_inverted(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.note_decision("i1", Decimal("0.55"))
    t.note_arrival("i1", "l1", Decimal("0.50"))
    rec = t.on_fill(
        fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
        strategy_id="s", venue="kalshi", market_id="m",
        side=Side.SELL, size=Decimal("1"), fill_price=Decimal("0.48"),
        fill_ts_ns=1, intent_price=Decimal("0.50"),
    )
    # raw decision-arrival = 0.05; SELL negates -> -0.05.
    assert rec.strategy_edge == Decimal("-0.05")
    # raw arrival-fill = 0.02; SELL negates -> -0.02 (we got 2c better than arrival mid).
    assert rec.execution_cost == Decimal("-0.02")
    t.close()


def test_settle_due_attaches_exit_price(tmp_path: Path) -> None:
    t = _tracker(tmp_path, exit_horizon_sec=1)
    t.note_decision("i1", Decimal("0.50"))
    t.note_arrival("i1", "l1", Decimal("0.50"))
    t.on_fill(
        fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
        strategy_id="s", venue="kalshi", market_id="m",
        side=Side.BUY, size=Decimal("1"), fill_price=Decimal("0.50"),
        fill_ts_ns=10, intent_price=Decimal("0.50"),
    )
    t.update_mid("kalshi", "m", Decimal("0.55"))
    settled = asyncio.run(t.settle_due(now_ns=10 + 2_000_000_000))
    assert len(settled) == 1
    # short_term_alpha for BUY: fill - exit = 0.50-0.55 = -0.05; signed for BUY = -0.05.
    assert settled[0].short_term_alpha == Decimal("-0.05")
    out = t.get_record("f1")
    assert out is not None and out.exit_price == Decimal("0.55")
    t.close()


def test_summary_aggregates_per_strategy(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    for i, sid in enumerate(["a", "a", "b"]):
        t.note_decision(f"i{i}", Decimal("0.50"))
        t.note_arrival(f"i{i}", "l", Decimal("0.50"))
        t.on_fill(
            fill_id=f"f{i}", order_id=f"o{i}", intent_id=f"i{i}", leg_id="l",
            strategy_id=sid, venue="kalshi", market_id="m",
            side=Side.BUY, size=Decimal("1"), fill_price=Decimal("0.51"),
            fill_ts_ns=i + 1, intent_price=Decimal("0.50"),
            fee=Decimal("0.001"),
        )
    s = t.summary(since_ns=0)
    assert s["total_fills"] == 3
    assert s["strategies"]["a"]["fills"] == 2
    assert s["strategies"]["b"]["fills"] == 1
    assert s["strategies"]["a"]["fee_sum"] == 0.002
    t.close()


def test_summary_filters_by_strategy(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    for i, sid in enumerate(["a", "b"]):
        t.note_decision(f"i{i}", Decimal("0.5"))
        t.note_arrival(f"i{i}", "l", Decimal("0.5"))
        t.on_fill(
            fill_id=f"f{i}", order_id=f"o{i}", intent_id=f"i{i}", leg_id="l",
            strategy_id=sid, venue="kalshi", market_id="m",
            side=Side.BUY, size=Decimal("1"), fill_price=Decimal("0.50"),
            fill_ts_ns=i + 1, intent_price=Decimal("0.50"),
        )
    s = t.summary(since_ns=0, strategy_id="a")
    assert "a" in s["strategies"] and "b" not in s["strategies"]
    t.close()


def test_pending_not_settled_before_horizon(tmp_path: Path) -> None:
    t = _tracker(tmp_path, exit_horizon_sec=300)
    t.on_fill(
        fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
        strategy_id="s", venue="kalshi", market_id="m",
        side=Side.BUY, size=Decimal("1"), fill_price=Decimal("0.50"),
        fill_ts_ns=1, intent_price=None,
    )
    settled = asyncio.run(t.settle_due(now_ns=1 + 1_000_000_000))
    assert settled == []
    t.close()


# ---------------------------------------------------------------------------
# Phase 4.9 Item 3: pruning to prevent unbounded cache growth.
# ---------------------------------------------------------------------------


def test_prune_on_fill_clears_decision_and_arrival(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.note_decision("i1", Decimal("0.50"))
    t.note_arrival("i1", "l1", Decimal("0.50"))
    t.note_arrival("i1", "l2", Decimal("0.50"))
    assert "i1" in t._decision
    assert ("i1", "l1") in t._arrival
    removed = t.prune_intent("i1")
    assert removed == 3  # 1 decision + 2 arrivals
    assert "i1" not in t._decision
    assert ("i1", "l1") not in t._arrival
    assert ("i1", "l2") not in t._arrival
    t.close()


def test_prune_on_reject_clears_caches(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.note_decision("rejected_intent", Decimal("0.40"))
    assert "rejected_intent" in t._decision
    t.prune_intent("rejected_intent")
    assert "rejected_intent" not in t._decision
    assert "rejected_intent" not in t._decision_ts_ns
    t.close()


def test_prune_on_expire_clears_caches(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.note_decision("expired", Decimal("0.60"))
    t.note_arrival("expired", "leg0", Decimal("0.60"))
    t.prune_intent("expired")
    assert "expired" not in t._decision
    assert not any(k[0] == "expired" for k in t._arrival)
    t.close()


def test_max_age_sweeper_prunes_orphans(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.note_decision("stale", Decimal("0.30"))
    t.note_decision("fresh", Decimal("0.40"))
    # Backdate "stale" to 2h old; leave "fresh" at now.
    t._decision_ts_ns["stale"] = 10**12  # 1970 — definitely older than cutoff
    pruned = t.prune_older_than(max_age_sec=3600.0)
    assert pruned >= 1
    assert "stale" not in t._decision
    assert "fresh" in t._decision
    t.close()


def test_prune_noop_on_unknown_intent(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    # No crash, returns 0.
    assert t.prune_intent("never-seen") == 0
    t.close()


# ---------------------------------------------------------------------------
# Phase 4.13.2: fee-aware PnL fed to gate_13 (daily_loss).
# ---------------------------------------------------------------------------


def test_settle_due_subtracts_fee_from_pnl(tmp_path: Path) -> None:
    """BUY fill with fee=0.50: gross PnL 10.00 → net 9.50 into record_pnl."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    asyncio.run(state.load())
    try:
        tracker = AttributionTracker(
            db_path=tmp_path / "attr.sqlite",
            exit_horizon_sec=0,
            risk_state=state,
        )
        tracker.update_mid("kalshi", "m", Decimal("0.50"))
        tracker.on_fill(
            fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
            strategy_id="s_fee", venue="kalshi", market_id="m",
            side=Side.BUY, size=Decimal("100"), fill_price=Decimal("0.40"),
            fill_ts_ns=1, intent_price=Decimal("0.40"),
            fee=Decimal("0.50"),
        )
        settle_ns = 2_000_000_000
        settled = asyncio.run(tracker.settle_due(now_ns=settle_ns))
        assert len(settled) == 1
        # Gross: (0.50 - 0.40) * 100 = 10.00 ; fee: 0.50 ; net: 9.50.
        pnl = state.daily_pnl("s_fee", now_ns=settle_ns)
        assert pnl == Decimal("9.50"), f"expected 9.50, got {pnl}"
        tracker.close()
    finally:
        state.close()


def test_settle_due_fee_none_treats_as_zero(tmp_path: Path) -> None:
    """fee=None → treated as 0; gross PnL 10.00 passes through unchanged."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    asyncio.run(state.load())
    try:
        tracker = AttributionTracker(
            db_path=tmp_path / "attr.sqlite",
            exit_horizon_sec=0,
            risk_state=state,
        )
        tracker.update_mid("kalshi", "m", Decimal("0.50"))
        tracker.on_fill(
            fill_id="f1", order_id="o1", intent_id="i1", leg_id="l1",
            strategy_id="s_nofee", venue="kalshi", market_id="m",
            side=Side.BUY, size=Decimal("100"), fill_price=Decimal("0.40"),
            fill_ts_ns=1, intent_price=Decimal("0.40"),
            # fee omitted => None
        )
        settle_ns = 2_000_000_000
        settled = asyncio.run(tracker.settle_due(now_ns=settle_ns))
        assert len(settled) == 1
        pnl = state.daily_pnl("s_nofee", now_ns=settle_ns)
        assert pnl == Decimal("10.00"), f"expected 10.00, got {pnl}"
        tracker.close()
    finally:
        state.close()
