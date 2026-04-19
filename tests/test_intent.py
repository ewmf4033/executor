"""Intent model — construction + validation."""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from executor.core.intent import Atomicity, BasketIntent, Intent, Leg
from executor.core.types import Side


def test_single_factory_builds_one_leg_basket():
    now = time.time_ns()
    b = Intent.single(
        strategy_id="t",
        venue="kalshi",
        market_id="M",
        outcome_id="YES",
        side=Side.BUY,
        target_exposure=10,
        price_limit=0.55,
        confidence=0.6,
        edge_estimate=0.02,
        time_horizon_sec=120,
        created_ts=now,
        expires_ts=now + 60_000_000_000,
    )
    assert len(b.legs) == 1
    assert b.atomicity == Atomicity.INDEPENDENT
    assert b.legs[0].side == Side.BUY
    assert b.legs[0].price_limit == Decimal("0.55")
    # UUID v7 is sortable and non-empty.
    assert isinstance(b.intent_id, str) and len(b.intent_id) >= 32


def test_basket_factory_requires_legs():
    now = time.time_ns()
    with pytest.raises(ValueError):
        Intent.basket(
            strategy_id="t",
            legs=[],
            atomicity=Atomicity.ALL_OR_NONE,
            max_slippage_per_leg=0.02,
            basket_target_exposure=10,
            created_ts=now,
            expires_ts=now + 60_000_000_000,
        )


def test_expires_after_created():
    with pytest.raises(ValueError):
        BasketIntent(
            intent_id="x",
            strategy_id="t",
            legs=(_make_leg(),),
            atomicity=Atomicity.INDEPENDENT,
            max_slippage_per_leg=Decimal("0.02"),
            basket_target_exposure=Decimal("10"),
            created_ts=1000,
            expires_ts=1000,
        )


def _make_leg() -> Leg:
    return Leg(
        venue="kalshi",
        market_id="M",
        outcome_id="YES",
        side=Side.BUY,
        target_exposure=Decimal("10"),
        price_limit=Decimal("0.55"),
        confidence=Decimal("0.6"),
        edge_estimate=Decimal("0.02"),
        time_horizon_sec=120,
    )
