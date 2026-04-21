"""Shared fixtures for risk-policy tests."""
from __future__ import annotations

import tempfile
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.event_bus import EventBus
from executor.core.intent import Atomicity, Intent, Leg
from executor.core.types import (
    Orderbook,
    OrderbookLevel,
    Side,
)
from executor.detectors.adverse_selection import NullAdverseSelectionDetector
from executor.risk import ConfigManager, RiskPolicy, RiskState


def make_leg(
    *,
    venue="kalshi",
    market_id="MKT-1",
    outcome_id="YES",
    side=Side.BUY,
    size=10,
    price=0.55,
    edge=0.03,
    horizon=120,
    caps=(),
) -> Leg:
    return Leg(
        venue=venue,
        market_id=market_id,
        outcome_id=outcome_id,
        side=side,
        target_exposure=Decimal(str(size)),
        price_limit=Decimal(str(price)),
        confidence=Decimal("0.6"),
        edge_estimate=Decimal(str(edge)),
        time_horizon_sec=horizon,
        required_capabilities=tuple(caps),
    )


def make_intent(*, strategy_id="s1", legs=None, expires_in_sec=60):
    now = time.time_ns()
    legs = legs or [make_leg()]
    if len(legs) == 1:
        leg = legs[0]
        return Intent.single(
            strategy_id=strategy_id,
            venue=leg.venue,
            market_id=leg.market_id,
            outcome_id=leg.outcome_id,
            side=leg.side,
            target_exposure=leg.target_exposure,
            price_limit=leg.price_limit,
            confidence=leg.confidence,
            edge_estimate=leg.edge_estimate,
            time_horizon_sec=leg.time_horizon_sec,
            required_capabilities=leg.required_capabilities,
            created_ts=now,
            expires_ts=now + expires_in_sec * 1_000_000_000,
        )
    return Intent.basket(
        strategy_id=strategy_id,
        legs=legs,
        atomicity=Atomicity.INDEPENDENT,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=sum(l.target_exposure for l in legs),
        created_ts=now,
        expires_ts=now + expires_in_sec * 1_000_000_000,
    )


def mk_orderbook(market_id, *, bids=((0.50, 50),), asks=((0.60, 50),)) -> Orderbook:
    now = time.time_ns()
    return Orderbook(
        market_id=market_id,
        venue="kalshi",
        outcome_id="YES",
        bids=tuple(OrderbookLevel(Decimal(str(p)), Decimal(str(s))) for p, s in bids),
        asks=tuple(OrderbookLevel(Decimal(str(p)), Decimal(str(s))) for p, s in asks),
        ts_ns=now,
        received_ts_ns=now,
    )


@pytest.fixture
def tmp_state_path(tmp_path):
    return tmp_path / "risk_state.sqlite"


@pytest.fixture
def cfg_mgr():
    return ConfigManager(path=None)


@pytest.fixture
async def policy(cfg_mgr, tmp_state_path):
    state = RiskState(db_path=tmp_state_path)
    await state.load()
    p = RiskPolicy(
        config_manager=cfg_mgr,
        state=state,
        adverse_selection=NullAdverseSelectionDetector(),
    )
    # Allow every market/venue capability for tests unless overridden.
    p.set_venue_capabilities({"kalshi": {"supports_limit", "supports_market"}})
    # Phase 4.7: tests predate market_universe registration — enable
    # the bootstrap escape hatch so StructuralGate default-allows when
    # the test doesn't explicitly call set_market_universe().
    p.set_allow_universe_bootstrap(True)
    yield p
    state.close()


@pytest.fixture
async def policy_with_bus(policy):
    bus = EventBus()
    await bus.start()
    received = []

    async def sink(e):
        received.append(e)

    await bus.subscribe("test", on_event=sink)
    policy.set_publish(bus.publish)
    # Phase 4.7 F5: event_concentration now fails closed when any leg
    # lacks an event_id. Register a catch-all mapping for test markets.
    _test_market_ids = (
        "MKT-1", "MKT-2", "MA", "MB", "MC", "MX", "MY0", "MY1", "MY2",
        "MY3", "MY4", "NEW-K", "NEW-P",
    )
    event_map = {("kalshi", m): f"EVT-{m}" for m in _test_market_ids}
    event_map.update({("poly", m): f"EVT-{m}" for m in _test_market_ids})
    policy.set_event_id_map(event_map)
    yield policy, bus, received
    await bus.stop()
