"""YESNOCrossDetect — quote intake, basket emission, sizing."""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from executor.core.events import Event, EventType
from executor.core.intent import Atomicity
from executor.strategies.yes_no_cross.strategy import (
    CrossPair,
    YESNOCrossDetect,
)


def _strategy(**kwargs) -> tuple[YESNOCrossDetect, list[Event]]:
    pair = CrossPair(
        yes_venue="kalshi", yes_market_id="K1",
        no_venue="polymarket", no_market_id="P1",
    )
    s = YESNOCrossDetect(pairs=[pair], emit_cooldown_sec=0.0, **kwargs)
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    s.attach(pub)
    return s, captured


def test_no_emit_without_quotes() -> None:
    s, _ = _strategy()
    pair = s._pairs[0]
    assert s.find_cross(pair) is None
    assert s.build_intent(pair) is None


def test_no_emit_when_sum_above_threshold() -> None:
    s, _ = _strategy()
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.55"), mid=Decimal("0.55"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.55"), mid=Decimal("0.55"))
    assert s.find_cross(s._pairs[0]) is None  # 0.55+0.55 = 1.10 > 0.98


def test_emit_when_sum_below_threshold() -> None:
    s, _ = _strategy()
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.40"), mid=Decimal("0.40"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.50"), mid=Decimal("0.50"))
    cross = s.find_cross(s._pairs[0])
    assert cross is not None
    sum_ask, edge, size = cross
    assert sum_ask == Decimal("0.90")
    assert edge == Decimal("0.08")
    assert size >= Decimal("1")


def test_build_intent_is_all_or_none_with_two_legs() -> None:
    s, _ = _strategy()
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.40"), mid=Decimal("0.40"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.50"), mid=Decimal("0.50"))
    intent = s.build_intent(s._pairs[0])
    assert intent is not None
    assert intent.atomicity is Atomicity.ALL_OR_NONE
    assert len(intent.legs) == 2
    yes_leg, no_leg = intent.legs
    assert yes_leg.outcome_id == "YES"
    assert yes_leg.venue == "kalshi"
    assert no_leg.outcome_id == "NO"
    assert no_leg.venue == "polymarket"
    assert intent.metadata["sum_ask"] == "0.90"


def test_attempt_emit_publishes_intent_emitted_event() -> None:
    s, events = _strategy()
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.40"), mid=Decimal("0.40"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.50"), mid=Decimal("0.50"))
    emitted = asyncio.run(s.attempt_emit())
    assert len(emitted) == 1
    assert any(e.event_type is EventType.INTENT_EMITTED for e in events)


def test_cooldown_blocks_repeat_emit() -> None:
    pair = CrossPair(
        yes_venue="kalshi", yes_market_id="K1",
        no_venue="polymarket", no_market_id="P1",
    )
    s = YESNOCrossDetect(pairs=[pair], emit_cooldown_sec=600.0)
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    s.attach(pub)
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.40"), mid=Decimal("0.40"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.50"), mid=Decimal("0.50"))
    e1 = asyncio.run(s.attempt_emit())
    e2 = asyncio.run(s.attempt_emit())
    assert len(e1) == 1
    assert e2 == []


def test_size_clamped_to_max() -> None:
    s, _ = _strategy(max_contracts=Decimal("5"))
    # huge edge: YES=0.01, NO=0.01 -> sum=0.02, edge = 0.96
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.01"), mid=Decimal("0.01"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.01"), mid=Decimal("0.01"))
    _, _, size = s.find_cross(s._pairs[0])
    assert size == Decimal("5")


def test_size_floor_at_one_contract() -> None:
    s, _ = _strategy()
    # tiny edge: 0.97+0.005 -> 0.975 sum, edge=0.005, kelly_cap=0.25 -> 0.001 -> floor 1
    s.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                   best_ask=Decimal("0.970"), mid=Decimal("0.97"))
    s.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                   best_ask=Decimal("0.005"), mid=Decimal("0.005"))
    _, _, size = s.find_cross(s._pairs[0])
    assert size == Decimal("1")
