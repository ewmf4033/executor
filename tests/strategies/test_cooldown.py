"""Phase 4.10 (4.9.1-c) — strategy rejection-aware cooldown.

Covers the Strategy base-class helpers:
    _record_rejection, _record_admit, _should_emit_for_market,
    on_gate_rejected, on_intent_admitted
plus the cooldown-block in emit().
"""
from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any

import pytest

from executor.core.events import Event, EventType, Source
from executor.core.intent import Atomicity, BasketIntent, Leg
from executor.core.types import Side
from executor.strategies import base as strategy_base


class _Probe(strategy_base.Strategy):
    """Minimal concrete strategy for tests."""

    strategy_id = "probe"
    required_capabilities = {}

    @property
    def markets(self) -> list[tuple[str, str]]:
        return [("kalshi", "MKT-A"), ("kalshi", "MKT-B")]

    async def run(self) -> None:  # pragma: no cover — not invoked
        pass


def _make_intent(market_id: str, intent_id: str = "i-1") -> Any:
    leg = Leg(
        venue="kalshi",
        market_id=market_id,
        outcome_id="YES",
        side=Side.BUY,
        target_exposure=Decimal("1"),
        price_limit=Decimal("0.50"),
        confidence=Decimal("0.8"),
        edge_estimate=Decimal("0.05"),
        time_horizon_sec=600,
        required_capabilities=("supports_limit",),
        kelly_fraction_used=Decimal("0.25"),
    )
    now = time.time_ns()
    return BasketIntent(
        intent_id=intent_id,
        strategy_id="probe",
        legs=(leg,),
        atomicity=Atomicity.ALL_OR_NONE,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=Decimal("1"),
        created_ts=now,
        expires_ts=now + 10_000_000_000,
    )


# ----------------------------------------------------------------------
# _record_rejection + _should_emit_for_market
# ----------------------------------------------------------------------

def test_below_threshold_does_not_engage_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 5)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    t0 = 1000.0
    for i in range(5):
        tripped = s._record_rejection(gate="market_exposure", market_id="MKT-A", now=t0 + i)
        assert tripped is False
    assert s._should_emit_for_market("MKT-A", now=t0 + 6)


def test_crossing_threshold_engages_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 5)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    t0 = 1000.0
    tripped = False
    for i in range(6):
        tripped = s._record_rejection(gate="market_exposure", market_id="MKT-A", now=t0 + i)
    assert tripped is True
    # Cooldown was engaged at the time of the 6th reject (t0+5).
    # Still in cooldown mid-window.
    assert not s._should_emit_for_market("MKT-A", now=t0 + 5.0 + 15.0)
    # Still in cooldown just before the cooldown expires.
    assert not s._should_emit_for_market("MKT-A", now=t0 + 5.0 + 29.9)
    # Out of cooldown once past the cooldown horizon.
    assert s._should_emit_for_market("MKT-A", now=t0 + 5.0 + 30.5)


def test_different_markets_do_not_cross_contaminate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 3)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    t0 = 1000.0
    for i in range(4):
        s._record_rejection(gate="market_exposure", market_id="MKT-A", now=t0 + i)
    # MKT-A in cooldown, MKT-B unaffected.
    assert not s._should_emit_for_market("MKT-A", now=t0 + 5)
    assert s._should_emit_for_market("MKT-B", now=t0 + 5)


def test_different_gates_accumulate_per_market(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same market, different gates: each gate has its own bucket; but a trip
    on ANY gate pushes the market into cooldown."""
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 3)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    t0 = 1000.0
    # 3 on gate A — doesn't trip (threshold is 3, needs 4 to trip).
    for i in range(3):
        tripped = s._record_rejection(gate="gate_a", market_id="MKT-A", now=t0 + i)
        assert tripped is False
    assert s._should_emit_for_market("MKT-A", now=t0 + 3)
    # 3 on gate B — same story.
    for i in range(3):
        s._record_rejection(gate="gate_b", market_id="MKT-A", now=t0 + 4 + i)
    # Now one more on gate A would trip.
    tripped = s._record_rejection(gate="gate_a", market_id="MKT-A", now=t0 + 10)
    assert tripped is True


def test_admit_clears_counter_for_market(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 5)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    t0 = 1000.0
    for i in range(4):
        s._record_rejection(gate="market_exposure", market_id="MKT-A", now=t0 + i)
    # Partial counter exists.
    assert ("market_exposure", "MKT-A") in s._reject_history
    s._record_admit("MKT-A")
    assert ("market_exposure", "MKT-A") not in s._reject_history
    assert s._should_emit_for_market("MKT-A", now=t0 + 5)


def test_window_evicts_old_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 3)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 10.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    t0 = 1000.0
    # 3 early rejects, outside window by the time we try #4.
    for i in range(3):
        s._record_rejection(gate="g", market_id="MKT-A", now=t0 + i)
    # Fire #4 well past the window — should NOT trip because old entries age out.
    tripped = s._record_rejection(gate="g", market_id="MKT-A", now=t0 + 100)
    assert tripped is False
    assert s._should_emit_for_market("MKT-A", now=t0 + 100)


# ----------------------------------------------------------------------
# on_gate_rejected / on_intent_admitted integration
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_gate_rejected_attributes_to_market_via_intent_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 2)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    # Simulate the strategy having emitted two intents.
    s._remember_intent("i-1", ("MKT-A",))
    s._remember_intent("i-2", ("MKT-A",))
    s._remember_intent("i-3", ("MKT-A",))

    for intent_id in ("i-1", "i-2", "i-3"):
        evt = Event.make(
            EventType.GATE_REJECTED,
            source=Source.RISK,
            intent_id=intent_id,
            strategy_id="probe",
            payload={"gate": "market_exposure", "reason": "cap"},
        )
        await s.on_gate_rejected(evt)

    # Threshold is 2 — 3 rejects on MKT-A trips cooldown.
    assert not s._should_emit_for_market("MKT-A")


@pytest.mark.asyncio
async def test_on_intent_admitted_clears_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 2)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    s = _Probe()
    s._remember_intent("i-1", ("MKT-A",))
    s._remember_intent("i-2", ("MKT-A",))
    s._remember_intent("i-3", ("MKT-A",))
    s._remember_intent("i-OK", ("MKT-A",))

    for intent_id in ("i-1", "i-2", "i-3"):
        evt = Event.make(
            EventType.GATE_REJECTED,
            source=Source.RISK,
            intent_id=intent_id,
            strategy_id="probe",
            payload={"gate": "market_exposure"},
        )
        await s.on_gate_rejected(evt)
    assert not s._should_emit_for_market("MKT-A")

    admit = Event.make(
        EventType.INTENT_ADMITTED,
        source=Source.RISK,
        intent_id="i-OK",
        strategy_id="probe",
        payload={},
    )
    await s.on_intent_admitted(admit)
    assert s._should_emit_for_market("MKT-A")


@pytest.mark.asyncio
async def test_other_strategys_events_are_ignored() -> None:
    s = _Probe()
    s._remember_intent("i-1", ("MKT-A",))
    evt = Event.make(
        EventType.GATE_REJECTED,
        source=Source.RISK,
        intent_id="i-1",
        strategy_id="other_strategy",
        payload={"gate": "g"},
    )
    await s.on_gate_rejected(evt)
    assert ("g", "MKT-A") not in s._reject_history


# ----------------------------------------------------------------------
# emit() is blocked while in cooldown
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_drops_when_market_in_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_THRESHOLD", 2)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_WINDOW_SEC", 60.0)
    monkeypatch.setattr(strategy_base, "STRATEGY_REJECT_COOLDOWN_SEC", 30.0)

    published: list[Event] = []

    async def _publish(ev: Event) -> None:
        published.append(ev)

    s = _Probe()
    s.attach(_publish)

    # Trip the cooldown directly.
    for _ in range(3):
        s._record_rejection(gate="market_exposure", market_id="MKT-A")
    assert not s._should_emit_for_market("MKT-A")

    intent = _make_intent("MKT-A", intent_id="blocked")
    await s.emit(intent)
    assert published == [], "emit must be dropped while market is in cooldown"


@pytest.mark.asyncio
async def test_emit_proceeds_when_not_in_cooldown() -> None:
    published: list[Event] = []

    async def _publish(ev: Event) -> None:
        published.append(ev)

    s = _Probe()
    s.attach(_publish)

    intent = _make_intent("MKT-B", intent_id="ok")
    await s.emit(intent)
    assert len(published) == 1
    assert published[0].event_type == EventType.INTENT_EMITTED
    assert published[0].intent_id == "ok"
    # Intent should be remembered for later reject attribution.
    assert s._intent_markets.get("ok") == ("MKT-B",)
