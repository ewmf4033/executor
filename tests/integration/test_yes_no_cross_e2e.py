"""
YESNOCrossDetect end-to-end:
- Strategy emits an ALL_OR_NONE BasketIntent into the event bus.
- The intent passes through the 13 risk gates with paper config.
- A simulated fill is observed by the AttributionTracker and Kill manager.
- A subsequent /kill hard cancels the still-open second leg.
"""
from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.attribution.tracker import AttributionTracker
from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType
from executor.core.intent import Atomicity
from executor.core.types import Side
from executor.kill.manager import KillManager
from executor.kill.state import KillMode, KillStateStore
from executor.risk import ConfigManager, RiskPolicy, RiskState
from executor.strategies.yes_no_cross.strategy import (
    CrossPair,
    YESNOCrossDetect,
)


pytestmark = pytest.mark.asyncio


class _FakeAdapter:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True


async def test_yes_no_cross_through_gates_then_kill(tmp_path: Path) -> None:
    # ----- Wiring -----
    bus = EventBus()
    await bus.start()
    received: list[Event] = []

    async def sink(ev: Event) -> None:
        received.append(ev)

    await bus.subscribe("sink", on_event=sink)

    cfg = ConfigManager(path=None)
    state = RiskState(db_path=tmp_path / "risk_state.sqlite")
    await state.load()
    policy = RiskPolicy(config_manager=cfg, state=state)
    policy.set_publish(bus.publish)
    policy.set_venue_capabilities({
        "kalshi": {"supports_limit", "supports_market"},
        "polymarket": {"supports_limit", "supports_market"},
    })

    kstore = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=kstore, publish=bus.publish)
    adapter_k = _FakeAdapter()
    adapter_p = _FakeAdapter()
    km.register_adapter("kalshi", adapter_k)
    km.register_adapter("polymarket", adapter_p)

    tracker = AttributionTracker(
        db_path=tmp_path / "attr.sqlite", exit_horizon_sec=1, publish=bus.publish
    )

    pair = CrossPair(
        yes_venue="kalshi", yes_market_id="K-RAINS",
        no_venue="polymarket", no_market_id="P-RAINS",
    )
    strategy = YESNOCrossDetect(pairs=[pair], emit_cooldown_sec=0.0)
    strategy.attach(bus.publish)

    # ----- Quote intake -----
    strategy.accept_quote(
        venue="kalshi", market_id="K-RAINS", outcome_id="YES",
        best_ask=Decimal("0.40"), mid=Decimal("0.40"),
    )
    strategy.accept_quote(
        venue="polymarket", market_id="P-RAINS", outcome_id="NO",
        best_ask=Decimal("0.50"), mid=Decimal("0.50"),
    )

    intents = await strategy.attempt_emit()
    assert len(intents) == 1
    intent = intents[0]
    assert intent.atomicity is Atomicity.ALL_OR_NONE

    # ----- Risk policy: feed the intent through the 13 gates -----
    verdict = await policy.evaluate(intent)
    assert verdict.admitted is True, f"expected admit, got reject_gate={verdict.reject_gate}"
    # 13 gates + structural pre-pass (per existing tests this is 14 entries).
    assert len(verdict.gates_passed) >= 13

    # ----- Track decision/arrival prices, simulate single-leg fill -----
    leg_yes, leg_no = intent.legs
    tracker.note_decision(intent.intent_id, Decimal("0.40"))
    tracker.note_arrival(intent.intent_id, leg_yes.leg_id, Decimal("0.40"))

    rec = tracker.on_fill(
        fill_id="fill-yes-1", order_id="ord-yes-1",
        intent_id=intent.intent_id, leg_id=leg_yes.leg_id,
        strategy_id="yes_no_cross", venue="kalshi", market_id="K-RAINS",
        side=Side.BUY, size=leg_yes.target_exposure,
        fill_price=Decimal("0.40"),
        fill_ts_ns=time.time_ns(),
        intent_price=Decimal("0.40"),
    )
    assert rec.execution_cost == Decimal("0.00")

    # Register basket bookkeeping in the kill manager. YES leg filled, NO leg open.
    km.record_basket(intent, open_orders={leg_no.leg_id: ["ord-no-1"]})
    km.mark_leg_filled(intent.intent_id, leg_yes.leg_id)

    # ----- Operator hits HARD kill -----
    await km.engage(KillMode.HARD, "operator panic")

    # NO leg cancelled on polymarket; YES already filled, no cancel attempted.
    assert adapter_p.cancelled == ["ord-no-1"]
    assert adapter_k.cancelled == []

    # Allow event subscribers to drain.
    await asyncio.sleep(0.05)
    types = [e.event_type for e in received]
    assert EventType.INTENT_EMITTED in types
    assert EventType.INTENT_ADMITTED in types
    assert EventType.BASKET_ORPHAN in types  # ALL_OR_NONE with one leg filled
    assert EventType.KILL_STATE_CHANGED in types

    await bus.stop()
    state.close()
    kstore.close()
    tracker.close()


async def test_yes_no_cross_no_emit_when_above_threshold(tmp_path: Path) -> None:
    """Sanity: above-threshold quotes never produce an INTENT_EMITTED."""
    bus = EventBus()
    await bus.start()
    captured: list[Event] = []
    async def _cap(ev: Event) -> None:
        captured.append(ev)
    await bus.subscribe("cap", on_event=_cap)

    pair = CrossPair(
        yes_venue="kalshi", yes_market_id="K1",
        no_venue="polymarket", no_market_id="P1",
    )
    strategy = YESNOCrossDetect(pairs=[pair], emit_cooldown_sec=0.0)
    strategy.attach(bus.publish)
    strategy.accept_quote(venue="kalshi", market_id="K1", outcome_id="YES",
                          best_ask=Decimal("0.55"), mid=Decimal("0.55"))
    strategy.accept_quote(venue="polymarket", market_id="P1", outcome_id="NO",
                          best_ask=Decimal("0.55"), mid=Decimal("0.55"))
    emitted = await strategy.attempt_emit()
    assert emitted == []
    await asyncio.sleep(0.02)
    assert not any(e.event_type is EventType.INTENT_EMITTED for e in captured)
    await bus.stop()
