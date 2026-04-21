"""Orchestrator — INTENT_EMITTED → risk → paper FILL → attribution."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from executor.attribution.tracker import AttributionTracker
from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType
from executor.core.intent import Atomicity, BasketIntent, Leg
from executor.core.orchestrator import Orchestrator, deserialize_intent
from executor.core.types import Side
from executor.risk.config import ConfigManager
from executor.risk.policy import RiskPolicy
from executor.risk.state import RiskState
from executor.strategies.base import _serialize_intent


def _synth_intent(now_ns: int) -> BasketIntent:
    legs = (
        Leg(
            venue="v1", market_id="m1", outcome_id="YES", side=Side.BUY,
            target_exposure=Decimal("2"), price_limit=Decimal("0.40"),
            confidence=Decimal("0.9"), edge_estimate=Decimal("0.05"),
            time_horizon_sec=60,
        ),
        Leg(
            venue="v2", market_id="m2", outcome_id="NO", side=Side.BUY,
            target_exposure=Decimal("2"), price_limit=Decimal("0.40"),
            confidence=Decimal("0.9"), edge_estimate=Decimal("0.05"),
            time_horizon_sec=60,
        ),
    )
    return BasketIntent(
        intent_id=f"t-{now_ns}",
        strategy_id="unit_test",
        legs=legs,
        atomicity=Atomicity.ALL_OR_NONE,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=Decimal("2"),
        created_ts=now_ns,
        expires_ts=now_ns + 60_000_000_000,
    )


def test_deserialize_intent_roundtrip():
    import time as _time
    intent = _synth_intent(_time.time_ns())
    payload = _serialize_intent(intent)
    back = deserialize_intent(payload)
    assert back.intent_id == intent.intent_id
    assert back.atomicity == intent.atomicity
    assert len(back.legs) == 2
    assert back.legs[0].venue == "v1"
    assert back.legs[1].outcome_id == "NO"
    assert back.legs[0].target_exposure == Decimal("2")


@pytest.mark.asyncio
async def test_orchestrator_admits_and_fills(tmp_path: Path):
    import time as _time
    bus = EventBus()
    await bus.start()
    cfg = ConfigManager(None)
    state = RiskState(db_path=tmp_path / "state.sqlite")
    await state.load()
    policy = RiskPolicy(config_manager=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({
        "v1": frozenset({"supports_limit"}),
        "v2": frozenset({"supports_limit"}),
    })
    attr = AttributionTracker(db_path=tmp_path / "attr.sqlite", publish=bus.publish)
    orch = Orchestrator(bus=bus, policy=policy, attribution=attr, paper_mode=True)
    await orch.start()

    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await bus.subscribe("cap", on_event=cap)

    intent = _synth_intent(_time.time_ns())
    await bus.publish(
        Event.make(
            EventType.INTENT_EMITTED,
            source="strategy:unit_test",
            payload=_serialize_intent(intent),
            intent_id=intent.intent_id,
            strategy_id="unit_test",
        )
    )

    # Allow fan-out + risk + fill.
    import asyncio as _asyncio
    await _asyncio.sleep(0.2)

    kinds = [e.event_type for e in seen]
    assert EventType.INTENT_EMITTED in kinds
    assert EventType.INTENT_ADMITTED in kinds
    assert EventType.ORDER_PLACED in kinds
    assert EventType.FILL in kinds
    assert EventType.INTENT_COMPLETE in kinds

    # Attribution persisted both legs.
    rows = attr._conn.execute(
        "SELECT COUNT(*) FROM attribution WHERE intent_id = ?",
        (intent.intent_id,),
    ).fetchone()
    assert rows[0] == 2

    s = orch.stats()
    assert s["intents_received"] == 1
    assert s["admitted"] == 1
    assert s["rejected"] == 0
    assert s["filled_legs"] == 2

    await bus.unsubscribe("cap")
    await orch.stop()
    await bus.stop()
    attr.close()
    state.close()


@pytest.mark.asyncio
async def test_orchestrator_counts_rejections(tmp_path: Path):
    """Reject path (zero-exposure leg) increments the reject counter and never fills."""
    import asyncio as _asyncio
    import time as _time
    bus = EventBus()
    await bus.start()
    cfg = ConfigManager(None)
    state = RiskState(db_path=tmp_path / "state.sqlite")
    await state.load()
    policy = RiskPolicy(config_manager=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({"v1": frozenset({"supports_limit"})})
    orch = Orchestrator(bus=bus, policy=policy, attribution=None, paper_mode=True)
    await orch.start()

    now = _time.time_ns()
    # zero target_exposure → structural gate rejects
    bad = BasketIntent(
        intent_id=f"bad-{now}",
        strategy_id="unit_test",
        legs=(
            Leg(
                venue="v1", market_id="m1", outcome_id="YES", side=Side.BUY,
                target_exposure=Decimal("0"), price_limit=Decimal("0.40"),
                confidence=Decimal("0.9"), edge_estimate=Decimal("0.05"),
                time_horizon_sec=60,
            ),
        ),
        atomicity=Atomicity.INDEPENDENT,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=Decimal("0"),
        created_ts=now,
        expires_ts=now + 60_000_000_000,
    )
    await bus.publish(
        Event.make(
            EventType.INTENT_EMITTED,
            source="strategy:unit_test",
            payload=_serialize_intent(bad),
            intent_id=bad.intent_id,
            strategy_id="unit_test",
        )
    )
    await _asyncio.sleep(0.15)

    s = orch.stats()
    assert s["rejected"] == 1
    assert s["admitted"] == 0
    assert s["filled_legs"] == 0

    await orch.stop()
    await bus.stop()
    state.close()
