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
async def test_orchestrator_emits_error_on_risk_crash(tmp_path: Path):
    """Phase 4.6: risk.evaluate raising must publish ORCHESTRATOR_CRASH and
    keep the stats invariant (received == admitted + rejected + crashed)."""
    import asyncio as _asyncio
    import time as _time

    bus = EventBus()
    await bus.start()

    class _CrashPolicy:
        async def evaluate(self, intent):  # noqa: ARG002
            raise RuntimeError("boom — gate subsystem crashed")

    orch = Orchestrator(bus=bus, policy=_CrashPolicy(), attribution=None, paper_mode=True)  # type: ignore[arg-type]
    await orch.start()

    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await bus.subscribe("cap", on_event=cap)
    try:
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
        await _asyncio.sleep(0.15)

        errors = [
            e for e in seen
            if e.event_type is EventType.ERROR
            and e.payload.get("kind") == "ORCHESTRATOR_CRASH"
        ]
        assert len(errors) == 1, [e.payload for e in seen]
        err = errors[0]
        assert err.payload["stage"] == "risk_evaluate"
        assert err.payload["intent_id"] == intent.intent_id
        assert err.payload["exc_type"] == "RuntimeError"
        assert "boom" in err.payload["error"]

        s = orch.stats()
        assert s["intent_crashes"] == 1
        assert s["intents_received"] == 1
        assert s["admitted"] == 0
        assert s["rejected"] == 0

        # Invariant: received == admitted + rejected + crashed.
        assert (
            s["intents_received"]
            == s["admitted"] + s["rejected"] + s["intent_crashes"]
        )
    finally:
        await bus.unsubscribe("cap")
        await orch.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_orchestrator_emits_error_on_deserialize_crash(tmp_path: Path):
    """Phase 4.6: undecodable INTENT_EMITTED must publish ORCHESTRATOR_CRASH
    and NOT increment n_intents_received (never a valid intent)."""
    import asyncio as _asyncio

    bus = EventBus()
    await bus.start()

    class _NeverCalledPolicy:
        async def evaluate(self, intent):  # pragma: no cover — must never fire
            raise AssertionError("policy.evaluate should not run on decode failure")

    orch = Orchestrator(bus=bus, policy=_NeverCalledPolicy(), attribution=None, paper_mode=True)  # type: ignore[arg-type]
    await orch.start()

    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await bus.subscribe("cap", on_event=cap)
    try:
        await bus.publish(
            Event.make(
                EventType.INTENT_EMITTED,
                source="strategy:unit_test",
                payload={"intent_id": "garbage-1", "legs": [{"bad": "payload"}]},
                intent_id="garbage-1",
                strategy_id="unit_test",
            )
        )
        await _asyncio.sleep(0.1)

        errors = [
            e for e in seen
            if e.event_type is EventType.ERROR
            and e.payload.get("kind") == "ORCHESTRATOR_CRASH"
        ]
        assert len(errors) == 1
        assert errors[0].payload["stage"] == "deserialize"
        assert errors[0].payload["intent_id"] == "garbage-1"

        s = orch.stats()
        assert s["intents_received"] == 0
        assert s["intent_crashes"] == 0
    finally:
        await bus.unsubscribe("cap")
        await orch.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_orchestrator_stats_invariant_holds_under_mixed_traffic(tmp_path: Path):
    """Admitted + rejected + crashed must equal received at all times."""
    import asyncio as _asyncio
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
    orch = Orchestrator(bus=bus, policy=policy, attribution=None, paper_mode=True)
    await orch.start()
    try:
        # one good
        good = _synth_intent(_time.time_ns())
        await bus.publish(Event.make(
            EventType.INTENT_EMITTED, source="strategy:unit_test",
            payload=_serialize_intent(good), intent_id=good.intent_id,
            strategy_id="unit_test",
        ))
        # one rejectable (zero exposure)
        now = _time.time_ns() + 1
        bad = BasketIntent(
            intent_id=f"bad-{now}", strategy_id="unit_test",
            legs=(Leg(
                venue="v1", market_id="m1", outcome_id="YES", side=Side.BUY,
                target_exposure=Decimal("0"), price_limit=Decimal("0.40"),
                confidence=Decimal("0.9"), edge_estimate=Decimal("0.05"),
                time_horizon_sec=60,
            ),),
            atomicity=Atomicity.INDEPENDENT, max_slippage_per_leg=Decimal("0.02"),
            basket_target_exposure=Decimal("0"),
            created_ts=now, expires_ts=now + 60_000_000_000,
        )
        await bus.publish(Event.make(
            EventType.INTENT_EMITTED, source="strategy:unit_test",
            payload=_serialize_intent(bad), intent_id=bad.intent_id,
            strategy_id="unit_test",
        ))
        await _asyncio.sleep(0.2)

        s = orch.stats()
        assert s["intents_received"] == s["admitted"] + s["rejected"] + s["intent_crashes"]
        assert s["admitted"] == 1
        assert s["rejected"] == 1
        assert s["intent_crashes"] == 0
    finally:
        await orch.stop()
        await bus.stop()
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
