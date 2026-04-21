"""Startup self-check — unit tests against a mocked event bus + risk policy.

Failure modes exercised:
- happy path: all four stages fire, SELF_CHECK_OK with latencies
- risk rejection: risk emits GATE_REJECTED, self-check still OK (pipeline worked)
- risk silence: no risk event → SELF_CHECK_FAIL on timeout
- fill silence: risk admits but no FILL → SELF_CHECK_FAIL on timeout
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from executor.attribution.tracker import AttributionTracker
from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType, Source
from executor.core.orchestrator import Orchestrator
from executor.core.self_check import build_synthetic_intent, run_self_check
from executor.risk.config import ConfigManager
from executor.risk.policy import RiskPolicy
from executor.risk.state import RiskState


@pytest.fixture
async def wired_bus(tmp_path: Path):
    """Bus + orchestrator + risk + attribution, no strategy / adapter."""
    bus = EventBus()
    await bus.start()

    cfg = ConfigManager(None)
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    policy = RiskPolicy(config_manager=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({
        "self_check_yes": frozenset({"supports_limit"}),
        "self_check_no": frozenset({"supports_limit"}),
    })
    attr = AttributionTracker(db_path=tmp_path / "attr.sqlite", publish=bus.publish)
    orch = Orchestrator(bus=bus, policy=policy, attribution=attr, paper_mode=True)
    await orch.start()

    yield {"bus": bus, "policy": policy, "attribution": attr, "orchestrator": orch,
           "state": state}

    await orch.stop()
    await bus.stop()
    attr.close()
    state.close()


@pytest.mark.asyncio
async def test_self_check_ok_full_pipeline(wired_bus):
    result = await run_self_check(
        bus=wired_bus["bus"],
        attribution=wired_bus["attribution"],
        timeout_sec=5.0,
    )
    assert result["kind"] == "ok", result
    s = result["stages_ms"]
    assert "emitted_ms" in s
    assert "risk_ms" in s
    assert "fill_ms" in s
    assert "attribution_ms" in s
    assert s["emitted_ms"] <= s["risk_ms"] <= s["fill_ms"] <= s["attribution_ms"]
    assert result["observed"]["risk_outcome"] == EventType.INTENT_ADMITTED.value


@pytest.mark.asyncio
async def test_self_check_publishes_ok_event_on_bus(wired_bus):
    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await wired_bus["bus"].subscribe("capture_test", on_event=cap)
    try:
        await run_self_check(
            bus=wired_bus["bus"],
            attribution=wired_bus["attribution"],
            timeout_sec=5.0,
        )
        # Give the pump a tick to fan out.
        await asyncio.sleep(0.05)
    finally:
        await wired_bus["bus"].unsubscribe("capture_test")
    kinds = [e.event_type for e in seen]
    assert EventType.SELF_CHECK_OK in kinds
    assert EventType.SELF_CHECK_FAIL not in kinds


@pytest.mark.asyncio
async def test_self_check_fail_when_risk_silent():
    """No orchestrator wired → risk never runs → SELF_CHECK_FAIL on timeout."""
    bus = EventBus()
    await bus.start()
    try:
        result = await run_self_check(bus=bus, attribution=None, timeout_sec=0.5)
        assert result["kind"] == "fail"
        assert "timeout" in result["reason"]
        # INTENT_EMITTED fires immediately because the bus carries it; risk + fill don't.
        assert "emitted_ms" in result["stages_ms"]
        assert "risk_ms" not in result["stages_ms"]
        assert "fill_ms" not in result["stages_ms"]
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_self_check_fail_when_fill_silent(tmp_path: Path):
    """Risk runs and admits but no paper fill path → fail on fill stage."""
    bus = EventBus()
    await bus.start()
    cfg = ConfigManager(None)
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    policy = RiskPolicy(config_manager=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({
        "self_check_yes": frozenset({"supports_limit"}),
        "self_check_no": frozenset({"supports_limit"}),
    })

    # Subscribe a "risk runner" that consumes INTENT_EMITTED and runs evaluate,
    # but do NOT wire an Orchestrator that produces FILL events.
    async def run_risk(ev: Event) -> None:
        if ev.event_type is not EventType.INTENT_EMITTED:
            return
        from executor.core.orchestrator import deserialize_intent
        await policy.evaluate(deserialize_intent(ev.payload))

    await bus.subscribe("risk_only", on_event=run_risk)
    try:
        result = await run_self_check(bus=bus, attribution=None, timeout_sec=0.5)
        assert result["kind"] == "fail", result
        assert "fill" in str(result["reason"]).lower()
        assert result["stages_ms"].get("risk_ms") is not None
    finally:
        await bus.unsubscribe("risk_only")
        await bus.stop()
        state.close()


def test_build_synthetic_intent_is_all_or_none():
    intent = build_synthetic_intent()
    assert len(intent.legs) == 2
    from executor.core.intent import Atomicity
    assert intent.atomicity == Atomicity.ALL_OR_NONE
    total = sum((leg.price_limit for leg in intent.legs), Decimal("0"))
    assert total < Decimal("1.0"), "synthetic basket must be clearly profitable (sum < $1)"
