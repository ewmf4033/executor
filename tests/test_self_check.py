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
from executor.detectors.adverse_selection import NullAdverseSelectionDetector
from executor.risk.config import ConfigManager
from executor.risk.policy import RiskPolicy
from executor.risk.state import RiskState


def _test_policy(*, cfg, state, publish):
    """Create a test RiskPolicy wired for self-check tests."""
    p = RiskPolicy(
        config_manager=cfg,
        state=state,
        adverse_selection=NullAdverseSelectionDetector(),
        publish=publish,
    )
    return p


@pytest.fixture
async def wired_bus(tmp_path: Path):
    """Bus + orchestrator + risk + attribution, no strategy / adapter."""
    bus = EventBus()
    await bus.start()

    cfg = ConfigManager(None)
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    policy = _test_policy(cfg=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({
        "self_check_yes": frozenset({"supports_limit"}),
        "self_check_no": frozenset({"supports_limit"}),
    })
    policy.register_self_check_markets()
    policy.set_event_id_map({
        ("self_check_yes", "SCYES"): "SELF_CHECK_EVENT",
        ("self_check_no", "SCNO"): "SELF_CHECK_EVENT",
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
    policy = _test_policy(cfg=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({
        "self_check_yes": frozenset({"supports_limit"}),
        "self_check_no": frozenset({"supports_limit"}),
    })
    policy.register_self_check_markets()
    policy.set_event_id_map({
        ("self_check_yes", "SCYES"): "SELF_CHECK_EVENT",
        ("self_check_no", "SCNO"): "SELF_CHECK_EVENT",
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


# -------------------------------------------------------------------------
# Phase 4.6 — self-check must require INTENT_ADMITTED, not accept rejections.
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_check_ok_when_admittable(wired_bus):
    """Happy path: SELF_CHECK_OK fires with all four stage latencies present."""
    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await wired_bus["bus"].subscribe("cap", on_event=cap)
    try:
        result = await run_self_check(
            bus=wired_bus["bus"],
            attribution=wired_bus["attribution"],
            timeout_sec=5.0,
        )
        await asyncio.sleep(0.05)
    finally:
        await wired_bus["bus"].unsubscribe("cap")

    assert result["kind"] == "ok", result
    stages = result["stages_ms"]
    for name in ("emitted_ms", "risk_ms", "fill_ms", "attribution_ms"):
        assert name in stages, f"missing stage {name} in {stages}"
    assert result["observed"]["risk_outcome"] == EventType.INTENT_ADMITTED.value
    kinds = [e.event_type for e in seen]
    assert EventType.SELF_CHECK_OK in kinds
    assert EventType.SELF_CHECK_FAIL not in kinds


@pytest.mark.asyncio
async def test_self_check_fails_on_gate_reject(tmp_path: Path):
    """If any risk gate rejects the synthetic intent (e.g. structural gate with
    mismatched market_universe), self-check must FAIL with the gate name,
    not silently pass."""
    bus = EventBus()
    await bus.start()
    cfg = ConfigManager(None)
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    policy = _test_policy(cfg=cfg, state=state, publish=bus.publish)
    # Populate market_universe with something that excludes the self-check
    # markets — this makes the structural gate reject the synthetic intent.
    policy.set_market_universe([("some_other_venue", "SOMETHING_ELSE")])
    policy.set_venue_capabilities({
        "self_check_yes": frozenset({"supports_limit"}),
        "self_check_no": frozenset({"supports_limit"}),
    })
    orch = Orchestrator(bus=bus, policy=policy, attribution=None, paper_mode=True)
    await orch.start()

    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await bus.subscribe("cap", on_event=cap)
    try:
        result = await run_self_check(bus=bus, attribution=None, timeout_sec=1.5)
        await asyncio.sleep(0.05)
    finally:
        await bus.unsubscribe("cap")
        await orch.stop()
        await bus.stop()
        state.close()

    assert result["kind"] == "fail", result
    assert "misconfigured" in result["reason"]
    assert "structural" in result["reason"]
    kinds = [e.event_type for e in seen]
    assert EventType.SELF_CHECK_FAIL in kinds
    assert EventType.SELF_CHECK_OK not in kinds


@pytest.mark.asyncio
async def test_self_check_run_daemon_returns_nonzero_on_gate_reject(tmp_path: Path, monkeypatch):
    """run_daemon must exit non-zero when the self-check fails on a gate reject."""
    from executor.core import daemon as daemon_mod

    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.setenv("EXECUTOR_ALLOW_NO_ORDERBOOK", "true")

    # Replace register_self_check_markets with a version that deliberately
    # excludes the self-check markets, forcing the structural gate to reject.
    def _bad_register(self) -> None:
        self.market_universe = {("some_other_venue", "SOMETHING_ELSE")}

    monkeypatch.setattr(daemon_mod.RiskPolicy, "register_self_check_markets", _bad_register)

    rc = await daemon_mod.run_daemon(
        self_check_only=True,
        audit_dir=tmp_path / "audit",
        risk_state_db=tmp_path / "rstate.sqlite",
        kill_db=tmp_path / "kill.sqlite",
        attribution_db=tmp_path / "attr.sqlite",
        telemetry_port=0,
        enable_quote_feeder=False,
    )
    assert rc != 0


@pytest.mark.asyncio
async def test_self_check_fails_on_crash(tmp_path: Path):
    """If risk.evaluate raises, orchestrator emits ORCHESTRATOR_CRASH and risk
    stage never signals ADMITTED/REJECTED → self-check times out → FAIL."""
    bus = EventBus()
    await bus.start()

    class _CrashPolicy:
        async def evaluate(self, intent):  # noqa: ARG002
            raise RuntimeError("risk subsystem crashed")

    orch = Orchestrator(bus=bus, policy=_CrashPolicy(), attribution=None, paper_mode=True)  # type: ignore[arg-type]
    await orch.start()

    seen: list[Event] = []

    async def cap(ev: Event) -> None:
        seen.append(ev)

    await bus.subscribe("cap", on_event=cap)
    try:
        result = await run_self_check(bus=bus, attribution=None, timeout_sec=0.75)
        await asyncio.sleep(0.05)
    finally:
        await bus.unsubscribe("cap")
        await orch.stop()
        await bus.stop()

    assert result["kind"] == "fail", result
    # ORCHESTRATOR_CRASH must have been published by the orchestrator.
    crashes = [
        e for e in seen
        if e.event_type is EventType.ERROR
        and e.payload.get("kind") == "ORCHESTRATOR_CRASH"
    ]
    assert len(crashes) >= 1
    kinds = [e.event_type for e in seen]
    assert EventType.SELF_CHECK_FAIL in kinds
