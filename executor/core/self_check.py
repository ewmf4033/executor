"""
Startup self-check — injects one synthetic BasketIntent and verifies every
stage of the pipeline fires within a short window.

Phase 4's burn-in produced 167 INTENT_EMITTED events but zero risk-gate or
fill events, and that went unnoticed because nothing was actually checking.
The self-check is the mechanism that prevents the same class of silent gap
from shipping again: if risk / fill / attribution don't fire, the daemon
exits non-zero and systemd surfaces the failure.

Verification (all must fire within ``timeout_sec`` of injection):
  (a) INTENT_EMITTED     — for the self-check intent_id
  (b) INTENT_ADMITTED or GATE_REJECTED — risk policy ran
  (c) FILL               — paper fill produced (only if admitted)
  (d) attribution row    — via AttributionTracker.get_record

On success publishes SELF_CHECK_OK with per-stage latencies.
On failure publishes SELF_CHECK_FAIL with the missing stage + partial
latencies, and caller typically exits non-zero.
"""
from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any

from ..attribution.tracker import AttributionTracker
from .event_bus import EventBus
from .events import Event, EventType, Source
from .intent import Atomicity, Intent
from .logging import get_logger
from .types import Side


log = get_logger("executor.self_check")


SELF_CHECK_STRATEGY_ID = "self_check"


def build_synthetic_intent(
    *,
    yes_venue: str = "self_check_yes",
    no_venue: str = "self_check_no",
    yes_market: str = "SCYES",
    no_market: str = "SCNO",
    yes_price: Decimal = Decimal("0.45"),
    no_price: Decimal = Decimal("0.45"),
    size: Decimal = Decimal("1"),
    now_ns: int | None = None,
):
    """A 2-leg, clearly-profitable ALL_OR_NONE basket: $0.45 YES + $0.45 NO = $0.90."""
    now_ns = now_ns or time.time_ns()
    from .intent import Leg, BasketIntent

    legs = (
        Leg(
            venue=yes_venue,
            market_id=yes_market,
            outcome_id="YES",
            side=Side.BUY,
            target_exposure=size,
            price_limit=yes_price,
            confidence=Decimal("0.99"),
            edge_estimate=Decimal("0.05"),
            time_horizon_sec=60,
            required_capabilities=(),
            kelly_fraction_used=Decimal("0.25"),
            metadata={"role": "YES", "self_check": True},
        ),
        Leg(
            venue=no_venue,
            market_id=no_market,
            outcome_id="NO",
            side=Side.BUY,
            target_exposure=size,
            price_limit=no_price,
            confidence=Decimal("0.99"),
            edge_estimate=Decimal("0.05"),
            time_horizon_sec=60,
            required_capabilities=(),
            kelly_fraction_used=Decimal("0.25"),
            metadata={"role": "NO", "self_check": True},
        ),
    )
    return BasketIntent(
        intent_id=f"self-check-{int(now_ns)}",
        strategy_id=SELF_CHECK_STRATEGY_ID,
        legs=legs,
        atomicity=Atomicity.ALL_OR_NONE,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=size,
        created_ts=now_ns,
        expires_ts=now_ns + 60_000_000_000,
        metadata={"self_check": True, "sum_ask": str(yes_price + no_price)},
    )


async def run_self_check(
    *,
    bus: EventBus,
    attribution: AttributionTracker | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    """
    Inject a synthetic intent and wait until all four stages fire.

    Returns a dict with the result (kind=ok/fail, stage latencies, reason).
    Also publishes SELF_CHECK_OK or SELF_CHECK_FAIL on the bus.
    """
    intent = build_synthetic_intent()
    stages: dict[str, float] = {}
    stage_events: dict[str, asyncio.Event] = {
        "emitted": asyncio.Event(),
        "risk": asyncio.Event(),
        "fill": asyncio.Event(),
    }
    observed: dict[str, Any] = {}
    t0 = time.perf_counter()

    async def _observer(ev: Event) -> None:
        if ev.intent_id != intent.intent_id:
            return
        if ev.event_type is EventType.INTENT_EMITTED and not stage_events["emitted"].is_set():
            stages["emitted_ms"] = (time.perf_counter() - t0) * 1000.0
            stage_events["emitted"].set()
        elif ev.event_type in (EventType.INTENT_ADMITTED, EventType.GATE_REJECTED) \
                and not stage_events["risk"].is_set():
            stages["risk_ms"] = (time.perf_counter() - t0) * 1000.0
            observed["risk_outcome"] = ev.event_type.value
            stage_events["risk"].set()
            if ev.event_type is EventType.GATE_REJECTED:
                # No fill will come; unblock fill waiter to short-circuit.
                stage_events["fill"].set()
        elif ev.event_type is EventType.FILL and not stage_events["fill"].is_set():
            stages["fill_ms"] = (time.perf_counter() - t0) * 1000.0
            stage_events["fill"].set()

    await bus.subscribe("self_check_observer", on_event=_observer, queue_maxsize=1024)
    try:
        # Inject as an INTENT_EMITTED event on the bus — same shape the strategy
        # would produce via Strategy.emit().
        from ..strategies.base import _serialize_intent

        injection_event = Event.make(
            EventType.INTENT_EMITTED,
            source=Source.strategy(SELF_CHECK_STRATEGY_ID),
            payload=_serialize_intent(intent),
            intent_id=intent.intent_id,
            strategy_id=SELF_CHECK_STRATEGY_ID,
        )
        await bus.publish(injection_event)

        # Wait for all three bus-observable stages.
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    stage_events["emitted"].wait(),
                    stage_events["risk"].wait(),
                    stage_events["fill"].wait(),
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            missing = [k for k, e in stage_events.items() if not e.is_set()]
            result = {
                "kind": "fail",
                "reason": f"timeout waiting on stages: {missing}",
                "stages_ms": dict(stages),
                "observed": dict(observed),
                "intent_id": intent.intent_id,
                "timeout_sec": timeout_sec,
            }
            await _emit_fail(bus, result)
            return result

        # (d) attribution — only when risk admitted and fill happened.
        attr_row = None
        if attribution is not None and observed.get("risk_outcome") == EventType.INTENT_ADMITTED.value:
            # Attribution writes synchronously in Orchestrator.on_fill; allow
            # a short grace window since our observer runs concurrently with it.
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                # Look up the first leg's expected fill by scanning the intent.
                for leg in intent.legs:
                    rows = attribution._conn.execute(  # noqa: SLF001 — internal read
                        "SELECT fill_id FROM attribution WHERE intent_id=? AND leg_id=? LIMIT 1",
                        (intent.intent_id, leg.leg_id),
                    ).fetchone()
                    if rows:
                        attr_row = rows[0]
                        break
                if attr_row:
                    break
                await asyncio.sleep(0.05)
            stages["attribution_ms"] = (time.perf_counter() - t0) * 1000.0
            if not attr_row:
                result = {
                    "kind": "fail",
                    "reason": "attribution row not found after fill",
                    "stages_ms": dict(stages),
                    "observed": dict(observed),
                    "intent_id": intent.intent_id,
                }
                await _emit_fail(bus, result)
                return result

        result = {
            "kind": "ok",
            "stages_ms": dict(stages),
            "observed": dict(observed),
            "intent_id": intent.intent_id,
        }
        await _emit_ok(bus, result)
        return result
    finally:
        try:
            await bus.unsubscribe("self_check_observer")
        except Exception:
            pass


async def _emit_ok(bus: EventBus, result: dict[str, Any]) -> None:
    await bus.publish(
        Event.make(
            EventType.SELF_CHECK_OK,
            source=Source.EXECUTOR,
            payload={k: v for k, v in result.items() if k != "kind"},
        )
    )
    log.info("self_check.ok", **{k: v for k, v in result.items() if k != "kind"})


async def _emit_fail(bus: EventBus, result: dict[str, Any]) -> None:
    await bus.publish(
        Event.make(
            EventType.SELF_CHECK_FAIL,
            source=Source.EXECUTOR,
            payload={k: v for k, v in result.items() if k != "kind"},
        )
    )
    log.error("self_check.fail", **{k: v for k, v in result.items() if k != "kind"})
