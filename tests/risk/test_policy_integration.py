"""
Policy integration — three strategies emit intents with mixed outcomes
(approve, clip, reject). Verify:
  - GATE_REJECTED for the bad one
  - GATE_CLIPPED + INTENT_ADMITTED for the clipped one
  - INTENT_ADMITTED for the clean one
  - event ordering
  - gates_passed + gate_timings_ms populated
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from executor.core.events import EventType

from .conftest import make_intent, make_leg


pytestmark = pytest.mark.asyncio


async def test_three_strategies_mixed_outcomes(policy_with_bus):
    policy, bus, received = policy_with_bus

    # --- Strategy A: clean single intent — ADMITTED ---
    a = make_intent(strategy_id="A", legs=[make_leg(market_id="MA", size=20, price=0.55)])
    va = await policy.evaluate(a)

    # --- Strategy B: oversize — CLIPPED by per-intent cap, stays above floor, then ADMITTED ---
    # 800 contracts * 0.5 = $400 notional, cap $250 → clip to 500 (62.5% of original).
    b = make_intent(strategy_id="B", legs=[make_leg(market_id="MB", size=800, price=0.5)])
    vb = await policy.evaluate(b)

    # --- Strategy C: below-min-edge — REJECTED on structural ---
    c = make_intent(strategy_id="C", legs=[make_leg(market_id="MC", size=10, edge=0.0)])
    vc = await policy.evaluate(c)

    assert va.admitted is True and va.reject_gate is None
    # Phase 4.14b added DeadManGate (order 8.5); it bypasses when
    # cfg.dead_man.enabled is False (the default) so it is counted as a
    # passed gate. 14 -> 15.
    assert len(va.gates_passed) == 15
    assert "structural" in va.gate_timings_ms
    assert "clip_floor" in va.gate_timings_ms

    assert vb.admitted is True and vb.reject_gate is None
    # clipped to about $250 notional at 0.5 = 500 contracts
    final_size = vb.intent.legs[0].target_exposure
    assert final_size == Decimal("500")

    assert vc.admitted is False
    assert vc.reject_gate == "structural"

    import asyncio
    await asyncio.sleep(0.05)

    types = [e.event_type for e in received]
    # Ordered: A admitted, B clipped+admitted, C rejected
    assert types.count(EventType.INTENT_ADMITTED) == 2
    assert types.count(EventType.GATE_CLIPPED) == 1
    assert types.count(EventType.GATE_REJECTED) == 1

    # B: GATE_CLIPPED precedes INTENT_ADMITTED
    b_events = [e for e in received if e.strategy_id == "B"]
    assert [e.event_type for e in b_events] == [
        EventType.GATE_CLIPPED, EventType.INTENT_ADMITTED,
    ]
    # INTENT_ADMITTED carries gates_passed + gate_timings_ms.
    admitted_b = b_events[-1]
    assert "gates_passed" in admitted_b.payload
    assert "gate_timings_ms" in admitted_b.payload
    assert admitted_b.payload["final_dollars"] != admitted_b.payload["original_dollars"]


async def test_reject_on_clip_floor_after_heavy_clip(policy_with_bus):
    """
    Start with an oversized single intent. Pre-load per-intent cap would
    only take us to $250, but if the market ceiling is tight we could clip
    harder. Here we engineer clip_floor rejection:
    original = $550, clipped to $100 would be 18% < 50% floor.
    """
    policy, bus, received = policy_with_bus
    # Tight market ceiling $100, clip floor 0.5 from defaults.
    policy.state.add_exposure(
        venue="kalshi", market_id="MX", outcome_id="YES", dollars=Decimal("400"),
    )
    # Pre-load $400 under a default $500 ceiling — remaining $100.
    # New intent 1000 contracts * 0.55 = $550 notional → clipped to ~181 contracts
    # ($100 / 0.55 ≈ 181). 181/1000 = 18.1% < 50% floor → REJECT.
    intent = make_intent(
        strategy_id="Z",
        legs=[make_leg(market_id="MX", size=1000, price=0.55)],
    )
    v = await policy.evaluate(intent)
    assert v.admitted is False
    assert v.reject_gate == "clip_floor"


async def test_basket_multi_venue_venue_exposure_clip(policy_with_bus):
    """Basket across two venues; kalshi near venue ceiling via many sub-markets."""
    policy, bus, received = policy_with_bus
    # Spread $450 across 5 kalshi markets → venue total $2250 (under $2500 ceiling)
    # but close enough that the new intent's kalshi leg gets clipped by venue_exposure.
    for i in range(5):
        policy.state.add_exposure(
            venue="kalshi", market_id=f"MY{i}", outcome_id="YES", dollars=Decimal("450"),
        )
    policy.set_venue_capabilities({
        "kalshi": {"supports_limit"},
        "poly": {"supports_limit"},
    })
    legs = [
        make_leg(venue="kalshi", market_id="NEW-K", size=200, price=0.55),
        make_leg(venue="poly", market_id="NEW-P", size=200, price=0.55),
    ]
    intent = make_intent(strategy_id="P", legs=legs)
    v = await policy.evaluate(intent)
    # original total = 2 * (200 * 0.55) = $220, under per-intent cap.
    # kalshi venue: current 2250 + leg 110 = 2360 < 2500 → no clip at gate 6. ADMITTED clean.
    assert v.admitted is True
    # Both legs at full size.
    sizes = {l.venue: l.target_exposure for l in v.intent.legs}
    assert sizes["kalshi"] == Decimal("200")
    assert sizes["poly"] == Decimal("200")
