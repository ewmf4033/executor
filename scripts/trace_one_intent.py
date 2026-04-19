"""
Trace one synthetic intent through all 13 gates and print each decision.

This script backs /root/executor/docs/gate_trace_example.md — re-run it when
gate logic changes to verify the documented trace still matches reality:

    .venv/bin/python scripts/trace_one_intent.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from decimal import Decimal
from pathlib import Path

from executor.core.event_bus import EventBus
from executor.core.intent import Atomicity, Intent, Leg
from executor.core.types import Orderbook, OrderbookLevel, Side
from executor.detectors.poisoning import PoisoningTracker, ZScoreDetector
from executor.risk import ConfigManager, KillSwitch, RiskPolicy, RiskState
from executor.risk.gates import (
    AdverseSelectionGate,
    ClipFloorGate,
    DailyLossGate,
    EventConcentrationGate,
    GlobalPortfolioGate,
    KillGate,
    LiquidityGate,
    MarketExposureGate,
    PerIntentDollarCapGate,
    PoisoningGate,
    StrategyAllocationGate,
    StructuralGate,
    VenueExposureGate,
    VenueHealthGate,
)
from executor.risk.exposure import intent_notional_dollars
from executor.risk.types import GateCtx, GateDecision


def make_orderbook(market_id: str) -> Orderbook:
    now = time.time_ns()
    return Orderbook(
        market_id=market_id,
        venue="kalshi",
        outcome_id="YES",
        bids=(OrderbookLevel(Decimal("0.54"), Decimal("60")),
              OrderbookLevel(Decimal("0.53"), Decimal("40")),
              OrderbookLevel(Decimal("0.52"), Decimal("30"))),
        asks=(OrderbookLevel(Decimal("0.56"), Decimal("80")),
              OrderbookLevel(Decimal("0.57"), Decimal("60")),
              OrderbookLevel(Decimal("0.58"), Decimal("50"))),
        ts_ns=now, received_ts_ns=now,
    )


async def main():
    tmp = tempfile.mkdtemp()
    state = RiskState(db_path=Path(tmp) / "rs.sqlite")
    await state.load()
    # Pre-load some exposure so market_exposure CLIPs this intent.
    state.add_exposure(venue="kalshi", market_id="KALSHI-PRES-2028-DEM",
                       outcome_id="YES", dollars=Decimal("300"),
                       event_id="PRES-2028")

    cfg = ConfigManager(path=None)
    bus = EventBus()
    await bus.start()
    events = []
    await bus.subscribe("sink", on_event=lambda e: events.append(e) or asyncio.sleep(0))

    policy = RiskPolicy(config_manager=cfg, state=state, publish=bus.publish)
    policy.set_venue_capabilities({"kalshi": {"supports_limit", "supports_replace"}})
    policy.set_market_universe([("kalshi", "KALSHI-PRES-2028-DEM")])
    policy.set_event_id_map({("kalshi", "KALSHI-PRES-2028-DEM"): "PRES-2028"})

    async def ob_provider(venue, market_id, outcome_id):
        return make_orderbook(market_id)
    policy.set_orderbook_provider(ob_provider)

    # Craft the intent: BUY 800 YES contracts at 0.56 on kalshi:KALSHI-PRES-2028-DEM
    now = time.time_ns()
    intent = Intent.single(
        strategy_id="YESNOCrossDetect",
        venue="kalshi",
        market_id="KALSHI-PRES-2028-DEM",
        outcome_id="YES",
        side=Side.BUY,
        target_exposure=800,
        price_limit=0.56,
        confidence=0.72,
        edge_estimate=0.04,
        time_horizon_sec=600,
        created_ts=now,
        expires_ts=now + 300_000_000_000,
        required_capabilities=("supports_limit",),
    )

    gates = [
        StructuralGate(), KillGate(), VenueHealthGate(), PoisoningGate(),
        AdverseSelectionGate(), PerIntentDollarCapGate(), LiquidityGate(),
        MarketExposureGate(), EventConcentrationGate(), VenueExposureGate(),
        GlobalPortfolioGate(), StrategyAllocationGate(), DailyLossGate(),
        ClipFloorGate(),
    ]

    ctx = GateCtx(original_intent=intent, current_intent=intent,
                  policy=policy, now_ns=time.time_ns())

    print(f"Intent: {intent.strategy_id}/{intent.intent_id}")
    print(f"  Leg: kalshi:KALSHI-PRES-2028-DEM BUY 800 YES @ 0.56")
    print(f"  Original notional: ${intent_notional_dollars(intent)}")
    print()

    for gate in gates:
        t0 = time.perf_counter()
        result = await gate.check(ctx)
        t_ms = (time.perf_counter() - t0) * 1000.0
        size = ctx.current_intent.legs[0].target_exposure
        print(f"Gate {gate.order:>4} {gate.name:<24} {result.decision.value:<8} "
              f"size={size:>6} ({t_ms:.3f} ms)  {result.reason}")

        if result.decision == GateDecision.CLIP:
            from dataclasses import replace
            new_legs = []
            for leg in ctx.current_intent.legs:
                new_size = result.new_leg_sizes.get(leg.leg_id, leg.target_exposure)
                new_legs.append(replace(leg, target_exposure=new_size))
            ctx.current_intent = replace(ctx.current_intent, legs=tuple(new_legs))
            ctx.gates_passed.append(gate.name)
            ctx.gate_timings_ms[gate.name] = t_ms
        elif result.decision == GateDecision.APPROVE:
            ctx.gates_passed.append(gate.name)
            ctx.gate_timings_ms[gate.name] = t_ms
        else:
            print(f"STOP — REJECTED at {gate.name}")
            break

    print()
    print(f"Final intent size: {ctx.current_intent.legs[0].target_exposure} contracts")
    print(f"Final notional: ${intent_notional_dollars(ctx.current_intent)}")
    print(f"Gates passed: {ctx.gates_passed}")

    await bus.stop()


if __name__ == "__main__":
    asyncio.run(main())
