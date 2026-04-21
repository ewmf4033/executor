"""Per-gate unit tests."""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from executor.core.types import Side
from executor.detectors.poisoning import PoisoningTracker, ZScoreDetector
from executor.risk import KillScope
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
from executor.risk.types import GateCtx, GateDecision

from .conftest import make_intent, make_leg, mk_orderbook


pytestmark = pytest.mark.asyncio


async def _ctx(policy, intent):
    return GateCtx(
        original_intent=intent, current_intent=intent, policy=policy,
        now_ns=time.time_ns(),
    )


# --- 1 Structural --------------------------------------------------------


async def test_structural_rejects_expired(policy):
    intent = make_intent(expires_in_sec=1)  # valid now, expired "later"
    # Simulate evaluation at a time after expiry.
    ctx = GateCtx(
        original_intent=intent, current_intent=intent, policy=policy,
        now_ns=intent.expires_ts + 1,
    )
    r = await StructuralGate().check(ctx)
    assert r.decision == GateDecision.REJECT
    assert "expired" in r.reason


async def test_structural_rejects_low_edge(policy):
    leg = make_leg(edge=0.001)
    intent = make_intent(legs=[leg])
    r = await StructuralGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert "edge" in r.reason


async def test_structural_rejects_unknown_market(policy):
    policy.set_market_universe([("kalshi", "OTHER")])
    intent = make_intent()
    r = await StructuralGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert "market not found" in r.reason


async def test_structural_rejects_missing_capability(policy):
    policy.set_venue_capabilities({"kalshi": set()})
    leg = make_leg(caps=("supports_limit",))
    intent = make_intent(legs=[leg])
    r = await StructuralGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert "capabilities" in r.reason


async def test_structural_approves_valid_intent(policy):
    r = await StructuralGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE


# --- 2 Kill ---------------------------------------------------------------


async def test_kill_global_rejects(policy):
    policy.kill_switch.engage(KillScope.GLOBAL, (), reason="maintenance")
    r = await KillGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT
    assert "GLOBAL" in r.reason


async def test_kill_strategy_venue_rejects(policy):
    policy.kill_switch.engage(KillScope.STRATEGY_VENUE, ("s1", "kalshi"), reason="debug")
    r = await KillGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT


async def test_kill_inactive_strategy_approves(policy):
    policy.kill_switch.engage(KillScope.STRATEGY, ("other-s",), reason="x")
    r = await KillGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE


# --- 2.5 Venue health -----------------------------------------------------


async def test_venue_health_rejects_when_tripped(policy):
    # Record 2 incidents within 60s → trip.
    now = time.time_ns()
    policy.venue_health.record_incident("kalshi", now_ns=now)
    policy.venue_health.record_incident("kalshi", now_ns=now + 1_000_000_000)
    r = await VenueHealthGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT
    assert "kalshi" in r.reason


async def test_venue_health_approves_below_threshold(policy):
    policy.venue_health.record_incident("kalshi")
    r = await VenueHealthGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE


# --- 2.6 Poisoning --------------------------------------------------------


async def test_poisoning_gate_rejects_paused_market(policy):
    assert policy.poisoning is not None
    # Prime the detector with enough quiet samples for a single outlier to pop.
    det = ZScoreDetector(window_sec=3600, z_threshold=2.0, min_samples=20)
    policy.poisoning = PoisoningTracker(det, pause_sec=300)
    await policy.poisoning.observe("MKT-1", Decimal("0.500"))
    for i in range(25):
        await policy.poisoning.observe("MKT-1", Decimal("0.500") + Decimal("0.0001") * i)
    anomaly = await policy.poisoning.observe("MKT-1", Decimal("0.95"))
    assert anomaly is not None
    r = await PoisoningGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT
    assert "poisoning-paused" in r.reason


async def test_poisoning_gate_approves_no_pause(policy):
    r = await PoisoningGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE


# --- 3 Adverse selection --------------------------------------------------


async def test_adverse_selection_null_detector_approves(policy):
    r = await AdverseSelectionGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE


async def test_adverse_selection_flagged_rejects(policy):
    class PauseAll:
        def is_flagged(self, *, strategy_id, market_id): return True
        def is_venue_paused(self, venue): return True
    policy.adverse_selection = PauseAll()
    r = await AdverseSelectionGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT
    assert "venue kalshi paused" in r.reason


# --- 4 Per-intent cap -----------------------------------------------------


async def test_per_intent_cap_clips(policy):
    # cap is $250 default. 1000 contracts * 0.55 = $550. Should clip.
    intent = make_intent(legs=[make_leg(size=1000, price=0.55)])
    r = await PerIntentDollarCapGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP
    new = list(r.new_leg_sizes.values())[0]
    # 250 / 0.55 ≈ 454
    assert 400 <= int(new) <= 500


async def test_per_intent_cap_approves_under(policy):
    intent = make_intent(legs=[make_leg(size=100, price=0.55)])
    r = await PerIntentDollarCapGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


# --- 4.5 Liquidity --------------------------------------------------------


async def test_liquidity_clips_to_depth(policy):
    async def ob(venue, market, outcome):
        return mk_orderbook(market, asks=((0.60, 10), (0.61, 5), (0.62, 5)))  # depth=20
    policy.set_orderbook_provider(ob)
    intent = make_intent(legs=[make_leg(size=50, price=0.62)])
    r = await LiquidityGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP
    assert list(r.new_leg_sizes.values())[0] == Decimal("20")


async def test_liquidity_approves_within_depth(policy):
    async def ob(venue, market, outcome):
        return mk_orderbook(market, asks=((0.60, 100),))
    policy.set_orderbook_provider(ob)
    intent = make_intent(legs=[make_leg(size=10, price=0.62)])
    r = await LiquidityGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_liquidity_rejects_empty_book(policy):
    async def ob(venue, market, outcome):
        return mk_orderbook(market, asks=())
    policy.set_orderbook_provider(ob)
    intent = make_intent(legs=[make_leg(size=10, price=0.62)])
    r = await LiquidityGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT


# --- 5 Market exposure ----------------------------------------------------


async def test_market_exposure_clips(policy):
    # ceiling default $500. Pre-load $400 exposure. 200 * 0.55 = $110 proposed,
    # remaining = $100. new_contracts = 100/0.55 = 181.
    policy.state.add_exposure(
        venue="kalshi", market_id="MKT-1", outcome_id="YES", dollars=Decimal("400"))
    intent = make_intent(legs=[make_leg(size=200, price=0.55)])
    r = await MarketExposureGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP
    assert Decimal("150") <= list(r.new_leg_sizes.values())[0] <= Decimal("200")


async def test_market_exposure_rejects_ceiling_exceeded(policy):
    policy.state.add_exposure(
        venue="kalshi", market_id="MKT-1", outcome_id="YES", dollars=Decimal("500"))
    intent = make_intent(legs=[make_leg(size=10)])
    r = await MarketExposureGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT


# --- 5.5 Event concentration ---------------------------------------------


async def test_event_concentration_clips(policy):
    policy.set_event_id_map({("kalshi", "MKT-1"): "EVT-A", ("kalshi", "MKT-2"): "EVT-A"})
    # Pre-load $900 under EVT-A (ceiling $1000).
    policy.state.add_exposure(venue="kalshi", market_id="MKT-1", outcome_id="YES",
                              dollars=Decimal("900"), event_id="EVT-A")
    intent = make_intent(legs=[make_leg(market_id="MKT-2", size=500, price=0.55)])
    r = await EventConcentrationGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP


async def test_event_concentration_rejects_unmapped(policy):
    """Phase 4.7 F5: legs without event_id are now rejected (fail-closed)."""
    intent = make_intent(legs=[make_leg()])
    r = await EventConcentrationGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert "missing event_id" in r.reason


# --- 6 Venue exposure -----------------------------------------------------


async def test_venue_exposure_clips(policy):
    policy.state.add_exposure(venue="kalshi", market_id="X", outcome_id="YES",
                              dollars=Decimal("2400"))
    intent = make_intent(legs=[make_leg(market_id="MKT-1", size=1000, price=0.55)])
    r = await VenueExposureGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP


# --- 7 Global portfolio ---------------------------------------------------


async def test_global_portfolio_clips(policy):
    policy.state.add_exposure(venue="kalshi", market_id="A", outcome_id="YES",
                              dollars=Decimal("9900"))
    intent = make_intent(legs=[make_leg(size=500, price=0.55)])
    r = await GlobalPortfolioGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP


# --- 7.5 Strategy allocation ---------------------------------------------


async def test_strategy_allocation_clips(policy):
    policy.state.add_strategy_exposure("s1", Decimal("900"))
    intent = make_intent(legs=[make_leg(size=500, price=0.55)])
    r = await StrategyAllocationGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.CLIP


# --- 8 Daily loss ---------------------------------------------------------


async def test_daily_loss_rejects(policy):
    policy.state.record_pnl("s1", Decimal("-300"))
    r = await DailyLossGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT


async def test_daily_loss_approves(policy):
    policy.state.record_pnl("s1", Decimal("-50"))
    r = await DailyLossGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE


# --- 9 Clip floor ---------------------------------------------------------


async def test_clip_floor_rejects_over_clip(policy):
    orig = make_intent(legs=[make_leg(size=100, price=0.55)])
    clipped = make_intent(legs=[make_leg(size=30, price=0.55)])
    ctx = GateCtx(
        original_intent=orig, current_intent=clipped, policy=policy,
        now_ns=time.time_ns(),
    )
    r = await ClipFloorGate().check(ctx)
    assert r.decision == GateDecision.REJECT
    assert "RISK_CLIPPED" in r.reason


async def test_clip_floor_approves_within_floor(policy):
    orig = make_intent(legs=[make_leg(size=100, price=0.55)])
    clipped = make_intent(legs=[make_leg(size=70, price=0.55)])
    ctx = GateCtx(
        original_intent=orig, current_intent=clipped, policy=policy,
        now_ns=time.time_ns(),
    )
    r = await ClipFloorGate().check(ctx)
    assert r.decision == GateDecision.APPROVE
