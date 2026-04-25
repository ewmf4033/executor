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


async def test_poisoning_gate_rejects_when_tracker_none_in_capital_mode(policy):
    """Phase 4.13 Finding #2: in capital mode, a missing poisoning tracker
    must fail-closed rather than silently approve."""
    from dataclasses import replace

    policy.poisoning = None
    policy._cfg_mgr._config = replace(policy._cfg_mgr._config, capital_mode=True)
    r = await PoisoningGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.REJECT
    assert "poisoning_unavailable_capital_mode" in r.reason


async def test_poisoning_gate_approves_when_tracker_none_in_paper_mode(policy):
    """Paper-mode behavior (capital_mode=False) preserved: missing tracker
    approves so tests that don't wire a tracker keep working."""
    policy.poisoning = None
    # Default RiskConfig has capital_mode=False.
    assert policy.config.capital_mode is False
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


# ---------------------------------------------------------------------------
# Phase 4.14b — DeadManGate (Gate 8.5).
# ---------------------------------------------------------------------------


async def _enable_dead_man(policy):
    """Flip dead_man.enabled=True on the policy's live config + wire a
    real OperatorLivenessStore backed by the policy's RiskState. Returns
    the store so the test can arm/disarm/heartbeat it directly."""
    from dataclasses import replace as _replace

    from executor.risk.config import DeadManCfg
    from executor.risk.state import OperatorLivenessStore

    cfg_mgr = policy._cfg_mgr
    new_cfg = _replace(
        cfg_mgr._config,
        dead_man=DeadManCfg(
            enabled=True,
            default_timeout_sec=600,
            min_timeout_sec=60,
            max_timeout_sec=3600,
        ),
    )
    cfg_mgr._config = new_cfg
    store = OperatorLivenessStore(policy.state.connection)
    policy.operator_liveness = store
    return store


async def test_dead_man_disabled_bypasses(policy):
    from executor.risk.gates import DeadManGate

    # Default cfg has dead_man.enabled=False — gate should approve
    # regardless of store state.
    intent = make_intent()
    r = await DeadManGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_dead_man_enabled_disarmed_rejects(policy):
    from executor.risk.gates import DeadManGate

    await _enable_dead_man(policy)
    intent = make_intent()
    r = await DeadManGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert "dead_man_disarmed" in r.reason


async def test_dead_man_enabled_armed_fresh_approves(policy):
    from executor.risk.gates import DeadManGate

    store = await _enable_dead_man(policy)
    now = time.time_ns()
    store.arm(timeout_sec=600, source="test", kill_mode="NONE", now_ns=now)
    intent = make_intent()
    ctx = GateCtx(
        original_intent=intent, current_intent=intent, policy=policy,
        now_ns=now + 1_000_000_000,  # +1s
    )
    r = await DeadManGate().check(ctx)
    assert r.decision == GateDecision.APPROVE


async def test_dead_man_enabled_armed_stale_rejects(policy):
    from executor.risk.gates import DeadManGate

    store = await _enable_dead_man(policy)
    now = time.time_ns()
    store.arm(timeout_sec=60, source="test", kill_mode="NONE", now_ns=now)
    intent = make_intent()
    # 120s after arm with 60s timeout => stale by ~60s
    ctx = GateCtx(
        original_intent=intent, current_intent=intent, policy=policy,
        now_ns=now + 120 * 1_000_000_000,
    )
    r = await DeadManGate().check(ctx)
    assert r.decision == GateDecision.REJECT
    assert "dead_man_stale" in r.reason


async def test_dead_man_enabled_armed_boundary_exact_timeout_rejects(policy):
    from executor.risk.gates import DeadManGate

    store = await _enable_dead_man(policy)
    now = time.time_ns()
    timeout_sec = 60
    store.arm(timeout_sec=timeout_sec, source="test", kill_mode="NONE", now_ns=now)
    intent = make_intent()
    # Exactly last_hb + timeout_sec*1e9 + 1ns => reject (spec: hard cutoff).
    exact_deadline_plus_one_ns = now + timeout_sec * 1_000_000_000 + 1
    ctx = GateCtx(
        original_intent=intent, current_intent=intent, policy=policy,
        now_ns=exact_deadline_plus_one_ns,
    )
    r = await DeadManGate().check(ctx)
    assert r.decision == GateDecision.REJECT
    assert "dead_man_stale" in r.reason

    # And at exactly the deadline (no extra ns), still approve — ">" not ">=".
    at_deadline_ctx = GateCtx(
        original_intent=intent, current_intent=intent, policy=policy,
        now_ns=now + timeout_sec * 1_000_000_000,
    )
    r2 = await DeadManGate().check(at_deadline_ctx)
    assert r2.decision == GateDecision.APPROVE


# ===========================================================================
# Phase 4.15 — FeeGate (1.5)
# ===========================================================================


from dataclasses import replace as _dc_replace

from executor.risk.config import FeeGateCfg, OrderPolicyCfg
from executor.risk.gates import FeeGate, OrderPolicyGate, _fee_lookup


def _set_cfg(policy, **kw):
    """Replace policy.config in-place with overrides applied.

    RiskPolicy reads cfg via property -> ConfigManager.config, so we mutate
    the manager's _config via dataclass.replace.
    """
    new = _dc_replace(policy.config, **kw)
    policy._cfg_mgr._config = new
    return new


async def test_fee_gate_disabled_approves(policy):
    _set_cfg(policy, fee_gate=FeeGateCfg(enabled=False))
    intent = make_intent()
    r = await FeeGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE
    assert r.metadata.get("bypassed") == "disabled"


async def test_fee_gate_paper_mode_default_bypasses(policy):
    # capital_mode defaults to False; apply_in_paper_mode defaults to False.
    _set_cfg(policy, fee_gate=FeeGateCfg(enabled=True, apply_in_paper_mode=False))
    r = await FeeGate().check(await _ctx(policy, make_intent()))
    assert r.decision == GateDecision.APPROVE
    assert r.metadata.get("bypassed") == "paper_mode"


async def test_fee_gate_active_rejects_negative_executable_edge(policy):
    # edge=0.001, size=10 => gross=0.01; fee=10bps of 10 = 0.01 => executable=0.
    leg = make_leg(edge=0.001, size=10)
    intent = make_intent(legs=[leg])
    _set_cfg(
        policy,
        fee_gate=FeeGateCfg(
            enabled=True, apply_in_paper_mode=True,
            default_fee_bps=Decimal("10"),
        ),
    )
    r = await FeeGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert "fee_gate" in r.reason
    assert r.metadata["kind"] == "fee_negative_edge"


async def test_fee_gate_active_approves_positive_executable_edge(policy):
    # edge=0.05, size=1000 => gross=50; fee=10bps = 1; safety=0; executable=49.
    leg = make_leg(edge=0.05, size=1000)
    intent = make_intent(legs=[leg])
    _set_cfg(
        policy,
        fee_gate=FeeGateCfg(
            enabled=True, apply_in_paper_mode=True,
            default_fee_bps=Decimal("10"),
        ),
    )
    r = await FeeGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE
    legs_meta = r.metadata["legs"]
    assert len(legs_meta) == 1
    lm = legs_meta[0]
    assert Decimal(lm["gross_edge_dollars"]) == Decimal("50.000")
    assert Decimal(lm["estimated_fee_dollars"]) == Decimal("1.0000")
    assert Decimal(lm["safety_margin_dollars"]) == Decimal("0")
    assert Decimal(lm["executable_edge_dollars"]) == Decimal("49.0000")
    assert lm["fee_source"] == "default"


async def test_fee_gate_unit_assertion(policy):
    """Spec-pinned assertion:
    edge_estimate=0.05, target_exposure=1000, fee_bps=10, safety=0
    => gross=50, fee=1, executable=49"""
    leg = make_leg(edge=0.05, size=1000)
    intent = make_intent(legs=[leg])
    _set_cfg(
        policy,
        fee_gate=FeeGateCfg(
            enabled=True, apply_in_paper_mode=True,
            default_fee_bps=Decimal("10"),
            safety_margin_bps=Decimal("0"),
        ),
    )
    r = await FeeGate().check(await _ctx(policy, intent))
    lm = r.metadata["legs"][0]
    assert Decimal(lm["gross_edge_dollars"]) == Decimal("50.000")
    assert Decimal(lm["estimated_fee_dollars"]) == Decimal("1.0000")
    assert Decimal(lm["executable_edge_dollars"]) == Decimal("49.0000")


async def test_fee_gate_per_market_override_wins(policy):
    leg = make_leg(market_id="MKT-1", edge=0.05, size=1000)
    intent = make_intent(legs=[leg])
    _set_cfg(
        policy,
        fee_gate=FeeGateCfg(
            enabled=True, apply_in_paper_mode=True,
            default_fee_bps=Decimal("100"),
            per_series_fee_bps={"MKT-": Decimal("50")},
            per_market_fee_bps={"kalshi:MKT-1": Decimal("7")},
        ),
    )
    r = await FeeGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE
    lm = r.metadata["legs"][0]
    assert lm["fee_bps"] == "7"
    assert lm["fee_source"] == "per_market"


async def test_fee_gate_per_series_override_wins_over_default(policy):
    leg = make_leg(market_id="MKT-1", edge=0.05, size=1000)
    intent = make_intent(legs=[leg])
    _set_cfg(
        policy,
        fee_gate=FeeGateCfg(
            enabled=True, apply_in_paper_mode=True,
            default_fee_bps=Decimal("100"),
            per_series_fee_bps={"MKT-": Decimal("3")},
        ),
    )
    r = await FeeGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE
    lm = r.metadata["legs"][0]
    assert lm["fee_bps"] == "3"
    assert lm["fee_source"] == "per_series:MKT-"


async def test_fee_gate_metadata_includes_all_fields(policy):
    leg = make_leg(edge=0.05, size=1000)
    intent = make_intent(legs=[leg])
    _set_cfg(
        policy,
        fee_gate=FeeGateCfg(
            enabled=True, apply_in_paper_mode=True,
            default_fee_bps=Decimal("5"),
            safety_margin_bps=Decimal("2"),
        ),
    )
    r = await FeeGate().check(await _ctx(policy, intent))
    lm = r.metadata["legs"][0]
    for key in (
        "fee_bps", "fee_source", "gross_edge_dollars",
        "estimated_fee_dollars", "safety_margin_dollars",
        "executable_edge_dollars", "leg_id",
    ):
        assert key in lm


# ===========================================================================
# Phase 4.15 — OrderPolicyGate (1.6)
# ===========================================================================


def _leg_with_meta(meta, **kw):
    leg = make_leg(**kw)
    return _dc_replace(leg, metadata=meta)


def _intent_with_leg(leg):
    """Build a BasketIntent directly so leg.metadata survives.

    The conftest make_intent helper routes through Intent.single, which
    drops Leg.metadata unless leg_metadata= is passed explicitly. This
    helper preserves it.
    """
    from executor.core.intent import Atomicity, BasketIntent
    from uuid6 import uuid7
    now = time.time_ns()
    return BasketIntent(
        intent_id=str(uuid7()),
        strategy_id="s1",
        legs=(leg,),
        atomicity=Atomicity.INDEPENDENT,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=leg.target_exposure,
        created_ts=now,
        expires_ts=now + 60 * 1_000_000_000,
    )


async def test_order_policy_disabled_approves(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg(enabled=False))
    intent = _intent_with_leg(_leg_with_meta({"tif": "GTC", "post_only": True}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_order_policy_paper_mode_missing_metadata_approves(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg())  # defaults
    intent = _intent_with_leg(_leg_with_meta({}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_order_policy_paper_mode_explicit_ioc_approves(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg())
    intent = _intent_with_leg(_leg_with_meta({"tif": "IOC"}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_order_policy_paper_mode_gtc_rejects(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg())
    intent = _intent_with_leg(_leg_with_meta({"tif": "GTC"}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert r.metadata["kind"] == "tif_not_allowed"


async def test_order_policy_paper_mode_post_only_rejects(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg())
    intent = _intent_with_leg(_leg_with_meta({"post_only": True}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert r.metadata["kind"] == "post_only_forbidden"


async def test_order_policy_paper_mode_reduce_only_allowed_when_not_forbidden(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg(forbid_reduce_only=False))
    intent = _intent_with_leg(_leg_with_meta({"reduce_only": True}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_order_policy_paper_mode_reduce_only_rejected_when_forbidden(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg(forbid_reduce_only=True))
    intent = _intent_with_leg(_leg_with_meta({"reduce_only": True}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert r.metadata["kind"] == "reduce_only_forbidden"


async def test_order_policy_capital_mode_missing_order_group_id_rejects(policy):
    _set_cfg(
        policy,
        capital_mode=True,
        order_policy=OrderPolicyCfg(),
    )
    intent = _intent_with_leg(_leg_with_meta({"tif": "IOC", "buy_max_cost": "100"}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert r.metadata["kind"] == "order_group_id_missing"


async def test_order_policy_capital_mode_buy_missing_buy_max_cost_rejects(policy):
    _set_cfg(
        policy,
        capital_mode=True,
        order_policy=OrderPolicyCfg(),
    )
    leg = _leg_with_meta(
        {"tif": "IOC", "order_group_id": "grp-1"},
        side=Side.BUY,
    )
    intent = _intent_with_leg(leg)
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.REJECT
    assert r.metadata["kind"] == "buy_max_cost_missing"


async def test_order_policy_capital_mode_buy_with_required_meta_approves(policy):
    _set_cfg(
        policy,
        capital_mode=True,
        order_policy=OrderPolicyCfg(),
    )
    leg = _leg_with_meta(
        {"tif": "IOC", "order_group_id": "grp-1", "buy_max_cost": "100.00"},
        side=Side.BUY,
    )
    intent = _intent_with_leg(leg)
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_order_policy_tif_lowercase_normalizes(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg())
    intent = _intent_with_leg(_leg_with_meta({"tif": "ioc"}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_order_policy_time_in_force_alias_recognized(policy):
    _set_cfg(policy, order_policy=OrderPolicyCfg())
    intent = _intent_with_leg(_leg_with_meta({"time_in_force": "fok"}))
    r = await OrderPolicyGate().check(await _ctx(policy, intent))
    assert r.decision == GateDecision.APPROVE


async def test_fee_lookup_longest_prefix_wins():
    cfg = FeeGateCfg(
        default_fee_bps=Decimal("1"),
        per_series_fee_bps={
            "MKT-": Decimal("5"),
            "MKT-A-": Decimal("9"),
        },
    )
    bps, src = _fee_lookup(cfg, "kalshi", "MKT-A-001")
    assert bps == Decimal("9")
    assert src == "per_series:MKT-A-"
