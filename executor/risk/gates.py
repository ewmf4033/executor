"""
14 risk gates.

Each Gate subclass has an async `check(ctx)` returning a GateResult. Gates are
stateless — they read policy + ctx but do not mutate shared state. The policy
is responsible for applying CLIPs (replacing ctx.current_intent) and for
recording clip history / emitting audit events.

Gate order (per Decision 3):
  1   Structural
  2   KillSwitch
  2.5 VenueHealth
  2.6 Poisoning          (0g — extended from the spec's adverse-selection slot)
  3   AdverseSelection   (0e — venue-level pause since Phase 4.7)
  4   PerIntentDollarCap
  4.5 Liquidity
  5   MarketExposure
  5.5 EventConcentration
  6   VenueExposure
  7   GlobalPortfolio
  7.5 StrategyAllocation
  8   DailyLoss
  9   ClipFloor
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..core.intent import BasketIntent, Leg
from ..core.types import Side
from .exposure import (
    ONE,
    ZERO,
    intent_notional_dollars,
    leg_notional_dollars,
    risk_per_contract,
)
from .types import GateCtx, GateResult

if TYPE_CHECKING:
    from .policy import RiskPolicy


# ---------------------------------------------------------------------------
# Gate base
# ---------------------------------------------------------------------------


class Gate(ABC):
    name: str = "gate"
    order: str = ""   # "1", "2", "2.5", ...

    @abstractmethod
    async def check(self, ctx: GateCtx) -> GateResult: ...


# ---------------------------------------------------------------------------
# 1. Structural
# ---------------------------------------------------------------------------


class StructuralGate(Gate):
    """Gate 1 — basic sanity + market/capability existence.

    Phase 4.7 F2: when market_universe is None, this gate default-REJECTS
    with "market_universe not configured" rather than silently allowing
    every market. Tests that don't need a registered universe must flip
    RiskPolicy.set_allow_universe_bootstrap(True) — that path logs a
    warning and is never taken from DaemonService.
    """

    name = "structural"
    order = "1"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        if not intent.legs:
            return GateResult.reject("intent has no legs")
        if intent.expires_ts <= ctx.now_ns:
            return GateResult.reject("intent expired before admission")
        pol = ctx.policy
        cfg = pol.config.structural
        min_edge = cfg.per_strategy_min_edge.get(
            intent.strategy_id, cfg.min_edge_default
        )
        # Phase 4.7 F2: default-deny when no universe is registered
        # (unless the test-only bootstrap flag is explicitly set).
        if pol.market_universe is None and not pol._allow_universe_bootstrap:
            return GateResult.reject(
                "structural_gate: market_universe not configured"
            )
        for leg in intent.legs:
            # market existence
            if pol.market_universe is not None:
                if (leg.venue, leg.market_id) not in pol.market_universe:
                    return GateResult.reject(
                        f"market not found: {leg.venue}:{leg.market_id}"
                    )
            # capabilities
            caps = pol.venue_capabilities.get(leg.venue)
            if caps is not None:
                missing = [c for c in leg.required_capabilities if c not in caps]
                if missing:
                    return GateResult.reject(
                        f"missing capabilities on {leg.venue}: {missing}"
                    )
            if not (ZERO <= leg.price_limit <= ONE):
                return GateResult.reject(
                    f"price_limit out of [0,1]: {leg.price_limit}"
                )
            if leg.target_exposure <= 0:
                return GateResult.reject(
                    f"target_exposure must be positive, got {leg.target_exposure}"
                )
            if leg.edge_estimate < min_edge:
                return GateResult.reject(
                    f"edge_estimate {leg.edge_estimate} < min {min_edge} for "
                    f"{intent.strategy_id}"
                )
        return GateResult.approve()


# ---------------------------------------------------------------------------
# 2. Kill switch
# ---------------------------------------------------------------------------


class KillGate(Gate):
    name = "kill_switch"
    order = "2"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        ks = ctx.policy.kill_switch
        # Evaluate against every venue in the basket. Any scope => reject.
        for leg in intent.legs:
            killed, reason = ks.is_killed(
                strategy_id=intent.strategy_id, venue=leg.venue
            )
            if killed:
                return GateResult.reject(f"kill_switch: {reason}")
        return GateResult.approve()


# ---------------------------------------------------------------------------
# 2.5 Venue health
# ---------------------------------------------------------------------------


class VenueHealthGate(Gate):
    name = "venue_health"
    order = "2.5"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        vh = ctx.policy.venue_health
        for leg in intent.legs:
            paused, until_ns = vh.is_paused(leg.venue, now_ns=ctx.now_ns)
            if paused:
                return GateResult.reject(
                    f"venue {leg.venue} auto-paused until ts_ns={until_ns}"
                )
        return GateResult.approve()


# ---------------------------------------------------------------------------
# 2.6 Poisoning (0g)
# ---------------------------------------------------------------------------


class PoisoningGate(Gate):
    name = "poisoning"
    order = "2.6"

    async def check(self, ctx: GateCtx) -> GateResult:
        tracker = ctx.policy.poisoning
        if tracker is None:
            # Phase 4.13: fail-closed in capital mode. Paper mode preserves
            # the pre-existing approve-on-None behavior so tests that don't
            # wire a tracker keep working.
            if ctx.policy.config.capital_mode:
                return GateResult.reject(
                    "poisoning_unavailable_capital_mode: "
                    "poisoning tracker not configured while capital_mode=True"
                )
            return GateResult.approve()
        intent = ctx.current_intent
        for leg in intent.legs:
            paused, until_ns, reason = tracker.is_paused(leg.market_id, now_ns=ctx.now_ns)
            if paused:
                return GateResult.reject(
                    f"market {leg.market_id} poisoning-paused ({reason}) until ts_ns={until_ns}"
                )
        return GateResult.approve()


# ---------------------------------------------------------------------------
# 3. Adverse selection (0e stub)
# ---------------------------------------------------------------------------


class AdverseSelectionGate(Gate):
    """Gate 3 — venue-level adverse-selection pause.

    Phase 4.7 F9: checks is_venue_paused(venue), not is_flagged(market).
    Rationale: adverse selection is typically driven by venue-side
    latency or information asymmetry, so a tripped signal should pause
    the whole venue, not just one market that tripped the metric.
    """

    name = "adverse_selection"
    order = "3"

    async def check(self, ctx: GateCtx) -> GateResult:
        det = ctx.policy.adverse_selection
        if det is None:
            return GateResult.approve()
        intent = ctx.current_intent
        for leg in intent.legs:
            if det.is_venue_paused(leg.venue):
                return GateResult.reject(
                    f"adverse_selection: venue {leg.venue} paused"
                )
        return GateResult.approve()


# ---------------------------------------------------------------------------
# 4. Per-intent dollar cap
# ---------------------------------------------------------------------------


class PerIntentDollarCapGate(Gate):
    name = "per_intent_dollar_cap"
    order = "4"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        cap = ctx.policy.config.per_intent.max_intent_dollars
        total = intent_notional_dollars(intent)
        if total <= cap:
            return GateResult.approve(metadata={"total_dollars": str(total)})
        # Clip proportionally.
        return GateResult.clip(
            _clip_proportional(intent, cap, total),
            reason=f"intent notional {total} > cap {cap}, clipping proportionally",
            metadata={"original_dollars": str(total), "cap_dollars": str(cap)},
        )


# ---------------------------------------------------------------------------
# 4.5 Liquidity
# ---------------------------------------------------------------------------


class LiquidityGate(Gate):
    name = "liquidity"
    order = "4.5"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        cfg = ctx.policy.config.liquidity
        ob_provider = ctx.policy.orderbook_provider
        if ob_provider is None:
            return GateResult.approve()
        new_sizes: dict[str, Decimal] = {}
        any_clip = False
        for leg in intent.legs:
            ob = await ob_provider(leg.venue, leg.market_id, leg.outcome_id)
            if ob is None:
                continue
            levels = _levels_for_side(ob, leg.side)[: cfg.depth_levels]
            depth = sum((lvl.size for lvl in levels), Decimal("0"))
            if depth <= 0:
                return GateResult.reject(
                    f"liquidity: no depth on {leg.venue}:{leg.market_id}:{leg.side.value}"
                )
            if leg.target_exposure <= depth:
                new_sizes[leg.leg_id] = leg.target_exposure
                continue
            # Clip to depth. If depth < min_remainder, reject leg → reject intent.
            if depth < cfg.min_remainder_contracts:
                return GateResult.reject(
                    f"liquidity: depth {depth} < min_remainder "
                    f"{cfg.min_remainder_contracts} on {leg.market_id}"
                )
            any_clip = True
            new_sizes[leg.leg_id] = depth
        if not any_clip:
            return GateResult.approve()
        return GateResult.clip(
            new_sizes,
            reason="liquidity: clipped to top-N depth",
            metadata={"depth_levels": cfg.depth_levels},
        )


def _levels_for_side(ob: Any, side: Side) -> tuple:
    # BUY crosses asks, SELL crosses bids.
    return ob.asks if side == Side.BUY else ob.bids


# ---------------------------------------------------------------------------
# 5. Market-level exposure ceiling
# ---------------------------------------------------------------------------


class MarketExposureGate(Gate):
    name = "market_exposure"
    order = "5"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        cfg = ctx.policy.config.market_exposure
        state = ctx.policy.state
        new_sizes: dict[str, Decimal] = {}
        any_clip = False
        for leg in intent.legs:
            key = f"{leg.venue}:{leg.market_id}:{leg.outcome_id}"
            ceiling = cfg.per_key.get(key, cfg.default_ceiling_dollars)
            current = state.exposure(leg.venue, leg.market_id, leg.outcome_id)
            added = leg_notional_dollars(leg)
            if current + added <= ceiling:
                new_sizes[leg.leg_id] = leg.target_exposure
                continue
            remaining = ceiling - current
            if remaining <= 0:
                return GateResult.reject(
                    f"market {key} exposure {current} >= ceiling {ceiling}"
                )
            rpc = risk_per_contract(leg.side, leg.price_limit)
            if rpc <= 0:
                return GateResult.reject(f"market_exposure: risk_per_contract 0 on {key}")
            new_contracts = (remaining / rpc).to_integral_value(rounding="ROUND_DOWN")
            if new_contracts <= 0:
                return GateResult.reject(
                    f"market {key} remaining {remaining} < 1 contract cost"
                )
            any_clip = True
            new_sizes[leg.leg_id] = new_contracts
        if not any_clip:
            return GateResult.approve()
        return GateResult.clip(new_sizes, reason="market-level exposure ceiling")


# ---------------------------------------------------------------------------
# 5.5 Event concentration
# ---------------------------------------------------------------------------


class EventConcentrationGate(Gate):
    """Gate 5.5 — event-level concentration ceiling.

    Phase 4.7 F5: fails closed when any leg has no event_id. Previously
    the gate silently skipped unmapped legs, which meant missing event
    metadata disabled the check entirely. Now: every leg must have an
    event_id registered via policy.set_event_id_map() or the intent is
    rejected.
    """

    name = "event_concentration"
    order = "5.5"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        pol = ctx.policy
        cfg = pol.config.event_concentration
        # Group legs by event_id; fail closed on any leg without one.
        by_event: dict[str, list[Leg]] = {}
        for leg in intent.legs:
            event_id = pol.event_id_for(leg.venue, leg.market_id)
            if event_id is None:
                return GateResult.reject(
                    f"event_concentration: leg {leg.leg_id} "
                    f"({leg.venue}:{leg.market_id}) missing event_id "
                    f"(required for concentration check)"
                )
            by_event.setdefault(event_id, []).append(leg)
        new_sizes: dict[str, Decimal] = {}
        any_clip = False
        for event_id, legs in by_event.items():
            ceiling = cfg.per_key.get(event_id, cfg.default_ceiling_dollars)
            current = pol.state.exposure_by_event(event_id)
            added = sum((leg_notional_dollars(l) for l in legs), ZERO)
            if current + added <= ceiling:
                continue
            remaining = ceiling - current
            if remaining <= 0:
                return GateResult.reject(
                    f"event {event_id} exposure {current} >= ceiling {ceiling}"
                )
            # Proportional clip across this event's legs.
            ratio = remaining / added
            for leg in legs:
                new_size = (leg.target_exposure * ratio).to_integral_value(rounding="ROUND_DOWN")
                if new_size <= 0:
                    return GateResult.reject(
                        f"event {event_id} leg {leg.leg_id} clipped to 0"
                    )
                new_sizes[leg.leg_id] = new_size
            any_clip = True
        if not any_clip:
            return GateResult.approve()
        # Preserve unaffected legs at their current size.
        for leg in intent.legs:
            new_sizes.setdefault(leg.leg_id, leg.target_exposure)
        return GateResult.clip(new_sizes, reason="event_concentration ceiling")


# ---------------------------------------------------------------------------
# 6. Venue exposure ceiling
# ---------------------------------------------------------------------------


class VenueExposureGate(Gate):
    name = "venue_exposure"
    order = "6"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        pol = ctx.policy
        cfg = pol.config.venue_exposure
        by_venue: dict[str, list[Leg]] = {}
        for leg in intent.legs:
            by_venue.setdefault(leg.venue, []).append(leg)
        new_sizes: dict[str, Decimal] = {}
        any_clip = False
        for venue, legs in by_venue.items():
            ceiling = cfg.per_key.get(venue, cfg.default_ceiling_dollars)
            current = pol.state.exposure_by_venue(venue)
            added = sum((leg_notional_dollars(l) for l in legs), ZERO)
            if current + added <= ceiling:
                continue
            remaining = ceiling - current
            if remaining <= 0:
                return GateResult.reject(
                    f"venue {venue} exposure {current} >= ceiling {ceiling}"
                )
            ratio = remaining / added
            for leg in legs:
                ns = (leg.target_exposure * ratio).to_integral_value(rounding="ROUND_DOWN")
                if ns <= 0:
                    return GateResult.reject(f"venue {venue} leg clipped to 0")
                new_sizes[leg.leg_id] = ns
            any_clip = True
        if not any_clip:
            return GateResult.approve()
        for leg in intent.legs:
            new_sizes.setdefault(leg.leg_id, leg.target_exposure)
        return GateResult.clip(new_sizes, reason="venue_exposure ceiling")


# ---------------------------------------------------------------------------
# 7. Global portfolio ceiling
# ---------------------------------------------------------------------------


class GlobalPortfolioGate(Gate):
    name = "global_portfolio"
    order = "7"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        pol = ctx.policy
        ceiling = pol.config.global_portfolio_dollars
        current = pol.state.total_exposure()
        added = intent_notional_dollars(intent)
        if current + added <= ceiling:
            return GateResult.approve()
        remaining = ceiling - current
        if remaining <= 0:
            return GateResult.reject(
                f"global portfolio {current} >= ceiling {ceiling}"
            )
        return GateResult.clip(
            _clip_proportional(intent, remaining, added),
            reason="global_portfolio ceiling",
        )


# ---------------------------------------------------------------------------
# 7.5 Strategy allocation
# ---------------------------------------------------------------------------


class StrategyAllocationGate(Gate):
    name = "strategy_allocation"
    order = "7.5"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        pol = ctx.policy
        cfg = pol.config.strategy_allocation
        ceiling = cfg.per_key.get(intent.strategy_id, cfg.default_ceiling_dollars)
        current = pol.state.strategy_exposure(intent.strategy_id)
        added = intent_notional_dollars(intent)
        if current + added <= ceiling:
            return GateResult.approve()
        remaining = ceiling - current
        if remaining <= 0:
            return GateResult.reject(
                f"strategy {intent.strategy_id} allocation {current} >= {ceiling}"
            )
        return GateResult.clip(
            _clip_proportional(intent, remaining, added),
            reason="strategy_allocation ceiling",
        )


# ---------------------------------------------------------------------------
# 8. Daily loss limit
# ---------------------------------------------------------------------------


class DailyLossGate(Gate):
    name = "daily_loss"
    order = "8"

    async def check(self, ctx: GateCtx) -> GateResult:
        intent = ctx.current_intent
        pol = ctx.policy
        cfg = pol.config.daily_loss
        limit = cfg.per_strategy.get(
            intent.strategy_id, cfg.default_max_loss_dollars
        )
        pol.state.reset_if_new_day(now_ns=ctx.now_ns)
        pnl = pol.state.daily_pnl(intent.strategy_id, now_ns=ctx.now_ns)
        if pnl < -limit:
            return GateResult.reject(
                f"daily loss {pnl} < -{limit} for strategy {intent.strategy_id}"
            )
        return GateResult.approve(metadata={"daily_pnl": str(pnl), "limit": str(limit)})


# ---------------------------------------------------------------------------
# 9. Clip floor — last gate
# ---------------------------------------------------------------------------


class ClipFloorGate(Gate):
    name = "clip_floor"
    order = "9"

    async def check(self, ctx: GateCtx) -> GateResult:
        original = intent_notional_dollars(ctx.original_intent)
        final = intent_notional_dollars(ctx.current_intent)
        if original <= 0:
            return GateResult.approve()
        ratio = final / original
        floor = ctx.policy.config.clip_floor.min_final_ratio
        if ratio < floor:
            return GateResult.reject(
                f"RISK_CLIPPED: final/original {ratio:.3f} < {floor}"
            )
        return GateResult.approve(metadata={"final_ratio": str(ratio)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip_proportional(
    intent: BasketIntent, cap_dollars: Decimal, total_dollars: Decimal
) -> dict[str, Decimal]:
    if total_dollars <= 0:
        return {leg.leg_id: leg.target_exposure for leg in intent.legs}
    ratio = cap_dollars / total_dollars
    out: dict[str, Decimal] = {}
    for leg in intent.legs:
        new = (leg.target_exposure * ratio).to_integral_value(rounding="ROUND_DOWN")
        if new <= 0:
            new = Decimal("0")
        out[leg.leg_id] = new
    return out


def default_gate_chain() -> list[Gate]:
    return [
        StructuralGate(),
        KillGate(),
        VenueHealthGate(),
        PoisoningGate(),
        AdverseSelectionGate(),
        PerIntentDollarCapGate(),
        LiquidityGate(),
        MarketExposureGate(),
        EventConcentrationGate(),
        VenueExposureGate(),
        GlobalPortfolioGate(),
        StrategyAllocationGate(),
        DailyLossGate(),
        ClipFloorGate(),
    ]
