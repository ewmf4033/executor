"""
14 risk gates.

Each Gate subclass has an async `check(ctx)` returning a GateResult. Gates are
stateless — they read policy + ctx but do not mutate shared state. The policy
is responsible for applying CLIPs (replacing ctx.current_intent) and for
recording clip history / emitting audit events.

Gate order (per Decision 3):
  1   Structural
  1.5 Fee                (Phase 4.15 — admission-time fee/edge protection)
  1.6 OrderPolicy        (Phase 4.15 — leg.metadata venue-write shape)
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
  8.5 DeadMan            (Phase 4.14b — operator availability)
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
# 1.5 Fee-aware executable-edge gate (Phase 4.15)
# ---------------------------------------------------------------------------


_BPS_DENOM = Decimal("10000")


def _fee_lookup(
    cfg: "Any",
    venue: str,
    market_id: str,
) -> tuple[Decimal, str]:
    """Resolve fee_bps for a leg with precedence:
      1. per_market_fee_bps["venue:market_id"]
      2. per_series_fee_bps[prefix]    (longest prefix match on market_id)
      3. default_fee_bps

    Returns (fee_bps, fee_source) where fee_source is one of
    "per_market" | "per_series:<prefix>" | "default".

    Encapsulated as a free function so Phase 5a can replace the body
    with a venue-native fee estimator without touching the gate.
    """
    market_key = f"{venue}:{market_id}"
    if market_key in cfg.per_market_fee_bps:
        return cfg.per_market_fee_bps[market_key], "per_market"
    # Longest matching prefix wins so "MKT-A-" beats "MKT-".
    best_prefix: str | None = None
    for prefix in cfg.per_series_fee_bps:
        if market_id.startswith(prefix):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix
    if best_prefix is not None:
        return cfg.per_series_fee_bps[best_prefix], f"per_series:{best_prefix}"
    return cfg.default_fee_bps, "default"


class FeeGate(Gate):
    """Gate 1.5 — Phase 4.15 fee-aware executable-edge gate.

    Computes (in dollars):
      gross_edge       = leg.edge_estimate * leg.target_exposure
      estimated_fee    = leg.target_exposure * fee_bps / 10000
      safety_margin    = leg.target_exposure * safety_margin_bps / 10000
      executable_edge  = gross_edge - estimated_fee - safety_margin

    Rejects the intent if any leg has executable_edge_dollars <= 0.

    Bypasses entirely when:
      - cfg.fee_gate.enabled is False, or
      - capital_mode is False AND apply_in_paper_mode is False.

    bps-of-notional is a Phase 4.15 placeholder. Phase 5a replaces
    `_fee_lookup` with a venue-native maker/taker estimator.
    """

    name = "fee_gate"
    order = "1.5"

    async def check(self, ctx: GateCtx) -> GateResult:
        cfg = ctx.policy.config.fee_gate
        if not cfg.enabled:
            return GateResult.approve(metadata={"bypassed": "disabled"})
        if not ctx.policy.config.capital_mode and not cfg.apply_in_paper_mode:
            return GateResult.approve(metadata={"bypassed": "paper_mode"})

        intent = ctx.current_intent
        leg_metas: list[dict[str, Any]] = []
        for leg in intent.legs:
            fee_bps, fee_source = _fee_lookup(cfg, leg.venue, leg.market_id)
            gross_edge_dollars = leg.edge_estimate * leg.target_exposure
            estimated_fee_dollars = (
                leg.target_exposure * fee_bps / _BPS_DENOM
            )
            safety_margin_dollars = (
                leg.target_exposure * cfg.safety_margin_bps / _BPS_DENOM
            )
            executable_edge_dollars = (
                gross_edge_dollars - estimated_fee_dollars - safety_margin_dollars
            )
            leg_meta = {
                "leg_id": leg.leg_id,
                "fee_bps": str(fee_bps),
                "fee_source": fee_source,
                "gross_edge_dollars": str(gross_edge_dollars),
                "estimated_fee_dollars": str(estimated_fee_dollars),
                "safety_margin_dollars": str(safety_margin_dollars),
                "executable_edge_dollars": str(executable_edge_dollars),
            }
            leg_metas.append(leg_meta)
            if executable_edge_dollars <= 0:
                return GateResult.reject(
                    f"fee_gate: executable_edge_dollars={executable_edge_dollars} "
                    f"<= 0 for leg {leg.leg_id} on {leg.venue}:{leg.market_id} "
                    f"(gross={gross_edge_dollars}, fee={estimated_fee_dollars}, "
                    f"margin={safety_margin_dollars}, fee_source={fee_source})",
                    metadata={
                        "kind": "fee_negative_edge",
                        "leg_id": leg.leg_id,
                        "fee_bps": str(fee_bps),
                        "fee_source": fee_source,
                        "gross_edge_dollars": str(gross_edge_dollars),
                        "estimated_fee_dollars": str(estimated_fee_dollars),
                        "safety_margin_dollars": str(safety_margin_dollars),
                        "executable_edge_dollars": str(executable_edge_dollars),
                    },
                )
        return GateResult.approve(metadata={"legs": leg_metas})


# ---------------------------------------------------------------------------
# 1.6 Order policy gate (Phase 4.15)
# ---------------------------------------------------------------------------


def _md_get(meta: dict[str, Any], *keys: str) -> Any:
    """Look up first present key in metadata. Returns None if none present."""
    for k in keys:
        if k in meta:
            return meta[k]
    return None


def _is_truthy_bool(v: Any) -> bool:
    """Strict-ish truthiness: True/"true"/"1"/1 are True; anything else False.
    Used so leg.metadata={"post_only": "true"} from a YAML/JSON path still
    counts as true."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    return False


class OrderPolicyGate(Gate):
    """Gate 1.6 — Phase 4.15 order-policy gate (leg.metadata bridge).

    Validates per-leg metadata for venue-write shape. Paper mode is
    permissive about ABSENCE of metadata but strict about PRESENCE of
    explicitly unsafe values. Capital mode adds required-key checks.

    Bridge keys recognized on leg.metadata:
      - "tif" or "time_in_force"  (case-insensitive)
      - "order_group_id"
      - "buy_max_cost"
      - "post_only"
      - "reduce_only"

    Does NOT auto-generate any field — strategies are responsible.
    """

    name = "order_policy"
    order = "1.6"

    async def check(self, ctx: GateCtx) -> GateResult:
        cfg = ctx.policy.config.order_policy
        if not cfg.enabled:
            return GateResult.approve(metadata={"bypassed": "disabled"})

        capital_mode = ctx.policy.config.capital_mode
        # paper-permissive: only enforce required-key checks when active.
        active = capital_mode or cfg.apply_in_paper_mode

        intent = ctx.current_intent
        for leg in intent.legs:
            meta = leg.metadata or {}

            # 1. time_in_force: default missing to IOC; reject explicit
            # unsafe values regardless of mode.
            raw_tif = _md_get(meta, "tif", "time_in_force")
            if raw_tif is None:
                tif_norm = "IOC"
                tif_explicit = False
            else:
                tif_norm = str(raw_tif).strip().upper()
                tif_explicit = True
            if tif_norm not in cfg.allowed_time_in_force:
                return GateResult.reject(
                    f"order_policy: tif={raw_tif!r} not in allowed "
                    f"{list(cfg.allowed_time_in_force)} for leg {leg.leg_id}",
                    metadata={
                        "kind": "tif_not_allowed",
                        "leg_id": leg.leg_id,
                        "tif": tif_norm,
                        "tif_explicit": tif_explicit,
                        "allowed": list(cfg.allowed_time_in_force),
                    },
                )

            # 2. post_only: reject if present-and-true and forbidden.
            if cfg.forbid_post_only and "post_only" in meta:
                if _is_truthy_bool(meta["post_only"]):
                    return GateResult.reject(
                        f"order_policy: post_only=true forbidden for leg {leg.leg_id}",
                        metadata={
                            "kind": "post_only_forbidden",
                            "leg_id": leg.leg_id,
                        },
                    )

            # 3. reduce_only: reject only if forbidden by config and present-and-true.
            if cfg.forbid_reduce_only and "reduce_only" in meta:
                if _is_truthy_bool(meta["reduce_only"]):
                    return GateResult.reject(
                        f"order_policy: reduce_only=true forbidden for leg {leg.leg_id}",
                        metadata={
                            "kind": "reduce_only_forbidden",
                            "leg_id": leg.leg_id,
                        },
                    )

            # 4. capital-mode required keys.
            if active and cfg.require_order_group_id_in_capital_mode:
                ogi = meta.get("order_group_id")
                if ogi is None or str(ogi).strip() == "":
                    return GateResult.reject(
                        f"order_policy: order_group_id required in capital mode "
                        f"for leg {leg.leg_id}",
                        metadata={
                            "kind": "order_group_id_missing",
                            "leg_id": leg.leg_id,
                        },
                    )

            if (
                active
                and cfg.require_buy_max_cost_for_buys_in_capital_mode
                and leg.side == Side.BUY
            ):
                bmc = meta.get("buy_max_cost")
                if bmc is None or str(bmc).strip() == "":
                    return GateResult.reject(
                        f"order_policy: buy_max_cost required for BUY leg "
                        f"{leg.leg_id} in capital mode",
                        metadata={
                            "kind": "buy_max_cost_missing",
                            "leg_id": leg.leg_id,
                        },
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


class DeadManGate(Gate):
    """Gate 8.5 — operator availability (dead-man).

    Phase 4.14b. GPT-5.5 architectural review #2 (2026-04-23) flagged that
    solo-operator trading had no availability precondition: if the operator
    is asleep, traveling, or unreachable, intents kept being admitted. This
    gate adds an explicit arm/disarm + periodic-heartbeat contract.

    Behavior matrix:
      - config.enabled is False  -> approve (paper default)
      - enabled, disarmed        -> reject (operator must arm to trade)
      - enabled, armed, fresh hb -> approve
      - enabled, armed, stale hb -> reject (hard cutoff, no grace)
    """

    name = "dead_man"
    order = "8.5"

    async def check(self, ctx: GateCtx) -> GateResult:
        cfg = ctx.policy.config.dead_man
        if not cfg.enabled:
            return GateResult.approve()
        store = ctx.policy.operator_liveness
        if store is None:
            # Enabled but no store wired: fail closed — configuration bug
            # should not silently admit intents.
            return GateResult.reject(
                "dead_man: enabled but operator_liveness store not wired"
            )
        snap = store.load()
        if not snap.armed:
            return GateResult.reject(
                "dead_man_disarmed: operator has not armed the dead-man; "
                "run executorctl arm or /arm before trading"
            )
        deadline_ns = (
            snap.last_heartbeat_ts_ns + snap.timeout_sec * 1_000_000_000
        )
        if ctx.now_ns > deadline_ns:
            stale_sec = (ctx.now_ns - deadline_ns) / 1e9
            # Phase 4.14d (Codex review): tag reject metadata so
            # RiskPolicy can emit a first-class DEAD_MAN_TRIPPED event
            # alongside the terminal GATE_REJECTED. DEAD_MAN_TRIPPED
            # exists for alerting to separate operator-availability
            # trips from generic gate rejections.
            return GateResult.reject(
                f"dead_man_stale: heartbeat stale by {stale_sec:.1f}s "
                f"(timeout={snap.timeout_sec}s, "
                f"last_hb_ns={snap.last_heartbeat_ts_ns})",
                metadata={
                    "kind": "dead_man_stale",
                    "last_heartbeat_ts_ns": int(snap.last_heartbeat_ts_ns),
                    "timeout_sec": int(snap.timeout_sec),
                    "deadline_ns": int(deadline_ns),
                    "now_ns": int(ctx.now_ns),
                    "stale_sec": round(stale_sec, 3),
                },
            )
        return GateResult.approve()


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
        FeeGate(),
        OrderPolicyGate(),
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
        DeadManGate(),
        ClipFloorGate(),
    ]
