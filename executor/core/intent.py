"""
Intent model — BasketIntent + Leg.

Per Decision 2 of /root/trading-wiki/specs/0d-executor.md:

- One intent type: BasketIntent. Singles are baskets of one leg.
- Atomicity: INDEPENDENT | ALL_OR_NONE. SEQUENTIAL deferred to v2.
- Factories: Intent.single(...) for one-leg, Intent.basket(...) for full form.
- intent_id is UUID v7 (time-sortable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable

from uuid6 import uuid7

from .types import Side


class Atomicity(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    ALL_OR_NONE = "ALL_OR_NONE"
    # SEQUENTIAL intentionally absent — deferred to v2 per spec.


@dataclass(frozen=True, slots=True)
class Leg:
    venue: str
    market_id: str
    outcome_id: str
    side: Side
    target_exposure: Decimal           # contracts
    price_limit: Decimal               # probability, [0, 1]
    confidence: Decimal                # strategy-reported, [0, 1]
    edge_estimate: Decimal             # in probability units
    time_horizon_sec: int
    required_capabilities: tuple[str, ...] = ()
    kelly_fraction_used: Decimal = Decimal("0")
    leg_id: str = field(default_factory=lambda: str(uuid7()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BasketIntent:
    intent_id: str
    strategy_id: str
    legs: tuple[Leg, ...]
    atomicity: Atomicity
    max_slippage_per_leg: Decimal          # in probability units
    basket_target_exposure: Decimal        # dollar-notional-level description (informational)
    created_ts: int                        # ns epoch
    expires_ts: int                        # ns epoch
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("BasketIntent must contain at least one leg")
        if self.expires_ts <= self.created_ts:
            raise ValueError("expires_ts must be after created_ts")


class Intent:
    """Factory namespace. Use Intent.single(...) / Intent.basket(...)."""

    @staticmethod
    def single(
        *,
        strategy_id: str,
        venue: str,
        market_id: str,
        outcome_id: str,
        side: Side,
        target_exposure: Decimal | int | float,
        price_limit: Decimal | float,
        confidence: Decimal | float,
        edge_estimate: Decimal | float,
        time_horizon_sec: int,
        created_ts: int,
        expires_ts: int,
        max_slippage: Decimal | float = Decimal("0.02"),
        basket_target_exposure: Decimal | float | None = None,
        required_capabilities: Iterable[str] = (),
        kelly_fraction_used: Decimal | float = Decimal("0"),
        leg_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BasketIntent:
        leg = Leg(
            venue=venue,
            market_id=market_id,
            outcome_id=outcome_id,
            side=side,
            target_exposure=Decimal(str(target_exposure)),
            price_limit=Decimal(str(price_limit)),
            confidence=Decimal(str(confidence)),
            edge_estimate=Decimal(str(edge_estimate)),
            time_horizon_sec=time_horizon_sec,
            required_capabilities=tuple(required_capabilities),
            kelly_fraction_used=Decimal(str(kelly_fraction_used)),
            metadata=dict(leg_metadata or {}),
        )
        bte = (
            Decimal(str(basket_target_exposure))
            if basket_target_exposure is not None
            else leg.target_exposure
        )
        return BasketIntent(
            intent_id=str(uuid7()),
            strategy_id=strategy_id,
            legs=(leg,),
            atomicity=Atomicity.INDEPENDENT,
            max_slippage_per_leg=Decimal(str(max_slippage)),
            basket_target_exposure=bte,
            created_ts=created_ts,
            expires_ts=expires_ts,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def basket(
        *,
        strategy_id: str,
        legs: Iterable[Leg],
        atomicity: Atomicity,
        max_slippage_per_leg: Decimal | float,
        basket_target_exposure: Decimal | float,
        created_ts: int,
        expires_ts: int,
        metadata: dict[str, Any] | None = None,
    ) -> BasketIntent:
        return BasketIntent(
            intent_id=str(uuid7()),
            strategy_id=strategy_id,
            legs=tuple(legs),
            atomicity=atomicity,
            max_slippage_per_leg=Decimal(str(max_slippage_per_leg)),
            basket_target_exposure=Decimal(str(basket_target_exposure)),
            created_ts=created_ts,
            expires_ts=expires_ts,
            metadata=dict(metadata or {}),
        )
