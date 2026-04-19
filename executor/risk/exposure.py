"""
Notional/exposure math for risk gates.

Kalshi binary contract: max payoff $1. Cost basis to open:
    BUY  YES @ p  -> risk = p    (you pay p, keep 1-p if NO)
    SELL YES @ p  -> risk = 1-p  (you collect p, owe 1 if YES hits)
Symmetric for NO leg. We collapse to a single formula using side:

    risk_per_contract = p   if BUY
    risk_per_contract = 1-p if SELL

leg notional = size * risk_per_contract
"""
from __future__ import annotations

from decimal import Decimal

from ..core.intent import BasketIntent, Leg
from ..core.types import Side


ONE = Decimal("1")
ZERO = Decimal("0")


def risk_per_contract(side: Side, price_prob: Decimal) -> Decimal:
    return price_prob if side == Side.BUY else (ONE - price_prob)


def leg_notional_dollars(leg: Leg) -> Decimal:
    return (leg.target_exposure * risk_per_contract(leg.side, leg.price_limit))


def leg_notional_dollars_with_size(leg: Leg, size: Decimal) -> Decimal:
    return size * risk_per_contract(leg.side, leg.price_limit)


def intent_notional_dollars(intent: BasketIntent) -> Decimal:
    return sum((leg_notional_dollars(leg) for leg in intent.legs), ZERO)
