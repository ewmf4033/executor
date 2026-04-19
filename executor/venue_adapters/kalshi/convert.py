"""
Kalshi <-> canonical conversions.

Per Decision 1 of /root/trading-wiki/specs/0d-executor.md:
  - Canonical price: Decimal in [0.0000, 1.0000].
  - outcome_id is "YES" or "NO" for binary markets.
  - Native cents (1..99) preserved on order/fill metadata.

Kalshi orderbook shape (REST GET /markets/{ticker}/orderbook):
    {"orderbook": {"yes": [[cents, size], ...], "no": [[cents, size], ...]}}
Both arrays are bid-side only:
    yes[i] = "I will pay <cents> for a YES share, in <size> contracts"
    no[i]  = "I will pay <cents> for a NO  share, in <size> contracts"
A YES ask is implied by a NO bid: ask_cents = 100 - no_bid_cents.
A NO  ask is implied by a YES bid: ask_cents = 100 - yes_bid_cents.

The arrays returned by Kalshi are sorted ascending by cents; "best bid" on the
YES side is the highest cents entry. We canonicalize internally to:
  bids: descending price (best first)
  asks: ascending price  (best first)

Outcome canonicalization:
  Kalshi binary markets always have YES and NO. The adapter exposes outcome_id
  as the literal string "YES" or "NO". yes_no_mapping in MarketSpec records
  the canonical -> venue-native mapping ({"YES": "yes", "NO": "no"}); Kalshi
  uses lowercase in API calls.
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from ...core.types import Orderbook, OrderbookLevel, Side


# Quantize all canonical prices to 4 decimal places (1 cent = 0.01, but reserve
# precision for venues that quote sub-cent like Polymarket).
_PRICE_QUANT = Decimal("0.0001")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Outcome canonicalization
# ---------------------------------------------------------------------------


CANONICAL_OUTCOMES: tuple[str, ...] = ("YES", "NO")
YES_NO_MAPPING: dict[str, str] = {"YES": "yes", "NO": "no"}


def canonicalize_outcome(raw: str) -> str:
    """
    Map a Kalshi-native outcome string ("yes"/"YES"/"Yes" ...) to canonical
    "YES" or "NO". Raises ValueError on anything else (Kalshi binary markets
    only have these two outcomes).
    """
    if raw is None:
        raise ValueError("outcome is None")
    u = raw.strip().upper()
    if u not in ("YES", "NO"):
        raise ValueError(f"unknown Kalshi outcome: {raw!r}")
    return u


def to_native_outcome(canonical: str) -> str:
    """Inverse of canonicalize_outcome — produce Kalshi's lowercase form."""
    c = canonicalize_outcome(canonical)
    return YES_NO_MAPPING[c]


# ---------------------------------------------------------------------------
# Cents <-> Decimal probability
# ---------------------------------------------------------------------------


def cents_to_prob(cents: int | str | Decimal) -> Decimal:
    """
    Convert Kalshi cents (1..99 typical, 0/100 reserved for settled) to a
    Decimal probability in [0, 1] quantized to 4dp.
    """
    c = Decimal(str(cents))
    if c < 0 or c > 100:
        raise ValueError(f"cents out of range: {cents}")
    return (c / _HUNDRED).quantize(_PRICE_QUANT)


def prob_to_cents(prob: Decimal | float | str) -> int:
    """
    Convert canonical probability -> integer cents 1..99 (Kalshi's tick).

    Refuses prices outside (0, 1) — Kalshi rejects 0 and 100 cent limit orders;
    we map that to InvalidPrice at the call site by re-raising ValueError.
    """
    p = Decimal(str(prob))
    if p <= 0 or p >= 1:
        raise ValueError(f"probability out of range for Kalshi cents: {p}")
    cents = int((p * _HUNDRED).to_integral_value(rounding="ROUND_HALF_EVEN"))
    if cents < 1 or cents > 99:
        raise ValueError(f"probability rounds to invalid cents: {p} -> {cents}")
    return cents


# ---------------------------------------------------------------------------
# Side / order action mapping
# ---------------------------------------------------------------------------


def side_to_action(side: Side) -> str:
    """Kalshi 'action' field: BUY -> 'buy', SELL -> 'sell'."""
    return "buy" if side == Side.BUY else "sell"


def action_to_side(action: str) -> Side:
    a = action.strip().lower()
    if a == "buy":
        return Side.BUY
    if a == "sell":
        return Side.SELL
    raise ValueError(f"unknown Kalshi action: {action!r}")


# ---------------------------------------------------------------------------
# Orderbook parsing
# ---------------------------------------------------------------------------


def _extract_bid_arrays(payload: dict[str, Any]) -> tuple[list, list, str]:
    """
    Normalize to (yes_bids, no_bids, scale) where scale is 'cents' or 'dollars'.

    Kalshi returns two shapes depending on endpoint/host:
      - {"orderbook": {"yes": [[cents, size], ...], "no": [[cents, size], ...]}}
        (integer cents; legacy shape still documented for trade-api)
      - {"orderbook_fp": {"yes_dollars": [["0.6200", "15.00"]], "no_dollars": [...]}}
        (dollar-formatted strings; elections host returns this)
    Both are bid-only per side.
    """
    book = payload.get("orderbook_fp") or payload.get("orderbook") or payload
    if not isinstance(book, dict):
        return [], [], "cents"
    if "yes_dollars" in book or "no_dollars" in book:
        return list(book.get("yes_dollars") or []), list(book.get("no_dollars") or []), "dollars"
    return list(book.get("yes") or []), list(book.get("no") or []), "cents"


def _level_price(raw_price: Any, scale: str) -> Decimal:
    """Convert a raw level price (int cents or string dollars) to canonical Decimal."""
    if scale == "dollars":
        return Decimal(str(raw_price)).quantize(_PRICE_QUANT)
    return cents_to_prob(raw_price)


def parse_orderbook(
    market_id: str,
    payload: dict[str, Any],
    *,
    outcome: str = "YES",
    venue_ts_ns: int | None = None,
) -> Orderbook:
    """
    Build a canonical Orderbook for the requested outcome (YES or NO).

    Handles both the cents-int shape (legacy /trade-api) and the dollar-string
    shape (orderbook_fp returned by api.elections.kalshi.com).
    """
    canon = canonicalize_outcome(outcome)
    yes_raw, no_raw, scale = _extract_bid_arrays(payload)

    own_bids = yes_raw if canon == "YES" else no_raw
    other_bids = no_raw if canon == "YES" else yes_raw

    def _bid_levels(rows: list) -> list[OrderbookLevel]:
        parsed: list[tuple[Decimal, Decimal]] = []
        for r in rows:
            try:
                price = _level_price(r[0], scale)
                size = Decimal(str(r[1]))
            except Exception:
                continue
            if size <= 0 or price <= 0 or price >= 1:
                continue
            parsed.append((price, size))
        parsed.sort(key=lambda ps: -ps[0])  # descending
        return [OrderbookLevel(price_prob=p, size=s) for p, s in parsed]

    def _ask_levels(other_rows: list) -> list[OrderbookLevel]:
        # Implied ask on this side = 1 - opposite-side bid. Lowest ask ↔ highest opposite bid.
        parsed: list[tuple[Decimal, Decimal]] = []
        for r in other_rows:
            try:
                other_price = _level_price(r[0], scale)
                size = Decimal(str(r[1]))
            except Exception:
                continue
            if size <= 0 or other_price <= 0 or other_price >= 1:
                continue
            ask = (_ONE - other_price).quantize(_PRICE_QUANT)
            parsed.append((ask, size))
        parsed.sort(key=lambda ps: ps[0])  # ascending
        return [OrderbookLevel(price_prob=p, size=s) for p, s in parsed]

    bid_levels = _bid_levels(own_bids)
    ask_levels = _ask_levels(other_bids)

    now_ns = time.time_ns()
    return Orderbook(
        market_id=market_id,
        venue="kalshi",
        outcome_id=canon,
        bids=tuple(bid_levels),
        asks=tuple(ask_levels),
        ts_ns=venue_ts_ns if venue_ts_ns is not None else now_ns,
        received_ts_ns=now_ns,
        native={"yes_raw": yes_raw, "no_raw": no_raw, "scale": scale},
    )
