"""
Core venue-adapter data types and the 6-item error taxonomy.

Per Decision 1 of /root/trading-wiki/specs/0d-executor.md:

- Canonical price: price_prob is Decimal in [0.0000, 1.0000] — probability of the
  outcome this order bets on. Native quote preserved in metadata.
- outcome_id: "YES"/"NO" for binary, leg-specific for multi-outcome markets.
- 6 exceptions: InsufficientFunds, InvalidPrice, MarketClosed, RateLimited,
  VenueDown, StaleQuote.
- Account shape: cash, unrealized_pnl, total_exposure, free_capital, currency,
  as_of_ts, native (raw venue blob).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TIF(str, Enum):
    """Time-in-force. Not all venues support all."""
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    DAY = "DAY"


class OrderState(str, Enum):
    PENDING = "PENDING"     # placed locally, not yet acked by venue
    OPEN = "OPEN"           # acked, live on venue
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# Error taxonomy — exactly 6 per spec.
# ---------------------------------------------------------------------------


class VenueError(Exception):
    """Base class for venue adapter errors."""


class InsufficientFunds(VenueError):
    pass


class InvalidPrice(VenueError):
    pass


class MarketClosed(VenueError):
    pass


class RateLimited(VenueError):
    def __init__(self, message: str = "", retry_after_sec: float | None = None):
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


class VenueDown(VenueError):
    pass


class StaleQuote(VenueError):
    pass


class NotSupportedError(VenueError):
    """Raised by adapters for capabilities the venue does not offer (e.g. stops on Kalshi)."""


# ---------------------------------------------------------------------------
# Market + orderbook
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Market:
    market_id: str
    venue: str
    event_id: str | None
    title: str
    status: str                  # OPEN, CLOSED, SETTLED, etc. (venue-normalized string)
    close_ts: int | None         # ns epoch
    outcomes: tuple[str, ...]    # canonical outcome ids; binary: ("YES", "NO")
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketSpec:
    market_id: str
    venue: str
    tick_size: Decimal            # in probability units
    lot_size: Decimal             # minimum contract increment
    min_notional: Decimal         # dollars
    fees_bps: Decimal             # round-trip fee estimate in bps of notional
    yes_no_mapping: dict[str, str]  # canonical -> venue-native outcome id
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderbookLevel:
    price_prob: Decimal
    size: Decimal


@dataclass(frozen=True, slots=True)
class Orderbook:
    market_id: str
    venue: str
    outcome_id: str
    bids: tuple[OrderbookLevel, ...]      # descending price
    asks: tuple[OrderbookLevel, ...]      # ascending price
    ts_ns: int                            # venue-reported timestamp
    received_ts_ns: int                   # local receive timestamp
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Account:
    cash: Decimal
    unrealized_pnl: Decimal
    total_exposure: Decimal
    free_capital: Decimal
    currency: str
    as_of_ts: int                  # ns epoch
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Position:
    market_id: str
    venue: str
    outcome_id: str
    size: Decimal                  # signed: positive = long this outcome
    avg_price_prob: Decimal
    unrealized_pnl: Decimal
    as_of_ts: int
    native: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capabilities — queried at strategy registration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Capabilities:
    venue: str
    supports_limit: bool
    supports_market: bool
    supports_stop: bool
    supports_replace: bool
    supports_orderbook_stream: bool
    supports_trade_stream: bool
    supports_fill_stream: bool
    supports_position_stream: bool
    min_tick: Decimal
    atomicity_scope: str           # "SINGLE", "MULTI_LEG_BEST_EFFORT", etc.
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orders + fills
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderStatus:
    order_id: str
    venue: str
    market_id: str
    outcome_id: str
    side: Side
    state: OrderState
    size: Decimal
    filled: Decimal
    price_prob: Decimal | None      # None for market orders
    tif: TIF | None
    placed_ts_ns: int
    last_update_ts_ns: int
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    venue: str
    market_id: str
    outcome_id: str
    side: Side
    size: Decimal
    price_prob: Decimal
    fee: Decimal
    ts_ns: int
    native_quote: dict[str, Any] = field(default_factory=dict)  # +150, 2.50, 0.62, raw implied, vig...
    native: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stream event envelopes — yielded by subscribe_* generators.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderbookEvent:
    orderbook: Orderbook


@dataclass(frozen=True, slots=True)
class TradeEvent:
    venue: str
    market_id: str
    outcome_id: str
    side: Side
    size: Decimal
    price_prob: Decimal
    ts_ns: int
    native: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FillEvent:
    fill: Fill


@dataclass(frozen=True, slots=True)
class PositionEvent:
    position: Position
