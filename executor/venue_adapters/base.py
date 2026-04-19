"""
VenueAdapter — abstract base for every venue integration.

Contract: Decision 1 of /root/trading-wiki/specs/0d-executor.md.
16 operations, 6 error types, canonical price representation.

Every adapter implements all 16 ops. Capabilities the venue does not
offer raise NotSupportedError (e.g. Kalshi has no native stop orders).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from decimal import Decimal

from ..core.types import (
    Account,
    Capabilities,
    Fill,
    FillEvent,
    Market,
    MarketSpec,
    Orderbook,
    OrderbookEvent,
    OrderStatus,
    Position,
    PositionEvent,
    Side,
    TIF,
    TradeEvent,
)


class VenueAdapter(ABC):
    """
    Canonical adapter interface. All 16 ops abstract; subclasses implement
    or raise NotSupportedError. Prices flow as Decimal probabilities in
    [0.0000, 1.0000]; native quotes preserved in order/fill metadata.
    """

    #: Unique venue identifier, e.g. "kalshi", "ibkr", "polymarket".
    venue_id: str = ""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_markets(self) -> list[Market]: ...

    @abstractmethod
    async def get_market_spec(self, market_id: str) -> MarketSpec: ...

    @abstractmethod
    async def get_orderbook(self, market_id: str) -> Orderbook: ...

    # ------------------------------------------------------------------
    # Account / capabilities
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_account(self) -> Account: ...

    @abstractmethod
    async def get_capabilities(self) -> Capabilities: ...

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def place_limit(
        self,
        market_id: str,
        side: Side,
        size: Decimal,
        price: Decimal,
        tif: TIF,
    ) -> str:
        """Returns venue order_id."""

    @abstractmethod
    async def place_market(
        self,
        market_id: str,
        side: Side,
        size: Decimal,
    ) -> str: ...

    @abstractmethod
    async def place_stop(
        self,
        market_id: str,
        side: Side,
        size: Decimal,
        trigger_price: Decimal,
    ) -> str:
        """Raise NotSupportedError on venues without native stops (Kalshi, sportsbooks)."""

    @abstractmethod
    async def replace_order(
        self,
        order_id: str,
        new_price: Decimal | None,
        new_size: Decimal | None,
    ) -> str: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus: ...

    @abstractmethod
    async def get_fills(self, since_ts: int) -> list[Fill]: ...

    # ------------------------------------------------------------------
    # Subscriptions (async generators)
    # ------------------------------------------------------------------

    @abstractmethod
    def subscribe_orderbook(self, market_ids: Iterable[str]) -> AsyncIterator[OrderbookEvent]: ...

    @abstractmethod
    def subscribe_trades(self, market_ids: Iterable[str]) -> AsyncIterator[TradeEvent]: ...

    @abstractmethod
    def subscribe_fills(self) -> AsyncIterator[FillEvent]: ...

    @abstractmethod
    def subscribe_positions(self) -> AsyncIterator[PositionEvent]: ...

    # ------------------------------------------------------------------
    # Optional helpers that many adapters want (not in the 16-op contract)
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Not in the 16-op contract but risk state rebuild uses it.
        Default: raise; concrete adapters override.
        """
        raise NotImplementedError("get_positions must be implemented by subclass")
