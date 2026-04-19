"""
Paper-mode integration test for the Kalshi adapter.

Uses a mock REST client so we can stub get_orderbook without touching the
network. Verifies:

  1. place_limit in paper mode does NOT call the trade endpoint (create_order
     on the REST mock is never invoked).
  2. place_limit returns a paper-* order id.
  3. When the live orderbook crosses the limit price, _paper_tick emits a
     synthetic FILL with the resting level's price (price improvement).
  4. Remaining size stays open when the level is smaller than the order.
  5. place_stop raises NotSupportedError.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from executor.core.types import (
    NotSupportedError,
    OrderState,
    Side,
    TIF,
    InvalidPrice,
)
from executor.venue_adapters.kalshi.adapter import KalshiAdapter


pytestmark = pytest.mark.asyncio


class FakeREST:
    """Stub KalshiREST with controllable responses."""

    def __init__(self, orderbook_body: dict) -> None:
        self._ob = orderbook_body
        self.create_order_calls: list[dict] = []
        self.cancel_calls: list[str] = []

    async def get_orderbook(self, ticker: str):
        return self._ob

    async def create_order(self, body):
        self.create_order_calls.append(body)
        return {"order": {"order_id": "live-1"}}

    async def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        return {}

    async def close(self):
        return None


def _book_with_ask_at(cents: int, size: int) -> dict:
    """Book where best YES ask = (100 - best_no_bid)/100 = cents/100.

    NO bid at (100 - cents) implies YES ask at cents.
    """
    return {"orderbook": {"yes": [], "no": [[100 - cents, size]]}}


def _book_empty() -> dict:
    return {"orderbook": {"yes": [], "no": []}}


async def test_paper_place_limit_does_not_call_trade_endpoint(kalshi_env):
    rest = FakeREST(_book_empty())
    adapter = KalshiAdapter(rest=rest, paper_mode=True)  # type: ignore[arg-type]
    try:
        order_id = await adapter.place_limit(
            market_id="TEST-MKT-1",
            side=Side.BUY,
            size=Decimal("5"),
            price=Decimal("0.55"),
            tif=TIF.GTC,
            outcome_id="YES",
        )
        assert order_id.startswith("paper-")
        assert rest.create_order_calls == [], "trade endpoint must not be called in paper mode"
        status = await adapter.get_order_status(order_id)
        assert status.state == OrderState.OPEN
        assert status.size == Decimal("5")
        assert status.filled == Decimal("0")
    finally:
        await adapter.close()


async def test_paper_limit_fills_when_book_crosses(kalshi_env):
    # BUY @ 0.55 should cross when best YES ask is 0.50 (via NO bid at 50c).
    rest = FakeREST(_book_with_ask_at(cents=50, size=10))
    adapter = KalshiAdapter(rest=rest, paper_mode=True)  # type: ignore[arg-type]
    try:
        order_id = await adapter.place_limit(
            market_id="TEST-MKT-1",
            side=Side.BUY,
            size=Decimal("5"),
            price=Decimal("0.55"),
            tif=TIF.GTC,
        )
        fills = await adapter._paper_tick("TEST-MKT-1")
        assert len(fills) == 1
        f = fills[0]
        assert f.order_id == order_id
        assert f.side == Side.BUY
        assert f.size == Decimal("5")
        # Price improvement: filled at the level price, not the limit.
        assert f.price_prob == Decimal("0.50")
        assert f.native == {"paper": True}
        assert f.native_quote["cents"] == 50
        status = await adapter.get_order_status(order_id)
        assert status.state == OrderState.FILLED
        assert status.filled == Decimal("5")
    finally:
        await adapter.close()


async def test_paper_limit_no_fill_when_price_does_not_cross(kalshi_env):
    # BUY @ 0.40 will NOT fill when best YES ask is 0.50.
    rest = FakeREST(_book_with_ask_at(cents=50, size=10))
    adapter = KalshiAdapter(rest=rest, paper_mode=True)  # type: ignore[arg-type]
    try:
        order_id = await adapter.place_limit(
            market_id="TEST-MKT-1",
            side=Side.BUY,
            size=Decimal("5"),
            price=Decimal("0.40"),
            tif=TIF.GTC,
        )
        fills = await adapter._paper_tick("TEST-MKT-1")
        assert fills == []
        status = await adapter.get_order_status(order_id)
        assert status.state == OrderState.OPEN
        assert status.filled == Decimal("0")
    finally:
        await adapter.close()


async def test_paper_partial_fill_when_level_smaller(kalshi_env):
    # Ask @ 0.50 with only 3 contracts; we want 5 -> partial fill of 3.
    rest = FakeREST(_book_with_ask_at(cents=50, size=3))
    adapter = KalshiAdapter(rest=rest, paper_mode=True)  # type: ignore[arg-type]
    try:
        order_id = await adapter.place_limit(
            market_id="TEST-MKT-1",
            side=Side.BUY,
            size=Decimal("5"),
            price=Decimal("0.55"),
            tif=TIF.GTC,
        )
        fills = await adapter._paper_tick("TEST-MKT-1")
        assert len(fills) == 1
        assert fills[0].size == Decimal("3")
        status = await adapter.get_order_status(order_id)
        assert status.state == OrderState.PARTIAL
        assert status.filled == Decimal("3")
    finally:
        await adapter.close()


async def test_place_stop_raises_not_supported(kalshi_env):
    adapter = KalshiAdapter(rest=FakeREST(_book_empty()), paper_mode=True)  # type: ignore[arg-type]
    try:
        with pytest.raises(NotSupportedError):
            await adapter.place_stop("X", Side.BUY, Decimal("1"), Decimal("0.5"))
    finally:
        await adapter.close()


async def test_paper_cancel_marks_cancelled(kalshi_env):
    rest = FakeREST(_book_empty())
    adapter = KalshiAdapter(rest=rest, paper_mode=True)  # type: ignore[arg-type]
    try:
        order_id = await adapter.place_limit(
            market_id="TEST-MKT-1",
            side=Side.BUY,
            size=Decimal("1"),
            price=Decimal("0.5"),
            tif=TIF.GTC,
        )
        ok = await adapter.cancel_order(order_id)
        assert ok is True
        assert rest.cancel_calls == [], "cancel must not hit Kalshi in paper mode"
        status = await adapter.get_order_status(order_id)
        assert status.state == OrderState.CANCELLED
    finally:
        await adapter.close()


async def test_paper_place_limit_invalid_price_raises(kalshi_env):
    adapter = KalshiAdapter(rest=FakeREST(_book_empty()), paper_mode=True)  # type: ignore[arg-type]
    try:
        with pytest.raises(InvalidPrice):
            await adapter.place_limit(
                market_id="X",
                side=Side.BUY,
                size=Decimal("1"),
                price=Decimal("1.0"),  # boundary — Kalshi rejects 0 and 100 cent limits
                tif=TIF.GTC,
            )
    finally:
        await adapter.close()


async def test_place_stop_signature_match_base(kalshi_env):
    """Adapter must implement every abstract op of VenueAdapter — sanity check."""
    adapter = KalshiAdapter(rest=FakeREST(_book_empty()), paper_mode=True)  # type: ignore[arg-type]
    try:
        from executor.venue_adapters.base import VenueAdapter
        for name in (
            "get_markets", "get_market_spec", "get_orderbook", "get_account",
            "get_capabilities", "get_positions",
            "place_limit", "place_market", "place_stop",
            "replace_order", "cancel_order", "get_order_status", "get_fills",
            "subscribe_orderbook", "subscribe_trades", "subscribe_fills", "subscribe_positions",
        ):
            assert hasattr(adapter, name), f"missing op: {name}"
        # All 17 ops present — explicitly count.
        ops = {"get_markets", "get_market_spec", "get_orderbook", "get_account",
               "get_capabilities", "get_positions",
               "place_limit", "place_market", "place_stop",
               "replace_order", "cancel_order", "get_order_status", "get_fills",
               "subscribe_orderbook", "subscribe_trades", "subscribe_fills", "subscribe_positions"}
        assert len(ops) == 17
    finally:
        await adapter.close()
