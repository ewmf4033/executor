"""
Live, read-only integration tests against api.elections.kalshi.com.

Guarded on /root/kalshi_sports.key existing. These tests use public endpoints
(/markets, /markets/{ticker}/orderbook) so auth is not strictly required, but
the adapter wires auth anyway.

These tests make real network calls. They are robust to Kalshi returning an
empty market list (test skips) but fail loudly if the response shape drifts.
"""
from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.types import RateLimited
from executor.venue_adapters.kalshi.adapter import KalshiAdapter


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not Path("/root/kalshi_sports.key").exists(),
        reason="Kalshi private key not present — live test skipped",
    ),
]


async def test_live_get_markets_returns_open_markets(kalshi_env):
    adapter = KalshiAdapter(paper_mode=True)
    try:
        try:
            markets = await adapter.get_markets()
        except RateLimited:
            pytest.skip("Kalshi rate-limited this run")
        assert len(markets) > 0, "expected at least one open Kalshi market"
        m0 = markets[0]
        assert m0.venue == "kalshi"
        assert m0.market_id
        assert m0.outcomes == ("YES", "NO")
    finally:
        await adapter.close()


async def test_live_get_orderbook_has_decimal_prices(kalshi_env):
    adapter = KalshiAdapter(paper_mode=True)
    try:
        try:
            markets = await adapter.get_markets()
        except RateLimited:
            pytest.skip("Kalshi rate-limited this run")
        if not markets:
            pytest.skip("no open markets")
        # Pick a market with at least one side quoted. Try up to 40.
        book = None
        for m in markets[:40]:
            try:
                ob = await adapter.get_orderbook(m.market_id)
            except RateLimited:
                pytest.skip("Kalshi rate-limited this run")
            if ob.bids or ob.asks:
                book = ob
                break
        if book is None:
            pytest.skip("no market with any quote in first 40 results")

        assert book.venue == "kalshi"
        assert book.outcome_id == "YES"
        # All quoted prices are Decimal in (0, 1).
        levels = list(book.bids) + list(book.asks)
        assert levels, "expected at least one quote level"
        for lvl in levels:
            assert isinstance(lvl.price_prob, Decimal)
            assert Decimal("0") < lvl.price_prob < Decimal("1")
            assert lvl.size > 0
        # Bids descending, asks ascending.
        bid_prices = [lvl.price_prob for lvl in book.bids]
        ask_prices = [lvl.price_prob for lvl in book.asks]
        assert bid_prices == sorted(bid_prices, reverse=True)
        assert ask_prices == sorted(ask_prices)
    finally:
        await adapter.close()
