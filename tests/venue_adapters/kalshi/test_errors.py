"""
Unit tests for Kalshi error -> canonical VenueError mapping.

Each of the 6 canonical exceptions must be reachable via at least one
representative Kalshi error body. Status-class fallbacks (429 -> RateLimited,
5xx -> VenueDown) are covered where body is empty.
"""
from __future__ import annotations

import pytest

from executor.core.types import (
    InsufficientFunds,
    InvalidPrice,
    MarketClosed,
    RateLimited,
    StaleQuote,
    VenueDown,
)
from executor.venue_adapters.kalshi.errors import map_http_error


def test_insufficient_funds_from_code():
    err = map_http_error(400, {"error": {"code": "insufficient_funds", "message": "no cash"}})
    assert isinstance(err, InsufficientFunds)


def test_insufficient_funds_from_message():
    err = map_http_error(400, {"error": {"message": "Insufficient funds for order"}})
    assert isinstance(err, InsufficientFunds)


def test_invalid_price_from_code():
    err = map_http_error(400, {"error": {"code": "invalid_price", "message": "bad"}})
    assert isinstance(err, InvalidPrice)


def test_invalid_price_from_message_tick():
    err = map_http_error(400, {"error": {"message": "price must be on tick_size"}})
    assert isinstance(err, InvalidPrice)


def test_market_closed_from_code():
    err = map_http_error(400, {"error": {"code": "market_closed", "message": "nope"}})
    assert isinstance(err, MarketClosed)


def test_market_closed_via_404_fallback():
    err = map_http_error(404, {})
    assert isinstance(err, MarketClosed)


def test_rate_limited_from_status():
    err = map_http_error(429, {})
    assert isinstance(err, RateLimited)


def test_rate_limited_from_body():
    err = map_http_error(400, {"error": {"message": "too many requests"}})
    assert isinstance(err, RateLimited)


def test_venue_down_from_5xx_fallback():
    err = map_http_error(503, {})
    assert isinstance(err, VenueDown)


def test_venue_down_from_body_internal_error():
    err = map_http_error(500, {"error": {"code": "internal_error"}})
    assert isinstance(err, VenueDown)


def test_stale_quote_from_body():
    err = map_http_error(400, {"error": {"code": "stale_quote", "message": "price_moved"}})
    assert isinstance(err, StaleQuote)


def test_all_six_canonical_exceptions_reachable():
    """Sanity: every canonical exception has at least one mapping rule."""
    reached = {
        type(map_http_error(400, {"error": {"code": "insufficient_funds"}})),
        type(map_http_error(400, {"error": {"code": "invalid_price"}})),
        type(map_http_error(400, {"error": {"code": "market_closed"}})),
        type(map_http_error(429, {})),
        type(map_http_error(400, {"error": {"code": "stale_quote"}})),
        type(map_http_error(503, {})),
    }
    assert reached == {InsufficientFunds, InvalidPrice, MarketClosed, RateLimited, StaleQuote, VenueDown}
