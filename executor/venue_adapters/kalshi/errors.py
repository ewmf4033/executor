"""
Map Kalshi HTTP/WebSocket errors to the 6 canonical executor exceptions.

Per Decision 1: InsufficientFunds, InvalidPrice, MarketClosed, RateLimited,
VenueDown, StaleQuote.

Kalshi error responses come in a few shapes; the most common is:
    {"error": {"code": "<short>", "message": "<human>"}}
Sometimes only "message" is present, sometimes top-level "code". Match
defensively on substrings of the message AND the structured code so a Kalshi
response-shape change does not silently drop into the wrong canonical bucket.
"""
from __future__ import annotations

from typing import Any

from ...core.types import (
    InsufficientFunds,
    InvalidPrice,
    MarketClosed,
    RateLimited,
    StaleQuote,
    VenueDown,
    VenueError,
)


def _extract(body: Any) -> tuple[str, str]:
    """Return (code, message) from a Kalshi error response. Both lowercase."""
    if not isinstance(body, dict):
        return "", str(body or "").lower()
    err = body.get("error")
    if isinstance(err, dict):
        code = str(err.get("code") or "").lower()
        msg = str(err.get("message") or "").lower()
        return code, msg
    code = str(body.get("code") or "").lower()
    msg = str(body.get("message") or body.get("error") or "").lower()
    return code, msg


# Substring matches against either code or message. First match wins.
_RULES: tuple[tuple[type[VenueError], tuple[str, ...]], ...] = (
    (InsufficientFunds, (
        "insufficient_funds", "insufficient funds", "not enough balance",
        "insufficient_balance", "not_enough_balance",
    )),
    (InvalidPrice, (
        "invalid_price", "price_out_of_range", "tick_size", "invalid price",
        "bad_price", "price must be", "yes_price", "no_price",
    )),
    (MarketClosed, (
        "market_closed", "market_not_open", "closed", "settled",
        "market_settled", "trading_halted",
    )),
    (RateLimited, ("rate_limit", "too many requests", "throttle")),
    (StaleQuote, ("stale_quote", "stale", "price_moved")),
    (VenueDown, ("internal_error", "service_unavailable", "bad_gateway", "gateway_timeout")),
)


def map_http_error(status: int, body: Any) -> VenueError:
    """
    Map an HTTP status + parsed JSON body to a canonical VenueError.

    Status precedence: 429 -> RateLimited, 5xx -> VenueDown unless the body's
    code/message points to something more specific. 4xx falls through to
    body-content matching, defaulting to InvalidPrice/VenueDown depending on
    code class.
    """
    code, msg = _extract(body)

    # Body-content rules first so a 400 with insufficient_funds is correctly tagged.
    for exc_cls, needles in _RULES:
        for n in needles:
            if n in code or n in msg:
                if exc_cls is RateLimited:
                    return RateLimited(message=str(body), retry_after_sec=None)
                return exc_cls(f"kalshi {status}: {msg or code or body!r}")

    # Status-class fallbacks.
    if status == 429:
        return RateLimited(message=str(body), retry_after_sec=None)
    if 500 <= status < 600:
        return VenueDown(f"kalshi {status}: {msg or body!r}")
    if status == 404:
        return MarketClosed(f"kalshi 404: {msg or body!r}")
    # Default 4xx: treat as invalid request shape.
    return InvalidPrice(f"kalshi {status}: {msg or body!r}")


def map_ws_error(msg: dict[str, Any]) -> VenueError:
    """Map a WebSocket {type:'error', msg:{...}} frame to a canonical error."""
    inner = msg.get("msg") if isinstance(msg.get("msg"), dict) else msg
    return map_http_error(status=400, body=inner)
